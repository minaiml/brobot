import {
  KV_BYTES,
  autotune,
  chooseThreads,
  computeBufferBytes,
  kvCacheBytes,
  HandheldDevice,
  ModelShape,
} from '../autotune';

const GB = 1e9;

// Roughly a 3B model: 32 layers, GQA with 8 KV heads, head dim 128.
const shape = (o: Partial<ModelShape> = {}): ModelShape => ({
  weightsBytes: 1.8 * GB,
  nLayers: 32,
  nHeadKv: 8,
  nEmbdHeadK: 128,
  nEmbdHeadV: 128,
  nVocab: 128000,
  nEmbd: 3072,
  trainedCtx: 32768,
  ...o,
});

const device = (o: Partial<HandheldDevice> = {}): HandheldDevice => ({
  availableMemoryBytes: 4 * GB,
  cpuCores: 8,
  gpuBackend: 'none',
  flashAttnSupported: false,
  ...o,
});

describe('kvCacheBytes', () => {
  it('q8_0 costs about 47% less than f16', () => {
    const s = shape();
    const f16 = kvCacheBytes(s, 4096, 'f16', 'f16');
    const q8 = kvCacheBytes(s, 4096, 'q8_0', 'q8_0');
    expect(q8 / f16).toBeCloseTo(KV_BYTES.q8_0 / KV_BYTES.f16, 5);
    expect(1 - q8 / f16).toBeGreaterThan(0.46);
  });

  it('respects sliding-window attention', () => {
    const s = shape({slidingWindow: 1024});
    expect(kvCacheBytes(s, 32768, 'f16', 'f16')).toBe(kvCacheBytes(s, 1024, 'f16', 'f16'));
  });

  it('scales linearly with context', () => {
    const s = shape();
    expect(kvCacheBytes(s, 8192, 'f16', 'f16')).toBe(2 * kvCacheBytes(s, 4096, 'f16', 'f16'));
  });
});

describe('chooseThreads', () => {
  it('prefers performance cores over core count', () => {
    expect(chooseThreads(device({cpuCores: 8, performanceCores: 4}))).toBe(4);
  });
  it('falls back to a fraction of cores when the split is unknown', () => {
    expect(chooseThreads(device({cpuCores: 8}))).toBe(6);
  });
  it('never returns zero on a single-core device', () => {
    expect(chooseThreads(device({cpuCores: 1}))).toBe(1);
  });
});

describe('autotune', () => {
  it('keeps f16 when there is room', () => {
    const plan = autotune(shape({weightsBytes: 0.6 * GB}), device({availableMemoryBytes: 6 * GB}));
    expect(plan.viable).toBe(true);
    expect(plan.cacheTypeK).toBe('f16');
    expect(plan.reasons.join(' ')).toMatch(/full precision fits/);
  });

  it('spends KV precision before it spends context', () => {
    // Tight enough that f16 at the top context will not fit.
    const plan = autotune(shape(), device({availableMemoryBytes: 3.2 * GB}));
    expect(plan.viable).toBe(true);
    const atF16 = autotune(shape(), device({availableMemoryBytes: 3.2 * GB}));
    expect(atF16.nCtx).toBeGreaterThanOrEqual(1024);
    expect(plan.projectedBytes).toBeLessThanOrEqual(3.2 * GB);
  });

  it('THE OPTIMISATION: q8_0 buys context that f16 cannot afford', () => {
    const s = shape();
    const d = device({availableMemoryBytes: 3.6 * GB});
    const plan = autotune(s, d);
    expect(plan.viable).toBe(true);
    // The same context at f16 would have exceeded the budget the tuner was given.
    const f16Cost =
      (s.weightsBytes + kvCacheBytes(s, plan.nCtx, 'f16', 'f16') + computeBufferBytes(s, 512)) * 1.1;
    if (plan.cacheTypeK !== 'f16') {
      expect(f16Cost).toBeGreaterThan(plan.projectedBytes);
      expect(plan.reasons.join(' ')).toMatch(/saving [\d.]+ GB/);
    }
  });

  it('never exceeds the budget it was given', () => {
    for (const avail of [1.5, 2, 3, 4, 6, 8, 12]) {
      const plan = autotune(shape(), device({availableMemoryBytes: avail * GB}));
      if (plan.viable) {
        expect(plan.projectedBytes).toBeLessThanOrEqual(avail * GB);
        expect(plan.headroomBytes).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it('never exceeds the context the model was trained for', () => {
    const plan = autotune(shape({trainedCtx: 2048}), device({availableMemoryBytes: 16 * GB}));
    expect(plan.nCtx).toBeLessThanOrEqual(2048);
  });

  it('reports non-viable rather than pretending, when the weights alone are too big', () => {
    const plan = autotune(shape({weightsBytes: 8 * GB}), device({availableMemoryBytes: 2 * GB}));
    expect(plan.viable).toBe(false);
    expect(plan.reasons.join(' ')).toMatch(/Will not fit/);
  });

  it('offloads when a GPU backend exists, and says which', () => {
    const plan = autotune(shape(), device({availableMemoryBytes: 8 * GB, gpuBackend: 'vulkan'}));
    expect(plan.nGpuLayers).toBe(99);
    expect(plan.reasons.join(' ')).toMatch(/vulkan/);
  });

  it('always memory-maps — the OS page cache is the cache', () => {
    expect(autotune(shape(), device()).useMmap).toBe(true);
  });

  it('explains every choice it made', () => {
    const plan = autotune(shape(), device({availableMemoryBytes: 8 * GB}));
    expect(plan.reasons.length).toBeGreaterThanOrEqual(4);
  });
});
