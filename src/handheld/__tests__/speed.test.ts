import {
  bytesPerToken,
  chooseBatchSizes,
  estimateSpeed,
  isMoE,
  shouldUseSpeculativeDraft,
  SpeedDevice,
  SpeedShape,
} from '../speed';

/** A dense 3B at Q4_K_M. */
const dense = (o: Partial<SpeedShape> = {}): SpeedShape => ({
  totalParamsB: 3,
  activeParamsB: 3,
  bytesPerWeight: 0.6,
  nLayers: 32,
  nHeadKv: 8,
  nEmbdHeadK: 128,
  nEmbdHeadV: 128,
  ...o,
});

/** Qwen3.5-35B-A3B — the model Flash-iOS runs on a phone. */
const moe = (o: Partial<SpeedShape> = {}): SpeedShape =>
  dense({totalParamsB: 35, activeParamsB: 3, nLayers: 40, ...o});

const device = (o: Partial<SpeedDevice> = {}): SpeedDevice => ({
  memoryBandwidthGBs: 28,
  ...o,
});

describe('the MoE asymmetry', () => {
  it('recognises a sparse model', () => {
    expect(isMoE(dense())).toBe(false);
    expect(isMoE(moe())).toBe(true);
  });

  it('reads only the ACTIVE weights per token', () => {
    const d = bytesPerToken(dense(), 4096, 'q8_0');
    const m = bytesPerToken(moe(), 4096, 'q8_0');
    // 35B held, 3B active — traffic tracks the 3B, not the 35B.
    expect(Math.abs(m - d) / d).toBeLessThan(0.3);
  });

  it('runs far faster than a dense model it costs the same to hold', () => {
    const sparse = estimateSpeed(moe(), device(), 4096, 'q8_0');
    const heavy = estimateSpeed(
      dense({totalParamsB: 35, activeParamsB: 35, nLayers: 40}),
      device(),
      4096,
      'q8_0',
    );
    expect(sparse.tokensPerSecond).toBeGreaterThan(heavy.tokensPerSecond * 5);
    expect(sparse.notes.join(' ')).toMatch(/Mixture of experts/);
  });
});

describe('estimateSpeed', () => {
  it('lands in the published mid-range band for a 3B', () => {
    const {tokensPerSecond} = estimateSpeed(dense(), device(), 4096, 'q8_0');
    expect(tokensPerSecond).toBeGreaterThan(5);
    expect(tokensPerSecond).toBeLessThan(15);
  });

  it('reports the sustained rate when the device is hot', () => {
    const cool = estimateSpeed(dense(), device(), 4096, 'q8_0').tokensPerSecond;
    const hot = estimateSpeed(dense(), device({thermallyThrottled: true}), 4096, 'q8_0');
    expect(hot.tokensPerSecond).toBeLessThan(cool);
    expect(hot.notes.join(' ')).toMatch(/sustained rate/);
  });

  it('slows with context, and says why', () => {
    const short = estimateSpeed(dense(), device(), 1024, 'f16').tokensPerSecond;
    const long = estimateSpeed(dense(), device(), 32768, 'f16');
    expect(long.tokensPerSecond).toBeLessThan(short);
    expect(long.notes.join(' ')).toMatch(/of the memory traffic/);
  });

  it('makes a quantized KV cache a SPEED win, not only a memory one', () => {
    const f16 = estimateSpeed(dense(), device(), 32768, 'f16').tokensPerSecond;
    const q8 = estimateSpeed(dense(), device(), 32768, 'q8_0').tokensPerSecond;
    expect(q8 / f16).toBeGreaterThan(1.4); // ~49% at 32k
  });
});

describe('chooseBatchSizes', () => {
  it('trades first-token latency for memory on small devices', () => {
    expect(chooseBatchSizes(8e9).nBatch).toBeGreaterThan(chooseBatchSizes(2e9).nBatch);
    expect(chooseBatchSizes(2e9).reason).toMatch(/leaves room/);
  });
});

describe('shouldUseSpeculativeDraft', () => {
  const headroom = {draftAvailable: true, memoryHeadroomBytes: 4e9, draftSizeBytes: 3e8};

  it('drafts a dense model', () => {
    expect(shouldUseSpeculativeDraft(dense(), headroom).use).toBe(true);
  });

  it('refuses to draft an MoE — measured break-even in the flash-moe log', () => {
    const r = shouldUseSpeculativeDraft(moe(), headroom);
    expect(r.use).toBe(false);
    expect(r.reason).toMatch(/break-even/);
  });

  it('refuses when the draft would not fit', () => {
    const r = shouldUseSpeculativeDraft(dense(), {
      draftAvailable: true,
      memoryHeadroomBytes: 1e8,
      draftSizeBytes: 9e8,
    });
    expect(r.use).toBe(false);
  });

  it('is silent when no draft is paired', () => {
    expect(shouldUseSpeculativeDraft(dense()).use).toBe(false);
  });
});
