import {
  balanced,
  bestByContext,
  bestByEfficiency,
  bestByThroughput,
  choose,
  explain,
  frontier,
  marginalCostOfContext,
  planHandheld,
} from '../efficiency';
import {ModelShape} from '../autotune';
import {SpeedShape} from '../speed';

const GB = 1e9;

const shape: ModelShape = {
  weightsBytes: 1.8 * GB,
  nLayers: 32,
  nHeadKv: 8,
  nEmbdHeadK: 128,
  nEmbdHeadV: 128,
  nVocab: 128000,
  nEmbd: 3072,
  trainedCtx: 32768,
};

const speed: SpeedShape = {
  totalParamsB: 3,
  activeParamsB: 3,
  bytesPerWeight: 0.6,
  nLayers: 32,
  nHeadKv: 8,
  nEmbdHeadK: 128,
  nEmbdHeadV: 128,
};

const device = (o: Record<string, unknown> = {}) =>
  ({
    memoryBandwidthGBs: 28,
    availableMemoryBytes: 4 * GB,
    cpuCores: 8,
    gpuBackend: 'none' as const,
    flashAttnSupported: false,
    ...o,
  } as Parameters<typeof planHandheld>[2]);

const points = () => frontier(shape, speed, {memoryBandwidthGBs: 28}, 4 * GB);

describe('RAM as effort', () => {
  it('only offers configurations that fit the budget', () => {
    for (const p of points()) {
      expect(p.ramBytes).toBeLessThanOrEqual(4 * GB);
    }
  });

  it('THE FINDING: the longest context is not the best return per gigabyte', () => {
    const pts = points();
    const eff = bestByEfficiency(pts)!;
    const ctx = bestByContext(pts)!;
    expect(eff.nCtx).toBeLessThan(ctx.nCtx);
    // Max context returns roughly a third of the tokens per GB.
    expect(eff.tokensPerSecondPerGB).toBeGreaterThan(ctx.tokensPerSecondPerGB * 2);
  });

  it('the longest context is also not the fastest', () => {
    const pts = points();
    expect(bestByThroughput(pts)!.tokensPerSecond).toBeGreaterThan(bestByContext(pts)!.tokensPerSecond);
  });

  it('balanced keeps throughput within a quarter of the fastest', () => {
    const pts = points();
    const b = balanced(pts)!;
    expect(b.tokensPerSecond).toBeGreaterThanOrEqual(bestByThroughput(pts)!.tokensPerSecond * 0.75);
    expect(b.nCtx).toBeGreaterThan(bestByThroughput(pts)!.nCtx);
  });

  it('prices a context doubling in both currencies', () => {
    const m = marginalCostOfContext(points(), 4096, 8192)!;
    expect(m.ramBytes).toBeGreaterThan(0);
    expect(m.speedLossRatio).toBeGreaterThan(0);
  });

  it('explains what it rejected', () => {
    const pts = points();
    expect(explain(balanced(pts)!, pts).join(' ')).toMatch(/tok\/s per GB/);
  });
});

describe('planHandheld', () => {
  it('defaults to balanced', () => {
    expect(planHandheld(shape, speed, device()).preference).toBe('balanced');
  });

  it('honours an explicit preference', () => {
    const fast = planHandheld(shape, speed, device(), 'speed');
    const long = planHandheld(shape, speed, device(), 'context');
    expect(fast.nCtx).toBeLessThan(long.nCtx);
    expect(fast.tokensPerSecond).toBeGreaterThan(long.tokensPerSecond);
  });

  it('never exceeds the memory it was given', () => {
    for (const avail of [2, 3, 4, 6, 8]) {
      const plan = planHandheld(shape, speed, device({availableMemoryBytes: avail * GB}));
      if (plan.viable) {
        expect(plan.ramBytes).toBeLessThanOrEqual(avail * GB);
      }
    }
  });

  it('says so plainly when nothing fits', () => {
    const plan = planHandheld({...shape, weightsBytes: 16 * GB}, speed, device({availableMemoryBytes: 2 * GB}));
    expect(plan.viable).toBe(false);
    expect(plan.reasons.join(' ')).toMatch(/No configuration/);
  });

  it('carries the GPU and batch decisions', () => {
    const plan = planHandheld(shape, speed, device({gpuBackend: 'vulkan'}));
    expect(plan.nGpuLayers).toBe(99);
    expect(plan.nBatch).toBeGreaterThan(0);
    expect(plan.useMmap).toBe(true);
  });
});
