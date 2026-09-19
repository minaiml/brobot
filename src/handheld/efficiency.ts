/**
 * BROBOT handheld efficiency model — tokens returned per gigabyte spent.
 *
 * On a handheld, RAM is the scarce resource, so treat it as EFFORT and ask what
 * each gigabyte buys. The three things a gigabyte can be spent on do not pay back
 * the same way:
 *
 *   weights   cost RAM, buy quality, and cost speed (more bytes read per token)
 *   KV cache  cost RAM, buy context, and cost speed (KV is re-read every token)
 *   batch     costs RAM, buys prefill speed, and is free during generation
 *
 * KV cache is the one that surprises: it is the only spend that is charged twice,
 * once in RAM and again in bandwidth on every single token. Which means the
 * configuration that fits the MOST context is reliably NOT the configuration that
 * produces the most tokens.
 *
 * autotune() maximises context subject to fitting. That is the right answer when a
 * user wants a long conversation and the wrong one when they want a fast one, so
 * this module makes the trade explicit instead of hiding it behind a default.
 *
 * Specification: https://github.com/minaiml/.github/blob/main/PLAYER.md
 */

import {
  KvCacheType,
  KV_LADDER,
  ModelShape,
  chooseThreads,
  computeBufferBytes,
  kvCacheBytes,
} from './autotune';
import {SpeedDevice, SpeedShape, chooseBatchSizes, estimateSpeed} from './speed';

export type Preference = 'speed' | 'context' | 'balanced';

export interface EfficiencyPoint {
  nCtx: number;
  kv: KvCacheType;
  /** Total projected footprint, bytes. The effort. */
  ramBytes: number;
  /** Generation throughput at this configuration. The return. */
  tokensPerSecond: number;
  /** The efficiency metric: tokens per second, per gigabyte of RAM spent. */
  tokensPerSecondPerGB: number;
}

const RUNTIME_OVERHEAD = 1.1;
const GB = 1e9;
const CTX_LADDER = [32768, 16384, 8192, 4096, 2048, 1024];

/**
 * Every configuration that fits the budget, with what it costs and what it returns.
 */
export function frontier(
  shape: ModelShape,
  speed: SpeedShape,
  device: SpeedDevice,
  budgetBytes: number,
  nUbatch = 512,
): EfficiencyPoint[] {
  const points: EfficiencyPoint[] = [];
  const ctxCeiling = shape.trainedCtx;

  for (const nCtx of CTX_LADDER.filter(c => c <= ctxCeiling)) {
    for (const kv of KV_LADDER) {
      const ramBytes =
        (shape.weightsBytes + kvCacheBytes(shape, nCtx, kv, kv) + computeBufferBytes(shape, nUbatch)) *
        RUNTIME_OVERHEAD;
      if (ramBytes > budgetBytes) {
        continue;
      }
      const {tokensPerSecond} = estimateSpeed(speed, device, nCtx, kv);
      points.push({
        nCtx,
        kv,
        ramBytes,
        tokensPerSecond,
        tokensPerSecondPerGB: tokensPerSecond / (ramBytes / GB),
      });
    }
  }
  return points;
}

/** The configuration returning the most tokens per gigabyte of effort. */
export function bestByEfficiency(points: EfficiencyPoint[]): EfficiencyPoint | null {
  return points.reduce<EfficiencyPoint | null>(
    (best, p) => (!best || p.tokensPerSecondPerGB > best.tokensPerSecondPerGB ? p : best),
    null,
  );
}

/** The fastest configuration that fits, regardless of what it spends. */
export function bestByThroughput(points: EfficiencyPoint[]): EfficiencyPoint | null {
  return points.reduce<EfficiencyPoint | null>(
    (best, p) => (!best || p.tokensPerSecond > best.tokensPerSecond ? p : best),
    null,
  );
}

/** The longest context that fits — what autotune() alone would pick. */
export function bestByContext(points: EfficiencyPoint[]): EfficiencyPoint | null {
  return points.reduce<EfficiencyPoint | null>(
    (best, p) =>
      !best || p.nCtx > best.nCtx || (p.nCtx === best.nCtx && p.tokensPerSecond > best.tokensPerSecond)
        ? p
        : best,
    null,
  );
}

/**
 * A middle that refuses the two extremes: the longest context whose throughput is
 * still within `tolerance` of the fastest configuration available.
 *
 * Context is worth paying for, but not at any price. The default tolerance of 0.75
 * means: take all the context you can get, so long as it does not cost more than a
 * quarter of your speed.
 */
export function balanced(points: EfficiencyPoint[], tolerance = 0.75): EfficiencyPoint | null {
  const fastest = bestByThroughput(points);
  if (!fastest) {
    return null;
  }
  const floor = fastest.tokensPerSecond * tolerance;
  const acceptable = points.filter(p => p.tokensPerSecond >= floor);
  return bestByContext(acceptable) ?? fastest;
}

export function choose(points: EfficiencyPoint[], preference: Preference): EfficiencyPoint | null {
  switch (preference) {
    case 'speed':
      return bestByThroughput(points);
    case 'context':
      return bestByContext(points);
    case 'balanced':
    default:
      return balanced(points);
  }
}

/**
 * What the next doubling of context actually costs, in both currencies.
 *
 * This is the number that makes the trade legible: "4k more context costs you
 * 0.4 GB and 18% of your speed" is a decision a person can make. "Context: 8192"
 * is not.
 */
export function marginalCostOfContext(
  points: EfficiencyPoint[],
  fromCtx: number,
  toCtx: number,
): {ramBytes: number; speedLossRatio: number; worthIt: boolean} | null {
  const at = (ctx: number) =>
    points.filter(p => p.nCtx === ctx).sort((a, b) => b.tokensPerSecond - a.tokensPerSecond)[0];
  const a = at(fromCtx);
  const b = at(toCtx);
  if (!a || !b) {
    return null;
  }
  const speedLossRatio = 1 - b.tokensPerSecond / a.tokensPerSecond;
  return {
    ramBytes: b.ramBytes - a.ramBytes,
    speedLossRatio,
    // More than a third of throughput for a doubling is a bad trade on a handheld.
    worthIt: speedLossRatio < 0.33,
  };
}

/**
 * Human account of a chosen point against the alternatives it beat.
 */
export function explain(chosen: EfficiencyPoint, points: EfficiencyPoint[]): string[] {
  const fastest = bestByThroughput(points);
  const longest = bestByContext(points);
  const out: string[] = [
    `${chosen.nCtx} tokens of context, KV at ${chosen.kv}: ` +
      `${chosen.tokensPerSecond.toFixed(1)} tok/s for ${(chosen.ramBytes / GB).toFixed(2)} GB ` +
      `(${chosen.tokensPerSecondPerGB.toFixed(2)} tok/s per GB).`,
  ];
  if (longest && longest.nCtx > chosen.nCtx) {
    const lost = 1 - longest.tokensPerSecond / chosen.tokensPerSecond;
    out.push(
      `${longest.nCtx} tokens would fit, but costs ${(lost * 100).toFixed(0)}% of the speed ` +
        `and ${((longest.ramBytes - chosen.ramBytes) / GB).toFixed(2)} GB more.`,
    );
  }
  if (fastest && fastest.nCtx < chosen.nCtx) {
    const gain = fastest.tokensPerSecond / chosen.tokensPerSecond - 1;
    out.push(
      `Dropping to ${fastest.nCtx} tokens would be ${(gain * 100).toFixed(0)}% faster, ` +
        'if speed matters more than memory of the conversation.',
    );
  }
  return out;
}

/**
 * The complete handheld plan, chosen on the efficiency frontier rather than by
 * maximising any single axis.
 *
 * This supersedes calling autotune() alone. autotune() answers "what is the most
 * context that fits", which measurably is not "what produces the most tokens" —
 * on a 3B model in a 4 GB budget, the longest context returns 1.29 tok/s per GB
 * where the efficient point returns 3.29, and costs 38% of the throughput to get
 * there. The default here is `balanced`: all the context available that does not
 * cost more than a quarter of the speed.
 */
export interface HandheldPlan {
  nCtx: number;
  cacheTypeK: KvCacheType;
  cacheTypeV: KvCacheType;
  nThreads: number;
  nGpuLayers: number;
  nBatch: number;
  nUbatch: number;
  flashAttnType: 'auto' | 'on' | 'off';
  useMmap: boolean;
  ramBytes: number;
  tokensPerSecond: number;
  tokensPerSecondPerGB: number;
  preference: Preference;
  reasons: string[];
  viable: boolean;
}

export function planHandheld(
  shape: ModelShape,
  speed: SpeedShape,
  device: SpeedDevice & {
    availableMemoryBytes: number;
    cpuCores: number;
    performanceCores?: number;
    gpuBackend: 'vulkan' | 'opencl' | 'metal' | 'none';
    flashAttnSupported: boolean;
  },
  preference: Preference = 'balanced',
): HandheldPlan {
  const {nBatch, nUbatch, reason: batchReason} = chooseBatchSizes(device.availableMemoryBytes);
  const budget = device.availableMemoryBytes - 256 * 1024 * 1024;
  const points = frontier(shape, speed, device, budget, nUbatch);
  const nThreads = chooseThreads(device);
  const gpu = device.gpuBackend !== 'none';

  const base = {
    nThreads,
    nGpuLayers: gpu ? 99 : 0,
    nBatch,
    nUbatch,
    flashAttnType: (device.flashAttnSupported ? 'on' : 'auto') as 'auto' | 'on' | 'off',
    useMmap: true,
    preference,
  };

  const chosen = choose(points, preference);
  if (!chosen) {
    return {
      ...base,
      nCtx: 1024,
      cacheTypeK: 'q4_0',
      cacheTypeV: 'q4_0',
      ramBytes: 0,
      tokensPerSecond: 0,
      tokensPerSecondPerGB: 0,
      viable: false,
      reasons: ['No configuration of this model fits the memory this device has free.'],
    };
  }

  return {
    ...base,
    nCtx: chosen.nCtx,
    cacheTypeK: chosen.kv,
    cacheTypeV: chosen.kv,
    ramBytes: chosen.ramBytes,
    tokensPerSecond: chosen.tokensPerSecond,
    tokensPerSecondPerGB: chosen.tokensPerSecondPerGB,
    viable: true,
    reasons: [
      ...explain(chosen, points),
      batchReason,
      gpu ? `Offloading to ${device.gpuBackend}.` : 'CPU only; no GPU backend available.',
      'Memory-mapped weights; the OS page cache does the caching.',
    ],
  };
}
