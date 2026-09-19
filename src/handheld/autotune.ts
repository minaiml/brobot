/**
 * BROBOT handheld autotuner.
 *
 * The app already knows how to PRICE a configuration: memoryEstimator computes the
 * KV cache exactly from GGUF metadata, and ModelStore calibrates a real
 * availableMemoryCeiling from loads that actually succeeded. What nothing does is
 * CHOOSE the configuration — `cache_type_k` and `cache_type_v` are hardcoded 'f16'
 * at every default, so the largest single lever on a handheld is never pulled.
 *
 * At f16 the KV cache costs 2.0 bytes per element; at q8_0 it costs 1.0625 — a 47%
 * reduction, which on a phone is frequently the difference between a 2k context and
 * a 16k one, or between a 1B model and a 3B one.
 *
 * This module is pure: it takes numbers and returns a plan with its reasoning, so
 * the decision is testable without a device. The caller supplies the device facts
 * from deviceCapabilities/ModelStore and applies the result to ContextInitParams.
 *
 * NOTE: autotune() maximises CONTEXT subject to fitting. Measured against the
 * efficiency frontier that is not the same as maximising tokens — see
 * efficiency.planHandheld(), which is the entry point callers should prefer.
 *
 * Specification: https://github.com/minaiml/.github/blob/main/PLAYER.md
 */

/** Mirrors llama.cpp cache types; values match memoryEstimator.getKVCacheTypeBytes. */
export type KvCacheType = 'f16' | 'q8_0' | 'q5_1' | 'q5_0' | 'q4_1' | 'q4_0';

export const KV_BYTES: Record<KvCacheType, number> = {
  f16: 2.0,
  q8_0: 1.0625,
  q5_1: 0.75,
  q5_0: 0.6875,
  q4_1: 0.625,
  q4_0: 0.5625,
};

/**
 * Preference order. q8_0 is effectively lossless for KV and is tried first; the
 * lower rungs trade measurable quality and are only reached under real pressure.
 */
export const KV_LADDER: KvCacheType[] = ['f16', 'q8_0', 'q5_1', 'q4_0'];

export interface ModelShape {
  /** Weights on disk, bytes. */
  weightsBytes: number;
  nLayers: number;
  nHeadKv: number;
  nEmbdHeadK: number;
  nEmbdHeadV: number;
  nVocab: number;
  nEmbd: number;
  /** Sliding-window attention bound (Gemma and friends), if the model has one. */
  slidingWindow?: number;
  /** Context the model was trained for; the tuner will not exceed it. */
  trainedCtx: number;
}

export interface HandheldDevice {
  /** Calibrated ceiling from ModelStore — bytes this device has actually proven it can give. */
  availableMemoryBytes: number;
  cpuCores: number;
  /** Cores that are not efficiency cores, when known. llama.cpp scales on these. */
  performanceCores?: number;
  gpuBackend: 'vulkan' | 'opencl' | 'metal' | 'none';
  flashAttnSupported: boolean;
}

export interface TunePlan {
  nCtx: number;
  cacheTypeK: KvCacheType;
  cacheTypeV: KvCacheType;
  nThreads: number;
  nGpuLayers: number;
  flashAttn: boolean;
  /** types.ts deprecates the flash_attn boolean in favour of this. */
  flashAttnType: 'auto' | 'on' | 'off';
  useMmap: boolean;
  /** Projected total footprint of this plan, bytes. */
  projectedBytes: number;
  /** Bytes left over after the plan loads. */
  headroomBytes: number;
  /** Ordered, human-readable account of every choice made. */
  reasons: string[];
  /** False when even the smallest viable plan does not fit. */
  viable: boolean;
}

/** Runtime overhead multiplier, matching memoryEstimator's RUNTIME_OVERHEAD. */
const RUNTIME_OVERHEAD = 1.1;
/** Never spend the last of the ceiling. */
const SAFETY_BYTES = 256 * 1024 * 1024;
/** Contexts worth offering, largest first. */
const CTX_LADDER = [32768, 16384, 8192, 4096, 2048, 1024];

export function kvCacheBytes(
  shape: ModelShape,
  nCtx: number,
  k: KvCacheType,
  v: KvCacheType,
): number {
  const effectiveCtx = shape.slidingWindow
    ? Math.min(nCtx, shape.slidingWindow)
    : nCtx;
  const key = shape.nLayers * effectiveCtx * shape.nEmbdHeadK * shape.nHeadKv * KV_BYTES[k];
  const val = shape.nLayers * effectiveCtx * shape.nEmbdHeadV * shape.nHeadKv * KV_BYTES[v];
  return key + val;
}

export function computeBufferBytes(shape: ModelShape, nUbatch: number): number {
  return (shape.nVocab + shape.nEmbd) * nUbatch * 4;
}

function projectBytes(
  shape: ModelShape,
  nCtx: number,
  k: KvCacheType,
  v: KvCacheType,
  nUbatch: number,
): number {
  const base =
    shape.weightsBytes + kvCacheBytes(shape, nCtx, k, v) + computeBufferBytes(shape, nUbatch);
  return base * RUNTIME_OVERHEAD;
}

/**
 * Threads: llama.cpp scales on performance cores, not on core count. Efficiency
 * cores drag the slowest thread and the whole step waits for it.
 */
export function chooseThreads(device: HandheldDevice): number {
  if (device.performanceCores && device.performanceCores > 0) {
    return Math.max(1, Math.min(device.performanceCores, 8));
  }
  const cores = device.cpuCores;
  return cores <= 4 ? Math.max(1, cores) : Math.max(1, Math.floor(cores * 0.75));
}

/**
 * Pick the largest context this device can actually hold, spending KV quantization
 * before spending context length.
 *
 * Order matters: dropping f16 -> q8_0 is close to free in quality and buys ~47% of
 * the KV cache back, so it is always tried before halving the context a user can
 * feel. Only if q8_0 at the smallest context still does not fit do the lossy rungs
 * come out.
 */
export function autotune(
  shape: ModelShape,
  device: HandheldDevice,
  opts: {nUbatch?: number; maxCtx?: number} = {},
): TunePlan {
  const nUbatch = opts.nUbatch ?? 512;
  const budget = device.availableMemoryBytes - SAFETY_BYTES;
  const reasons: string[] = [];

  const nThreads = chooseThreads(device);
  reasons.push(
    device.performanceCores
      ? `${nThreads} threads on ${device.performanceCores} performance cores (efficiency cores excluded — the slowest thread sets the pace).`
      : `${nThreads} threads from ${device.cpuCores} cores.`,
  );

  const gpu = device.gpuBackend !== 'none';
  const nGpuLayers = gpu ? 99 : 0;
  reasons.push(
    gpu
      ? `Offloading to ${device.gpuBackend}.`
      : 'No GPU backend available; running on CPU.',
  );

  const flashAttn = device.flashAttnSupported;
  if (flashAttn) {
    reasons.push('Flash attention on — less memory per token of context.');
  }

  const ctxCeiling = Math.min(opts.maxCtx ?? shape.trainedCtx, shape.trainedCtx);
  const ctxOptions = CTX_LADDER.filter(c => c <= ctxCeiling);
  if (ctxOptions.length === 0) {
    ctxOptions.push(Math.max(512, ctxCeiling));
  }

  // Context is the outer loop: for a given context, spend KV precision before
  // giving up length. A user feels a shorter conversation; q8_0 KV is close to
  // free. Walking KV outermost would have picked f16 at 2k over q8_0 at 16k.
  for (const nCtx of ctxOptions) {
    for (const kv of KV_LADDER) {
      const projectedBytes = projectBytes(shape, nCtx, kv, kv, nUbatch);
      if (projectedBytes <= budget) {
        const headroomBytes = budget - projectedBytes;
        if (kv !== 'f16') {
          const atF16 = projectBytes(shape, nCtx, 'f16', 'f16', nUbatch);
          const saved = atF16 - projectedBytes;
          reasons.push(
            `KV cache at ${kv} instead of f16, saving ${(saved / 1e9).toFixed(2)} GB — ` +
              `f16 would not have fit this context.`,
          );
        } else {
          reasons.push('KV cache at f16 — full precision fits.');
        }
        reasons.push(
          `Context ${nCtx} tokens${nCtx < ctxCeiling ? ` (model supports ${ctxCeiling})` : ''}.`,
        );
        reasons.push('Memory-mapped weights; the OS page cache does the caching.');
        return {
          nCtx,
          cacheTypeK: kv,
          cacheTypeV: kv,
          nThreads,
          nGpuLayers,
          flashAttn,
          flashAttnType: flashAttn ? 'on' : 'auto',
          useMmap: true,
          projectedBytes,
          headroomBytes,
          reasons,
          viable: true,
        };
      }
    }
  }

  const smallestCtx = ctxOptions[ctxOptions.length - 1];
  const floorKv = KV_LADDER[KV_LADDER.length - 1];
  const projectedBytes = projectBytes(shape, smallestCtx, floorKv, floorKv, nUbatch);
  reasons.push(
    `Will not fit: even ${smallestCtx} tokens of context with a ${floorKv} KV cache needs ` +
      `${(projectedBytes / 1e9).toFixed(2)} GB against a ${(budget / 1e9).toFixed(2)} GB budget.`,
  );
  return {
    nCtx: smallestCtx,
    cacheTypeK: floorKv,
    cacheTypeV: floorKv,
    nThreads,
    nGpuLayers,
    flashAttn,
    flashAttnType: flashAttn ? 'on' : 'auto',
    useMmap: true,
    projectedBytes,
    headroomBytes: budget - projectedBytes,
    reasons,
    viable: false,
  };
}
