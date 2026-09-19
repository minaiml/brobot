/**
 * BROBOT handheld speed model.
 *
 * The asymmetry this module exists to encode, which comes straight out of the
 * mixture-of-experts work:
 *
 *     MEMORY scales with TOTAL parameters.
 *     SPEED  scales with ACTIVE parameters.
 *
 * Qwen3.5-35B-A3B holds 35B weights and wakes about 3B of them per token. It costs
 * a 35B model to hold and runs at roughly 3B speed. A player that sizes a model by
 * one number gets one of the two answers wrong, and on a handheld both matter.
 *
 * Token generation is memory-BANDWIDTH bound, not compute bound: each token reads
 * the active weights out of RAM, so throughput is roughly bandwidth divided by
 * bytes-read-per-token. Prompt processing (prefill) is the opposite — compute
 * bound, and it is what n_batch/n_ubatch are for.
 *
 * Specification: https://github.com/minaiml/.github/blob/main/PLAYER.md
 * Background:    https://github.com/minaiml/.github/blob/main/HYPOTHESIS.md
 */

import {KvCacheType, KV_BYTES} from './autotune';

export interface SpeedShape {
  /** Every parameter the model holds, in billions. Drives MEMORY. */
  totalParamsB: number;
  /** Parameters woken per token, in billions. Drives SPEED. Equals total for a dense model. */
  activeParamsB: number;
  /** Bytes per weight at the chosen quantization, e.g. 0.6 for Q4_K_M. */
  bytesPerWeight: number;
  nLayers: number;
  nHeadKv: number;
  nEmbdHeadK: number;
  nEmbdHeadV: number;
}

export interface SpeedDevice {
  /** Usable memory bandwidth, GB/s. Mid-range phones sit near 25-30; flagships 50+. */
  memoryBandwidthGBs: number;
  /** Fraction of peak bandwidth actually reachable. Real kernels never hit the spec sheet. */
  efficiency?: number;
  /** Device is already hot; sustained throughput is what the user experiences. */
  thermallyThrottled?: boolean;
}

export interface SpeedEstimate {
  /** Tokens per second during generation, the number a user feels. */
  tokensPerSecond: number;
  /** Bytes read from memory for each token produced. */
  bytesPerToken: number;
  /** True when the model wakes fewer parameters than it holds. */
  isMixtureOfExperts: boolean;
  /** How much faster this is than a dense model of the same total size. */
  moeSpeedup: number;
  notes: string[];
}

/** Sparse enough to be worth treating as MoE rather than dense. */
const MOE_ACTIVE_RATIO_THRESHOLD = 0.75;
/** Sustained clock under thermal pressure, as a fraction of burst. */
const THERMAL_DERATE = 0.7;
const DEFAULT_EFFICIENCY = 0.5;

export function isMoE(shape: SpeedShape): boolean {
  return shape.activeParamsB / shape.totalParamsB < MOE_ACTIVE_RATIO_THRESHOLD;
}

/**
 * Bytes pulled from memory to produce one token: the active weights, plus the KV
 * cache that attention has to read back across the whole context.
 */
export function bytesPerToken(
  shape: SpeedShape,
  nCtx: number,
  kv: KvCacheType,
): number {
  const weights = shape.activeParamsB * 1e9 * shape.bytesPerWeight;
  const kvRead =
    shape.nLayers * nCtx * (shape.nEmbdHeadK + shape.nEmbdHeadV) * shape.nHeadKv * KV_BYTES[kv];
  return weights + kvRead;
}

/**
 * Estimate generation throughput.
 *
 * This is a bandwidth model, not a benchmark. It is here so the player can rank
 * models and warn before a download, and it should be replaced by measured
 * telemetry the moment a real number exists for the device.
 */
export function estimateSpeed(
  shape: SpeedShape,
  device: SpeedDevice,
  nCtx: number,
  kv: KvCacheType,
): SpeedEstimate {
  const efficiency = device.efficiency ?? DEFAULT_EFFICIENCY;
  const thermal = device.thermallyThrottled ? THERMAL_DERATE : 1;
  const effectiveBandwidth = device.memoryBandwidthGBs * 1e9 * efficiency * thermal;

  const perToken = bytesPerToken(shape, nCtx, kv);
  const tokensPerSecond = effectiveBandwidth / perToken;

  const moe = isMoE(shape);
  const denseEquivalent =
    shape.totalParamsB * 1e9 * shape.bytesPerWeight + (perToken - shape.activeParamsB * 1e9 * shape.bytesPerWeight);
  const moeSpeedup = moe ? denseEquivalent / perToken : 1;

  const notes: string[] = [];
  if (moe) {
    notes.push(
      `Mixture of experts: ${shape.activeParamsB}B active of ${shape.totalParamsB}B held — ` +
        `costs a ${shape.totalParamsB}B model to hold, runs about ${moeSpeedup.toFixed(1)}x faster than one.`,
    );
  }
  if (device.thermallyThrottled) {
    notes.push('Device is hot; this is the sustained rate, not the burst rate.');
  }
  const kvShare = 1 - (shape.activeParamsB * 1e9 * shape.bytesPerWeight) / perToken;
  if (kvShare > 0.25) {
    notes.push(
      `At ${nCtx} tokens the KV cache is ${(kvShare * 100).toFixed(0)}% of the memory traffic — ` +
        'a shorter context would generate faster.',
    );
  }

  return {tokensPerSecond, bytesPerToken: perToken, isMixtureOfExperts: moe, moeSpeedup, notes};
}

/**
 * Prefill batch sizes.
 *
 * Prompt processing is compute bound, so a bigger batch is faster — but the compute
 * buffer is (n_vocab + n_embd) * n_ubatch * 4 bytes, so it is paid for in RAM. Small
 * devices take the smaller batch and the slower prefill; it is the difference between
 * a slow first token and no model at all.
 */
export function chooseBatchSizes(availableMemoryBytes: number): {
  nBatch: number;
  nUbatch: number;
  reason: string;
} {
  const gb = availableMemoryBytes / 1e9;
  if (gb >= 6) {
    return {nBatch: 2048, nUbatch: 512, reason: 'Large prefill batch — fastest time to first token.'};
  }
  if (gb >= 3) {
    return {nBatch: 1024, nUbatch: 256, reason: 'Moderate prefill batch, balancing first-token latency against memory.'};
  }
  return {nBatch: 512, nUbatch: 128, reason: 'Small prefill batch — slower first token, but it leaves room for the model.'};
}

/**
 * Should this model use a speculative draft?
 *
 * Speculative decoding runs a small draft model ahead of the real one and verifies
 * its guesses in a batch. For a DENSE model that is a clear win: verification reads
 * the same weights once for several tokens.
 *
 * For an MoE it is not. The flash-moe experiment log records MTP speculative decoding
 * as **break-even**, with the reason that MoE I/O scales per token in a way dense
 * models' does not — each speculated token routes to its own experts, so verifying a
 * batch of guesses touches a batch of different expert sets and the I/O saving never
 * materialises.
 *
 * So: draft dense models, do not draft MoE.
 */
export function shouldUseSpeculativeDraft(
  shape: SpeedShape,
  opts: {draftAvailable: boolean; memoryHeadroomBytes: number; draftSizeBytes?: number} = {
    draftAvailable: false,
    memoryHeadroomBytes: 0,
  },
): {use: boolean; reason: string} {
  if (!opts.draftAvailable) {
    return {use: false, reason: 'No draft model paired with this one.'};
  }
  if (isMoE(shape)) {
    return {
      use: false,
      reason:
        'Mixture-of-experts model: speculative decoding measured break-even on MoE, because each ' +
        'speculated token routes to its own experts and the I/O saving never arrives. Not worth the memory.',
    };
  }
  const needed = (opts.draftSizeBytes ?? 0) * 1.1;
  if (needed > opts.memoryHeadroomBytes) {
    return {
      use: false,
      reason: 'A draft model would not fit in the memory left after the main model loads.',
    };
  }
  return {use: true, reason: 'Dense model with room for a draft — speculative decoding should raise throughput.'};
}
