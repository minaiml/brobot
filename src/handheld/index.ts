/**
 * BROBOT handheld module — the device side of the model player.
 *
 * What it adds to the app, and what it deliberately does not:
 *
 *   - autotune.ts  CHOOSES the configuration. The app could already price one
 *                  (memoryEstimator) and knew what the device had free
 *                  (ModelStore.availableMemoryCeiling), but cache_type_k/v were
 *                  hardcoded 'f16' at every default. Nothing picked them.
 *   - speed.ts     Encodes the MoE asymmetry: memory scales with TOTAL parameters,
 *                  speed scales with ACTIVE ones. Generation is memory-bandwidth
 *                  bound, so throughput is bandwidth / bytes-read-per-token.
 *   - efficiency.ts Treats RAM as effort and asks what each gigabyte returns.
 *                  planHandheld() is the entry point: it picks on the frontier
 *                  instead of maximising one axis.
 *   - tiers.ts     Classifies a device and says which models are worth offering,
 *                  and estimates a file size before anything is downloaded.
 *
 * Memory checking itself is NOT reimplemented here. memoryEstimator computes the
 * KV cache exactly from GGUF metadata, and useMemoryCheck compares it against a
 * ceiling calibrated from loads that actually succeeded. Both are better than a
 * static formula and are used as-is.
 *
 * Specification: https://github.com/minaiml/.github/blob/main/PLAYER.md
 */

export {
  autotune,
  chooseThreads,
  computeBufferBytes,
  kvCacheBytes,
  KV_BYTES,
  KV_LADDER,
} from './autotune';
export type {HandheldDevice, KvCacheType, ModelShape, TunePlan} from './autotune';

export {
  bytesPerToken,
  chooseBatchSizes,
  estimateSpeed,
  isMoE,
  shouldUseSpeculativeDraft,
} from './speed';
export type {SpeedDevice, SpeedEstimate, SpeedShape} from './speed';

export {
  balanced,
  bestByContext,
  bestByEfficiency,
  bestByThroughput,
  choose,
  explain,
  frontier,
  marginalCostOfContext,
  planHandheld,
} from './efficiency';
export type {EfficiencyPoint, HandheldPlan, Preference} from './efficiency';

export {TIERS, classifyDevice, estimateFileSizeBytes} from './tiers';
export type {
  DeviceSnapshot,
  EngineId,
  GpuBackend,
  ModelDescriptor,
  Quant,
  Tier,
  TierId,
} from './types';
