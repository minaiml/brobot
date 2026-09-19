/**
 * BROBOT handheld module — the device side of the model player.
 *
 * What it adds to the app, and what it deliberately does not:
 *
 *   - autotune.ts  CHOOSES the configuration. The app could already price one
 *                  (memoryEstimator) and knew what the device had free
 *                  (ModelStore.availableMemoryCeiling), but cache_type_k/v were
 *                  hardcoded 'f16' at every default. Nothing picked them.
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
