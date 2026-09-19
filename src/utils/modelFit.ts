import DeviceInfo from 'react-native-device-info';

/**
 * Will this model actually play on this device?
 *
 * The rule BROBOT refuses to break: budget against the memory that is FREE,
 * never the memory printed on the box. An "8 GB phone" has already given
 * 3–4 GB to the OS before a single weight is loaded, so `getTotalMemory()`
 * — which is what `isHighEndDevice()` uses — systematically overpromises.
 *
 * A player that offers a model and then OOMs is worse than one that says no
 * and explains why, so every verdict here carries its reason.
 */

/**
 * Runtime memory is larger than the file on disk: KV cache, compute buffers
 * and allocator slack all sit on top of the weights. Published mid-range
 * Android measurement puts the practical figure at about 1.5x the file.
 */
export const RUNTIME_MEMORY_MULTIPLIER = 1.5;

/** Never hand the last of a device's memory to a model; the OS will reclaim it. */
export const SAFETY_MARGIN_BYTES = 512 * 1024 * 1024;

const GB = 1000 * 1000 * 1000;

export type FitVerdict = 'fits' | 'tight' | 'too_large' | 'no_storage' | 'unknown';

export interface ModelFit {
  /** Whether the player should offer this model at all. */
  canPlay: boolean;
  verdict: FitVerdict;
  /** One sentence, written for a person, explaining the verdict. */
  reason: string;
  /** Bytes of RAM the model is projected to need while running. */
  projectedRuntimeBytes: number;
  /** Bytes of RAM actually available right now. */
  availableMemoryBytes: number;
  /** Headroom after the model loads; negative means it cannot. */
  headroomBytes: number;
}

export interface MemorySnapshot {
  totalBytes: number;
  usedBytes: number;
  availableBytes: number;
  freeDiskBytes: number;
}

const gb = (bytes: number): string => `${(bytes / GB).toFixed(1)} GB`;

/**
 * What the device has right now — not what it was sold as.
 */
export async function getMemorySnapshot(): Promise<MemorySnapshot> {
  const [totalBytes, usedBytes, freeDiskBytes] = await Promise.all([
    DeviceInfo.getTotalMemory(),
    DeviceInfo.getUsedMemory(),
    DeviceInfo.getFreeDiskStorage(),
  ]);

  // getUsedMemory can exceed getTotalMemory on some Android builds; clamp
  // rather than report negative availability.
  const availableBytes = Math.max(0, totalBytes - usedBytes);

  return {totalBytes, usedBytes, availableBytes, freeDiskBytes};
}

/**
 * Projected RAM for a model of `fileSizeBytes` on disk.
 */
export function projectRuntimeMemory(fileSizeBytes: number): number {
  return Math.ceil(fileSizeBytes * RUNTIME_MEMORY_MULTIPLIER);
}

/**
 * Decide whether a model can play, given a file size and a memory snapshot.
 *
 * Pure so it can be tested without a device; `canModelPlay` wraps it for
 * runtime use.
 */
export function assessFit(
  fileSizeBytes: number,
  snapshot: MemorySnapshot,
  opts: {requireStorage?: boolean} = {},
): ModelFit {
  const projectedRuntimeBytes = projectRuntimeMemory(fileSizeBytes);
  const {availableBytes} = snapshot;
  const headroomBytes = availableBytes - projectedRuntimeBytes;

  const base = {projectedRuntimeBytes, availableMemoryBytes: availableBytes, headroomBytes};

  if (!Number.isFinite(fileSizeBytes) || fileSizeBytes <= 0) {
    return {
      ...base,
      canPlay: false,
      verdict: 'unknown',
      reason: 'The size of this model is unknown, so it cannot be checked against your free memory.',
    };
  }

  if (opts.requireStorage && snapshot.freeDiskBytes < fileSizeBytes) {
    return {
      ...base,
      canPlay: false,
      verdict: 'no_storage',
      reason: `Needs ${gb(fileSizeBytes)} of storage and only ${gb(snapshot.freeDiskBytes)} is free.`,
    };
  }

  if (headroomBytes < 0) {
    return {
      ...base,
      canPlay: false,
      verdict: 'too_large',
      reason:
        `Needs about ${gb(projectedRuntimeBytes)} of memory while running and only ` +
        `${gb(availableBytes)} is free. Try a smaller model or a lower quantization.`,
    };
  }

  if (headroomBytes < SAFETY_MARGIN_BYTES) {
    return {
      ...base,
      canPlay: true,
      verdict: 'tight',
      reason:
        `Will fit, but barely — about ${gb(headroomBytes)} to spare. Expect slowdowns, and ` +
        'close other apps before starting.',
    };
  }

  return {
    ...base,
    canPlay: true,
    verdict: 'fits',
    reason: `Fits with about ${gb(headroomBytes)} to spare.`,
  };
}

/**
 * Runtime convenience: read the device, then assess.
 */
export async function canModelPlay(
  fileSizeBytes: number,
  opts: {requireStorage?: boolean} = {},
): Promise<ModelFit> {
  const snapshot = await getMemorySnapshot();
  return assessFit(fileSizeBytes, snapshot, opts);
}
