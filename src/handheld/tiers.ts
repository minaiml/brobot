import {DeviceSnapshot, Quant, Tier, TierId} from './types';

/**
 * Device tiers.
 *
 * The boundaries come from published mid-range Android measurement (2026): a 4 GB
 * phone runs roughly a 1B model, 6 GB runs 1.7B, 8 GB runs 3-4B, and 12 GB with a
 * GPU backend reaches 7-8B. Mid-range throughput lands at 5-15 tok/s.
 *
 * They are expressed against AVAILABLE memory, not installed memory, because the
 * OS has already taken 3-4 GB before the first weight loads.
 */
export const TIERS: Record<TierId, Tier> = {
  floor: {
    id: 'floor',
    label: 'Runs small models only',
    maxParamsB: 1.2,
    quantLadder: ['Q4_K_M', 'Q3_K_M', 'Q2_K'],
    expectedTokensPerSecond: [2, 6],
  },
  modest: {
    id: 'modest',
    label: 'Runs everyday models',
    maxParamsB: 2,
    quantLadder: ['Q4_K_M', 'Q3_K_M'],
    expectedTokensPerSecond: [5, 12],
  },
  capable: {
    id: 'capable',
    label: 'Runs mid-sized models',
    maxParamsB: 4,
    quantLadder: ['Q4_K_M', 'Q5_K_M', 'Q3_K_M'],
    expectedTokensPerSecond: [5, 15],
  },
  flagship: {
    id: 'flagship',
    label: 'Runs large models with GPU offload',
    maxParamsB: 8,
    quantLadder: ['Q4_K_M', 'Q5_K_M', 'Q6_K'],
    expectedTokensPerSecond: [10, 30],
  },
};

const GB = 1000 * 1000 * 1000;

/**
 * Classify a device from what it has free, not what it claims.
 *
 * A GPU backend lifts a borderline device one step, because Vulkan/OpenCL offload
 * is what makes the largest tier reachable at all — but it can never lift a device
 * that lacks the memory to hold the weights.
 */
export function classifyDevice(snapshot: DeviceSnapshot): Tier {
  const availGB = snapshot.availableMemoryBytes / GB;
  const hasGpu = snapshot.gpuBackend !== 'none';

  let id: TierId;
  if (availGB >= 8) {
    id = 'flagship';
  } else if (availGB >= 5) {
    id = hasGpu ? 'flagship' : 'capable';
  } else if (availGB >= 3) {
    id = 'capable';
  } else if (availGB >= 1.8) {
    id = 'modest';
  } else {
    id = 'floor';
  }

  // A device with very few cores cannot sustain the upper tiers regardless of RAM.
  if (snapshot.cpuCores <= 4 && (id === 'flagship' || id === 'capable')) {
    id = id === 'flagship' ? 'capable' : 'modest';
  }

  return TIERS[id];
}

/** Approximate bytes per parameter for each GGUF quantization. */
const BYTES_PER_PARAM: Record<Quant, number> = {
  Q2_K: 0.42,
  Q3_K_M: 0.49,
  Q4_K_M: 0.6,
  Q5_K_M: 0.71,
  Q6_K: 0.82,
  Q8_0: 1.06,
};

/**
 * Estimate the on-disk size of a model before it is downloaded, so the shelf can
 * refuse an impossible model without fetching a byte.
 */
export function estimateFileSizeBytes(paramsB: number, quant: Quant): number {
  return Math.round(paramsB * 1e9 * BYTES_PER_PARAM[quant]);
}
