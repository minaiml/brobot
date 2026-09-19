/**
 * BROBOT handheld module — the device side of the model player.
 *
 * Everything here is pure and data-driven so a decision can be tested without a
 * phone in hand. The thin layer that actually reads hardware lives in probe.ts.
 *
 * Specification: https://github.com/minaiml/.github/blob/main/PLAYER.md
 */

/** What the device can actually be asked to do. Derived, never advertised. */
export type TierId = 'floor' | 'modest' | 'capable' | 'flagship';

/** GPU compute path available to an inference engine. */
export type GpuBackend = 'vulkan' | 'opencl' | 'metal' | 'none';

/** GGUF quantization levels the player will choose between. */
export type Quant = 'Q2_K' | 'Q3_K_M' | 'Q4_K_M' | 'Q5_K_M' | 'Q6_K' | 'Q8_0';

/** An engine that can be asked to play a model. */
export type EngineId = 'mnn' | 'llama.rn' | 'mediapipe' | 'executorch';

export interface DeviceSnapshot {
  /** Bytes of RAM free right now — the only memory figure that may drive a decision. */
  availableMemoryBytes: number;
  /** Bytes of RAM the device advertises. Recorded for telemetry, never for budgeting. */
  totalMemoryBytes: number;
  freeDiskBytes: number;
  cpuCores: number;
  gpuBackend: GpuBackend;
  /** e.g. "Snapdragon 695". Free-text, for the record only. */
  chipset?: string;
  platform: 'android' | 'ios';
}

export interface Tier {
  id: TierId;
  /** Human sentence for the UI. */
  label: string;
  /** Largest parameter count, in billions, this tier should be offered. */
  maxParamsB: number;
  /** Quantizations to try, best-fitting first. */
  quantLadder: Quant[];
  /** Realistic throughput range on this tier, tokens/second. */
  expectedTokensPerSecond: [number, number];
}

export interface ModelDescriptor {
  id: string;
  name: string;
  /** Parameters in billions. 0.135 for a 135M model. */
  paramsB: number;
  /** Where the weights come from. */
  repo: string;
  quant: Quant;
  /** Exact size when known; otherwise it is estimated from paramsB + quant. */
  fileSizeBytes?: number;
  /** Bundled with the app and always playable. */
  preloaded?: boolean;
  /** Formats the model is published in, to match against engines. */
  formats?: Array<'gguf' | 'mnn' | 'task' | 'pte'>;
}
