import {TIERS, classifyDevice, estimateFileSizeBytes} from '../tiers';
import {DeviceSnapshot} from '../types';

const GB = 1000 * 1000 * 1000;

const dev = (o: Partial<DeviceSnapshot> = {}): DeviceSnapshot => ({
  availableMemoryBytes: 4 * GB,
  totalMemoryBytes: 8 * GB,
  freeDiskBytes: 32 * GB,
  cpuCores: 8,
  gpuBackend: 'none',
  platform: 'android',
  ...o,
});

describe('classifyDevice', () => {
  it('classifies on AVAILABLE memory, not the number on the box', () => {
    // An "8 GB phone" with 4 GB actually free is not a flagship.
    const tier = classifyDevice(dev({totalMemoryBytes: 8 * GB, availableMemoryBytes: 4 * GB}));
    expect(tier.id).toBe('capable');
    expect(tier.id).not.toBe('flagship');
  });

  it('puts a 2 GB-free device on the floor tier', () => {
    expect(classifyDevice(dev({availableMemoryBytes: 1.5 * GB})).id).toBe('floor');
  });

  it('lifts a borderline device when a GPU backend exists', () => {
    const noGpu = classifyDevice(dev({availableMemoryBytes: 5.5 * GB, gpuBackend: 'none'}));
    const gpu = classifyDevice(dev({availableMemoryBytes: 5.5 * GB, gpuBackend: 'vulkan'}));
    expect(noGpu.id).toBe('capable');
    expect(gpu.id).toBe('flagship');
  });

  it('will not let a GPU rescue a device that lacks the memory', () => {
    expect(classifyDevice(dev({availableMemoryBytes: 1.2 * GB, gpuBackend: 'vulkan'})).id).toBe(
      'floor',
    );
  });

  it('demotes a device with too few cores to sustain the tier', () => {
    expect(classifyDevice(dev({availableMemoryBytes: 8 * GB, cpuCores: 4})).id).toBe('capable');
  });

  it('every tier offers Q4_K_M, the 2026 mobile default', () => {
    for (const tier of Object.values(TIERS)) {
      expect(tier.quantLadder).toContain('Q4_K_M');
      expect(tier.expectedTokensPerSecond[0]).toBeLessThan(tier.expectedTokensPerSecond[1]);
    }
  });
});

describe('estimateFileSizeBytes', () => {
  it('estimates before a byte is downloaded', () => {
    // A 3B model at Q4_K_M lands near 1.8 GB.
    const bytes = estimateFileSizeBytes(3, 'Q4_K_M');
    expect(bytes).toBeGreaterThan(1.6e9);
    expect(bytes).toBeLessThan(2.0e9);
  });

  it('is monotonic across the quantization ladder', () => {
    const q2 = estimateFileSizeBytes(3, 'Q2_K');
    const q4 = estimateFileSizeBytes(3, 'Q4_K_M');
    const q8 = estimateFileSizeBytes(3, 'Q8_0');
    expect(q2).toBeLessThan(q4);
    expect(q4).toBeLessThan(q8);
  });

  it('sizes the 135M floor model at well under 200 MB', () => {
    expect(estimateFileSizeBytes(0.135, 'Q4_K_M')).toBeLessThan(200e6);
  });
});
