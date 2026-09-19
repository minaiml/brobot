import {
  RUNTIME_MEMORY_MULTIPLIER,
  SAFETY_MARGIN_BYTES,
  assessFit,
  projectRuntimeMemory,
  MemorySnapshot,
} from '../modelFit';

const GB = 1000 * 1000 * 1000;

const snapshot = (overrides: Partial<MemorySnapshot> = {}): MemorySnapshot => ({
  totalBytes: 8 * GB,
  usedBytes: 4 * GB, // the OS has already taken half
  availableBytes: 4 * GB,
  freeDiskBytes: 32 * GB,
  ...overrides,
});

describe('projectRuntimeMemory', () => {
  it('applies the runtime multiplier to the file size', () => {
    expect(projectRuntimeMemory(2 * GB)).toBe(2 * GB * RUNTIME_MEMORY_MULTIPLIER);
  });
});

describe('assessFit', () => {
  it('plays a small model with room to spare', () => {
    const fit = assessFit(1 * GB, snapshot());
    expect(fit.canPlay).toBe(true);
    expect(fit.verdict).toBe('fits');
    expect(fit.headroomBytes).toBe(4 * GB - 1.5 * GB);
  });

  it('refuses a model that does not fit in AVAILABLE memory', () => {
    // 3 GB file -> 4.5 GB projected, but only 4 GB is free on an "8 GB phone"
    const fit = assessFit(3 * GB, snapshot());
    expect(fit.canPlay).toBe(false);
    expect(fit.verdict).toBe('too_large');
    expect(fit.reason).toMatch(/only 4\.0 GB is free/);
  });

  it('would have accepted that model if it trusted TOTAL memory — the bug this prevents', () => {
    const s = snapshot();
    expect(projectRuntimeMemory(3 * GB)).toBeLessThan(s.totalBytes); // 4.5 < 8
    expect(projectRuntimeMemory(3 * GB)).toBeGreaterThan(s.availableBytes); // 4.5 > 4
    expect(assessFit(3 * GB, s).canPlay).toBe(false);
  });

  it('warns when it fits but only barely', () => {
    // leave less than the safety margin free
    const free = 1.5 * GB + SAFETY_MARGIN_BYTES / 2;
    const fit = assessFit(1 * GB, snapshot({availableBytes: free}));
    expect(fit.canPlay).toBe(true);
    expect(fit.verdict).toBe('tight');
    expect(fit.reason).toMatch(/barely/);
  });

  it('refuses when there is not enough storage for the download', () => {
    const fit = assessFit(2 * GB, snapshot({freeDiskBytes: 1 * GB}), {requireStorage: true});
    expect(fit.canPlay).toBe(false);
    expect(fit.verdict).toBe('no_storage');
  });

  it('ignores storage unless asked', () => {
    const fit = assessFit(1 * GB, snapshot({freeDiskBytes: 0}));
    expect(fit.verdict).toBe('fits');
  });

  it('refuses an unknown size rather than guessing', () => {
    for (const bad of [0, -1, NaN, Infinity]) {
      const fit = assessFit(bad, snapshot());
      expect(fit.canPlay).toBe(false);
      expect(fit.verdict).toBe('unknown');
    }
  });

  it('always explains itself', () => {
    const cases = [
      assessFit(1 * GB, snapshot()),
      assessFit(9 * GB, snapshot()),
      assessFit(0, snapshot()),
    ];
    for (const fit of cases) {
      expect(fit.reason.length).toBeGreaterThan(10);
    }
  });
});
