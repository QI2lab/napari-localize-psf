# Repository instructions

- Keep changes local to this repository. Do not edit localize-psf or its batch example.
- Export JSON and provide a copyable batch command; never launch batch processing
  or own batch subprocesses from the plugin.
- Resolve only Linux x86-64 and Windows x64, using the same declared GPU stack.
- Use uv and the locked localize-psf `opm-ptv` GPU and example dependencies.
- Follow upstream's Python minimum (currently >=3.12) without a plugin-specific
  minor-version ceiling or interpreter pin. User installation targets an existing
  active napari environment; the separate host project is for local development.
- Keep napari, Qt/QtPy, npe2, and reader plugins out of the plugin's runtime
  dependencies. The independent `environments/napari` host provides them;
  test-only dependency groups may also include them. Preserve the bundled
  `napari.yaml` and `napari.manifest` entry point. Do not publish to PyPI.
- CUDA toolkits and runtime libraries must come from the installed uv environment;
  the NVIDIA driver is the host prerequisite. Never fall back to system CUDA.
- Reuse upstream readers, U-FISH, PSF models, ROI utilities, bounds validation,
  GPUfit orchestration, and uncertainty calculations. Do not duplicate algorithms.
- Add only unit and integration tests, executing real production code without
  mocks, stubs, spies, replacement modules, or fake GPU backends. Simulated input
  data are allowed; reuse production models to generate them.
- Test plugin-owned behavior and package integration; do not duplicate upstream
  GPUfit solver, CUDA kernel, ABI, or installation validation tests.
- Keep helper functions minimal. Annotate production functions and document
  coordinate conventions, units, and interfaces.
- Keep the batch handoff aligned with the exported configuration schema.
