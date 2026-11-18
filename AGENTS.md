# Repository instructions

- Configuration, detection, fitting policy, and batch processing belong to localize-psf.
  Keep this plugin limited to its UI, source-layer adapters, JSON export, and
  adapters to upstream tracking and motion-analysis packages.
- Localization settings export only JSON with a copyable batch command; never
  write shell scripts, launch batch processing, or own batch subprocesses.
- Full tracking may run Trackpy on CPU and save tracks and motion analyses as
  Parquet, PDF plots, and tracking metadata as JSON. Reuse upstream scientific algorithms.
- Resolve only Linux x86-64 and Windows x64, using the same declared GPU stack.
- Use uv and the locked localize-psf `opm-ptv` GPU and example dependencies.
- Follow upstream's Python minimum (currently >=3.12) without a plugin-specific
  minor-version ceiling or interpreter pin. User installation targets an existing
  active napari environment. Keep development and test dependencies in the root
  pyproject's dependency groups rather than a separate environment project.
- Keep napari, Qt/QtPy, npe2, and reader plugins out of the plugin's runtime
  dependencies. The test dependency group may include them. Preserve the bundled
  `napari.yaml` and `napari.manifest` entry point. Do not publish to PyPI.
- CUDA toolkits and runtime libraries must come from the installed uv environment;
  the NVIDIA driver is the host prerequisite. Never fall back to system CUDA.
- Reuse upstream readers, U-FISH, PSF models, ROI utilities, bounds validation,
  GPUfit orchestration, and uncertainty calculations. Do not duplicate algorithms.
- Add only unit and integration tests, executing real production code without
  mocks, stubs, spies, replacement modules, or fake GPU backends. Simulated input
  data are allowed; reuse production models to generate them.
- Do not add regression tests or regression suites. If planned or completed work
  violates these testing rules, stop and report the violation before continuing.
- Test plugin-owned behavior and package integration; do not duplicate upstream
  GPUfit solver, CUDA kernel, ABI, or installation validation tests.
- GitHub Actions installs only test/development dependencies and the plugin;
  upstream native backends must not be built or validated by these jobs.
- Keep helper functions minimal. Annotate production functions and document
  coordinate conventions, units, and interfaces.
