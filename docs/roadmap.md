# Roadmap — NIRSPY

> Lightweight roadmap, **no fixed dates**, ordered by priority. The public
> commitment is on ordering, not scheduling.

## v0.0 — Bootstrap (pre-code)

> Foundation work. Output: a public repository with clear identity, nothing
> implemented yet.

- [x] Create repository `github.com/BrunoFurlanetto/nirspy` (public)
- [x] Add LICENSE BSD-3-Clause
- [x] Initial README (vision, status, roadmap)
- [x] `pyproject.toml` with basic metadata and planned dependencies
- [x] Folder structure as described in [`architecture.md`](architecture.md)
- [x] Minimal CI (GitHub Actions: ruff + mypy + pytest on PRs)
- [x] CODE_OF_CONDUCT.md (Contributor Covenant)
- [x] CONTRIBUTING.md (local setup, opening issues, PRs)
- [ ] Verify `nirspy` name availability on PyPI
- [ ] Branch protection on `main`

## v0.1 — Functional MVP

> Linear pipeline builder covering the minimal path: **load SNIRF → QC →
> preprocessing → visual HRF → export**.

**Domain:**
- [x] `DataType`, `Block` (Protocol), `Pipeline` (linear list)
- [x] I/O type validation between blocks
- [x] Topological execution with cache (`diskcache`)

**Engine:**
- [x] MNE-NIRS adapter for the core operations

**Priority blocks:**
- [x] `LoadSnirf` — loads a SNIRF file
- [x] `OpticalDensity` — Intensity → OD
- [x] `BeerLambert` — OD → HbO/HbR
- [x] `BandpassFilter` — IIR with configurable cutoffs
- [x] `ScalpCouplingIndex` — QC metric
- [x] `PruneChannels` — removes bad channels
- [x] `BlockAverage` — HRF per event

**GUI:**
- [x] Main layout with catalog sidebar + pipeline area (5A)
- [ ] Reorderable vertical block list (drag-and-drop) — 5B
- [ ] Block card with expandable parameter panel — 5B
- [ ] Inline indicator for incompatible I/O types — 5B
- [ ] Probe visualization (sources/detectors on the scalp) — 5C
- [ ] QC panel (SCI/PSP/SNR per channel) — 5C
- [ ] Mean HRF plot per condition — 5C

**IO:**
- [x] Save pipeline as YAML
- [x] Load pipeline from YAML
- [x] `.nirs ↔ .snirf` converter

**CLI:**
- [x] `nirspy serve` — runs Dash on `127.0.0.1:8050`
- [x] `nirspy run pipeline.yml --input X --output Y`
- [x] `nirspy --version`

**Distribution:**
- [ ] Publish on PyPI: `pip install nirspy`
- [ ] Tag `v0.1.0`
- [ ] GitHub release notes

## v0.2 — Robustness and Best Practices

> Focus: harden what already exists. No big new features.

- [ ] Full motion correction: TDDR, Spline, Wavelet (3 blocks)
- [ ] Starter pipeline templates (YAML in `examples/`):
    - "Best Practices Block Design"
    - "Resting State Connectivity (preview)"
    - "Motion Heavy Recording"
- [ ] In-GUI guided tutorial (first 5 steps)
- [ ] Test coverage ≥80% on the domain layer
- [ ] `mkdocs-material` documentation on GitHub Pages
- [ ] Human-friendly error messages (no stack trace for end users)

## v0.3 — Statistical analysis

- [ ] `GLM` block with short-channel regression
- [ ] `EpochsExtraction` block with automatic rejection
- [ ] Per-channel and ROI t-test plot
- [ ] Export results as CSV/Parquet
- [ ] Automatic HTML report

## v0.4 — Batch and extended CLI

- [ ] `nirspy run pipeline.yml --input data/*.snirf --output results/`
- [ ] Parallel processing (multiprocessing)
- [ ] Consolidated QC report across multiple subjects
- [ ] BIDS (Brain Imaging Data Structure) integration

## v1.0 — Graph builder

> Delivers on the promise of Path C (ADR-003): a direct Python replacement
> for Homer3.

- [ ] Migrate `Pipeline.blocks` (list) to `Pipeline.nodes + edges` (DAG)
- [ ] `dash-cytoscape` component replaces the vertical list
- [ ] Cycle validation in the graph
- [ ] Support for branching (parallel, conditional)
- [ ] Automatic migration of YAML pipelines from v0.x to v1.0
- [ ] Versioned documentation

## v2.0+ — Long-term vision (no commitment)

> Speculative ideas, no guarantee of implementation. Open to contributions.

- Cedalion adapter as an alternative engine
- Plugin system: custom blocks via entry points
- Hyperscanning support (two subjects in parallel)
- Basic DOT (Diffuse Optical Tomography)
- EEG integration (multimodal)
- Packaged desktop app (PyInstaller or Tauri)
- UI i18n — starting with Portuguese

## Evolution principles

1. **Do not inflate MVP scope.** Every feature added before v0.1 delays
   release.
2. **Do not break saved pipelines.** YAML v0.1 must keep working in v0.2,
   v0.3, etc. Breaking changes only in v1.0 with migration tooling.
3. **Best practices by default.** Templates and default parameters reflect
   current scientific consensus (Yücel et al., 2021).
4. **Communicate uncertainty.** If an algorithm has trade-offs, the UI
   explains them rather than hiding them.

## Progress metrics

| Signal           | What it means                                       |
| ---------------- | --------------------------------------------------- |
| GitHub stars     | Passive community interest                          |
| Open issues      | Active engagement (positive, even when complaints)  |
| External PRs     | Community willing to contribute                     |
| PyPI downloads   | Real adoption                                       |
| Citations        | Formal academic adoption                            |

> The metric that does **not** matter early on: feature count. Depth over
> surface.
