# Changelog

Todas as mudanças relevantes do projeto serão documentadas aqui.

Formato baseado em [Keep a Changelog](https://keepachangelog.com/),
versionamento por [SemVer](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-05-24

### Added
- **T-023 — PipelineRunner stepable** (PR #36): refactor `execution.py` into stepable executor supporting block-by-block interactive runs.
- **T-024 — ConditionGroup domain + engine** (PR #37): `ConditionGroup` dataclass, `BlockAverageParams` extension with `event_indices`, `create_epochs_per_group` in `MNEAdapter`, YAML round-trip.
- **T-025 — Condition groups editor** (PR #39): radio toggle HRF mode + builder editor for condition groups in ParamEditor.
- **T-027 — Run Interactive button + generic dialog** (PR #40): toggle button, per-block runtime dialog with ParamEditor.
- **T-028 — HRF specialized 2-stage dialog** (PR #41): grouped conditions + time windows per group in dedicated HRF dialog.
- **T-030 — Condition timeline selection** (PR #43): individual event occurrence selection via Plotly scatter timeline.
- **Probe distance check on .nirs→.snirf conversion**: warns about inter-optode distances outside physiological range during format conversion.

### Fixed
- Preserve `0.0` values in group time fields (falsy fallback bug).
- Auto-select new group + timeline-first card layout.
- Wire `snirf_path` through to T-030 timeline in builder.
- Use correct per-instance key for active group lookup.
- Clear condition windows/groups on SNIRF path change.
- Merge probe click callbacks to fix anchor race condition.

### Removed
- Probe head silhouette, 10-20 grid, and channel interaction (T-026/T-029) — reverted for redesign in future milestone.

## [0.2.0] - 2026-05-22

### Added
- **T-015 — Motion Correction: TDDR** (PR #25): `TDDRBlock` delegando a `mne_nirs.signal_enhancement.temporal_derivative_distribution_repair` (Fishburn et al., 2019).
- **T-016 — Motion Correction: Spline** (PR #28): `SplineBlock` + `SplineParams` (threshold z-score, spline_order), implementação custom em `MNEAdapter.spline_motion_correction` (Scholkmann et al., 2010).
- **T-017 — Motion Correction: Wavelet** (PR #29): `WaveletBlock` + `WaveletParams` (wavelet, iqr_multiplier), DWT via PyWavelets, implementação custom em `MNEAdapter.wavelet_motion_correction` (Molavi & Dumont, 2012). Validação de wavelet via `pywt.wavelist()`.
- **T-018 — Pipeline Templates** (PR #30): 3 YAMLs em `examples/pipelines/` (best-practices block design atualizado, resting-state-connectivity preview, motion-heavy-recording). Job CI `templates-smoke` gated por label `run-templates`. 9 integration tests parametrizados.
- **T-019 — Documentação mkdocs-material** (PR #26): site em `https://brunofurlanetto.github.io/nirspy/`, landing + getting-started + tutorial + reference auto-gerada. Workflow `docs.yml` para deploy automático.
- **T-020 — Mensagens de erro humanizadas** (PR #27): 13 subclasses de `NirspyError`, `UI_ERROR_MESSAGES` completo em EN (ADR-018), `get_user_message()` MRO-based, integração GUI execution + converter callbacks.
- **T-021 — Tutorial guiado na GUI** (PR #31): overlay modal sequencial com 5 passos, dbc.Switch + navegação, `assets/tutorial.css`.
- **ADR-024 — Filter bads no adapter** (PR #32): `MNEAdapter.average_epochs(*, filter_bads=True)` + `_drop_bads_from_evoked()` helper. Default True.
- **T-022 — Prune channels telemetry** (PR #33): metadata `n_bads_total`, `n_channels_total`, `fraction_bads` em `PruneChannelsBlock.run`, warning quando fração excede `bad_fraction_warning` (default 0.5).
- HRF plot: vrect overlay de região descartável (toggle + tmin/tmax), exclusão de bads do cálculo da média HRF.

### Changed
- `best-practices-block-design.yml`: reordenação científica (TDDR + Bandpass antes de SCI/Prune/BeerLambert), schema 0.1, baseline expandido (tmin -5, tmax 25).
- Sanitização de filenames em `store_input_file` via `os.path.basename` (SEC-INFO-01).
- Tutorial template loading sem `importlib.resources` traversal — usa `Path(__file__).resolve()` + cwd candidates (SEC-INFO-02).
- `MNEAdapter` motion correction: assert de shape antes de assign em `corrected._data` (SEC-INFO-03).

### Fixed
- mypy 3.10: `_label_segments` em `mne_adapter.py` agora declara `np.ndarray[Any, Any]` (CI fix).

## [0.1.0] - 2026-05-21

### Added
- Camada `domain/`: `Pipeline`, `Block` Protocol, `DataType`, `BlockSpec`, `BlockResult`, `Context`, `CacheProtocol`, exceções (`NirspyError`, `ValidationError`, `ExecutionError`).
- Camada `engine/`: `MNEAdapter` com `load_snirf`, `raw_to_od`, `beer_lambert`, `bandpass_filter`, `scalp_coupling_index`, `prune_channels`, `block_average`. `DiskCacheAdapter` com `JSONDisk` (sem pickle, S-01).
- Blocos: `LoadSnirf`, `OpticalDensity`, `BeerLambert`, `BandpassFilter`, `ScalpCouplingIndex`, `PruneChannels`, `BlockAverage`, `ManualChannelExclude`.
- IO: `yaml_serializer.dump_pipeline`/`load_pipeline`, conversor `.nirs ↔ .snirf` (`io/converters.py`), conversor Oxysoft `.txt → .snirf` (`io/oxysoft_txt.py`).
- CLI: `nirspy run pipeline.yml --input X --output Y`, `nirspy serve`, `nirspy --version`.
- GUI: Dash app factory, layout de três painéis, builder de pipeline (catalog + reorder + remove + toggle), param editor enriquecido (labels, tooltips, ranges, Optional checkbox, multiselect de canais), visualização de execução (raw, probe, QC heatmap, HRF μM), tab Convert (`.nirs↔.snirf` + Oxysoft `.txt→.snirf`).
- **T-012 — Janelas temporais por condição em `BlockAverage`**: novo dataclass `ConditionWindow`, parâmetro `per_condition_windows`, método `MNEAdapter.create_epochs_per_condition`, editor GUI auto-populado a partir do SNIRF, YAML round-trip retrocompatível.
- Segurança Etapa 5A: validação de path SNIRF (S-02), sample-count guard (S-001), bloqueio de h5py external links (S-002), `O_EXCL` em escrita (S-003), opção `strip_pii` em conversores (I-001), serialização JSON em cache (S-01).
- Templates de issue/PR, política de segurança, dependabot, cache pip e cobertura no CI.

### Changed
- CI: removido `pytest-dash` das dev deps (unmaintained, incompatível com selenium ≥4.10).
- CI: `mypy` agora bloqueia merge (era `continue-on-error`).
- `JSONDisk.store/fetch` delegam ao `Disk` base — corrige overwrite e persistência entre instâncias.
- `BlockAverageBlock.run`: keys desconhecidas em `per_condition_windows` viram `UserWarning` (em vez de raise) — pipeline robusta a troca de SNIRF com conjunto de condições diferente.

### Removed
- `pytest-dash` das dev dependencies.
