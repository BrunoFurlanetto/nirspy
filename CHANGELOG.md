# Changelog

Todas as mudanças relevantes do projeto serão documentadas aqui.

Formato baseado em [Keep a Changelog](https://keepachangelog.com/),
versionamento por [SemVer](https://semver.org/).

## [Unreleased]

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
