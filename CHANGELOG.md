# Changelog

Todas as mudanças relevantes do projeto serão documentadas aqui.

Formato baseado em [Keep a Changelog](https://keepachangelog.com/),
versionamento por [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Camada `domain/`: `Pipeline`, `Block` Protocol, `DataType`, `BlockSpec`, `BlockResult`, `Context`, `CacheProtocol`, exceções (`NirspyError`, `ValidationError`, `ExecutionError`).
- Camada `engine/`: `MNEAdapter` com `load_snirf`, `raw_to_od`, `beer_lambert`, `bandpass_filter`, `scalp_coupling_index`, `prune_channels`, `block_average`. `DiskCacheAdapter` com `JSONDisk` (sem pickle, S-01).
- Blocos: `LoadSnirf`, `OpticalDensity`, `BeerLambert`, `BandpassFilter`, `ScalpCouplingIndex`, `PruneChannels`, `BlockAverage`.
- IO: `yaml_serializer.dump_pipeline`/`load_pipeline`, conversor `.nirs ↔ .snirf` (`io/converters.py`).
- CLI: `nirspy run pipeline.yml --input X --output Y`, `nirspy serve`, `nirspy --version`.
- GUI: Dash app factory, layout base de três painéis, callback de execução com background callback + diskcache.
- Segurança Etapa 5A: validação de path SNIRF (S-02), sample-count guard (S-001), bloqueio de h5py external links (S-002), `O_EXCL` em escrita (S-003), opção `strip_pii` em conversores (I-001), serialização JSON em cache (S-01).
- Templates de issue/PR, política de segurança, dependabot, cache pip e cobertura no CI.

### Changed
- CI: removido `pytest-dash` das dev deps (unmaintained, incompatível com selenium ≥4.10).
- CI: `mypy` agora bloqueia merge (era `continue-on-error`).
- `JSONDisk.store/fetch` delegam ao `Disk` base — corrige overwrite e persistência entre instâncias.

### Removed
- `pytest-dash` das dev dependencies.
