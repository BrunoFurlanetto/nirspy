# Roadmap — NIRSPY

> Roadmap leve, **sem datas fixas**, em ordem de prioridade. Compromisso público é só de ordem, não de cronograma.

## v0.0 — Bootstrap (pré-código)

> Trabalho de fundação. Saída: repositório público com identidade clara, nada implementado ainda.

- [x] Criar repositório `github.com/BrunoFurlanetto/nirspy` (público)
- [x] Adicionar LICENSE BSD-3-Clause
- [x] README inicial (visão, status, roadmap)
- [x] `pyproject.toml` com metadados básicos e dependências planejadas
- [x] Estrutura de pastas conforme [`architecture.md`](architecture.md)
- [x] CI mínimo (GitHub Actions: ruff + mypy + pytest em PRs)
- [ ] Verificar disponibilidade do nome `nirspy` no PyPI
- [ ] CODE_OF_CONDUCT.md (Contributor Covenant)
- [ ] CONTRIBUTING.md (como rodar local, abrir issues, PRs)
- [ ] Branch protection na `main`

## v0.1 — MVP funcional

> Builder linear de pipeline cobrindo o caminho mínimo: **carregar SNIRF → QC → pré-processamento → HRF visual → exportar**.

**Domínio:**
- [ ] `DataType`, `Block` (Protocol), `Pipeline` (lista linear)
- [ ] Validação de tipos I/O entre blocos
- [ ] Execução topológica com cache (`diskcache`)

**Engine:**
- [ ] Adapter MNE-NIRS para operações básicas

**Blocos prioritários:**
- [ ] `LoadSnirf` — carrega arquivo SNIRF
- [ ] `OpticalDensity` — Intensity → OD
- [ ] `BeerLambert` — OD → HbO/HbR
- [ ] `BandpassFilter` — IIR padrão
- [ ] `ScalpCouplingIndex` — QC métrica
- [ ] `PrunChannels` — remove canais ruins
- [ ] `BlockAverage` — HRF por evento

**GUI:**
- [ ] Layout principal com sidebar de catálogo + área de pipeline
- [ ] Lista vertical reordenável de blocos (drag-and-drop)
- [ ] Card de bloco com painel de parâmetros expansível
- [ ] Indicador visual de tipos I/O incompatíveis
- [ ] Visualização de probe (sources/detectors no escalpo)
- [ ] Painel de QC (SCI/PSP/SNR por canal)
- [ ] Plot de HRF médio por condição

**IO:**
- [ ] Salvar pipeline como YAML
- [ ] Carregar pipeline de YAML

**CLI:**
- [ ] `nirspy serve` — sobe Dash em `127.0.0.1:8050`
- [ ] `nirspy --version`

**Distribuição:**
- [ ] Publicar no PyPI: `pip install nirspy`
- [ ] Tag `v0.1.0`
- [ ] Release notes no GitHub

## v0.2 — Robustez e Best Practices

> Foco: tornar robusto e bem testado o que já existe. Sem features novas grandes.

- [ ] Motion correction completa: TDDR, Spline, Wavelet (3 blocos)
- [ ] Templates de pipeline iniciais (YAML em `examples/`):
    - "Best Practices Block Design"
    - "Resting State Connectivity (preview)"
    - "Motion Heavy Recording"
- [ ] Tutorial guiado dentro da GUI (primeiros 5 passos)
- [ ] Cobertura de testes ≥80% no domínio
- [ ] Documentação `mkdocs-material` no GitHub Pages
- [ ] Mensagens de erro humanizadas (sem stack trace para o usuário final)

## v0.3 — Análise estatística

- [ ] Bloco `GLM` com short-channel regression
- [ ] Bloco `EpochsExtraction` com rejeição automática
- [ ] Plot de t-test por canal e ROI
- [ ] Exportação de resultados em CSV/Parquet
- [ ] Relatório HTML automático

## v0.4 — Batch e CLI estendida

- [ ] `nirspy run pipeline.yml --input data/*.snirf --output results/`
- [ ] Processamento em paralelo (multiprocessing)
- [ ] Relatório de QC consolidado para múltiplos sujeitos
- [ ] Integração com BIDS (Brain Imaging Data Structure)

## v1.0 — Builder de grafo

> Cumpre a promessa do Caminho C (ADR-003): substituto direto do Homer3 em Python.

- [ ] Migrar `Pipeline.blocks` (lista) para `Pipeline.nodes + edges` (DAG)
- [ ] Componente `dash-cytoscape` substitui lista vertical
- [ ] Validação de ciclos no grafo
- [ ] Suporte a ramificações (paralelo, condicional)
- [ ] Migração automática de pipelines YAML v0.x para formato v1.0
- [ ] Documentação migrada e versionada

## v2.0+ — Visão de longo prazo (sem compromisso)

> Ideias especulativas, sem garantia de implementação. Lista aberta a contribuições.

- Adapter para Cedalion como engine alternativo
- Plugin system: blocos custom via entry points
- Suporte a hyperscanning (dois sujeitos em paralelo)
- DOT (Diffuse Optical Tomography) básico
- Integração com EEG (multimodal)
- App desktop empacotado (PyInstaller ou Tauri)
- i18n da UI — começando com português

## Princípios para evolução

1. **Não inflar o escopo do MVP.** Cada feature adicionada antes de v0.1 atrasa o lançamento.
2. **Não quebrar pipelines salvas.** YAML v0.1 deve funcionar em v0.2, v0.3 etc. Quebra só em v1.0 com tooling de migração.
3. **Best Practices por padrão.** Templates e parâmetros padrões refletem consenso científico atual (Yücel et al., 2021).
4. **Comunicar incertezas.** Se um algoritmo tem trade-offs, a UI explica — não esconde.

## Métricas de progresso

| Sinal              | O que significa                                   |
| ------------------ | ------------------------------------------------- |
| Stars no GitHub    | Interesse passivo da comunidade                   |
| Issues abertas     | Engajamento ativo (positivo, mesmo se reclamação) |
| PRs externos       | Comunidade dispondo a contribuir                  |
| Downloads PyPI     | Adoção real                                       |
| Citações           | Adoção acadêmica formal                           |

> Métrica que **não** importa no início: número de features. Profundidade > superfície.
