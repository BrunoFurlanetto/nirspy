# Arquitetura — NIRSPY

> Fonte de verdade arquitetural do projeto. Implementação respeita as três camadas e a regra de dependências unidirecionais.

## Visão em 3 camadas

```
┌─────────────────────────────────────────────────────────────┐
│  GUI (gui/)                                                 │
│  Dash + Plotly + dash-bootstrap-components                  │
│  ──────────────────────────────────────────────────────     │
│  Componentes, callbacks, layouts, drag-and-drop visual      │
│  Conhece: domain, engine                                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Engine (engine/)                                           │
│  Adapter para MNE-NIRS                                      │
│  ──────────────────────────────────────────────────────     │
│  Implementa interfaces declaradas em domain                 │
│  Conhece: domain, MNE-NIRS, MNE-Python                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Domain (domain/)                                           │
│  Pipeline, Block, DataType, validação, execução abstrata    │
│  ──────────────────────────────────────────────────────     │
│  Sem dependências de UI ou de engine                        │
│  Conhece: nada além de stdlib + tipagem                     │
└─────────────────────────────────────────────────────────────┘
```

**Regra de ouro das dependências:** seta aponta para baixo. `gui` importa `engine` e `domain`. `engine` importa `domain`. `domain` não importa nada do projeto.

Esse desenho permite trocar GUI (Reflex, PyQt) ou adicionar engine alternativo (Cedalion) sem reescrever o core. Justificativa em ADR-005.

## Estrutura de pastas

```
nirspy/
├── pyproject.toml
├── README.md
├── LICENSE                          # BSD-3-Clause
├── CHANGELOG.md
├── third_party/
│   └── licenses/
│       ├── MNE-NIRS-LICENSE
│       └── MNE-Python-LICENSE
├── docs/                            # mkdocs-material
│   ├── index.md
│   ├── architecture.md
│   ├── roadmap.md
│   ├── tutorials/
│   └── reference/
├── src/
│   └── nirspy/
│       ├── __init__.py
│       ├── domain/                  # camada 1 — pura
│       │   ├── data_types.py        # Enum: Intensity, OpticalDensity, Hemoglobin, ...
│       │   ├── block.py             # Protocol Block, BlockSpec, BlockResult
│       │   ├── pipeline.py          # Pipeline (lista hoje, DAG amanhã)
│       │   ├── validation.py        # checagem de tipos I/O entre blocos
│       │   └── execution.py         # ordem topológica, cache de resultados
│       ├── engine/                  # camada 2 — adapter MNE-NIRS
│       │   ├── mne_adapter.py
│       │   └── exceptions.py
│       ├── blocks/                  # blocos concretos
│       │   ├── load.py              # LoadSnirfBlock, LoadNirsBlock
│       │   ├── preprocessing.py     # OpticalDensity, BeerLambert, Bandpass
│       │   ├── motion.py            # TDDR, Spline, Wavelet
│       │   ├── quality.py           # SCI, PSP, ChannelPruning
│       │   ├── analysis.py          # BlockAverage, GLM
│       │   └── export.py            # ExportCSV, ExportSnirf, ExportReport
│       ├── io/                      # serialização de pipelines
│       │   ├── yaml_serializer.py
│       │   └── json_serializer.py
│       ├── gui/                     # camada 3 — Dash app
│       │   ├── app.py               # cria app Dash, registra callbacks
│       │   ├── components/
│       │   │   ├── pipeline_view.py
│       │   │   ├── block_card.py
│       │   │   ├── probe_viewer.py
│       │   │   ├── qc_dashboard.py
│       │   │   └── hrf_plot.py
│       │   ├── pages/
│       │   ├── callbacks/
│       │   └── layouts.py
│       └── cli/
│           └── main.py              # entry point: nirspy serve / nirspy run
├── tests/
│   ├── domain/                      # testes da camada pura — rápidos
│   ├── engine/                      # testes do adapter — fixtures de Raw
│   ├── blocks/                      # testes de cada bloco
│   ├── io/
│   └── gui/                         # smoke tests da GUI
└── examples/
    ├── pipelines/                   # YAML de pipelines de exemplo
    │   ├── best-practices-block-design.yml
    │   ├── resting-state.yml
    │   └── motion-heavy.yml
    └── data/                        # dados de exemplo (SNIRF públicos)
```

## Modelos da camada de domínio

### `DataType`

```python
from enum import Enum

class DataType(Enum):
    INTENSITY = "intensity"           # raw light intensity
    OPTICAL_DENSITY = "od"            # log(I0/I)
    HEMOGLOBIN = "hbo_hbr"            # HbO + HbR concentrations
    EVOKED = "evoked"                 # block-averaged HRF
    GLM_RESULT = "glm"                # beta + statistics
```

Cada bloco declara `inputs: list[DataType]` e `outputs: list[DataType]`. Validação garante compatibilidade antes de executar.

### `Block` (Protocol)

```python
from typing import Protocol, Any

class Block(Protocol):
    id: str                           # identifier único na pipeline
    name: str                         # display name
    inputs: list[DataType]
    outputs: list[DataType]
    params: dict[str, Any]            # parâmetros editáveis na UI

    def validate_params(self) -> list[str]: ...
    def execute(self, data: Any, context: ExecutionContext) -> Any: ...
```

`data` é tipado em runtime pelo `engine` (tipicamente `mne.io.Raw`). A camada `domain` trata como `Any` para não importar MNE.

### `Pipeline`

```python
from dataclasses import dataclass, field

@dataclass
class Pipeline:
    name: str
    blocks: list[Block] = field(default_factory=list)  # v0.1 lista linear

    def validate(self) -> list[ValidationError]:
        """Tipo de saída do bloco N deve ser aceito pela entrada do bloco N+1."""
        ...

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, data: dict) -> "Pipeline": ...
```

No v1.0, `blocks` vira `nodes: dict[str, Block]` + `edges: list[tuple[str, str]]` para suportar grafo. Interface pública (`validate`, `to_dict`) permanece — UI muda, domínio evolui sem breaking change.

## Fluxo de execução

```
Usuário arrasta blocos na UI
        │
        ▼
GUI atualiza Pipeline (objeto domain) via callback
        │
        ▼
Pipeline.validate() — checagem de tipos I/O entre blocos consecutivos
        │
        ├── Erro? → GUI mostra inline em vermelho, bloqueia execução
        │
        ▼ OK
Usuário clica "Run pipeline"
        │
        ▼
Engine cria ExecutionContext (cache, logger)
        │
        ▼
Para cada bloco em ordem:
        block.execute(data_anterior, context)
        cache.store(block.id, resultado)
        │
        ▼
GUI consome resultados do cache para renderizar (probe viewer, HRF, QC dashboard)
```

**Cache de resultados:** `diskcache` indexado por hash de `(block_id, params, hash_dos_inputs)`. Mudar parâmetro de um bloco invalida cache desse bloco e dos posteriores, mantém cache dos anteriores. Permite iteração rápida sem recomputar tudo.

## Estratégia de testes

| Camada       | Tipo                    | Velocidade | O que testa                                      |
| ------------ | ----------------------- | ---------- | ------------------------------------------------ |
| `domain/`    | Unit puro               | <100ms     | Validação, serialização, ordem de execução       |
| `engine/`    | Integração com fixture  | ~segundos  | Adapter retorna `Raw` correto para cada op       |
| `blocks/`    | Integração              | ~segundos  | Cada bloco produz output esperado em SNIRF teste |
| `io/`        | Unit + golden file      | <100ms     | YAML round-trip preserva pipeline                |
| `gui/`       | Smoke (`pytest-dash`)   | minutos    | App sobe, callback principal não crasha          |

Dados de teste: SNIRF públicos do MNE-NIRS sample dataset (BSD-3, redistribuíveis).

## Packaging e distribuição

| Item                  | Decisão                                            |
| --------------------- | -------------------------------------------------- |
| Build backend         | `hatchling`                                        |
| Versionamento         | SemVer (`0.1.0`, `0.2.0`, `1.0.0`)                 |
| Publicação            | PyPI via `uv publish` em GitHub Actions            |
| Trigger de release    | Tag `v*.*.*` no Git                                |
| Documentação          | `mkdocs-material` em GitHub Pages                  |
| CI                    | GitHub Actions: ruff + mypy + pytest               |
| Dependências mínimas  | Python ≥3.10, mne-nirs ≥0.7, dash ≥2.x             |

## Pontos de extensão futura (v2.0+)

- **Adapter para Cedalion:** nova classe em `engine/cedalion_adapter.py`, configuração em `config.engine`
- **Builder de grafo:** trocar `gui/components/pipeline_view.py` por `dash-cytoscape`, `Pipeline.blocks` evolui para DAG
- **Pipelines em múltiplos sujeitos:** novo módulo `batch/` consumindo as mesmas pipelines YAML
- **Plugin system:** blocos custom carregáveis via entry points do `pyproject.toml`

## ADRs relacionados

- ADR-001 — Engine MNE-NIRS, não Cedalion
- ADR-002 — GUI Dash, não PyQt nem Streamlit
- ADR-003 — Filosofia builder modular linear → grafo (Caminho C)
- ADR-005 — Camada de domínio UI-agnóstica
