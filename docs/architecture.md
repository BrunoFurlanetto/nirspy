# Architecture — NIRSPY

> The architectural source of truth for the project. The implementation
> respects the three layers and the one-directional dependency rule.

## Three-layer view

```
┌─────────────────────────────────────────────────────────────┐
│  GUI (gui/)                                                 │
│  Dash + Plotly + dash-bootstrap-components                  │
│  ──────────────────────────────────────────────────────     │
│  Components, callbacks, layouts, visual drag-and-drop       │
│  Knows about: domain, engine                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Engine (engine/)                                           │
│  MNE-NIRS adapter                                           │
│  ──────────────────────────────────────────────────────     │
│  Implements the interfaces declared in domain               │
│  Knows about: domain, MNE-NIRS, MNE-Python                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Domain (domain/)                                           │
│  Pipeline, Block, DataType, validation, abstract execution  │
│  ──────────────────────────────────────────────────────     │
│  No UI or engine dependencies                               │
│  Knows about: nothing beyond stdlib + typing                │
└─────────────────────────────────────────────────────────────┘
```

**Golden rule of dependencies:** the arrow points downward. `gui` imports
`engine` and `domain`. `engine` imports `domain`. `domain` imports nothing
from the project.

This shape lets us swap the GUI (Reflex, PyQt) or add an alternative engine
(Cedalion) without rewriting the core. The rationale lives in ADR-005.

## Folder layout

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
│       ├── domain/                  # layer 1 — pure
│       │   ├── data_types.py        # Enum: Intensity, OpticalDensity, Hemoglobin, ...
│       │   ├── block.py             # Protocol Block, BlockSpec, BlockResult
│       │   ├── pipeline.py          # Pipeline (list today, DAG tomorrow)
│       │   ├── validation.py        # I/O type checks between blocks
│       │   └── execution.py         # topological order, result cache
│       ├── engine/                  # layer 2 — MNE-NIRS adapter
│       │   ├── mne_adapter.py
│       │   └── exceptions.py
│       ├── blocks/                  # concrete blocks
│       │   ├── load.py              # LoadSnirfBlock, LoadNirsBlock
│       │   ├── preprocessing.py     # OpticalDensity, BeerLambert, Bandpass
│       │   ├── motion.py            # TDDR, Spline, Wavelet
│       │   ├── quality.py           # SCI, PSP, ChannelPruning
│       │   ├── analysis.py          # BlockAverage, GLM
│       │   └── export.py            # ExportCSV, ExportSnirf, ExportReport
│       ├── io/                      # pipeline serialization
│       │   ├── yaml_serializer.py
│       │   └── json_serializer.py
│       ├── gui/                     # layer 3 — Dash app
│       │   ├── app.py               # creates Dash app, registers callbacks
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
│   ├── domain/                      # pure layer tests — fast
│   ├── engine/                      # adapter tests — Raw fixtures
│   ├── blocks/                      # per-block tests
│   ├── io/
│   └── gui/                         # GUI smoke tests
└── examples/
    ├── pipelines/                   # example pipeline YAMLs
    │   ├── best-practices-block-design.yml
    │   ├── resting-state.yml
    │   └── motion-heavy.yml
    └── data/                        # sample data (public SNIRF)
```

## Domain layer models

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

Each block declares `inputs: list[DataType]` and `outputs: list[DataType]`.
Validation guarantees compatibility before execution.

### `Block` (Protocol)

```python
from typing import Protocol, Any

class Block(Protocol):
    id: str                           # unique identifier within the pipeline
    name: str                         # display name
    inputs: list[DataType]
    outputs: list[DataType]
    params: dict[str, Any]            # UI-editable parameters

    def validate_params(self) -> list[str]: ...
    def execute(self, data: Any, context: ExecutionContext) -> Any: ...
```

`data` is typed at runtime by the `engine` (typically `mne.io.Raw`). The
`domain` layer treats it as `Any` so it does not need to import MNE.

### `Pipeline`

```python
from dataclasses import dataclass, field

@dataclass
class Pipeline:
    name: str
    blocks: list[Block] = field(default_factory=list)  # v0.1 linear list

    def validate(self) -> list[ValidationError]:
        """Block N's output type must be accepted by block N+1's input."""
        ...

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, data: dict) -> "Pipeline": ...
```

In v1.0 `blocks` becomes `nodes: dict[str, Block]` + `edges: list[tuple[str, str]]`
to support a graph. The public interface (`validate`, `to_dict`) is
preserved — the UI changes, the domain evolves without a breaking change.

## Execution flow

```
User drags blocks in the UI
        │
        ▼
GUI updates Pipeline (domain object) via callback
        │
        ▼
Pipeline.validate() — I/O type checks between consecutive blocks
        │
        ├── Error? → GUI shows inline red message, blocks execution
        │
        ▼ OK
User clicks "Run pipeline"
        │
        ▼
Engine creates ExecutionContext (cache, logger)
        │
        ▼
For each block in order:
        block.execute(previous_data, context)
        cache.store(block.id, result)
        │
        ▼
GUI consumes results from the cache to render (probe viewer, HRF, QC dashboard)
```

**Result cache:** `diskcache` indexed by hash of `(block_id, params, hash_of_inputs)`.
Changing a block's parameter invalidates that block's cache and every
downstream block, while keeping upstream caches intact. This enables fast
iteration without recomputing everything.

## Test strategy

| Layer        | Type                          | Speed       | What it tests                                       |
| ------------ | ----------------------------- | ----------- | --------------------------------------------------- |
| `domain/`    | Pure unit                     | <100 ms     | Validation, serialization, execution order          |
| `engine/`    | Integration with fixture      | ~seconds    | Adapter returns the correct `Raw` for each op       |
| `blocks/`    | Integration                   | ~seconds    | Each block produces the expected output on SNIRF    |
| `io/`        | Unit + golden file            | <100 ms     | YAML round-trip preserves the pipeline              |
| `gui/`       | Smoke (Dash test client)      | seconds     | App boots, main callback does not crash             |

Test data: public SNIRF files from the MNE-NIRS sample dataset (BSD-3,
redistributable).

## Packaging and distribution

| Item                  | Decision                                           |
| --------------------- | -------------------------------------------------- |
| Build backend         | `hatchling`                                        |
| Versioning            | SemVer (`0.1.0`, `0.2.0`, `1.0.0`)                 |
| Publication           | PyPI via `uv publish` in GitHub Actions            |
| Release trigger       | Git tag `v*.*.*`                                   |
| Documentation         | `mkdocs-material` on GitHub Pages                  |
| CI                    | GitHub Actions: ruff + mypy + pytest               |
| Minimum dependencies  | Python ≥3.10, mne-nirs ≥0.7, dash ≥2.x             |

## Future extension points (v2.0+)

- **Cedalion adapter:** new class in `engine/cedalion_adapter.py`,
  configuration in `config.engine`.
- **Graph builder:** swap `gui/components/pipeline_view.py` for
  `dash-cytoscape`; `Pipeline.blocks` evolves to a DAG.
- **Multi-subject pipelines:** new `batch/` module consuming the same YAML
  pipelines.
- **Plugin system:** custom blocks loaded via entry points in
  `pyproject.toml`.

## Related ADRs

- ADR-001 — Engine is MNE-NIRS, not Cedalion
- ADR-002 — GUI in Dash, not PyQt or Streamlit
- ADR-003 — Builder philosophy: linear → graph (Path C)
- ADR-005 — UI-agnostic domain layer
