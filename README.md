# NIRSPY — NIRS Processing in Python

> GUI fNIRS-first em Python. Builder modular de pipeline rodando localmente no browser, construído como wrapper sobre [MNE-NIRS](https://mne.tools/mne-nirs/).

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)]()

## Status

⚠️ **Pre-alpha — em bootstrap.** Nenhum bloco implementado ainda. Veja o [roadmap](docs/roadmap.md) para a ordem de entrega planejada.

## Por quê

Pesquisadores que precisam de GUI para fNIRS hoje dependem do [Homer3](https://openfnirs.org/software/homer/) — MATLAB pago. No ecossistema Python, MNE-NIRS, Cedalion e pysnirf2 são excelentes mas exigem programação.

NIRSPY preenche essa lacuna: GUI fNIRS-first, open source (BSD-3), pipelines reproduzíveis em YAML, rodando localmente no browser via Dash.

## Roadmap (v0.1)

- Builder modular linear de pipeline (drag-and-drop)
- Quality Control automático (SCI/PSP/pruning)
- Visualização de probe (montage)
- Block Average / HRF
- Salvar/carregar pipelines em YAML

Detalhes em [`docs/architecture.md`](docs/architecture.md).

## Instalação (planejada)

```bash
pip install nirspy
nirspy serve
# acesse http://127.0.0.1:8050
```

## Desenvolvimento

```bash
git clone https://github.com/BrunoFurlanetto/nirspy.git
cd nirspy
uv venv
source .venv/bin/activate    # PowerShell: .venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"

pytest
ruff check .
mypy src/nirspy
```

## Arquitetura

Três camadas com dependências unidirecionais:

- `domain/` — modelos puros (Pipeline, Block, DataType). Sem imports de UI/engine.
- `engine/` — adapter para MNE-NIRS.
- `gui/` — Dash app.

Detalhes em [`docs/architecture.md`](docs/architecture.md).

## Licença

[BSD-3-Clause](LICENSE) — mesma do MNE-NIRS. Ver [`third_party/licenses/`](third_party/licenses/) para licenças de dependências redistribuídas.

## Agradecimentos

Construído sobre [MNE-NIRS](https://github.com/mne-tools/mne-nirs) e [MNE-Python](https://github.com/mne-tools/mne-python).
