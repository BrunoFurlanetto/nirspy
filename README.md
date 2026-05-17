# NIRSPY — NIRS Processing in Python

> GUI-first, fNIRS-focused pipeline builder in Python. Runs locally in your
> browser. Wraps [MNE-NIRS](https://mne.tools/mne-nirs/) so every pipeline
> remains reproducible and inspectable.

[![CI](https://github.com/BrunoFurlanetto/nirspy/actions/workflows/ci.yml/badge.svg)](https://github.com/BrunoFurlanetto/nirspy/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)]()

## Status

**Pre-alpha — v0.1.0 in progress.** Core pipeline engine, preprocessing,
quality control, and HRF block-average blocks are implemented and exercised
by 280+ tests. The CLI (`nirspy run`, `nirspy serve`) works end-to-end. The
Dash GUI foundation is in place; the visual block builder ships in the next
sub-release. See [`docs/roadmap.md`](docs/roadmap.md) for the full plan.

## Why

Researchers who need a GUI for fNIRS today rely on
[Homer3](https://openfnirs.org/software/homer/), which requires a paid MATLAB
license. The Python ecosystem already has excellent libraries — MNE-NIRS,
Cedalion, pysnirf2 — but every one of them assumes the user can program.

NIRSPY fills that gap: a GUI-first, open-source (BSD-3) tool with
reproducible YAML pipelines, running locally in the browser via Dash.

## Features (v0.1)

- Linear, modular pipeline builder (drag-and-drop, coming in 5B)
- Optical-density conversion + modified Beer–Lambert law
- Bandpass filtering (configurable cutoffs)
- Automated quality control: Scalp Coupling Index + channel pruning
- HRF Block Average across conditions
- Probe / montage visualization (planned for 5C)
- Save and reload pipelines as YAML (round-trip stable)
- `.nirs ↔ .snirf` converter

Architecture details: [`docs/architecture.md`](docs/architecture.md).

## Installation

NIRSPY is not yet on PyPI. Install from source:

```bash
git clone https://github.com/BrunoFurlanetto/nirspy.git
cd nirspy
python -m venv .venv
source .venv/bin/activate    # PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Once installed:

```bash
nirspy --version
nirspy serve                              # GUI on http://127.0.0.1:8050
nirspy run pipeline.yml --input data.snirf --output result.snirf
```

## Development

```bash
pytest                       # 280+ tests, ~5 s on a laptop
ruff check .                 # lint
mypy src/nirspy              # type check (strict)
```

The Python 3.10 / 3.11 / 3.12 matrix runs on every PR via GitHub Actions.

## Architecture

Three layers with strictly one-directional dependencies:

- `domain/` — pure types (`Pipeline`, `Block`, `DataType`). No UI / MNE imports.
- `engine/` — thin adapter over MNE-NIRS, plus the cache layer.
- `gui/` — the Dash application.

Pipelines are persisted as YAML and treated as a public API: a breaking
schema change bumps the minor version and ships an ADR.

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md)
first. Bug reports and feature requests use the
[issue templates](.github/ISSUE_TEMPLATE/).

## Security

Found a vulnerability? Do **not** open a public issue. Use the private
[security advisory flow](SECURITY.md).

## License

[BSD-3-Clause](LICENSE) — same license as MNE-NIRS. Third-party license
notices live in [`third_party/licenses/`](third_party/licenses/).

## Acknowledgements

Built on top of [MNE-NIRS](https://github.com/mne-tools/mne-nirs) and
[MNE-Python](https://github.com/mne-tools/mne-python).
