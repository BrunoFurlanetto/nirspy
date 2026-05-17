# Contributing to NIRSPY

Thanks for your interest in contributing. This document describes the minimal
flow for running the project, opening issues, and submitting pull requests.

## Local setup

Prerequisites: Python 3.10+. [`uv`](https://docs.astral.sh/uv/) is supported
but not required.

```bash
git clone https://github.com/BrunoFurlanetto/nirspy.git
cd nirspy
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

### Run tests, lint, types

```bash
pytest
ruff check .
mypy src/nirspy
```

### Run the app

```bash
nirspy serve
# open http://127.0.0.1:8050
```

## Issues

Before opening an issue:

1. Check that a similar issue is not already open or closed.
2. For bugs: describe reproduction steps, expected vs observed behaviour,
   Python version, and `nirspy` version.
3. For features: review the [roadmap](docs/roadmap.md) — it may already be
   planned. Otherwise describe the problem before the solution.

Use the [issue templates](.github/ISSUE_TEMPLATE/) — blank issues are
disabled.

## Branches

Always branched off `main`:

```
feature/T-xxx-short-name
fix/T-xxx-description
chore/T-xxx-description
docs/T-xxx-description
```

`main` is protected — no direct commits. All changes go through a PR.

## Commits

Short Conventional Commits style:

```
type(scope): short imperative subject

Optional body explaining the why.
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `style`.
Suggested scopes: `domain`, `engine`, `blocks`, `gui`, `io`, `cli`, `ci`, `docs`.

## Pull requests

1. Branch from `main`.
2. Lint, types and tests pass locally.
3. Open a PR against `main` with a clear description: what changes, why, and
   how to test.
4. CI must pass (ruff + mypy + pytest matrix on Python 3.10/3.11/3.12).
5. Wait for review. Squash-merge is the default.

## Architecture

Before creating new files, read [`docs/architecture.md`](docs/architecture.md).
The golden rule:

- `domain/` must not import Dash, Plotly, or MNE.
- `engine/` may import `domain` + MNE/MNE-NIRS, but not Dash.
- `gui/` may import everything from the project.

PRs that violate this rule will be rejected.

## Tests

- `tests/domain/` — pure unit tests, <100 ms.
- `tests/engine/` and `tests/blocks/` — integration with fixtures.
- `tests/io/` — round-trip golden files.
- `tests/gui/` — smoke tests via Dash test client.

Minimum coverage target for `domain/`: 80%.

## Code of conduct

By participating in this project you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

Contributions are accepted under the project's
[BSD-3-Clause license](LICENSE).
