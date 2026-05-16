# Contributing to nirspy

Obrigado pelo interesse em contribuir. Este documento descreve o fluxo mínimo pra rodar o projeto, abrir issues e enviar PRs.

## Setup local

Pré-requisitos: Python 3.10+ e [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/BrunoFurlanetto/nirspy.git
cd nirspy
uv venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

uv pip install -e ".[dev]"
```

### Rodar testes, lint, types

```bash
pytest
ruff check .
mypy src/nirspy
```

### Rodar app (quando GUI existir)

```bash
nirspy serve
# acesse http://127.0.0.1:8050
```

## Issues

Antes de abrir issue:

1. Confira se já não existe issue similar aberta ou fechada.
2. Para bug: descreva passos pra reproduzir, comportamento esperado vs observado, versão do Python e do `nirspy`.
3. Para feature: confira o [roadmap](docs/roadmap.md) — pode já estar planejada. Caso não esteja, descreva o problema antes da solução.

## Branches

Sempre saindo de `main`:

```
feature/T-xxx-nome-curto
fix/T-xxx-descricao
chore/T-xxx-descricao
docs/T-xxx-descricao
```

`main` é protegida — sem commits diretos. Toda mudança via PR.

## Commits

Padrão Conventional Commits curto:

```
tipo(escopo): mensagem imperativa curta

Corpo opcional explicando o porquê.
```

Tipos: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `style`.
Escopos sugeridos: `domain`, `engine`, `blocks`, `gui`, `io`, `cli`, `ci`, `docs`.

## Pull Requests

1. Branch a partir de `main`.
2. Lint, types e testes passando localmente.
3. PR pra `main` com descrição clara: o que muda, por quê, como testar.
4. CI precisa passar (ruff + mypy + pytest matrix).
5. Aguardar revisão. Squash-merge é o padrão.

## Arquitetura

Antes de criar arquivo novo, leia [`docs/architecture.md`](docs/architecture.md). Regra de ouro:

- `domain/` → não importa Dash, Plotly, MNE
- `engine/` → importa `domain` + MNE/MNE-NIRS, não Dash
- `gui/` → pode importar tudo do projeto

PR que viole essa regra será rejeitado.

## Testes

- `tests/domain/` → unit puro, <100 ms
- `tests/engine/` e `tests/blocks/` → integração com fixtures
- `tests/io/` → round-trip golden file
- `tests/gui/` → smoke via `pytest-dash`

Cobertura mínima alvo no `domain/`: 80%.

## Conduta

Por participar deste projeto você concorda com o [Code of Conduct](CODE_OF_CONDUCT.md).

## Licença

Contribuições são aceitas sob a [licença BSD-3-Clause](LICENSE) do projeto.
