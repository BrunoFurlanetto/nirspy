# CLAUDE.md — nirspy

> Configurações do projeto para o time de agentes.
> Complementa e sobrescreve o CLAUDE.md global quando necessário.

---

## Projeto

**Nome:** nirspy
**Descrição:** GUI fNIRS-first em Python — builder modular de pipeline sobre MNE-NIRS, distribuída como open source (BSD-3).
**Stack:** Python 3.10+ · Dash · Plotly · MNE-NIRS · uv · hatchling · pytest · ruff · mypy
**Repositório:** https://github.com/BrunoFurlanetto/nirspy.git
**Ambiente local:** `nirspy serve` → `http://127.0.0.1:8050`

---

## Vault (Obsidian)

**Path do projeto:** `Dev projects/nirspy`
**Memory:** `Dev projects/nirspy/Memory.md`
**Session Log:** `Dev projects/nirspy/session-log.md`
**Planejamentos:** `Dev projects/nirspy/features/`
**Visão e arquitetura:** `Dev projects/nirspy/visao-do-produto.md`, `Dev projects/nirspy/arquitetura.md`, `Dev projects/nirspy/roadmap.md`

### Abertura de sessão

Ler em paralelo antes de qualquer task:

```
mcp_obsidian: view  →  Dev projects/nirspy/Memory.md
mcp_obsidian: view  →  Dev projects/nirspy/session-log.md
mcp_obsidian: view  →  Dev projects/_memory.md      (ADRs globais)
```

Se a task envolve uma feature, ler também:
```
mcp_obsidian: view  →  Dev projects/nirspy/features/<nome-da-feature>.md
```

Sempre consultar `Dev projects/nirspy/arquitetura.md` antes de criar arquivos novos — a regra das três camadas (`domain` → `engine` → `gui`) é estrita.

### Fechamento de sessão

Seguir protocolo condicional do CLAUDE.md global. Quando registrar:

```
mcp_obsidian: str_replace  →  Dev projects/nirspy/session-log.md
```

ADR novo (decisão de design): adicionar **antes** de fechar log:
```
mcp_obsidian: str_replace  →  Dev projects/nirspy/Memory.md
```

---

## Branches protegidas

Nunca commitar diretamente nem criar branches a partir de:
- `main`

Orchestrator sempre confirma branch base com Lead antes de criar nova branch.

### Convenção de branches
```
feature/T-xxx-nome-curto
fix/T-xxx-descricao
chore/T-xxx-descricao
```

---

## Como rodar

```bash
# setup
uv venv
uv pip install -e ".[dev]"

# dev server
nirspy serve

# testes
pytest

# lint + types
ruff check .
mypy src/nirspy
```

---

## Ownership dos agentes

Arquitetura em três camadas (`domain` → `engine` → `gui`) determina ownership.

| Agente | Pode criar/editar | Somente leitura |
|--------|-------------------|-----------------|
| `dev` | `src/nirspy/gui/`, `src/nirspy/cli/`, `src/nirspy/blocks/` (lógica não-engine), `src/nirspy/io/`, `examples/`, `docs/` | `domain/`, `engine/` (precisa contrato dba) |
| `dba` | `src/nirspy/domain/`, `src/nirspy/engine/`, `src/nirspy/blocks/` (camada de execução MNE), `examples/pipelines/*.yml` (schema de pipeline) | `gui/`, `cli/` |
| `qa` | `tests/` | todo o código de produção |
| `security` | `docs/security/` (criar se necessário) | todo o codebase |
| `reviewer` | — | todo o codebase |

> Note: neste projeto `dba` cobre **camada de dados/domínio + adapter MNE-NIRS**, não banco relacional. Ele detém o contrato `Pipeline`/`Block`/`DataType` (ADR-005) e o adapter `engine/mne_adapter.py`. Qualquer mudança em estrutura de domínio passa por contrato dele antes do dev tocar na GUI.

---

## Convenções

### Commits

```
tipo(escopo): mensagem imperativa curta

Agente: <Nome> (<role>)
Task: T-xxx
```

Tipos válidos: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `style`

Escopos sugeridos: `domain`, `engine`, `blocks`, `gui`, `io`, `cli`, `ci`, `docs`.

### Estilo Python
- `ruff` configurado em `pyproject.toml` (E, F, I, B, UP, SIM)
- `mypy --strict` no pacote `nirspy`
- Linha máx 100 caracteres
- Type hints obrigatórios em código novo (camada `domain` em particular)

### Regra de ouro arquitetural (ADR-005)
- `domain/` **não importa** Dash, Plotly, MNE, MNE-NIRS — apenas stdlib + tipagem
- `engine/` importa `domain` + MNE/MNE-NIRS — nunca Dash
- `gui/` pode importar tudo do projeto
- Antes de criar arquivo novo, identificar a camada e respeitar a direção dos imports

### Pipelines como contrato
- Pipelines salvas em YAML são **API pública** do projeto
- Mudança breaking no schema de pipeline requer ADR e bump de minor (v0.x → v0.x+1)
- Round-trip YAML → Pipeline → YAML deve ser idêntico (golden tests em `tests/io/`)

### Open source
Toda decisão técnica deve considerar:
- Reprodutibilidade (mesma pipeline + mesmo SNIRF = mesmo resultado em qualquer máquina)
- Compatibilidade com Best Practices fNIRS (Yücel et al., 2021)
- Acessibilidade para usuários que não programam
- Manutenibilidade por dev solo

### Dependências
- Não adicionar dependência sem justificativa em PR
- Preferir biblioteca já presente no ecossistema MNE/scipy
- Nunca acoplar a algo GPL — projeto é BSD-3

---

## Stack-específico

### Estrutura de pastas (src layout)
```
src/nirspy/
├── domain/      # camada 1 — pura
├── engine/      # camada 2 — adapter MNE-NIRS
├── blocks/      # blocos concretos (load, OD, mBLL, motion, QC, analysis, export)
├── io/          # serializadores YAML/JSON de pipelines
├── gui/         # Dash app (components/, pages/, callbacks/)
└── cli/         # entry point `nirspy`
```

### Testes
- `tests/domain/` — unit puro, <100ms
- `tests/engine/` — integração com fixtures MNE
- `tests/blocks/` — integração por bloco
- `tests/io/` — round-trip golden files
- `tests/gui/` — smoke tests via `pytest-dash`

Dataset de referência: MNE-NIRS sample dataset (BSD-3, redistribuível).

### CI
GitHub Actions em `.github/workflows/ci.yml` — matrix Python 3.10/3.11/3.12, ruff + mypy + pytest.
