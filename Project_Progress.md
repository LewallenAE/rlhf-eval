# RLHF Eval Platform -- Project Progress

## CURRENT STATUS (keep this to ~15 lines)
- Last updated: 2026-02-01
- Operating mode: teach-first (source: AGENTS.md)
- Current module: 2 -- Database Layer
- Current milestone: 2.0 -- Dependencies + Toolchain (IN PROGRESS)
- Current objective (1 sentence): Lock down DB toolchain and dependencies, then implement async engine/session layer with minimal diffs.
- Last verification: `python -c "import rlhf_eval; print('ok')"`
- Evidence: `ok`
- Next single step: Confirm SQLAlchemy is installed and discover version.
- Verification command: `python -c "import sqlalchemy; print(sqlalchemy.__version__)"`
- Blockers: None

---

## Repo Facts (source of truth)
- Repo root: `rlhf-eval/`
- Python: 3.13.9 (venv)
- Package import: `rlhf_eval` (verified)
- Shell: PowerShell (primary) | Git Bash/MINGW (secondary)
- Dependency manager: UNKNOWN (pip/uv/poetry -- to be confirmed)
- Tests: UNKNOWN (to be confirmed)
- Lint/format: UNKNOWN (to be confirmed)
- Progress file: `Project_Progress.md`

---

## Project Overview
Building a production-grade RLHF evaluation harness/toolkit featuring:
- Detection logic for response quality evaluation
- Storage layer (PostgreSQL)
- CLI interface (Typer)
- REST API (FastAPI)
- Modern Python + clean architecture (minimal complexity, explicit design)

## Tech Stack (target)
- Python 3.13+
- PostgreSQL
- SQLAlchemy (async)
- Pydantic + pydantic-settings (settings/validation)
- Typer (CLI)
- FastAPI (API)

## Learning / Execution Approach
- SEE -> SAY -> DO micro-chunks
- Student runs commands + types small edits (unless IMPLEMENT selected)
- Verification after every change
- Minimal diffs; no broad refactors

---

## Roadmap and Milestones

### 1. Configuration Management (APPROVED)
- [x] Settings class with Pydantic
- [x] Environment variable loading
- [x] Validation logic
- Status: APPROVED

### 2. Database Layer (PLANNED -> IN PROGRESS)
Milestone 2.0: Dependencies + Toolchain
- [ ] Confirm dependency manager (pip/uv/poetry)
- [ ] Confirm SQLAlchemy installed (and version)
- [ ] Confirm async driver (asyncpg) installed
- [ ] Confirm migration tool (alembic) installed (if used)
- Status: IN PROGRESS

Milestone 2.1: Connection + Session Management (async)
- [ ] Async engine factory
- [ ] Async session factory
- [ ] Session lifecycle helper (DI-friendly)
- [ ] Clean shutdown
- Status: NOT STARTED

Milestone 2.2: Base Model + Mixins
- [ ] Declarative Base (SQLAlchemy 2.x style)
- [ ] Timestamp mixin (created_at, updated_at)
- [ ] UUID primary key mixin (if desired)
- Status: NOT STARTED

Milestone 2.3: Domain Models
- [ ] Evaluation model
- [ ] Response model
- [ ] Comparison model
- Status: NOT STARTED

Milestone 2.4: Repository Pattern
- [ ] Repository interfaces
- [ ] Concrete implementations
- [ ] Minimal tests
- Status: NOT STARTED

### 3. Core Domain Models
- [ ] Evaluation types
- [ ] Response models
- [ ] Comparison models
- Status: NOT STARTED

### 4. Detection/Scoring Logic
- [ ] Base scorer interface
- [ ] Concrete implementations
- [ ] Scorer registry
- Status: NOT STARTED

### 5. CLI Interface
- [ ] Typer app structure
- [ ] Commands
- [ ] Output formatting
- Status: NOT STARTED

### 6. API Layer
- [ ] FastAPI setup
- [ ] Endpoints
- [ ] Authentication
- Status: NOT STARTED

---

## Progress Snapshots (append-only)

### Snapshot -- 2026-02-01
- Timestamp: 2026-02-01
- Current goal: Initialize teaching workflow + verify environment + verify repo import
- What we verified (commands + results):
  - `python -c "import sys; print(sys.version)"`
    -> `3.13.9 (tags/v3.13.9:8183fa5, Oct 14 2025, 14:09:13) [MSC v.1944 64 bit (AMD64)]`
  - `python -c "import rlhf_eval; print('ok')"`
    -> `ok`
- What changed (files + brief summary):
  - `AGENTS.md` updated to enforce: source-of-truth mapping, no redundant verification, session-start behavior.
  - `Project_Progress.md` upgraded to: CURRENT STATUS + Repo Facts + milestone breakdown + evidence-driven snapshots.
- What’s working:
  - Codex can resume from CURRENT STATUS cleanly.
  - Repo package import works in venv.
- What’s not working / blockers:
  - None.
- Next single step:
  - Confirm SQLAlchemy presence/version: `python -c "import sqlalchemy; print(sqlalchemy.__version__)"`
