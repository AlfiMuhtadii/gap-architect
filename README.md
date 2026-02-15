# Gap Architect (Option B)
Full-stack MVP for resume vs JD gap analysis using Next.js + FastAPI + PostgreSQL.

## TL;DR Architecture
This MVP is intentionally built as a deterministic decision system augmented by AI, ensuring stable scoring, idempotent caching, and recoverable async processing within true MVP scope.

- AI performs structured skill extraction from Resume/JD.
- Backend deterministically verifies AI extraction (canonical mapping + text evidence) and computes final diff from verified sets.
- `missing_skills`, `match_percent`, `match_reason` are deterministic from verified skill sets.
- AI generates guidance content (`action_steps`, `interview_questions`, `roadmap_markdown`) under schema validation.
- Caching is idempotent by normalized fingerprint + unique DB constraint.
- Async lifecycle is explicit: `PENDING -> DONE/FAILED_VALIDATION/FAILED_LLM/FAILED_TIMEOUT`.

## End-to-End Flow
1. Submit Resume + JD to `POST /api/v1/gap-analyses`.
2. Normalize input, validate quality, compute fingerprint, check cache.
3. If cache hit `DONE`: return result immediately.
4. If miss: create `PENDING`, enqueue background processing.
5. Worker runs AI extraction -> deterministic verification/normalization -> final diff/scoring -> constrained generation.
6. Persist result and expose status via `GET /api/v1/gap-analyses/{id}`.

## Why This Design
- Stable scoring: deterministic decision layer prevents LLM math drift.
- Spec compliance: AI performs extraction; backend validates and decides final scoring deterministically.
- Reliability: retry/fallback chain + timeout sweeper avoids stuck `PENDING`.
- Auditability: `llm_runs` + explicit statuses + persisted error messages.

## Production Boundary
This submission focuses on production-aware MVP reliability:
deterministic scoring, idempotent caching, async lifecycle, and timeout recovery.

Enterprise durability features (distributed queues, shared cache, full metrics/alerting pipelines) are intentionally out of scope to keep implementation aligned with assignment requirements.

## Domain Scope
This MVP targets software engineering and technical roles,
where deterministic skill normalization and scoring are reliable.

Extending coverage to additional domains is supported
through taxonomy expansion without changing the scoring pipeline.

## Edge Cases Handled
- Malformed AI output: strict parse + schema validation + repair attempt.
- Provider issues: fallback (`primary -> local_llm -> heuristic`).
- Long/noisy JD: adaptive clipping.
- Duplicate concurrent requests: race-safe fingerprint flow and in-flight guard.
- Stuck jobs: TTL-based transition to `FAILED_TIMEOUT`.

## Data & Async Structure
- Core tables:
  - `gap_analyses`: fingerprint, status, error.
  - `gap_results`: final missing skills, score/reason, roadmap outputs.
  - `llm_runs`: provider/model/request/response/status/duration.
  - `jd_clean_runs`: cleaning strategy audit.
- Async behavior:
  - API returns non-blocking.
  - Worker concurrency capped by `MAX_CONCURRENT_GAP_JOBS`.
  - Stuck TTL configured by `PENDING_TIMEOUT_SECONDS`.
  - Retry backoff configured by `RETRY_COOLDOWN_SECONDS`.

## Concurrency & Test Guarantees
The system enforces idempotent and race-safe processing for identical submissions:

- A unique fingerprint constraint ensures only one analysis row per normalized Resume+JD.
- Atomic status transitions prevent duplicate background execution.
- Parallel identical requests must produce:
  - exactly one persisted `gap_result`
  - exactly one successful `llm_run`
  - all other requests returning the same cached analysis id or `PENDING`.

Concurrency correctness is validated at the database level.

SQLite in-memory is not suitable for this verification because:
- in-memory databases are connection-scoped and not shared,
- transactional and locking semantics differ from PostgreSQL.

Therefore, concurrency invariants are verified against PostgreSQL,
which is the target production database and source of transactional truth.

## Status State Machine
| Status | What it means | Transition |
|--------|---------------|------------|
| `PENDING` | Job queued or processing in background | `DONE`, `FAILED_VALIDATION`, `FAILED_LLM`, `FAILED_TIMEOUT` |
| `DONE` | Analysis complete and persisted | n/a |
| `FAILED_VALIDATION` | AI output failed contract validation | retry -> `PENDING` |
| `FAILED_LLM` | Provider call or processing failure | retry -> `PENDING` |
| `FAILED_TIMEOUT` | Processing exceeded timeout threshold | retry -> `PENDING` |

## AI Output Validation
- Pydantic schema gate for AI result contract.
- Final decision fields are computed deterministically from verified extraction:
  - `missing_skills`
  - `top_priority_skills`
  - `match_percent`
  - `match_reason`
- AI extraction is accepted only when it passes deterministic verification against taxonomy and source text evidence.

## AI Tooling Workflow
- Used Cursor, Windsurf, GPT, and Claude for:
  - architecture discussion and trade-off checks
  - code review on race/consistency/parser failures
  - refactoring and boilerplate acceleration
- Final behavior was verified by tests + runtime logs (not prompt assumptions).

## Local Setup (Docker)
1. `cp .env.dev.example .env`
2. `docker compose up -d --build`
3. Optional local Ollama: `docker compose --profile local-llm up -d --build`

Endpoints:
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- Health: `http://localhost:8000/health`

## Production Compose
1. `cp .env.prod.example .env.production`
2. `docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build`

## App-Level Setup Docs
- Backend: `backend/README.md`
- Frontend: `frontend/README.md`
