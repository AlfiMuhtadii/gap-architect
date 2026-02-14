# Gap Architect (Option B)
Full-stack MVP for resume vs JD gap analysis using Next.js + FastAPI + PostgreSQL.

## What I Built
- Semantic diff flow: Resume + JD -> missing skills, match score/reason, 3 action steps, 3 interview questions, roadmap markdown.
- Deterministic-first extraction and scoring:
  - skill extraction and normalization from taxonomy
  - `match_percent = matched / total_required` (not LLM-generated math)
- Cached analysis: same normalized input fingerprint returns existing result from DB.

## Design Rationale (Idea and Solution)
The core idea is to treat this as a decision system, not a pure text-generation system.

- Deterministic layer is the source of truth:
  - extract/normalize skills from Resume and JD
  - compute `missing_skills`, `top_priority_skills`, `match_percent`, `match_reason`
- Generative layer is constrained:
  - LLM is used to generate learning artifacts only (`action_steps`, `interview_questions`, `roadmap_markdown`)
  - generated output must pass schema validation before persistence

Why this architecture:
- LLMs are strong at language generation but weaker at stable numeric reasoning.
- Recruiter-facing systems need repeatable outputs for trust and auditability.
- Separating deterministic vs generative concerns prevents score drift and logic contradictions.

Problem -> Solution mapping:
- AI returns malformed JSON:
  - strict parse + schema validation + one repair prompt attempt + fail-safe message
- Provider downtime/rate limits/timeouts:
  - fallback chain (`primary -> local_llm -> heuristic`) to keep service responsive
- Noisy/long JD text:
  - adaptive clipping before prompting to focus on requirements signal
- Duplicate concurrent processing:
  - in-flight guard + bounded concurrency + atomic state transitions
- Repeat submissions:
  - input fingerprint + DB cache for fast retrieval and cost control

Trade-offs and scope:
- This is intentionally more robust than a minimal MVP.
- Additional complexity (fallbacks, llm run audit, concurrency guards) is used to reduce operational risk.
- Core product flow remains simple for users: submit -> pending -> done/failed.

## Future Extensibility (Why This Foundation Can Grow)
This codebase is intentionally structured so it can evolve from MVP into a richer talent intelligence platform without breaking the core API contract.

Why it is already support-ready:
- Deterministic and generative concerns are separated:
  - deterministic layer owns skill truth and scoring
  - LLM layer owns content generation
  - this allows replacing skill intelligence without rewriting roadmap generation
- Canonical normalization exists as a stable intermediate layer:
  - current canonical skill pipeline can be swapped from flat list to richer ontology mappings
- Data model already captures analysis lifecycle:
  - `gap_analyses`, `gap_results`, `llm_runs` provide state, outputs, and audit trail
  - this supports future calibration, governance, and replay/reproducibility workflows
- Async/background architecture is in place:
  - heavier future matching/linking logic can be added in worker path without changing frontend behavior

Practical upgrades enabled by this design:
- Ontology expansion:
  - move from simple canonical set to ESCO/O*NET-backed concepts
  - add broader/narrower/synonym relations for better recall/precision
- Cross-taxonomy mapping:
  - internal canonical IDs mapped to ESCO URIs and O*NET concept keys
- Multilingual semantic matching:
  - add embedding linker for abstract skills and non-English phrasing
- Human-in-the-loop learning:
  - reviewer decisions can be persisted and used for threshold calibration
- Production scaling:
  - introduce shared cache/worker queues and richer metrics without changing core endpoints

In short: the MVP solves today’s assignment scope, and the current boundaries make tomorrow’s ontology-grade upgrades feasible with controlled refactoring.

## How I Handled Edge Cases The AI Missed
- Malformed JSON from LLM:
  - strict parsing + schema validation
  - one repair attempt with constrained prompt
  - fail-safe message when still invalid
- Input quality issues:
  - resume/JD validation before full analysis
  - short/low-signal input rejected with explicit message
- JD noise and long text:
  - adaptive clipping with requirements-first heuristics
  - fallback clipping strategy when section headers are missing
- Network/provider failures:
  - controlled fallback path (primary -> local -> deterministic heuristic)

## Database and Async Design
- PostgreSQL tables (core):
  - `gap_analyses`: request fingerprint, status, error_message
  - `gap_results`: missing skills, steps, questions, roadmap, match fields
  - `llm_runs`: provider/model/request_hash/response/status/duration
  - `jd_clean_runs`: input cleaning strategy + status
- Async/background processing:
  - API returns quickly, heavy LLM work runs in background task
  - in-flight guard prevents duplicate processing per analysis id
  - semaphore caps concurrent jobs (`MAX_CONCURRENT_GAP_JOBS`)
- Consistency safeguards:
  - atomic status transitions and conflict-safe DB writes
  - migration at container startup (`alembic upgrade head`)

## How I Validate AI Output
- Pydantic schema validation for AI result contract.
- Deterministic fields remain source-of-truth:
  - missing skills, top priority, match percent, match reason.
- LLM used for generative content quality:
  - action steps
  - interview questions
  - roadmap formatting

## Engineering Constraints (How I Handled Them)
- Background processing:
  - request path is non-blocking; user receives id/status immediately
  - polling endpoint tracks `PENDING -> DONE/FAILED_*`
- Reliability:
  - retries and fallback chains avoid total failure on provider issues
  - Docker health checks for postgres/redis/backend/frontend
- Config parity:
  - dev/prod compose follow same structure, values via env files

## AI Tooling Workflow
- I used AI assistants across the workflow (Cursor, Windsurf, GPT, and Claude), not only for code generation but also for technical discussion and design trade-off exploration.
- How AI was used in practice:
  - solution discussion: compared deterministic-vs-generative boundaries, fallback strategies, and async processing patterns
  - code review: asked for risk-focused review (race conditions, transaction consistency, parser edge cases)
  - refactoring: reorganized service modules and simplified interfaces while preserving behavior
  - boilerplate acceleration: generated repetitive scaffolding (compose/env/test skeletons) faster, then manually tightened logic
- Main value was faster bug isolation:
  - traced async pending-loop issue to a mis-indented provider execution path
  - validated fixes using `pytest`
- Guardrails I kept to avoid hidden AI side effects:
  - deterministic logic remains explicit and testable
  - schema boundaries are enforced with validation
  - final behavior is verified by tests and runtime checks, not prompt assumptions

## Local Setup (Docker)
1. Copy env:
   - `cp .env.dev.example .env`
2. Start app:
   - `docker compose up -d --build`
3. Optional local Ollama:
   - `docker compose --profile local-llm up -d --build`

Endpoints:
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- Health: `http://localhost:8000/health`

## Production Compose
1. `cp .env.prod.example .env.production`
2. `docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build`

## App-Level Setup Docs
- Backend setup: `backend/README.md`
- Frontend setup: `frontend/README.md`
