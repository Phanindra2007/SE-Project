# Banking Requirements & SDLC Explorer

A two-stage local LLM workbench that generates grounded banking software requirements from curated Markdown domain specifications and recommends the best-fitting Software Development Life Cycle (SDLC) model with ranked comparisons, engineering justification, and requirement-linked risk mitigations.

---

## Architecture Overview

1. **Banking Reference Specifications (`data/banking/`)**
   - `account_transactions_reqs.md`: balance checks, debit/spending limits, idempotency, atomic posting & rollback.
   - `security_compliance_reqs.md`: MFA/step-up auth, least-privilege authorization, TLS/encryption, immutable audits, AML/sanctions/fraud screening.
   - `non_functional_operations_reqs.md`: availability (99.95%), latency SLAs, error handling, structured JSON logging, 7-year retention, PII redaction.

2. **Stage 1 — Requirements Generation (`src/step1_generate_reqs.py`)**
   - Loads `data/banking/*.md` with source file and section labels.
   - Prompts the configured OpenAI-compatible/Ollama LLM to produce a `functionality_profile` (`target_functionality`, `estimated_criticality`, `expected_change_frequency`) and structured Functional (`REQ-F-*`) and Non-Functional (`REQ-NF-*`) requirements grounded in the source rules.
   - Validates and saves output atomically to `outputs/step1_requirements.json`.

3. **Stage 2 — SDLC Model Selection (`src/step2_select_sdlc.py`)**
   - Reads `outputs/step1_requirements.json` and runs deterministic factor extraction (`analyze_step1_factors`) and candidate scoring (`score_sdlc_models`) across 10 pure and hybrid SDLC models (`Waterfall`, `V-Shaped`, `Prototype`, `RAD`, `Spiral`, `Agile`, `Agile-DevSecOps`, `Agile + V-Model`, `Prototype + V-Shaped`, `Spiral + Agile`).
   - Prompts the local LLM for grounded scores, rationales, engineering justification, alternatives, and risks/mitigations (with one automatic strict-JSON retry if needed), then blends LLM scores with factor-derived scores weighted by cited requirement evidence.
   - Saves outputs atomically to `outputs/step2_sdlc_justification.json` and `outputs/step2_sdlc_justification.md`.

4. **Web Application (`backend/main.py` & `frontend/`)**
   - **Backend**: FastAPI service exposing `/api/health`, `/api/overview`, `/api/documents`, `/api/step1`, `/api/step2`, and `/api/results/download/{artifact}` while serving the static frontend at `/`.
   - **Frontend**: Pure HTML, CSS, and Vanilla JavaScript single-page dark-themed dashboard with Overview, Requirements Generation, SDLC Analysis, Workflow Pipeline, and Results & Artifacts views.

---

## Project Structure

```text
final/
├── backend/
│   ├── __init__.py
│   └── main.py                  # FastAPI API + static frontend server
├── frontend/
│   ├── index.html               # Single-page UI layout
│   ├── style.css                # Dark-themed SaaS styling & responsive layout
│   └── script.js                # State, API integration, charts & filtering
├── src/
│   ├── step1_generate_reqs.py   # Stage 1 LLM requirements generator
│   ├── step2_select_sdlc.py     # Stage 2 SDLC evaluator & ranker
│   └── run_pipeline.py          # CLI pipeline runner
├── data/
│   └── banking/                 # Banking domain Markdown rules
├── outputs/                     # Generated JSON & Markdown artifacts
├── app.py                       # Original Streamlit UI (preserved)
├── requirements.txt
└── README.md
```

---

## Prerequisites & Local LLM Setup (Ollama)

Both Stage 1 and Stage 2 connect to a local OpenAI-compatible LLM endpoint (defaulting to Ollama).

1. **Install and start Ollama** (if not already running):
   ```bash
   ollama serve
   ```
2. **Pull the default model** (or configure another model via `LLM_MODEL`):
   ```bash
   ollama pull qwen3:8b
   ```
3. **Optional `.env` configuration**:
   Create a `.env` file in the project root if you want to override defaults:
   ```env
   LLM_BASE_URL=http://localhost:11434/v1/
   LLM_API_KEY=ollama
   LLM_MODEL=qwen3:8b
   LLM_TEMPERATURE=0.2
   ```

---

## Quick Start (Web Application)

1. **Create/activate virtual environment & install dependencies** (if not already set up):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Start the FastAPI backend + frontend**:
   ```bash
   .venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```
3. **Open the website** in your browser:
   ```text
   http://127.0.0.1:8000
   ```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves `frontend/index.html` |
| `GET` | `/api/health` | Returns backend & local LLM reachability status |
| `GET` | `/api/overview` | Returns project metadata, reference docs, and saved outputs |
| `GET` | `/api/documents` | Returns parsed banking Markdown reference files |
| `POST` | `/api/step1` | Runs `generate_requirements` (`{"functionality": "..."}`) |
| `POST` | `/api/step2` | Runs `select_sdlc` (`{"project_context": "..."}`) |
| `GET` | `/api/results/download/{artifact}` | Downloads `step1_requirements.json`, `step2_sdlc_justification.json`, or `step2_sdlc_justification.md` |

---

## CLI & Legacy UI (Preserved)

- **CLI two-stage pipeline**:
  ```bash
  .venv/bin/python -m src.run_pipeline --functionality "Domestic Instant Fund Transfer"
  ```
- **Original Streamlit UI**:
  ```bash
  .venv/bin/streamlit run app.py
  ```
