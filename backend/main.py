"""FastAPI backend wrapping the existing two-stage Banking Requirements & SDLC pipeline."""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.step1_generate_reqs import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_PATH,
    _markdown_sections,
    generate_requirements,
)
from src.step2_select_sdlc import (  # noqa: E402
    DEFAULT_JSON_PATH,
    DEFAULT_MARKDOWN_PATH,
    HYBRID_COMPONENTS,
    MODEL_SIGNAL_FIT,
    SCORING_CANDIDATES,
    analyze_step1_factors,
    load_requirements,
    select_sdlc,
)

FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_DIR = DEFAULT_DATA_DIR
STEP1_OUTPUT = DEFAULT_OUTPUT_PATH
STEP2_JSON = DEFAULT_JSON_PATH
STEP2_MARKDOWN = DEFAULT_MARKDOWN_PATH


class Step1Request(BaseModel):
    functionality: str = Field(..., description="Target banking functionality description.")


class Step2Request(BaseModel):
    project_context: str | None = Field(
        default=None,
        description="Optional project, team, or schedule context for SDLC evaluation.",
    )


def _safe_llm_config() -> dict[str, Any]:
    """Return non-secret LLM configuration values from environment/.env."""
    load_dotenv()
    return {
        "base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434/v1/"),
        "model": os.getenv("LLM_MODEL", "qwen3:8b"),
        "temperature": os.getenv("LLM_TEMPERATURE", "0.2"),
    }


def _check_llm_status(config: dict[str, Any]) -> dict[str, Any]:
    """Check whether the configured OpenAI-compatible/Ollama server is reachable."""
    base_url = config["base_url"].rstrip("/")
    parsed = urlparse(base_url)
    ollama_root = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""
    candidates: list[str] = []
    if ollama_root:
        candidates.append(f"{ollama_root}/api/tags")
    candidates.append(f"{base_url}/models")

    for url in candidates:
        try:
            with httpx.Client(timeout=1.5) as client:
                resp = client.get(url)
            if resp.status_code < 500:
                models: list[str] = []
                try:
                    data = resp.json()
                    if isinstance(data, dict) and isinstance(data.get("models"), list):
                        models = [
                            str(m.get("name"))
                            for m in data["models"]
                            if isinstance(m, dict) and m.get("name")
                        ]
                    elif isinstance(data, dict) and isinstance(data.get("data"), list):
                        models = [
                            str(m.get("id"))
                            for m in data["data"]
                            if isinstance(m, dict) and m.get("id")
                        ]
                except ValueError:
                    pass
                return {
                    "reachable": True,
                    "checked_url": url,
                    "available_models": models,
                }
        except httpx.HTTPError:
            continue

    return {
        "reachable": False,
        "checked_url": candidates[0] if candidates else base_url,
        "available_models": [],
    }


def _explain_error(exc: Exception) -> tuple[int, str]:
    """Map backend exceptions to appropriate HTTP status codes and user-friendly messages."""
    message = str(exc)
    lowered = message.lower()
    if isinstance(exc, FileNotFoundError):
        return 404, f"Missing required file: {message}"
    if isinstance(exc, json.JSONDecodeError):
        return 502, f"Invalid JSON output: {message}"
    if (
        "cannot reach" in lowered
        or "connection" in lowered
        or "connect" in lowered
        or "timed out" in lowered
    ):
        cfg = _safe_llm_config()
        return (
            503,
            f"Cannot reach the local LLM service at {cfg['base_url']}. "
            f"Ensure Ollama is running (e.g. `ollama serve`) and model '{cfg['model']}' is pulled.",
        )
    if isinstance(exc, ValueError):
        return 400, f"Validation error: {message}"
    if isinstance(exc, RuntimeError):
        return 502, message
    return 500, f"Unexpected server error: {message}"


def _read_reference_documents() -> list[dict[str, Any]]:
    """Read banking Markdown reference documents from data/banking."""
    if not DATA_DIR.is_dir():
        return []
    docs: list[dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        rules = [
            line.lstrip("- ").strip()
            for line in content.splitlines()
            if line.strip().startswith("-")
        ]
        sections = [
            {"title": title, "body": body}
            for title, body in _markdown_sections(content, path.name)
        ]
        docs.append(
            {
                "filename": path.name,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "rule_count": len(rules),
                "rules": rules,
                "sections": sections,
                "content": content,
            }
        )
    return docs


def _load_saved_state() -> dict[str, Any]:
    """Load persisted Step 1 and Step 2 outputs if present and consistent."""
    step1_data: dict[str, Any] | None = None
    step1_factors: dict[str, Any] | None = None
    step1_mtime: float | None = None

    if STEP1_OUTPUT.is_file():
        try:
            step1_data = load_requirements(STEP1_OUTPUT)
            step1_factors = analyze_step1_factors(step1_data)
            step1_mtime = STEP1_OUTPUT.stat().st_mtime
        except (ValueError, OSError, json.JSONDecodeError):
            step1_data = None

    step2_data: dict[str, Any] | None = None
    step2_markdown: str | None = None
    step2_stale = False

    if STEP2_JSON.is_file():
        try:
            parsed_step2 = json.loads(STEP2_JSON.read_text(encoding="utf-8"))
            if isinstance(parsed_step2, dict) and "recommended_sdlc" in parsed_step2:
                step2_mtime = STEP2_JSON.stat().st_mtime
                if step1_mtime is not None and step2_mtime + 0.01 < step1_mtime:
                    step2_stale = True
                else:
                    step2_data = parsed_step2
                    if STEP2_MARKDOWN.is_file():
                        step2_markdown = STEP2_MARKDOWN.read_text(encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            step2_data = None

    return {
        "step1": step1_data,
        "step1_factors": step1_factors,
        "step2": step2_data,
        "step2_markdown": step2_markdown,
        "step2_stale": step2_stale,
    }


app = FastAPI(
    title="Banking Requirements & SDLC Explorer API",
    description="Two-stage local LLM pipeline for banking requirements engineering and SDLC selection.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def get_health() -> dict[str, Any]:
    """Return backend health and local LLM server reachability."""
    config = _safe_llm_config()
    llm_status = await run_in_threadpool(_check_llm_status, config)
    return {
        "status": "ok",
        "llm": {
            **config,
            **llm_status,
        },
        "data_dir_exists": DATA_DIR.is_dir(),
        "step1_output_exists": STEP1_OUTPUT.is_file(),
        "step2_output_exists": STEP2_JSON.is_file(),
    }


@app.get("/api/overview")
async def get_overview() -> dict[str, Any]:
    """Return project overview metadata, reference docs, SDLC candidates, and saved outputs."""
    config = _safe_llm_config()
    llm_status = await run_in_threadpool(_check_llm_status, config)
    documents = await run_in_threadpool(_read_reference_documents)
    saved = await run_in_threadpool(_load_saved_state)

    return {
        "project": {
            "name": "Banking Requirements & SDLC Explorer",
            "tagline": "Two-Stage Local LLM Pipeline for Grounded Requirements & Lifecycle Engineering",
            "description": (
                "Generates grounded functional and non-functional software requirements from "
                "curated banking specification documents, then evaluates requirement signals, "
                "criticality, and volatility to rank and justify the optimal SDLC model."
            ),
            "stages": [
                {
                    "id": "stage1",
                    "name": "Requirements Generation",
                    "module": "src/step1_generate_reqs.py",
                    "function": "generate_requirements",
                    "output_file": "outputs/step1_requirements.json",
                },
                {
                    "id": "stage2",
                    "name": "SDLC Selection & Justification",
                    "module": "src/step2_select_sdlc.py",
                    "function": "select_sdlc",
                    "output_files": [
                        "outputs/step2_sdlc_justification.json",
                        "outputs/step2_sdlc_justification.md",
                    ],
                },
            ],
            "candidate_models": list(SCORING_CANDIDATES),
            "pure_models": list(MODEL_SIGNAL_FIT.keys()),
            "hybrid_models": {k: list(v) for k, v in HYBRID_COMPONENTS.items()},
        },
        "llm": {
            **config,
            **llm_status,
        },
        "documents": documents,
        "saved": saved,
    }


@app.get("/api/documents")
async def get_documents() -> dict[str, Any]:
    """Return all banking Markdown reference documents."""
    documents = await run_in_threadpool(_read_reference_documents)
    if not documents:
        raise HTTPException(
            status_code=404,
            detail=f"No banking Markdown reference files found in {DATA_DIR}",
        )
    return {"documents": documents}


@app.post("/api/step1")
async def run_step1(payload: Step1Request) -> dict[str, Any]:
    """Execute Stage 1: generate grounded banking software requirements."""
    functionality = payload.functionality.strip()
    if not functionality:
        raise HTTPException(
            status_code=400,
            detail="Target banking functionality must not be empty.",
        )

    def _execute() -> tuple[dict[str, Any], dict[str, Any], list[str], int]:
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = generate_requirements(
                functionality,
                data_dir=DATA_DIR,
                output_path=STEP1_OUTPUT,
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        factors = analyze_step1_factors(result)
        warning_messages = [str(w.message) for w in caught]
        return result, factors, warning_messages, elapsed_ms

    try:
        result, factors, warning_messages, elapsed_ms = await run_in_threadpool(_execute)
    except Exception as exc:
        status_code, detail = _explain_error(exc)
        raise HTTPException(status_code=status_code, detail=detail) from exc

    return {
        "result": result,
        "factors": factors,
        "warnings": warning_messages,
        "duration_ms": elapsed_ms,
        "output_path": str(STEP1_OUTPUT.relative_to(PROJECT_ROOT)),
    }


@app.post("/api/step2")
async def run_step2(payload: Step2Request) -> dict[str, Any]:
    """Execute Stage 2: select and justify an SDLC model from Stage 1 requirements."""
    if not STEP1_OUTPUT.is_file():
        raise HTTPException(
            status_code=400,
            detail=(
                "Stage 1 requirements have not been generated yet "
                f"(missing {STEP1_OUTPUT.relative_to(PROJECT_ROOT)}). "
                "Run Stage 1 (Generate Requirements) first."
            ),
        )

    project_context = payload.project_context.strip() if payload.project_context else None
    if project_context == "":
        project_context = None

    def _execute() -> tuple[dict[str, Any], dict[str, Any], str, list[str], int]:
        started = time.perf_counter()
        step1_data = load_requirements(STEP1_OUTPUT)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = select_sdlc(
                STEP1_OUTPUT,
                project_context,
                json_output_path=STEP2_JSON,
                markdown_output_path=STEP2_MARKDOWN,
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        factors = analyze_step1_factors(step1_data, project_context)
        markdown_text = (
            STEP2_MARKDOWN.read_text(encoding="utf-8")
            if STEP2_MARKDOWN.is_file()
            else ""
        )
        warning_messages = [str(w.message) for w in caught]
        return result, factors, markdown_text, warning_messages, elapsed_ms

    try:
        result, factors, markdown_text, warning_messages, elapsed_ms = await run_in_threadpool(
            _execute
        )
    except Exception as exc:
        status_code, detail = _explain_error(exc)
        raise HTTPException(status_code=status_code, detail=detail) from exc

    return {
        "result": result,
        "factors": factors,
        "markdown": markdown_text,
        "warnings": warning_messages,
        "duration_ms": elapsed_ms,
        "output_paths": {
            "json": str(STEP2_JSON.relative_to(PROJECT_ROOT)),
            "markdown": str(STEP2_MARKDOWN.relative_to(PROJECT_ROOT)),
        },
    }


@app.get("/api/results/download/{artifact_name}")
async def download_artifact(artifact_name: str):
    """Download one of the generated JSON or Markdown output artifacts."""
    allowed = {
        "step1_requirements.json": (STEP1_OUTPUT, "application/json"),
        "step2_sdlc_justification.json": (STEP2_JSON, "application/json"),
        "step2_sdlc_justification.md": (STEP2_MARKDOWN, "text/markdown"),
    }
    entry = allowed.get(artifact_name)
    if not entry:
        raise HTTPException(status_code=404, detail="Unknown output artifact.")
    file_path, media_type = entry
    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Artifact '{artifact_name}' has not been generated yet.",
        )
    if media_type == "text/markdown":
        return PlainTextResponse(
            content=file_path.read_text(encoding="utf-8"),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{artifact_name}"'},
        )
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=artifact_name,
    )


if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def serve_index():
    """Serve the single-page web frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path)
