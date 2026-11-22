"""Select an SDLC model for generated banking requirements."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS_PATH = PROJECT_ROOT / "outputs" / "step1_requirements.json"
DEFAULT_JSON_PATH = PROJECT_ROOT / "outputs" / "step2_sdlc_justification.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "outputs" / "step2_sdlc_justification.md"


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract one JSON object from fences or extra text from a local model."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("The local model returned an empty response instead of JSON.")

    text = raw_text.strip().lstrip("\ufeff")
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                candidates.append(value)
        if not candidates:
            preview = " ".join(text.split())
            if len(preview) > 160:
                preview = preview[:157] + "..."
            raise ValueError(
                "The local model returned invalid JSON. "
                f"Response preview: {preview or '<empty>'}"
            )
        parsed = max(candidates, key=lambda item: len(json.dumps(item)))

    if not isinstance(parsed, dict):
        raise ValueError("The local model must return a JSON object.")
    return parsed


def load_requirements(path: str | Path = DEFAULT_REQUIREMENTS_PATH) -> list[dict[str, Any]]:
    requirements_path = Path(path)
    if not requirements_path.is_file():
        raise FileNotFoundError(
            f"Generated requirements not found: {requirements_path}"
        )
    try:
        requirements = json.loads(requirements_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Generated requirements contain invalid JSON: {requirements_path}"
        ) from exc

    if not isinstance(requirements, list) or not requirements:
        raise ValueError("Generated requirements must be a non-empty JSON array.")
    if not all(isinstance(item, dict) for item in requirements):
        raise ValueError("Every generated requirement must be a JSON object.")
    return requirements


def temperature_from_environment() -> float:
    raw_value = os.getenv("LLM_TEMPERATURE", "0.2")
    try:
        temperature = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"LLM_TEMPERATURE must be a number, received: {raw_value!r}"
        ) from exc
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("LLM_TEMPERATURE must be between 0.0 and 2.0")
    return temperature


def build_sdlc_prompt(
    requirements: list[dict[str, Any]],
    project_context: str | None = None,
) -> str:
    context = project_context.strip() if project_context else ""
    context_text = f"\nOptional project/team context:\n{context}\n" if context else ""
    return f"""Evaluate these generated requirements and recommend one SDLC model.

Use only the requirements and optional context below. Do not introduce unrelated
requirements or claim capabilities that are not supported. Ground the decision
in requirement clarity/volatility, security and compliance criticality,
testing/verification needs, and stakeholder/delivery cadence.

Choose from Agile, Iterative/Incremental, V-Model, Spiral, Waterfall,
DevSecOps, or a clearly named hybrid. Avoid vague recommendations.

Return exactly one valid JSON object and no Markdown fences or prose:
{{
  "recommended_sdlc": "specific model name",
  "alternative_considered": [
    {{"model": "alternative name", "reason": "why it is less suitable"}},
    {{"model": "alternative name", "reason": "why it is less suitable"}}
  ],
  "se_justification": "thorough paragraph covering requirement volatility/clarity, risk/security/compliance criticality, testing/verification needs, and stakeholder/delivery cadence",
  "key_risks_and_mitigations": [
    {{"risk": "practical execution risk", "mitigation": "specific mitigation"}}
  ]
}}

Provide one or two alternatives and at least three practical risks.
{context_text}
Generated requirements:
{json.dumps(requirements, indent=2, ensure_ascii=False)}
"""


def validate_sdlc_result(result: dict[str, Any]) -> dict[str, Any]:
    required = {
        "recommended_sdlc",
        "alternative_considered",
        "se_justification",
        "key_risks_and_mitigations",
    }
    missing = required - result.keys()
    if missing:
        raise ValueError(
            "The local model response is missing keys: " + ", ".join(sorted(missing))
        )

    if not isinstance(result["recommended_sdlc"], str) or not result["recommended_sdlc"].strip():
        raise ValueError("recommended_sdlc must be a non-empty string.")
    if not isinstance(result["se_justification"], str) or not result["se_justification"].strip():
        raise ValueError("se_justification must be a non-empty string.")

    alternatives = result["alternative_considered"]
    if not isinstance(alternatives, list) or not 1 <= len(alternatives) <= 2:
        raise ValueError("alternative_considered must contain one or two alternatives.")
    normalized_alternatives: list[dict[str, str]] = []
    for index, alternative in enumerate(alternatives, start=1):
        if not isinstance(alternative, dict):
            raise ValueError(f"Alternative {index} must be a JSON object.")
        model = alternative.get("model")
        reason = alternative.get("reason")
        if not isinstance(model, str) or not model.strip() or not isinstance(
            reason, str
        ) or not reason.strip():
            raise ValueError(
                f"Alternative {index} must contain non-empty model and reason strings."
            )
        normalized_alternatives.append(
            {"model": model.strip(), "reason": reason.strip()}
        )

    risks = result["key_risks_and_mitigations"]
    if not isinstance(risks, list) or not risks:
        raise ValueError("key_risks_and_mitigations must be a non-empty list.")
    normalized_risks: list[dict[str, str]] = []
    for index, item in enumerate(risks, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Risk {index} must be a JSON object.")
        risk = item.get("risk")
        mitigation = item.get("mitigation")
        if not isinstance(risk, str) or not risk.strip() or not isinstance(
            mitigation, str
        ) or not mitigation.strip():
            raise ValueError(
                f"Risk {index} must contain non-empty risk and mitigation strings."
            )
        normalized_risks.append(
            {"risk": risk.strip(), "mitigation": mitigation.strip()}
        )

    return {
        "recommended_sdlc": result["recommended_sdlc"].strip(),
        "alternative_considered": normalized_alternatives,
        "se_justification": result["se_justification"].strip(),
        "key_risks_and_mitigations": normalized_risks,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# SDLC Recommendation",
        "",
        f"## Recommended SDLC: {result['recommended_sdlc']}",
        "",
        "## Alternatives Considered",
        "",
    ]
    for alternative in result["alternative_considered"]:
        lines.append(f"- **{alternative['model']}**: {alternative['reason']}")

    lines.extend(["", "## Software Engineering Justification", "", result["se_justification"], "",
                  "## Key Risks and Mitigations", ""])
    for item in result["key_risks_and_mitigations"]:
        lines.append(f"- **Risk:** {item['risk']}")
        lines.append(f"  - **Mitigation:** {item['mitigation']}")
    return "\n".join(lines).rstrip() + "\n"


def write_atomic(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)


def select_sdlc(
    requirements_path: str | Path = DEFAULT_REQUIREMENTS_PATH,
    project_context: str | None = None,
    *,
    json_output_path: str | Path = DEFAULT_JSON_PATH,
    markdown_output_path: str | Path = DEFAULT_MARKDOWN_PATH,
) -> dict[str, Any]:
    """Evaluate requirements and save/return the structured SDLC recommendation."""
    load_dotenv()
    requirements = load_requirements(requirements_path)
    prompt = build_sdlc_prompt(requirements, project_context)

    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1/")
    api_key = os.getenv("LLM_API_KEY", "ollama")
    model = os.getenv("LLM_MODEL", "qwen3:8b")
    temperature = temperature_from_environment()

    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a software engineering lifecycle analyst. "
                        "Use the requirements and return one valid JSON object only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
    except (APIConnectionError, APITimeoutError) as exc:
        raise RuntimeError(
            f"Cannot reach the local LLM at {base_url}. "
            f"Verify the endpoint and that model '{model}' is running."
        ) from exc
    except APIError as exc:
        raise RuntimeError(
            f"The local LLM at {base_url} returned an API error for model "
            f"'{model}': {exc}"
        ) from exc

    if not response.choices:
        raise ValueError("The local model returned no choices.")
    raw_text = response.choices[0].message.content
    if not raw_text:
        raise ValueError("The local model returned an empty response.")

    result = validate_sdlc_result(extract_json_object(raw_text))
    write_atomic(
        json_output_path,
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
    )
    write_atomic(markdown_output_path, render_markdown(result))
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recommend an SDLC model for generated banking requirements."
    )
    parser.add_argument(
        "requirements_pos",
        nargs="?",
        help="Optional positional path to step 1 requirements JSON.",
    )
    parser.add_argument(
        "context_pos",
        nargs="?",
        help="Optional positional project/team context.",
    )
    parser.add_argument(
        "--requirements",
        default=str(DEFAULT_REQUIREMENTS_PATH),
        help="Path to step 1 requirements JSON.",
    )
    parser.add_argument("--context", default=None, help="Optional project/team context.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    requirements_path = args.requirements_pos or args.requirements
    project_context = args.context_pos or args.context

    try:
        result = select_sdlc(requirements_path, project_context)
    except (FileNotFoundError, ValueError, RuntimeError, APIError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Recommended {result['recommended_sdlc']} "
        f"and saved JSON/Markdown outputs under {DEFAULT_JSON_PATH.parent}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())