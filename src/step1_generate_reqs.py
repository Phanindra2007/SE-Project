"""Generate grounded banking requirements with a local OpenAI-compatible LLM."""

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
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "banking"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "step1_requirements.json"

REQUIRED_KEYS = {
    "id",
    "type",
    "title",
    "requirement",
    "reasoning",
    "source_references",
}


def _markdown_sections(markdown: str, filename: str) -> list[tuple[str, str]]:
    """Split Markdown into labeled sections while retaining unheaded content."""
    lines = markdown.strip().splitlines()
    if not lines:
        return [(filename, "")]

    sections: list[tuple[str, str]] = []
    current_title = "Full document"
    current_lines: list[str] = []

    for line in lines:
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if heading:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = heading.group(1)
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return [(title, body) for title, body in sections if body]


def load_banking_sources(data_dir: str | Path = DEFAULT_DATA_DIR) -> str:
    """Load all Markdown files with filename and section labels."""
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Banking reference directory not found: {directory}")

    markdown_files = sorted(directory.glob("*.md"))
    if not markdown_files:
        raise FileNotFoundError(f"No .md files found in: {directory}")

    source_blocks: list[str] = []
    for path in markdown_files:
        markdown = path.read_text(encoding="utf-8")
        sections = _markdown_sections(markdown, path.name)
        for section_title, body in sections:
            source_blocks.append(
                f"[SOURCE FILE: {path.name} | SECTION: {section_title}]\n{body}"
            )

    return "\n\n".join(source_blocks)


def _temperature_from_environment() -> float:
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


def _extract_json_text(raw_text: str) -> Any:
    """Extract a JSON array or object from fenced or extra-text model output."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("The local model returned an empty response instead of JSON.")

    text = raw_text.strip().lstrip("\ufeff")

    # Smaller models commonly leave Markdown around an otherwise valid payload.
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (list, dict)):
            candidates.append(value)

    if candidates:
        # Prefer the largest top-level payload; nested objects are unlikely to
        # be the intended requirement list.
        return max(
            candidates,
            key=lambda candidate: len(json.dumps(candidate, ensure_ascii=False)),
        )

    preview = " ".join(text.split())
    if len(preview) > 160:
        preview = preview[:157] + "..."
    raise ValueError(
        "The local model returned invalid JSON. "
        f"Response preview: {preview or '<empty>'}"
    )


def normalize_source_references(
    item: dict[str, Any],
    fallback_references: list[str],
) -> None:
    """Normalize local-model source references before strict validation."""
    references = item.get("source_references")

    if isinstance(references, str):
        normalized = [references.strip()] if references.strip() else []
    elif isinstance(references, list):
        normalized = [
            str(reference).strip()
            for reference in references
            if reference is not None and str(reference).strip()
        ]
    else:
        normalized = []

    item["source_references"] = normalized or list(fallback_references)


def fallback_source_references(data_dir: str | Path) -> list[str]:
    directory = Path(data_dir)
    if not directory.is_dir():
        return []
    return [
        path.name
        for path in sorted(directory.glob("*.md"))
        if path.is_file()
    ]


def parse_requirements_json(
    raw_text: str,
    target_functionality: str | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict[str, Any]:
    """Parse and validate a profile-bearing requirements response."""
    parsed = _extract_json_text(raw_text)

    if isinstance(parsed, dict):
        raw_profile = parsed.get("functionality_profile")
        parsed = parsed.get("requirements")
    else:
        raw_profile = None

    if not isinstance(parsed, list) or not parsed:
        raise ValueError("The LLM response must be a non-empty JSON array.")

    if raw_profile is None:
        profile = {
            "target_functionality": target_functionality or "Unspecified functionality",
            "estimated_criticality": "Medium",
            "expected_change_frequency": "Medium",
        }
    else:
        if not isinstance(raw_profile, dict):
            raise ValueError("functionality_profile must be a JSON object.")
        profile = dict(raw_profile)

    fallback_references = fallback_source_references(data_dir)
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Requirement at index {index} is not a JSON object.")

        normalize_source_references(item, fallback_references)

        missing = REQUIRED_KEYS - item.keys()
        if missing:
            raise ValueError(
                f"Requirement at index {index} is missing keys: "
                + ", ".join(sorted(missing))
            )

        if item["type"] not in {"Functional", "Non-Functional"}:
            raise ValueError(
                f"Requirement at index {index} has invalid type: {item['type']!r}"
            )
        if not isinstance(item["id"], str) or not item["id"].strip():
            raise ValueError(f"Requirement at index {index} has an invalid id.")
        item_id = item["id"].strip()
        expected_prefix = "REQ-F-" if item["type"] == "Functional" else "REQ-NF-"
        if not re.fullmatch(re.escape(expected_prefix) + r"\d{3}", item_id):
            raise ValueError(
                f"Requirement at index {index} has invalid id: {item_id!r}"
            )
        for key in ("title", "requirement", "reasoning"):
            if not isinstance(item[key], str) or not item[key].strip():
                raise ValueError(
                    f"Requirement at index {index} has an invalid {key}."
                )
        if (
            not isinstance(item["source_references"], list)
            or not item["source_references"]
            or not all(
                isinstance(reference, str) and reference.strip()
                for reference in item["source_references"]
            )
        ):
            raise ValueError(
                f"Requirement at index {index} must contain source_references "
                "as a non-empty list of strings."
            )

        validated.append(
            {
                "id": item["id"].strip(),
                "type": item["type"],
                "title": item["title"].strip(),
                "requirement": item["requirement"].strip(),
                "reasoning": item["reasoning"].strip(),
                "source_references": [
                    reference.strip() for reference in item["source_references"]
                ],
            }
        )

    target = profile.get("target_functionality")
    if not isinstance(target, str) or not target.strip():
        target = target_functionality or "Unspecified functionality"
    criticality = profile.get("estimated_criticality")
    if criticality not in {"Low", "Medium", "High"}:
        raise ValueError(
            "functionality_profile estimated_criticality must be Low, Medium, or High."
        )
    change_frequency = profile.get("expected_change_frequency")
    if change_frequency not in {"Low", "Medium", "High"}:
        raise ValueError(
            "functionality_profile expected_change_frequency must be Low, Medium, or High."
        )

    return {
        "functionality_profile": {
            "target_functionality": target.strip(),
            "estimated_criticality": criticality,
            "expected_change_frequency": change_frequency,
        },
        "requirements": validated,
    }


def _build_prompt(target_functionality: str, banking_sources: str) -> str:
    return f"""Create requirements for: "{target_functionality}".

STRICT SOURCE RULES:
- Use only facts stated in the Banking source material below.
- Select only source rules that directly apply to "{target_functionality}".
- Do not force unrelated bank-wide controls onto low-risk or UI/preference
  features. For example, do not require transaction limits, AML/sanctions,
  atomic posting, or 7-year financial ledgers unless the target performs or
  directly stores a transaction/compliance-sensitive action.
- Conversely, include transaction, AML/compliance, audit, rollback, encryption,
  and retention rules when the target materially performs those operations.
- Do not add laws, limits, SLA values, design choices, or bank policies that are
  not explicitly present in the source material.
- Every requirement must cite one or more source filenames and section labels.
- If a detail is unstated or inferred, begin the relevant sentence in
  "reasoning" with "[Assumption]" and keep the requirement directly testable.
- Do not use generic filler such as "system should be robust", "use best practices",
  or "ensure a good experience".
- Do not duplicate a requirement or mix multiple independent behaviors in one
  requirement.

OUTPUT:
- Return exactly one valid JSON object and no Markdown fences or prose.
- Top-level keys:
  - functionality_profile: target_functionality, estimated_criticality
    (Low/Medium/High), and expected_change_frequency (Low/Medium/High)
  - requirements: the requirement objects
- Estimate profile fields only from direct applicability and the generated
  requirements. UI/preference features are usually Low criticality and may have
  Medium/High change frequency; transactional/compliance features are normally
  High criticality.
- Produce roughly 8 Functional and 6 Non-Functional requirements.
- Use REQ-F-001 onward for Functional and REQ-NF-001 onward for Non-Functional.
- Every object must contain exactly:
  id, type, title, requirement, reasoning, source_references.
- "title" must be short and specific.
- "requirement" must be clear and measurable/testable.
- "reasoning" must name the supporting source points, explain relevance to the
  target functionality, and identify any unstated detail with "[Assumption]".
- "source_references" must contain only filename plus section labels grounded in
  the Banking source material, for example:
  ["account_transactions_reqs.md | Full document"].
- The response shape is:
  {{
    "functionality_profile": {{
      "target_functionality": "{target_functionality}",
      "estimated_criticality": "Low|Medium|High",
      "expected_change_frequency": "Low|Medium|High"
    }},
    "requirements": [ ...requirement objects... ]
  }}

Banking source material:
{banking_sources}
"""


def generate_requirements(
    target_functionality: str,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Generate, validate, and save requirements for a target functionality."""
    target_functionality = target_functionality.strip()
    if not target_functionality:
        raise ValueError("target_functionality must not be empty.")

    load_dotenv()

    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1/")
    api_key = os.getenv("LLM_API_KEY", "ollama")
    model = os.getenv("LLM_MODEL", "qwen3:8b")
    temperature = _temperature_from_environment()

    banking_sources = load_banking_sources(data_dir)
    prompt = _build_prompt(target_functionality, banking_sources)

    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate grounded banking requirements. "
                        "Follow the source rules and return JSON only."
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
        raise ValueError("The LLM returned an empty response.")

    result = parse_requirements_json(
        raw_text,
        target_functionality,
        data_dir,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)

    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate banking requirements using a local LLM."
    )
    parser.add_argument(
        "target_functionality",
        nargs="?",
        default="Domestic Instant Fund Transfer",
        help='Target functionality (default: "Domestic Instant Fund Transfer")',
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        result = generate_requirements(args.target_functionality)
    except (FileNotFoundError, ValueError, RuntimeError, APIError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Generated {len(result['requirements'])} requirements at "
        f"{DEFAULT_OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())