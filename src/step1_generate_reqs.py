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
from openai import APIError, OpenAI


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
    text = raw_text.strip()
    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (list, dict)):
            return value

    raise ValueError("The LLM response did not contain valid JSON.")


def parse_requirements_json(raw_text: str) -> list[dict[str, Any]]:
    """Safely parse, normalize, and validate the LLM JSON response."""
    parsed = _extract_json_text(raw_text)

    if isinstance(parsed, dict):
        if "requirements" in parsed:
            parsed = parsed["requirements"]
        else:
            raise ValueError(
                "The JSON object must contain a 'requirements' array."
            )

    if not isinstance(parsed, list) or not parsed:
        raise ValueError("The LLM response must be a non-empty JSON array.")

    validated: list[dict[str, Any]] = []
    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Requirement at index {index} is not a JSON object.")

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

    return validated


def _build_prompt(target_functionality: str, banking_sources: str) -> str:
    return f"""Generate a concise set of specific, testable banking software requirements
for the target functionality: "{target_functionality}".

Ground every requirement in the provided banking source material. Do not invent
bank policies or regulatory requirements beyond the source material. If an
assumption is necessary, state it explicitly in "reasoning".

Create roughly 8 functional requirements and 6 non-functional requirements.
Use REQ-F-001 onward for Functional requirements and REQ-NF-001 onward for
Non-Functional requirements.

Return only a valid JSON array. Do not add Markdown fences or prose outside the
array. Each object must contain exactly these keys:
- id: string
- type: "Functional" or "Non-Functional"
- title: string
- requirement: clear, measurable, testable statement
- reasoning: supporting source points, relevance to the target functionality,
  and assumptions or edge cases
- source_references: non-empty list of source filenames and section labels,
  such as ["account_transactions_reqs.md | Full document"]

Banking source material:
{banking_sources}
"""


def generate_requirements(
    target_functionality: str,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, Any]]:
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
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise banking software requirements analyst. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw_text = response.choices[0].message.content
    if not raw_text:
        raise ValueError("The LLM returned an empty response.")

    requirements = parse_requirements_json(raw_text)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(requirements, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)

    return requirements


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
        requirements = generate_requirements(args.target_functionality)
    except (FileNotFoundError, ValueError, RuntimeError, APIError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Generated {len(requirements)} requirements at "
        f"{DEFAULT_OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())