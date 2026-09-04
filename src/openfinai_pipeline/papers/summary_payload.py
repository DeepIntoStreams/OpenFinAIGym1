from typing import Any


def extract_summary_payload(paper_doc: dict[str, Any]) -> dict[str, Any] | None:
    """Return the paper summary payload from a paper.json document."""
    summary = paper_doc.get("summary")
    return summary if isinstance(summary, dict) else None


def has_summary_payload(paper_doc: dict[str, Any]) -> bool:
    payload = extract_summary_payload(paper_doc)
    return isinstance(payload, dict) and bool(payload)
