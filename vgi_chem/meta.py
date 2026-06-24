"""Shared per-object discovery/description metadata helpers for the strict profile.

The ``vgi-lint`` strict profile expects these tags on **every** function and
table. Each function/table surfaces them in its ``Meta.tags`` mapping:

- ``vgi.title`` (VGI124)          -- human-friendly display name (must differ
  from the machine name once normalized, or VGI125 fires).
- ``vgi.description_llm`` (VGI112) -- a Markdown narrative aimed at an LLM/agent.
- ``vgi.description_md`` (VGI113)  -- a Markdown narrative aimed at human docs.
- ``vgi.keywords`` (VGI126)        -- comma-separated search terms / synonyms.
- ``vgi.source_url`` (VGI128)      -- link to the implementing source file.

``source_url(path)`` builds the canonical GitHub blob URL for a source file so
every object points at exactly where it is implemented.
"""

from __future__ import annotations

# Base GitHub blob URL for source files in this repo (pinned to ``main``).
_SOURCE_BASE = "https://github.com/Query-farm/vgi-chem/blob/main"


def source_url(relative_path: str) -> str:
    """Build the implementation ``vgi.source_url`` for a repo-relative file.

    For example ``source_url("vgi_chem/scalars.py")``.
    """
    return f"{_SOURCE_BASE}/{relative_path}"


def object_tags(
    *,
    title: str,
    description_llm: str,
    description_md: str,
    keywords: str,
    relative_path: str,
) -> dict[str, str]:
    """Build the five standard per-object discovery/description tags.

    ``relative_path`` is the implementing file relative to the repo root.
    """
    return {
        "vgi.title": title,
        "vgi.description_llm": description_llm,
        "vgi.description_md": description_md,
        "vgi.keywords": keywords,
        "vgi.source_url": source_url(relative_path),
    }
