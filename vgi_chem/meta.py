"""Shared per-object discovery/description metadata helpers for the strict profile.

The ``vgi-lint`` strict profile expects these tags on **every** function and
table. Each function/table surfaces them in its ``Meta.tags`` mapping:

- ``vgi.title`` (VGI124)     -- human-friendly display name (must differ
  from the machine name once normalized, or VGI125 fires).
- ``vgi.doc_llm`` (VGI112)   -- a Markdown narrative aimed at an LLM/agent.
- ``vgi.doc_md`` (VGI113)    -- a Markdown narrative aimed at human docs.
- ``vgi.keywords`` (VGI138)  -- search terms / synonyms as a **JSON array of
  strings** (e.g. ``["a","b"]``), never a comma-separated string.

``keywords_json(...)`` serializes a list of keyword strings into the JSON-array
form the linter requires. ``vgi.source_url`` is intentionally **not** set
per-object (VGI139): the source link lives only on the catalog object.
"""

from __future__ import annotations

import json
from collections.abc import Sequence


def keywords_json(keywords: Sequence[str]) -> str:
    """Serialize keyword strings into the ``vgi.keywords`` JSON-array form.

    The linter (VGI138) requires ``vgi.keywords`` to be a JSON array of strings
    like ``["a","b"]`` rather than a comma-separated string.
    """
    return json.dumps(list(keywords))


def object_tags(
    *,
    title: str,
    description_llm: str,
    description_md: str,
    keywords: Sequence[str],
    relative_path: str,
) -> dict[str, str]:
    """Build the standard per-object discovery/description tags.

    ``relative_path`` is accepted for call-site documentation of where the
    object is implemented, but is intentionally not emitted as a per-object
    ``vgi.source_url`` (VGI139 keeps the source link on the catalog only).
    """
    del relative_path  # documented at the call site; not emitted per-object (VGI139)
    return {
        "vgi.title": title,
        "vgi.doc_llm": description_llm,
        "vgi.doc_md": description_md,
        "vgi.keywords": keywords_json(keywords),
    }
