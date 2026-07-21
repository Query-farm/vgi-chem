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
from collections.abc import Iterable, Sequence
from typing import Any


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
    category: str,
) -> dict[str, str]:
    """Build the standard per-object discovery/description tags.

    ``relative_path`` is accepted for call-site documentation of where the
    object is implemented, but is intentionally not emitted as a per-object
    ``vgi.source_url`` (VGI139 keeps the source link on the catalog only).

    ``category`` is the object's primary ``vgi.category`` -- it must name one of
    the ``vgi.categories`` registry entries declared on the owning schema
    (VGI409/VGI411).
    """
    del relative_path  # documented at the call site; not emitted per-object (VGI139)
    return {
        "vgi.title": title,
        "vgi.doc_llm": description_llm,
        "vgi.doc_md": description_md,
        "vgi.keywords": keywords_json(keywords),
        "vgi.category": category,
    }


def attach_example_queries(functions: Iterable[Any]) -> None:
    """Mirror each function's ``Meta.examples`` into a ``vgi.example_queries`` tag.

    The native ``duckdb_functions().examples`` carrier surfaces a function's
    ``Meta.examples`` SQL but **drops the per-example description**, so the linter
    (VGI515) flags every example as description-less. The ``vgi.example_queries``
    tag is the parallel carrier that keeps descriptions, and the loader dedupes
    the two by SQL. We therefore emit a JSON list of ``{"description","sql"}``
    objects whose SQL is byte-identical to each ``FunctionExample`` so the two
    carriers collapse to one described example per query.

    For arity overloads that share a ``Meta.name`` (``morgan_fingerprint``,
    ``tanimoto``), the described examples are aggregated by name across the
    overload classes so the single merged catalog object is fully described.
    """
    by_name: dict[str, list[dict[str, str]]] = {}
    classes_by_name: dict[str, list[Any]] = {}
    for fn in functions:
        meta = fn.Meta
        name = meta.name
        entries = by_name.setdefault(name, [])
        classes_by_name.setdefault(name, []).append(fn)
        seen = {" ".join(e["sql"].split()).lower() for e in entries}
        for ex in getattr(meta, "examples", ()):
            key = " ".join(ex.sql.split()).lower()
            if key in seen:
                continue
            seen.add(key)
            entries.append({"description": ex.description, "sql": ex.sql})
    for name, entries in by_name.items():
        payload = json.dumps(entries)
        for fn in classes_by_name[name]:
            # ``Meta.tags`` may be a fresh dict per class; set the aggregated tag on
            # every overload so the merged object is described regardless of which
            # overload's tags the loader reads first.
            fn.Meta.tags = {**fn.Meta.tags, "vgi.example_queries": payload}
