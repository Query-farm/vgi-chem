"""Composed example queries: placement in the catalog metadata and pinned E2E coverage."""

from __future__ import annotations

import json
from pathlib import Path

from vgi_chem.examples import (
    ACID_FILTER_SQL,
    COMPOSED_EXAMPLES,
    INCHIKEY_LOOKUP_SQL,
    NEAREST_PAIRS_SQL,
    SIMILARITY_SEARCH_SQL,
)
from vgi_chem.worker import _CHEM_CATALOG

_COMPOSED_TEST = Path(__file__).resolve().parent.parent / "test" / "sql" / "composed.test"


def _schema():
    return _CHEM_CATALOG.schemas[0]


def _function_examples(name: str) -> list[str]:
    for fn in _schema().functions:
        if fn.Meta.name == name:
            return [e["sql"] for e in json.loads(fn.Meta.tags["vgi.example_queries"])]
    raise AssertionError(f"no function {name}")


def test_every_composed_example_is_in_the_schema_list() -> None:
    schema_sql = [e["sql"] for e in json.loads(_schema().tags["vgi.example_queries"])]
    for ex in COMPOSED_EXAMPLES:
        assert ex["sql"] in schema_sql


def test_composed_examples_sit_in_the_driving_objects_first_five() -> None:
    assert SIMILARITY_SEARCH_SQL in _function_examples("tanimoto")[:5]
    assert ACID_FILTER_SQL in _function_examples("substructure_match")[:5]
    assert INCHIKEY_LOOKUP_SQL in _function_examples("inchikey")[:5]
    table = next(t for t in _schema().tables if t.name == "example_molecules")
    table_sql = [e["sql"] for e in json.loads(table.tags["vgi.example_queries"])]
    assert NEAREST_PAIRS_SQL in table_sql[:5]


def test_every_composed_example_is_pinned_by_the_sql_suite() -> None:
    suite = _COMPOSED_TEST.read_text()
    for ex in COMPOSED_EXAMPLES:
        assert ex["sql"] in suite, ex["description"]
