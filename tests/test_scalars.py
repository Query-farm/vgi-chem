"""End-to-end tests for the per-row scalar chem functions.

These spawn ``chem_worker.py`` as a subprocess via ``vgi.client.Client`` and
call each scalar exactly as DuckDB would after ``ATTACH``, exercising the arity
overloads (``morgan_fingerprint(smiles)`` / ``(smiles, radius, nbits)`` and
``tanimoto(a, b)`` / ``(a, b, radius)``). The per-row SMILES column travels in
the input batch (a ``Param``); only the constant ``radius`` / ``nbits``
arguments go in ``positional``.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pytest
from vgi import Arguments
from vgi.client import Client

_WORKER = str(Path(__file__).resolve().parent.parent / "chem_worker.py")

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
BENZENE = "c1ccccc1"
ETHANOL = "CCO"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


@pytest.fixture(scope="module")
def client() -> Iterator[Client]:
    # Current interpreter (deps already installed) + worker_limit=1 so output
    # order matches input order for deterministic per-row assertions.
    with Client(f"{sys.executable} {_WORKER}", worker_limit=1) as c:
        yield c


def _scalar(
    client: Client,
    name: str,
    columns: dict[str, list],
    *,
    positional: list[pa.Scalar] | None = None,
) -> list:
    batch = pa.RecordBatch.from_pydict({k: pa.array(v, type=pa.string()) for k, v in columns.items()})
    results = list(
        client.scalar_function(
            function_name=name,
            input=iter([batch]),
            arguments=Arguments(positional=positional or []),
        )
    )
    return results[0]["result"].to_pylist()


class TestValidity:
    def test_is_valid(self, client: Client) -> None:
        assert _scalar(client, "is_valid_smiles", {"s": [ETHANOL, "xyz", None]}) == [
            True,
            False,
            None,
        ]

    def test_canonical(self, client: Client) -> None:
        assert _scalar(client, "canonical_smiles", {"s": ["OCC", "xyz"]}) == ["CCO", None]


class TestDescriptors:
    def test_formula(self, client: Client) -> None:
        assert _scalar(client, "mol_formula", {"s": [ASPIRIN, "xyz"]}) == ["C9H8O4", None]

    def test_weight(self, client: Client) -> None:
        out = _scalar(client, "mol_weight", {"s": [ASPIRIN]})
        assert math.isclose(out[0], 180.16, abs_tol=0.05)

    def test_counts(self, client: Client) -> None:
        assert _scalar(client, "num_atoms", {"s": [BENZENE]}) == [6]
        assert _scalar(client, "num_rings", {"s": [BENZENE]}) == [1]
        assert _scalar(client, "num_h_donors", {"s": [ETHANOL]}) == [1]

    def test_invalid_null(self, client: Client) -> None:
        assert _scalar(client, "mol_weight", {"s": ["xyz", None]}) == [None, None]


class TestInchi:
    def test_inchikey(self, client: Client) -> None:
        assert _scalar(client, "inchikey", {"s": [ASPIRIN]}) == ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N"]


class TestFingerprintsAndSimilarity:
    def test_morgan_default(self, client: Client) -> None:
        out = _scalar(client, "morgan_fingerprint", {"s": [ETHANOL]})
        assert out[0] is not None and len(out[0]) == 2048 // 4

    def test_morgan_params(self, client: Client) -> None:
        out = _scalar(
            client,
            "morgan_fingerprint",
            {"s": [ETHANOL]},
            positional=[pa.scalar(2), pa.scalar(1024)],
        )
        assert out[0] is not None and len(out[0]) == 1024 // 4

    def test_tanimoto_self(self, client: Client) -> None:
        out = _scalar(client, "tanimoto", {"a": [ASPIRIN], "b": [ASPIRIN]})
        assert out == [1.0]

    def test_tanimoto_distinct(self, client: Client) -> None:
        out = _scalar(client, "tanimoto", {"a": [ASPIRIN], "b": [CAFFEINE]})
        assert 0.0 <= out[0] < 1.0

    def test_tanimoto_radius(self, client: Client) -> None:
        out = _scalar(
            client,
            "tanimoto",
            {"a": [ASPIRIN], "b": [ASPIRIN]},
            positional=[pa.scalar(3)],
        )
        assert out == [1.0]


class TestSubstructure:
    def test_match(self, client: Client) -> None:
        out = _scalar(client, "substructure_match", {"s": ["c1ccccc1O"], "p": [BENZENE]})
        assert out == [True]

    def test_no_match(self, client: Client) -> None:
        out = _scalar(client, "substructure_match", {"s": [ETHANOL], "p": [BENZENE]})
        assert out == [False]

    def test_invalid_smiles_null(self, client: Client) -> None:
        out = _scalar(client, "substructure_match", {"s": ["xyz"], "p": [BENZENE]})
        assert out == [None]
