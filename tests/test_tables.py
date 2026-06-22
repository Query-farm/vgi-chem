"""Integration tests for the ``lipinski`` table function.

Drives the function through the real bind -> init -> process lifecycle
in-process (no worker subprocess). The per-row functions are *scalars* and are
covered in ``test_scalars.py``.
"""

from __future__ import annotations

import pyarrow as pa

from vgi_chem.tables import LipinskiFunction

from .harness import invoke_table_function

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"


class TestLipinski:
    def test_columns_and_rows(self) -> None:
        table = invoke_table_function(LipinskiFunction, positional=(pa.scalar(ASPIRIN),))
        assert table.column_names == ["rule", "value", "passes"]
        assert table.num_rows == 4

    def test_rules_and_pass(self) -> None:
        table = invoke_table_function(LipinskiFunction, positional=(pa.scalar(ASPIRIN),))
        rules = table.column("rule").to_pylist()
        passes = table.column("passes").to_pylist()
        assert set(rules) == {
            "molecular_weight",
            "logp",
            "h_bond_donors",
            "h_bond_acceptors",
        }
        # Aspirin is drug-like: passes all four rules.
        assert all(passes)

    def test_invalid_smiles_no_rows(self) -> None:
        table = invoke_table_function(LipinskiFunction, positional=(pa.scalar("xyz"),))
        assert table.num_rows == 0
