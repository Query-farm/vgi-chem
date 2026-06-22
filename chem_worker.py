# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python",
#     "rdkit>=2024.3",
#     "pyarrow",
# ]
#
# [tool.uv.sources]
# vgi-python = { path = "../vgi-python" }
# ///
"""VGI worker exposing RDKit cheminformatics to SQL.

Assembles the chem functions in ``vgi_chem`` into a single ``chem`` catalog and
runs the worker over stdio (DuckDB subprocess) or HTTP. It brings molecular
descriptors, fingerprints/similarity, substructure search and InChI -- computed
with RDKit over SMILES strings -- into DuckDB as scalar functions, plus a
``lipinski`` rule-of-five table function.

Usage:
    uv run chem_worker.py               # serve over stdio (DuckDB subprocess)

    INSTALL vgi FROM community; LOAD vgi;
    ATTACH 'chem' (TYPE vgi, LOCATION 'uv run chem_worker.py');

    SELECT chem.is_valid_smiles('CCO');                              -- true
    SELECT chem.mol_formula('CC(=O)OC1=CC=CC=C1C(=O)O');             -- 'C9H8O4'
    SELECT chem.mol_weight('CC(=O)OC1=CC=CC=C1C(=O)O');              -- ~180.16
    SELECT chem.inchikey('CC(=O)OC1=CC=CC=C1C(=O)O');                -- 'BSYNRYMUTXBXSQ-...'
    SELECT chem.tanimoto('CCO', 'CCO');                              -- 1.0
    SELECT chem.substructure_match('c1ccccc1O', 'c1ccccc1');         -- true
    SELECT * FROM chem.lipinski('CC(=O)OC1=CC=CC=C1C(=O)O');
"""

from __future__ import annotations

from vgi import Worker
from vgi.catalog import Catalog, Schema

from vgi_chem.scalars import SCALAR_FUNCTIONS
from vgi_chem.tables import TABLE_FUNCTIONS

_FUNCTIONS: list[type] = [
    *SCALAR_FUNCTIONS,
    *TABLE_FUNCTIONS,
]

_CHEM_CATALOG = Catalog(
    name="chem",
    default_schema="main",
    schemas=[
        Schema(
            name="main",
            comment="RDKit cheminformatics for SQL: descriptors, fingerprints, substructure, InChI",
            functions=list(_FUNCTIONS),
        ),
    ],
)


class ChemWorker(Worker):
    """Worker process hosting the ``chem`` catalog."""

    catalog = _CHEM_CATALOG


def main() -> None:
    """Run the chem worker process (stdio or, via flags, HTTP)."""
    ChemWorker.main()


if __name__ == "__main__":
    main()
