# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.8.5",
#     "rdkit>=2024.3",
#     "pyarrow",
# ]
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

from vgi_chem.meta import keywords_json
from vgi_chem.scalars import SCALAR_FUNCTIONS
from vgi_chem.tables import TABLE_FUNCTIONS

_FUNCTIONS: list[type] = [
    *SCALAR_FUNCTIONS,
    *TABLE_FUNCTIONS,
]

_CHEM_CATALOG = Catalog(
    name="chem",
    default_schema="main",
    comment="RDKit cheminformatics for SQL: molecular descriptors, fingerprints, substructure, InChI.",
    tags={
        "vgi.title": "Cheminformatics for SQL (RDKit)",
        "vgi.keywords": keywords_json(
            [
                "cheminformatics",
                "chemistry",
                "rdkit",
                "smiles",
                "smarts",
                "molecule",
                "descriptor",
                "molecular weight",
                "logp",
                "tpsa",
                "fingerprint",
                "morgan",
                "ecfp",
                "tanimoto",
                "similarity",
                "substructure",
                "inchi",
                "inchikey",
                "lipinski",
                "druglikeness",
            ]
        ),
        "vgi.doc_llm": (
            "Cheminformatics over SMILES strings, computed with RDKit. Validate and canonicalize "
            "SMILES; compute molecular descriptors (molecular weight, exact mass, Crippen logP, "
            "TPSA, heavy-atom/ring/rotatable-bond/H-bond-donor/acceptor counts); derive molecular "
            "formula, InChI and InChIKey identifiers; build Morgan (ECFP-like) fingerprints and "
            "measure Tanimoto similarity between molecules; run SMARTS substructure matches; and "
            "break a molecule down against the Lipinski rule of five. Use for molecular property "
            "calculation, similarity search, substructure filtering and drug-likeness screening in SQL."
        ),
        "vgi.doc_md": (
            "# chem\n\n"
            "Cheminformatics for DuckDB, computed from SMILES strings with "
            "[RDKit](https://www.rdkit.org/).\n\n"
            "- **Validity & identity**: `is_valid_smiles`, `canonical_smiles`, `mol_formula`, "
            "`inchi`, `inchikey`.\n"
            "- **Descriptors**: `mol_weight`, `exact_mass`, `logp`, `tpsa`, `num_atoms`, "
            "`num_rings`, `num_rotatable_bonds`, `num_h_donors`, `num_h_acceptors`.\n"
            "- **Fingerprints & similarity**: `morgan_fingerprint`, `tanimoto`.\n"
            "- **Substructure**: `substructure_match` (SMARTS).\n"
            "- **Drug-likeness**: `lipinski(smiles)` table function (rule-of-five breakdown).\n"
        ),
        "vgi.author": "Query.Farm",
        "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
        "vgi.license": "MIT",
        "vgi.support_contact": "https://github.com/Query-farm/vgi-chem/issues",
        "vgi.support_policy_url": "https://github.com/Query-farm/vgi-chem/blob/main/README.md",
    },
    source_url="https://github.com/Query-farm/vgi-chem",
    schemas=[
        Schema(
            name="main",
            comment="RDKit cheminformatics for SQL: descriptors, fingerprints, substructure, InChI.",
            tags={
                "vgi.title": "Chem — main",
                "vgi.keywords": keywords_json(
                    [
                        "cheminformatics",
                        "chemistry",
                        "smiles",
                        "smarts",
                        "molecule",
                        "descriptor",
                        "mol_weight",
                        "exact_mass",
                        "logp",
                        "tpsa",
                        "fingerprint",
                        "morgan",
                        "tanimoto",
                        "substructure",
                        "inchi",
                        "inchikey",
                        "lipinski",
                        "druglikeness",
                    ]
                ),
                # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
                "domain": "cheminformatics",
                "category": "molecular-analysis",
                "topic": "descriptors-fingerprints-substructure",
                # VGI506: representative, catalog-qualified example queries for the schema.
                "vgi.example_queries": (
                    "SELECT chem.main.is_valid_smiles('CCO');\n"
                    "SELECT chem.main.canonical_smiles('OCC');\n"
                    "SELECT chem.main.mol_formula('CC(=O)OC1=CC=CC=C1C(=O)O');\n"
                    "SELECT ROUND(chem.main.mol_weight('CC(=O)OC1=CC=CC=C1C(=O)O'), 2);\n"
                    "SELECT chem.main.inchikey('CC(=O)OC1=CC=CC=C1C(=O)O');\n"
                    "SELECT chem.main.tanimoto('CCO', 'CCO');\n"
                    "SELECT chem.main.substructure_match('c1ccccc1O', 'c1ccccc1');\n"
                    "SELECT * FROM chem.main.lipinski('CC(=O)OC1=CC=CC=C1C(=O)O');"
                ),
                "vgi.doc_llm": (
                    "Cheminformatics functions over SMILES strings: validate/canonicalize SMILES, "
                    "compute molecular descriptors (weight, exact mass, logP, TPSA, atom/ring/bond "
                    "counts), derive formula/InChI/InChIKey identifiers, build Morgan fingerprints "
                    "and Tanimoto similarity, run SMARTS substructure matches, and evaluate the "
                    "Lipinski rule of five."
                ),
                "vgi.doc_md": (
                    "# chem.main\n\n"
                    "The single schema of the `chem` catalog. It groups every cheminformatics "
                    "function exposed by this worker, all computed from SMILES strings via "
                    "RDKit.\n\n"
                    "## Contents\n\n"
                    "- **Validity & identity**: `is_valid_smiles`, `canonical_smiles`, "
                    "`mol_formula`, `inchi`, `inchikey`.\n"
                    "- **Numeric descriptors**: `mol_weight`, `exact_mass`, `logp`, `tpsa`.\n"
                    "- **Count descriptors**: `num_atoms`, `num_rings`, `num_rotatable_bonds`, "
                    "`num_h_donors`, `num_h_acceptors`.\n"
                    "- **Fingerprints & similarity**: `morgan_fingerprint`, `tanimoto`.\n"
                    "- **Substructure**: `substructure_match` (SMARTS).\n"
                    "- **Drug-likeness**: `lipinski(smiles)` table function.\n\n"
                    "## Usage\n\n"
                    "Reference functions as `chem.main.<fn>(...)` (or `chem.<fn>` since `main` is "
                    "the default schema). Scalars take positional arguments; `lipinski` is a "
                    "set-returning table function.\n\n"
                    "## Notes\n\n"
                    "Every function is total: `NULL` in yields `NULL` out, and invalid (non-NULL) "
                    "SMILES yields `NULL`/`false`/no-rows rather than raising."
                ),
            },
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
