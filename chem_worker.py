# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.9.0",
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

import json

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
            "# Cheminformatics in SQL with RDKit\n\n"
            "![RDKit logo](https://www.rdkit.org/Images/logo.png)\n\n"
            "**Run molecular descriptors, fingerprints, Tanimoto similarity, SMARTS "
            "substructure search and InChI/InChIKey generation directly in DuckDB SQL — "
            "computed from SMILES strings by [RDKit](https://www.rdkit.org/), the "
            "industry-standard open-source cheminformatics toolkit.**\n\n"
            "The `chem` catalog turns DuckDB into a cheminformatics engine. Instead of "
            "exporting a column of SMILES to Python, looping through RDKit, and joining the "
            "results back, you call chemistry functions inline in a query: validate and "
            "canonicalize structures, compute physicochemical properties, generate molecular "
            "fingerprints, score structural similarity, match SMARTS patterns, and screen for "
            "drug-likeness — all on whole tables of compounds at once. It is built for "
            "chemists, computational chemists, data scientists and drug-discovery teams who "
            "already keep their compound libraries in DuckDB, Parquet or CSV and want "
            "molecular analytics without leaving SQL.\n\n"
            "Every function is powered by [RDKit](https://github.com/rdkit/rdkit) "
            "(BSD-3-Clause), the open-source toolkit that underpins much of modern "
            "cheminformatics. RDKit is imported as an unmodified dependency and parses each "
            "SMILES (or SMARTS) string into a molecule before computing the requested "
            "property. The worker is deliberately robust: a malformed structure never raises "
            "or crashes a query — `NULL` input yields `NULL` output, and invalid (non-NULL) "
            "SMILES yields `NULL`, `false`, or no rows depending on the function. RDKit runs "
            "fully offline with no network access or model downloads, so results are "
            "deterministic and reproducible across runs. See the official "
            "[RDKit documentation](https://www.rdkit.org/docs/) for the underlying algorithms.\n\n"
            "## What you can compute\n\n"
            "The catalog spans the everyday cheminformatics workflow, grouped into navigation "
            "sections: structure **validity**, canonical **identity** (canonical SMILES plus "
            "InChI/InChIKey for deduplication and joins), numeric and count **descriptors** for "
            "property filtering and QSAR, circular **fingerprints** and Tanimoto **similarity** "
            "for search and clustering, SMARTS **substructure** matching for structural filters, "
            "and Lipinski **drug-likeness** screening. Everything operates on SMILES strings and "
            "runs inline over whole tables. List the schema to discover the individual functions "
            "and their signatures.\n\n"
            "## Example\n\n"
            "```sql\n"
            "INSTALL vgi FROM community; LOAD vgi;\n"
            "ATTACH 'chem' (TYPE vgi, LOCATION 'uv run chem_worker.py');\n\n"
            "SELECT chem.mol_formula('CC(=O)OC1=CC=CC=C1C(=O)O');   -- 'C9H8O4'\n"
            "SELECT ROUND(chem.mol_weight('CC(=O)OC1=CC=CC=C1C(=O)O'), 2);  -- 180.16\n"
            "SELECT chem.tanimoto('CCO', 'CCO');                    -- 1.0\n"
            "SELECT chem.substructure_match('c1ccccc1O', 'c1ccccc1');  -- true\n"
            "SELECT * FROM chem.lipinski('CC(=O)OC1=CC=CC=C1C(=O)O');\n"
            "```\n\n"
            "Source and issues: "
            "[Query-farm/vgi-chem](https://github.com/Query-farm/vgi-chem).\n"
        ),
        "vgi.author": "Query.Farm",
        "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
        "vgi.license": "MIT",
        "vgi.support_contact": "https://github.com/Query-farm/vgi-chem/issues",
        "vgi.support_policy_url": "https://github.com/Query-farm/vgi-chem/blob/main/README.md",
        # VGI152/VGI920: analyst-suitability suite. Each task's `prompt` is the only
        # field shown to the analyst; `reference_sql` is a deterministic grader-only
        # canonical solution. `ignore_column_names` relaxes the result-column labels
        # (analysts alias freely). All references are backend-deterministic RDKit calls.
        "vgi.agent_test_tasks": json.dumps(
            [
                {
                    "name": "molecular_formula",
                    "prompt": ("The SMILES for aspirin is CC(=O)OC1=CC=CC=C1C(=O)O. What is its molecular formula?"),
                    "reference_sql": "SELECT chem.mol_formula('CC(=O)OC1=CC=CC=C1C(=O)O')",
                    "ignore_column_names": True,
                },
                {
                    "name": "validate_smiles",
                    "prompt": "Is the string 'xyz' a valid SMILES molecule? Return a boolean.",
                    "reference_sql": "SELECT chem.is_valid_smiles('xyz')",
                    "ignore_column_names": True,
                },
                {
                    "name": "molecular_weight",
                    "prompt": (
                        "Compute the molecular weight of aspirin (SMILES "
                        "CC(=O)OC1=CC=CC=C1C(=O)O), rounded to two decimal places."
                    ),
                    "reference_sql": "SELECT ROUND(chem.mol_weight('CC(=O)OC1=CC=CC=C1C(=O)O'), 2)",
                    "ignore_column_names": True,
                },
                {
                    "name": "tanimoto_self_similarity",
                    "prompt": "What is the Tanimoto similarity between ethanol (SMILES CCO) and itself?",
                    "reference_sql": "SELECT chem.tanimoto('CCO', 'CCO')",
                    "ignore_column_names": True,
                },
                {
                    "name": "inchikey",
                    "prompt": ("Give the InChIKey identifier for aspirin, whose SMILES is CC(=O)OC1=CC=CC=C1C(=O)O."),
                    "reference_sql": "SELECT chem.inchikey('CC(=O)OC1=CC=CC=C1C(=O)O')",
                    "ignore_column_names": True,
                },
                {
                    "name": "substructure_search",
                    "prompt": (
                        "Does phenol (SMILES c1ccccc1O) contain a benzene ring, "
                        "expressed as the SMARTS pattern c1ccccc1? Return a boolean."
                    ),
                    "reference_sql": "SELECT chem.substructure_match('c1ccccc1O', 'c1ccccc1')",
                    "ignore_column_names": True,
                },
            ]
        ),
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
                "topic": "descriptors-fingerprints-substructure",
                # VGI413: ordered navigation registry; array order is display order.
                # Each function carries a matching `vgi.category` (VGI409/VGI411).
                "vgi.categories": json.dumps(
                    [
                        {
                            "name": "validity",
                            "title": "Validity",
                            "description": "Check whether a SMILES string parses to a real molecule.",
                        },
                        {
                            "name": "identity",
                            "title": "Identity & canonicalization",
                            "description": (
                                "Canonical SMILES and InChI/InChIKey identifiers for deduplication and stable joins."
                            ),
                        },
                        {
                            "name": "descriptors",
                            "title": "Molecular descriptors",
                            "description": (
                                "Physicochemical properties and structural counts "
                                "(weight, mass, logP, TPSA, atom/ring/bond counts, formula)."
                            ),
                        },
                        {
                            "name": "fingerprint",
                            "title": "Fingerprints",
                            "description": "Morgan (ECFP-style) circular fingerprints for vectorizing molecules.",
                        },
                        {
                            "name": "similarity",
                            "title": "Similarity",
                            "description": "Tanimoto similarity scoring between molecules for search and clustering.",
                        },
                        {
                            "name": "substructure",
                            "title": "Substructure search",
                            "description": "SMARTS pattern matching for structural filtering.",
                        },
                        {
                            "name": "druglikeness",
                            "title": "Drug-likeness",
                            "description": "Lipinski rule-of-five screening for drug-likeness.",
                        },
                    ]
                ),
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
                    "Functions are organized into navigation categories: validity, identity and "
                    "canonicalization, molecular descriptors, fingerprints, similarity, "
                    "substructure search, and drug-likeness. List the schema to enumerate the "
                    "functions in each category, including their signatures and column docs.\n\n"
                    "## Usage\n\n"
                    "Call a function by its qualified name, `chem.main.<fn>` — or `chem.<fn>`, since "
                    "`main` is the default schema. Scalars take positional arguments; `lipinski` is a "
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
