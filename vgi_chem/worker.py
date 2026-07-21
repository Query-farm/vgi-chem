# Copyright 2026 Query Farm LLC - https://query.farm
"""VGI worker exposing RDKit cheminformatics to SQL.

Assembles the chem functions in ``vgi_chem`` into a single ``chem`` catalog and
runs the worker over stdio (DuckDB subprocess) or HTTP. It brings molecular
descriptors, fingerprints/similarity, substructure search and InChI -- computed
with RDKit over SMILES strings -- into DuckDB as scalar functions, plus a
``lipinski`` rule-of-five table function.

This module is wheel-importable; the repo-root ``chem_worker.py`` is a thin
PEP 723 shim that re-exports :class:`ChemWorker` and :func:`main` from here, so
``uv run chem_worker.py`` (Makefile / ci/run-integration.sh / tests) keeps
working unchanged while the Docker image serves ``vgi_chem.worker:ChemWorker``.

Usage:
    uv run chem_worker.py               # serve over stdio (DuckDB subprocess)
    vgi-chem-worker                     # console-script entry (same, from the wheel)

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
from vgi.catalog import Catalog, Schema, Table

from vgi_chem.meta import keywords_json
from vgi_chem.scalars import SCALAR_FUNCTIONS
from vgi_chem.tables import TABLE_FUNCTIONS, ExampleMoleculesFunction

_FUNCTIONS: list[type] = [
    *SCALAR_FUNCTIONS,
    *TABLE_FUNCTIONS,
]

# VGI146/VGI311: `example_molecules` takes no arguments, so it always returns the
# same rows. Expose it as a regular browsable table (backed by the identically
# named table function) so an agent can `SELECT ... FROM chem.main.example_molecules`
# without parentheses — the credential-free discovery entry point into the catalog.
# The table's schema is derived from the function's bind(), so the two stay in
# lockstep, and the table carries its own discovery tags + natural primary key.
_DISCOVERY_TABLES: list[Table] = [
    Table(
        name="example_molecules",
        function=ExampleMoleculesFunction,
        comment="Curated registry of well-known molecules with pre-computed descriptors (discovery table).",
        primary_key=(("name",),),
        not_null=("name", "smiles", "formula"),
        column_comments={
            "name": "Common name of the molecule.",
            "smiles": "SMILES string you can pass to any chem function.",
            "formula": "Hill-system molecular formula.",
            "mol_weight": "Average molecular weight in g/mol.",
            "logp": "Crippen logP (octanol-water partition estimate).",
            "tpsa": "Topological polar surface area in Angstrom^2.",
            "h_bond_donors": "Lipinski hydrogen-bond donor count.",
            "h_bond_acceptors": "Lipinski hydrogen-bond acceptor count.",
            "num_rings": "Ring count (SSSR).",
            "drug_like": "True if the molecule passes all four Lipinski rules.",
        },
        tags={
            "vgi.title": "Example Molecule Registry Table",
            "vgi.doc_llm": (
                "## `example_molecules` (table)\n\n"
                "A browsable registry of well-known molecules (water, ethanol, benzene, aspirin, "
                "caffeine, ibuprofen, glucose, paracetamol, nicotine, penicillin G, ...) with "
                "ready-computed descriptors, exposed as a regular table you can read without "
                "parentheses.\n\n"
                "Columns: `name` (primary key), `smiles`, `formula`, `mol_weight` (g/mol), `logp`, "
                "`tpsa` (Angstrom^2), `h_bond_donors`, `h_bond_acceptors`, `num_rings`, and "
                "`drug_like`. Read it to obtain valid SMILES strings to feed the scalar functions, "
                "or to compare descriptor ranges across common drugs. Backed by the identically "
                "named table function, so the rows are identical."
            ),
            "vgi.doc_md": (
                "# `example_molecules`\n\n"
                "A curated, browsable registry of common molecules and their descriptors, exposed "
                "as a regular table you can read without parentheses -- the zero-argument discovery "
                "entry point for the `chem` catalog.\n\n"
                "## Columns\n\n"
                "- `name` (`VARCHAR`, primary key), `smiles`, `formula`\n"
                "- `mol_weight` (`DOUBLE`, g/mol), `logp` (`DOUBLE`), `tpsa` (`DOUBLE`, Angstrom^2)\n"
                "- `h_bond_donors`, `h_bond_acceptors`, `num_rings` (`INTEGER`)\n"
                "- `drug_like` (`BOOLEAN`) -- passes all four Lipinski rules\n\n"
                "Read it directly to discover valid SMILES inputs to feed the scalar functions, or "
                "to compare descriptor ranges across common drugs."
            ),
            "vgi.keywords": keywords_json(
                [
                    "example",
                    "molecules",
                    "registry",
                    "reference",
                    "discovery",
                    "smiles",
                    "sample",
                    "drug-like",
                    "browse",
                    "catalog",
                ]
            ),
            "vgi.category": "reference",
            "domain": "cheminformatics",
            "vgi.example_queries": json.dumps(
                [
                    {
                        "description": "Browse the built-in molecules from lightest to heaviest.",
                        "sql": (
                            "SELECT name, formula, mol_weight FROM chem.main.example_molecules ORDER BY mol_weight"
                        ),
                    },
                    {
                        "description": "How many example molecules are drug-like (pass all four Lipinski rules)?",
                        "sql": "SELECT count(*) AS drug_like FROM chem.main.example_molecules WHERE drug_like",
                    },
                ]
            ),
        },
    ),
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
            "runs inline over whole tables, so a column of compounds becomes a column of "
            "properties, identifiers, fingerprints or pass/fail flags in a single query.\n\n"
            "## Getting started\n\n"
            "Attach the catalog with `ATTACH 'chem' (TYPE vgi, LOCATION 'uv run chem_worker.py')`, "
            "then call any function inline in a query. Pass a SMILES string to get a molecular "
            "formula (`C9H8O4` for aspirin), a molecular weight (~180.16 g/mol), an InChIKey, a "
            "Tanimoto similarity score, or a Lipinski drug-likeness breakdown -- no export to "
            "Python required. New here? Browse the built-in `example_molecules()` table for "
            "ready-to-use SMILES and their pre-computed descriptors, then see each function's own "
            "example queries for ready-to-run SQL.\n\n"
            "Source and issues: "
            "[Query-farm/vgi-chem](https://github.com/Query-farm/vgi-chem).\n"
        ),
        # VGI509: at least one catalog-level, guaranteed-runnable example. These execute
        # against the attached worker under `--execute`; each is fully catalog-qualified.
        "vgi.executable_examples": json.dumps(
            [
                {
                    "description": "Molecular formula of aspirin.",
                    "sql": "SELECT chem.main.mol_formula('CC(=O)OC1=CC=CC=C1C(=O)O') AS formula",
                    "expected_result": [["C9H8O4"]],
                },
                {
                    "description": "Tanimoto self-similarity is exactly 1.0.",
                    "sql": "SELECT chem.main.tanimoto('CCO', 'CCO') AS sim",
                    "expected_result": [[1.0]],
                },
                {
                    "description": "Browse the built-in example molecules by molecular weight.",
                    "sql": ("SELECT name, formula FROM chem.main.example_molecules() ORDER BY mol_weight LIMIT 3"),
                },
            ]
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
                    "reference_sql": "SELECT chem.main.mol_formula('CC(=O)OC1=CC=CC=C1C(=O)O')",
                    "ignore_column_names": True,
                },
                {
                    "name": "validate_smiles",
                    "prompt": "Is the string 'xyz' a valid SMILES molecule? Return a boolean.",
                    "reference_sql": "SELECT chem.main.is_valid_smiles('xyz')",
                    "ignore_column_names": True,
                },
                {
                    "name": "canonicalize_smiles",
                    "prompt": ("Rewrite the SMILES 'OCC' into its canonical form (it denotes ethanol)."),
                    "reference_sql": "SELECT chem.main.canonical_smiles('OCC')",
                    "ignore_column_names": True,
                },
                {
                    "name": "inchi_of_ethanol",
                    "prompt": "What is the standard InChI string for ethanol, whose SMILES is CCO?",
                    "reference_sql": "SELECT chem.main.inchi('CCO')",
                    "ignore_column_names": True,
                },
                {
                    "name": "inchikey",
                    "prompt": ("Give the InChIKey identifier for aspirin, whose SMILES is CC(=O)OC1=CC=CC=C1C(=O)O."),
                    "reference_sql": "SELECT chem.main.inchikey('CC(=O)OC1=CC=CC=C1C(=O)O')",
                    "ignore_column_names": True,
                },
                {
                    "name": "molecular_weight",
                    "prompt": (
                        "Compute the molecular weight of aspirin (SMILES "
                        "CC(=O)OC1=CC=CC=C1C(=O)O), rounded to two decimal places."
                    ),
                    "reference_sql": "SELECT ROUND(chem.main.mol_weight('CC(=O)OC1=CC=CC=C1C(=O)O'), 2)",
                    "ignore_column_names": True,
                },
                {
                    "name": "exact_mass_threshold",
                    "prompt": (
                        "Is the monoisotopic (exact) mass of ethanol (SMILES CCO) between "
                        "46.0 and 46.1 daltons? Return a boolean."
                    ),
                    "reference_sql": "SELECT chem.main.exact_mass('CCO') BETWEEN 46.0 AND 46.1",
                    "ignore_column_names": True,
                },
                {
                    "name": "logp_lipophilic",
                    "prompt": (
                        "Is benzene (SMILES c1ccccc1) lipophilic, i.e. is its Crippen logP "
                        "greater than 1? Return a boolean."
                    ),
                    "reference_sql": "SELECT chem.main.logp('c1ccccc1') > 1",
                    "ignore_column_names": True,
                },
                {
                    "name": "tpsa_threshold",
                    "prompt": (
                        "Is the topological polar surface area (TPSA) of aspirin (SMILES "
                        "CC(=O)OC1=CC=CC=C1C(=O)O) greater than 50 square angstroms? Return a boolean."
                    ),
                    "reference_sql": "SELECT chem.main.tpsa('CC(=O)OC1=CC=CC=C1C(=O)O') > 50",
                    "ignore_column_names": True,
                },
                {
                    "name": "heavy_atom_count",
                    "prompt": "How many heavy (non-hydrogen) atoms does benzene (SMILES c1ccccc1) have?",
                    "reference_sql": "SELECT chem.main.num_atoms('c1ccccc1')",
                    "ignore_column_names": True,
                },
                {
                    "name": "ring_count",
                    "prompt": ("How many rings does caffeine (SMILES CN1C=NC2=C1C(=O)N(C(=O)N2C)C) have?"),
                    "reference_sql": "SELECT chem.main.num_rings('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')",
                    "ignore_column_names": True,
                },
                {
                    "name": "rotatable_bond_count",
                    "prompt": ("How many rotatable bonds does ibuprofen (SMILES CC(C)Cc1ccc(cc1)C(C)C(=O)O) have?"),
                    "reference_sql": "SELECT chem.main.num_rotatable_bonds('CC(C)Cc1ccc(cc1)C(C)C(=O)O')",
                    "ignore_column_names": True,
                },
                {
                    "name": "h_bond_donors",
                    "prompt": ("How many hydrogen-bond donors does glucose (SMILES OCC1OC(O)C(O)C(O)C1O) have?"),
                    "reference_sql": "SELECT chem.main.num_h_donors('OCC1OC(O)C(O)C(O)C1O')",
                    "ignore_column_names": True,
                },
                {
                    "name": "h_bond_acceptors",
                    "prompt": (
                        "How many hydrogen-bond acceptors does caffeine (SMILES CN1C=NC2=C1C(=O)N(C(=O)N2C)C) have?"
                    ),
                    "reference_sql": "SELECT chem.main.num_h_acceptors('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')",
                    "ignore_column_names": True,
                },
                {
                    "name": "morgan_fingerprint_stable",
                    "prompt": (
                        "Do two copies of ethanol (SMILES CCO) produce the same Morgan fingerprint? Return a boolean."
                    ),
                    "reference_sql": (
                        "SELECT chem.main.morgan_fingerprint('CCO') = chem.main.morgan_fingerprint('CCO')"
                    ),
                    "ignore_column_names": True,
                },
                {
                    "name": "tanimoto_self_similarity",
                    "prompt": "What is the Tanimoto similarity between ethanol (SMILES CCO) and itself?",
                    "reference_sql": "SELECT chem.main.tanimoto('CCO', 'CCO')",
                    "ignore_column_names": True,
                },
                {
                    "name": "substructure_search",
                    "prompt": (
                        "Does phenol (SMILES c1ccccc1O) contain a benzene ring, "
                        "expressed as the SMARTS pattern c1ccccc1? Return a boolean."
                    ),
                    "reference_sql": "SELECT chem.main.substructure_match('c1ccccc1O', 'c1ccccc1')",
                    "ignore_column_names": True,
                },
                {
                    "name": "lipinski_drug_like",
                    "prompt": (
                        "Does aspirin (SMILES CC(=O)OC1=CC=CC=C1C(=O)O) satisfy all four "
                        "Lipinski rule-of-five criteria? Return a single boolean."
                    ),
                    "reference_sql": ("SELECT bool_and(passes) FROM chem.main.lipinski('CC(=O)OC1=CC=CC=C1C(=O)O')"),
                    "ignore_column_names": True,
                },
                {
                    "name": "drug_like_predicate",
                    "prompt": (
                        "Using a single scalar drug-likeness predicate, is aspirin (SMILES "
                        "CC(=O)OC1=CC=CC=C1C(=O)O) drug-like? Return a boolean."
                    ),
                    "reference_sql": "SELECT chem.main.drug_like('CC(=O)OC1=CC=CC=C1C(=O)O')",
                    "ignore_column_names": True,
                },
                {
                    "name": "browse_example_molecules",
                    "prompt": (
                        "How many molecules are listed in the worker's built-in example molecule reference table?"
                    ),
                    "reference_sql": "SELECT count(*) FROM chem.main.example_molecules()",
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
                        {
                            "name": "reference",
                            "title": "Reference & discovery",
                            "description": (
                                "Browsable registries of example molecules and their descriptors "
                                "for discovering valid SMILES inputs."
                            ),
                        },
                    ]
                ),
                # VGI506/VGI515: representative, catalog-qualified example queries for
                # the schema, each carrying a human-readable description.
                "vgi.example_queries": json.dumps(
                    [
                        {
                            "description": "Check that a SMILES string parses to a real molecule.",
                            "sql": "SELECT chem.main.is_valid_smiles('CCO')",
                        },
                        {
                            "description": "Canonicalize a SMILES so equivalent spellings compare equal.",
                            "sql": "SELECT chem.main.canonical_smiles('OCC')",
                        },
                        {
                            "description": "Molecular formula of aspirin in Hill notation.",
                            "sql": "SELECT chem.main.mol_formula('CC(=O)OC1=CC=CC=C1C(=O)O')",
                        },
                        {
                            "description": "Molecular weight of aspirin, rounded to two decimals.",
                            "sql": "SELECT ROUND(chem.main.mol_weight('CC(=O)OC1=CC=CC=C1C(=O)O'), 2)",
                        },
                        {
                            "description": "InChIKey structure hash of aspirin for joins and lookup.",
                            "sql": "SELECT chem.main.inchikey('CC(=O)OC1=CC=CC=C1C(=O)O')",
                        },
                        {
                            "description": "Tanimoto self-similarity is exactly 1.0.",
                            "sql": "SELECT chem.main.tanimoto('CCO', 'CCO')",
                        },
                        {
                            "description": "Does phenol contain a benzene ring (SMARTS match)?",
                            "sql": "SELECT chem.main.substructure_match('c1ccccc1O', 'c1ccccc1')",
                        },
                        {
                            "description": "Lipinski rule-of-five breakdown for aspirin, one row per rule.",
                            "sql": (
                                "SELECT rule, value, passes "
                                "FROM chem.main.lipinski('CC(=O)OC1=CC=CC=C1C(=O)O') ORDER BY rule"
                            ),
                        },
                    ]
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
                    "substructure search, and drug-likeness. Scalars return one value per SMILES "
                    "row; the sole set-returning function, `lipinski`, expands one molecule into "
                    "one row per rule.\n\n"
                    "## Usage\n\n"
                    "Call a function by its qualified name, `chem.main.<fn>` — or `chem.<fn>`, since "
                    "`main` is the default schema. Scalars take positional arguments; `lipinski` is a "
                    "set-returning table function.\n\n"
                    "## Notes\n\n"
                    "Every function is total: `NULL` in yields `NULL` out, and invalid (non-NULL) "
                    "SMILES yields `NULL`/`false`/no-rows rather than raising."
                ),
            },
            tables=list(_DISCOVERY_TABLES),
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
