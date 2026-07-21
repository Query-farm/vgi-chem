"""Set-returning chem table functions for DuckDB.

The Lipinski rule-of-five breakdown expands one molecule into **several rows**
(one per rule), so it is exposed as a **table function** -- the form that accepts
DuckDB ``name := value`` arguments. The per-row, single-value cheminformatics
functions are *scalars* and live in :mod:`vgi_chem.scalars`.

    SELECT * FROM chem.main.lipinski('CC(=O)OC1=CC=CC=C1C(=O)O');
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, ClassVar

import pyarrow as pa
from vgi.arguments import Arg
from vgi.metadata import FunctionExample
from vgi.table_function import (
    BindParams,
    ProcessParams,
    TableCardinality,
    TableFunctionGenerator,
    bind_fixed_schema,
    init_single_worker,
)
from vgi_rpc.rpc import OutputCollector

from . import chem
from .meta import attach_example_queries, object_tags
from .schema_utils import field


@dataclass(kw_only=True)
class _LipinskiArgs:
    """``lipinski(smiles)``."""

    smiles: Annotated[str, Arg(0, arrow_type=pa.string(), doc="SMILES string to analyze.")]


_LIPINSKI_SCHEMA = pa.schema(
    [
        field("rule", pa.string(), "Lipinski rule name (molecular_weight, logp, ...).", nullable=False),
        field("value", pa.float64(), "Computed descriptor value for the rule.", nullable=False),
        field("passes", pa.bool_(), "True if this rule's threshold is satisfied.", nullable=False),
    ]
)


@init_single_worker
@bind_fixed_schema
class LipinskiFunction(TableFunctionGenerator[_LipinskiArgs]):
    """Lipinski rule-of-five breakdown: one ``(rule, value, passes)`` row per rule.

    Rules: MW <= 500, logP <= 5, HBD <= 5, HBA <= 10. An invalid SMILES yields
    **no rows** (rather than raising).
    """

    FIXED_SCHEMA: ClassVar[pa.Schema] = _LIPINSKI_SCHEMA

    class Meta:
        """VGI table function metadata (name, description, categories, examples)."""

        name = "lipinski"
        description = "Lipinski rule-of-five breakdown (MW<=500, logP<=5, HBD<=5, HBA<=10), one row per rule"
        categories = ["chem", "druglikeness"]
        tags = {
            **object_tags(
                title="Lipinski Rule-of-Five Breakdown",
                description_llm=(
                    "## lipinski\n\n"
                    "A **set-returning** table function that breaks a molecule (given as SMILES) "
                    "down against the **Lipinski rule of five**, emitting one row per rule with "
                    "its computed value and whether the threshold is satisfied.\n\n"
                    "**Use it** for drug-likeness screening: keep compounds that pass all four "
                    "rules, or inspect exactly which rule a compound violates.\n\n"
                    "Rules evaluated: molecular weight <= 500, logP <= 5, H-bond donors <= 5, "
                    "H-bond acceptors <= 10.\n\n"
                    "- **Input**: one SMILES string (`VARCHAR`) as a positional table argument.\n"
                    "- **Output**: rows of `(rule VARCHAR, value DOUBLE, passes BOOLEAN)`.\n\n"
                    "**Edge cases**: an invalid SMILES yields **no rows** (rather than raising), so "
                    "`bool_and(passes)` over an empty result behaves accordingly. Aggregate with "
                    "`bool_and(passes)` to get a single drug-like flag."
                ),
                description_md=(
                    "# lipinski\n\n"
                    "Lipinski rule-of-five breakdown for a SMILES string, one row per rule.\n\n"
                    "## Result\n\n"
                    "Four rows -- `molecular_weight`, `logp`, `h_bond_donors`, `h_bond_acceptors` "
                    "-- each with its computed `value` and a `passes` flag. Aggregate the `passes` "
                    "column with `bool_and` for a single drug-like verdict, or keep the "
                    "non-passing rows to see which criterion a compound violates.\n\n"
                    "## Notes\n\n"
                    "Rules: MW <= 500, logP <= 5, HBD <= 5, HBA <= 10. An invalid SMILES returns "
                    "no rows."
                ),
                keywords=[
                    "lipinski",
                    "rule of five",
                    "ro5",
                    "druglikeness",
                    "drug-like",
                    "screening",
                    "molecular weight",
                    "logp",
                    "hbd",
                    "hba",
                    "filter",
                ],
                relative_path="vgi_chem/tables.py",
                category="druglikeness",
            ),
            # VGI414/VGI307: the retired free-form `vgi.result_columns_md` is migrated to
            # the structured `vgi.result_columns_schema` (validated JSON: name/type/description).
            "vgi.result_columns_schema": json.dumps(
                [
                    {
                        "name": "rule",
                        "type": "VARCHAR",
                        "description": (
                            "Lipinski rule name: molecular_weight, logp, h_bond_donors, or h_bond_acceptors."
                        ),
                    },
                    {
                        "name": "value",
                        "type": "DOUBLE",
                        "description": "Computed descriptor value for the rule.",
                    },
                    {
                        "name": "passes",
                        "type": "BOOLEAN",
                        "description": "True if this rule's rule-of-five threshold is satisfied.",
                    },
                ]
            ),
        }
        examples = [
            FunctionExample(
                sql=("SELECT rule, value, passes FROM chem.main.lipinski('CC(=O)OC1=CC=CC=C1C(=O)O') ORDER BY rule"),
                description="Rule-of-five breakdown for aspirin, one row per rule",
            ),
            FunctionExample(
                sql=("SELECT bool_and(passes) AS drug_like FROM chem.main.lipinski('CC(=O)OC1=CC=CC=C1C(=O)O')"),
                description="Does aspirin pass all four rules?",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_LipinskiArgs]) -> TableCardinality:
        """Report the fixed row count (one row per Lipinski rule)."""
        return TableCardinality(estimate=4, max=4)

    @classmethod
    def process(cls, params: ProcessParams[_LipinskiArgs], state: None, out: OutputCollector) -> None:
        """Emit one row per Lipinski rule for the input SMILES."""
        rows = chem.lipinski(params.args.smiles) or []
        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "rule": [r[0] for r in rows],
                    "value": [r[1] for r in rows],
                    "passes": [r[2] for r in rows],
                },
                schema=params.output_schema,
            )
        )
        out.finish()


@dataclass(kw_only=True)
class _NoArgs:
    """A discovery table function that takes no arguments."""


_EXAMPLE_MOLECULES_SCHEMA = pa.schema(
    [
        field("name", pa.string(), "Common name of the molecule.", nullable=False),
        field("smiles", pa.string(), "SMILES string you can pass to any chem function.", nullable=False),
        field("formula", pa.string(), "Hill-system molecular formula.", nullable=False),
        field("mol_weight", pa.float64(), "Average molecular weight in g/mol.", nullable=False),
        field("logp", pa.float64(), "Crippen logP (octanol-water partition estimate).", nullable=False),
        field("tpsa", pa.float64(), "Topological polar surface area in Angstrom^2.", nullable=False),
        field("h_bond_donors", pa.int32(), "Lipinski hydrogen-bond donor count.", nullable=False),
        field("h_bond_acceptors", pa.int32(), "Lipinski hydrogen-bond acceptor count.", nullable=False),
        field("num_rings", pa.int32(), "Ring count (SSSR).", nullable=False),
        field("drug_like", pa.bool_(), "True if the molecule passes all four Lipinski rules.", nullable=False),
    ]
)


@init_single_worker
@bind_fixed_schema
class ExampleMoleculesFunction(TableFunctionGenerator[_NoArgs]):
    """A curated, browsable registry of well-known molecules with ready descriptors.

    Takes no arguments, so an agent can scan it directly (``SELECT * FROM
    chem.main.example_molecules()``) to discover real SMILES strings and their
    pre-computed properties before calling any per-molecule function. Every
    descriptor column is computed live from the ``smiles`` via the same functions
    the catalog exposes, so the table can never drift from them.
    """

    FIXED_SCHEMA: ClassVar[pa.Schema] = _EXAMPLE_MOLECULES_SCHEMA

    class Meta:
        """VGI table function metadata (name, description, categories, examples)."""

        name = "example_molecules"
        description = "Curated registry of well-known molecules with pre-computed descriptors (no arguments)"
        categories = ["chem", "reference"]
        tags = {
            **object_tags(
                title="Example Molecule Registry",
                description_llm=(
                    "## example_molecules()\n\n"
                    "A **no-argument table function** returning a small curated set of well-known "
                    "molecules (water, ethanol, benzene, aspirin, caffeine, ibuprofen, glucose, "
                    "paracetamol, nicotine, penicillin G, ...) with ready-computed descriptors.\n\n"
                    "**Use it** as the browsable entry point to this catalog: scan it to obtain "
                    "valid SMILES strings to feed the scalar functions, or to compare descriptor "
                    "ranges across common drugs without computing anything yourself.\n\n"
                    "Columns: `name`, `smiles`, `formula`, `mol_weight` (g/mol), `logp`, `tpsa` "
                    "(Angstrom^2), `h_bond_donors`, `h_bond_acceptors`, `num_rings`, and "
                    "`drug_like` (passes all four Lipinski rules). Every column is derived from the "
                    "`smiles` via this catalog's own functions."
                ),
                description_md=(
                    "# example_molecules\n\n"
                    "A curated, browsable registry of common molecules and their descriptors -- the "
                    "zero-argument discovery entry point for the `chem` catalog.\n\n"
                    "## Columns\n\n"
                    "- `name`, `smiles`, `formula`\n"
                    "- `mol_weight` (g/mol), `logp`, `tpsa` (Angstrom^2)\n"
                    "- `h_bond_donors`, `h_bond_acceptors`, `num_rings`\n"
                    "- `drug_like` -- passes all four Lipinski rules\n\n"
                    "Every descriptor column is computed live from `smiles` via this catalog's own "
                    "functions, so the registry can never drift from them."
                ),
                keywords=[
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
                ],
                relative_path="vgi_chem/tables.py",
                category="reference",
            ),
            "vgi.result_columns_schema": json.dumps(
                [
                    {"name": "name", "type": "VARCHAR", "description": "Common name of the molecule."},
                    {
                        "name": "smiles",
                        "type": "VARCHAR",
                        "description": "SMILES string you can pass to any chem function.",
                    },
                    {"name": "formula", "type": "VARCHAR", "description": "Hill-system molecular formula."},
                    {"name": "mol_weight", "type": "DOUBLE", "description": "Average molecular weight in g/mol."},
                    {
                        "name": "logp",
                        "type": "DOUBLE",
                        "description": "Crippen logP (octanol-water partition estimate).",
                    },
                    {"name": "tpsa", "type": "DOUBLE", "description": "Topological polar surface area (Angstrom^2)."},
                    {"name": "h_bond_donors", "type": "INTEGER", "description": "Lipinski H-bond donor count."},
                    {"name": "h_bond_acceptors", "type": "INTEGER", "description": "Lipinski H-bond acceptor count."},
                    {"name": "num_rings", "type": "INTEGER", "description": "Ring count (SSSR)."},
                    {
                        "name": "drug_like",
                        "type": "BOOLEAN",
                        "description": "True if the molecule passes all four Lipinski rules.",
                    },
                ]
            ),
        }
        examples = [
            FunctionExample(
                sql="SELECT name, smiles, formula, mol_weight FROM chem.main.example_molecules() ORDER BY mol_weight",
                description="Browse the built-in molecules by molecular weight",
            ),
            FunctionExample(
                sql="SELECT name, logp FROM chem.main.example_molecules() WHERE drug_like ORDER BY logp DESC",
                description="Drug-like example molecules ranked by lipophilicity",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_NoArgs]) -> TableCardinality:
        """Report the fixed row count of the curated registry."""
        n = len(chem.example_molecules())
        return TableCardinality(estimate=n, max=n)

    @classmethod
    def process(cls, params: ProcessParams[_NoArgs], state: None, out: OutputCollector) -> None:
        """Emit the curated molecule registry as a single batch."""
        rows = chem.example_molecules()
        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "name": [r["name"] for r in rows],
                    "smiles": [r["smiles"] for r in rows],
                    "formula": [r["formula"] for r in rows],
                    "mol_weight": [r["mol_weight"] for r in rows],
                    "logp": [r["logp"] for r in rows],
                    "tpsa": [r["tpsa"] for r in rows],
                    "h_bond_donors": [r["h_bond_donors"] for r in rows],
                    "h_bond_acceptors": [r["h_bond_acceptors"] for r in rows],
                    "num_rings": [r["num_rings"] for r in rows],
                    "drug_like": [r["drug_like"] for r in rows],
                },
                schema=params.output_schema,
            )
        )
        out.finish()


TABLE_FUNCTIONS: list[type] = [
    LipinskiFunction,
    ExampleMoleculesFunction,
]

# VGI515: mirror each table function's Meta.examples into a vgi.example_queries
# tag so every example carries its description.
attach_example_queries(TABLE_FUNCTIONS)
