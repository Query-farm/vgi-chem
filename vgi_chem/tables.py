"""Set-returning chem table functions for DuckDB.

The Lipinski rule-of-five breakdown expands one molecule into **several rows**
(one per rule), so it is exposed as a **table function** -- the form that accepts
DuckDB ``name := value`` arguments. The per-row, single-value cheminformatics
functions are *scalars* and live in :mod:`vgi_chem.scalars`.

    SELECT * FROM chem.lipinski('CC(=O)OC1=CC=CC=C1C(=O)O');
"""

from __future__ import annotations

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
from .meta import object_tags
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
                    "## Usage\n\n"
                    "```sql\n"
                    "SELECT * FROM chem.lipinski('CC(=O)OC1=CC=CC=C1C(=O)O');\n"
                    "SELECT bool_and(passes) AS drug_like\n"
                    "FROM chem.lipinski('CC(=O)OC1=CC=CC=C1C(=O)O');\n"
                    "```\n\n"
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
            "vgi.result_columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `rule` | VARCHAR | Lipinski rule name (`molecular_weight`, `logp`, "
                "`h_bond_donors`, `h_bond_acceptors`). |\n"
                "| `value` | DOUBLE | Computed descriptor value for the rule. |\n"
                "| `passes` | BOOLEAN | True if this rule's rule-of-five threshold is satisfied. |"
            ),
        }
        examples = [
            FunctionExample(
                sql="SELECT * FROM chem.lipinski('CC(=O)OC1=CC=CC=C1C(=O)O')",
                description="Rule-of-five breakdown for aspirin",
            ),
            FunctionExample(
                sql=("SELECT bool_and(passes) AS drug_like FROM chem.lipinski('CC(=O)OC1=CC=CC=C1C(=O)O')"),
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


TABLE_FUNCTIONS: list[type] = [
    LipinskiFunction,
]
