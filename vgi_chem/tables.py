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
        name = "lipinski"
        description = "Lipinski rule-of-five breakdown (MW<=500, logP<=5, HBD<=5, HBA<=10), one row per rule"
        categories = ["chem", "druglikeness"]
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
        return TableCardinality(estimate=4, max=4)

    @classmethod
    def process(cls, params: ProcessParams[_LipinskiArgs], state: None, out: OutputCollector) -> None:
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
