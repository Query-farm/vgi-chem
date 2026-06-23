"""Per-row scalar cheminformatics functions.

Every function here is a true DuckDB **scalar** -- one value (per row) in, one
value out -- so it can be used inline in any projection or predicate:

    SELECT is_valid_smiles(smiles)              FROM compounds;
    SELECT id, mol_weight(smiles)               FROM compounds;
    SELECT inchikey(smiles)                     FROM compounds;
    SELECT tanimoto(a.smiles, b.smiles)         FROM a, b;

A note on argument syntax
-------------------------
VGI / DuckDB *scalar* functions take **positional** arguments and resolve
overloads by *arity* (the ``name := value`` named-argument syntax is a property
of table functions and macros, not scalars). The optional ``radius`` / ``nbits``
arguments on the fingerprint / similarity functions therefore cannot have
Python-style defaults on a single class; instead each optional-argument form is
exposed as its own arity overload that shares the function ``name`` -- the same
idiom the sibling ``vgi-conform`` / ``vgi-calendar`` workers use. So, e.g.:

    morgan_fingerprint(smiles)                  -- radius=2, nbits=2048
    morgan_fingerprint(smiles, radius, nbits)   -- explicit
    tanimoto(a, b)                              -- radius=2
    tanimoto(a, b, radius)                      -- explicit radius

NULL semantics: a NULL input row yields NULL output. An INVALID (non-NULL)
SMILES yields NULL for every value function and ``false`` for
``is_valid_smiles``; an invalid SMARTS in ``substructure_match`` yields NULL
(documented). Nothing here ever raises on bad chemical input.

The ``lipinski(smiles)`` rule-of-five breakdown is set-returning and lives in
:mod:`vgi_chem.tables`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

import pyarrow as pa
from vgi.arguments import ConstParam, Param, Returns
from vgi.metadata import FunctionExample
from vgi.scalar_function import ScalarFunction

from . import chem

# ---------------------------------------------------------------------------
# Small mapping helpers: apply a pure ``str -> X`` function across an array,
# passing NULL straight through.
# ---------------------------------------------------------------------------


def _map_bool(arr: pa.StringArray, fn: Callable[[str], bool | None]) -> pa.BooleanArray:
    out = [None if x is None else fn(x) for x in arr.to_pylist()]
    return pa.array(out, type=pa.bool_())


def _map_str(arr: pa.StringArray, fn: Callable[[str], str | None]) -> pa.StringArray:
    out = [None if x is None else fn(x) for x in arr.to_pylist()]
    return pa.array(out, type=pa.string())


def _map_double(arr: pa.StringArray, fn: Callable[[str], float | None]) -> pa.DoubleArray:
    out = [None if x is None else fn(x) for x in arr.to_pylist()]
    return pa.array(out, type=pa.float64())


def _map_int(arr: pa.StringArray, fn: Callable[[str], int | None]) -> pa.Int32Array:
    out = [None if x is None else fn(x) for x in arr.to_pylist()]
    return pa.array(out, type=pa.int32())


# ===========================================================================
# Validity + identity
# ===========================================================================


class IsValidSmilesFunction(ScalarFunction):
    """``is_valid_smiles(smiles)`` -- True if the string parses to a molecule."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "is_valid_smiles"
        description = "True if the SMILES string parses to a valid molecule (false if not, NULL if NULL)"
        categories = ["chem", "validity"]
        examples = [
            FunctionExample(
                sql="SELECT chem.is_valid_smiles('CCO')",
                description="A valid SMILES (ethanol)",
            ),
            FunctionExample(
                sql="SELECT chem.is_valid_smiles('xyz')",
                description="An invalid SMILES -> false",
            ),
        ]

    @classmethod
    def compute(
        cls, smiles: Annotated[pa.StringArray, Param(doc="SMILES string to validate.")]
    ) -> Annotated[pa.BooleanArray, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        return _map_bool(smiles, chem.is_valid_smiles)


class CanonicalSmilesFunction(ScalarFunction):
    """``canonical_smiles(smiles)`` -- RDKit canonical SMILES, or NULL if invalid."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "canonical_smiles"
        description = "RDKit canonical SMILES form, or NULL if the input is invalid"
        categories = ["chem", "identity"]
        examples = [
            FunctionExample(
                sql="SELECT chem.canonical_smiles('OCC')",
                description="Canonicalize ethanol -> 'CCO'",
            ),
        ]

    @classmethod
    def compute(
        cls, smiles: Annotated[pa.StringArray, Param(doc="SMILES string to canonicalize.")]
    ) -> Annotated[pa.StringArray, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        return _map_str(smiles, chem.canonical_smiles)


class MolFormulaFunction(ScalarFunction):
    """``mol_formula(smiles)`` -- Hill-system molecular formula, or NULL if invalid."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "mol_formula"
        description = "Hill-system molecular formula, e.g. 'C9H8O4', or NULL if invalid"
        categories = ["chem", "descriptors"]
        examples = [
            FunctionExample(
                sql="SELECT chem.mol_formula('CC(=O)OC1=CC=CC=C1C(=O)O')",
                description="Molecular formula of aspirin",
            ),
        ]

    @classmethod
    def compute(
        cls, smiles: Annotated[pa.StringArray, Param(doc="SMILES string.")]
    ) -> Annotated[pa.StringArray, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        return _map_str(smiles, chem.mol_formula)


class InchiFunction(ScalarFunction):
    """``inchi(smiles)`` -- standard InChI string, or NULL if invalid."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "inchi"
        description = "Standard InChI string, or NULL if invalid"
        categories = ["chem", "identity"]
        examples = [
            FunctionExample(
                sql="SELECT chem.inchi('CCO')",
                description="InChI of ethanol",
            ),
        ]

    @classmethod
    def compute(
        cls, smiles: Annotated[pa.StringArray, Param(doc="SMILES string.")]
    ) -> Annotated[pa.StringArray, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        return _map_str(smiles, chem.inchi_of)


class InchiKeyFunction(ScalarFunction):
    """``inchikey(smiles)`` -- standard InChIKey, or NULL if invalid."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "inchikey"
        description = "Standard InChIKey (27-char hashed InChI), or NULL if invalid"
        categories = ["chem", "identity"]
        examples = [
            FunctionExample(
                sql="SELECT chem.inchikey('CC(=O)OC1=CC=CC=C1C(=O)O')",
                description="InChIKey of aspirin -> 'BSYNRYMUTXBXSQ-UHFFFAOYSA-N'",
            ),
        ]

    @classmethod
    def compute(
        cls, smiles: Annotated[pa.StringArray, Param(doc="SMILES string.")]
    ) -> Annotated[pa.StringArray, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        return _map_str(smiles, chem.inchikey_of)


# ===========================================================================
# Numeric descriptors -- DOUBLE
# ===========================================================================


class MolWeightFunction(ScalarFunction):
    """``mol_weight(smiles)`` -- average molecular weight (g/mol)."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "mol_weight"
        description = "Average molecular weight in g/mol, or NULL if invalid"
        categories = ["chem", "descriptors"]
        examples = [
            FunctionExample(
                sql="SELECT chem.mol_weight('CC(=O)OC1=CC=CC=C1C(=O)O')",
                description="Molecular weight of aspirin (~180.16)",
            ),
        ]

    @classmethod
    def compute(
        cls, smiles: Annotated[pa.StringArray, Param(doc="SMILES string.")]
    ) -> Annotated[pa.DoubleArray, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        return _map_double(smiles, chem.mol_weight)


class ExactMassFunction(ScalarFunction):
    """``exact_mass(smiles)`` -- monoisotopic (exact) mass."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "exact_mass"
        description = "Monoisotopic (exact) mass, or NULL if invalid"
        categories = ["chem", "descriptors"]
        examples = [
            FunctionExample(
                sql="SELECT chem.exact_mass('CCO')",
                description="Exact mass of ethanol",
            ),
        ]

    @classmethod
    def compute(
        cls, smiles: Annotated[pa.StringArray, Param(doc="SMILES string.")]
    ) -> Annotated[pa.DoubleArray, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        return _map_double(smiles, chem.exact_mass)


class LogPFunction(ScalarFunction):
    """``logp(smiles)`` -- Crippen MolLogP."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "logp"
        description = "Crippen MolLogP (octanol-water partition coefficient estimate), or NULL if invalid"
        categories = ["chem", "descriptors"]
        examples = [
            FunctionExample(
                sql="SELECT chem.logp('CCO')",
                description="logP of ethanol",
            ),
        ]

    @classmethod
    def compute(
        cls, smiles: Annotated[pa.StringArray, Param(doc="SMILES string.")]
    ) -> Annotated[pa.DoubleArray, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        return _map_double(smiles, chem.logp)


class TpsaFunction(ScalarFunction):
    """``tpsa(smiles)`` -- topological polar surface area."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "tpsa"
        description = "Topological polar surface area (TPSA) in Angstrom^2, or NULL if invalid"
        categories = ["chem", "descriptors"]
        examples = [
            FunctionExample(
                sql="SELECT chem.tpsa('CC(=O)OC1=CC=CC=C1C(=O)O')",
                description="TPSA of aspirin",
            ),
        ]

    @classmethod
    def compute(
        cls, smiles: Annotated[pa.StringArray, Param(doc="SMILES string.")]
    ) -> Annotated[pa.DoubleArray, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        return _map_double(smiles, chem.tpsa)


# ===========================================================================
# Count descriptors -- INT
# ===========================================================================


class NumAtomsFunction(ScalarFunction):
    """``num_atoms(smiles)`` -- number of heavy (non-H) atoms."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "num_atoms"
        description = "Number of heavy (non-hydrogen) atoms, or NULL if invalid"
        categories = ["chem", "descriptors"]
        examples = [
            FunctionExample(
                sql="SELECT chem.num_atoms('c1ccccc1')",
                description="Heavy-atom count of benzene (6)",
            ),
        ]

    @classmethod
    def compute(
        cls, smiles: Annotated[pa.StringArray, Param(doc="SMILES string.")]
    ) -> Annotated[pa.Int32Array, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        return _map_int(smiles, chem.num_atoms)


class NumRingsFunction(ScalarFunction):
    """``num_rings(smiles)`` -- ring count (SSSR)."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "num_rings"
        description = "Number of rings (SSSR ring count), or NULL if invalid"
        categories = ["chem", "descriptors"]
        examples = [
            FunctionExample(
                sql="SELECT chem.num_rings('c1ccccc1')",
                description="Ring count of benzene (1)",
            ),
        ]

    @classmethod
    def compute(
        cls, smiles: Annotated[pa.StringArray, Param(doc="SMILES string.")]
    ) -> Annotated[pa.Int32Array, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        return _map_int(smiles, chem.num_rings)


class NumRotatableBondsFunction(ScalarFunction):
    """``num_rotatable_bonds(smiles)`` -- rotatable bond count."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "num_rotatable_bonds"
        description = "Number of rotatable bonds, or NULL if invalid"
        categories = ["chem", "descriptors"]
        examples = [
            FunctionExample(
                sql="SELECT chem.num_rotatable_bonds('CC(=O)OC1=CC=CC=C1C(=O)O')",
                description="Rotatable bonds of aspirin",
            ),
        ]

    @classmethod
    def compute(
        cls, smiles: Annotated[pa.StringArray, Param(doc="SMILES string.")]
    ) -> Annotated[pa.Int32Array, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        return _map_int(smiles, chem.num_rotatable_bonds)


class NumHDonorsFunction(ScalarFunction):
    """``num_h_donors(smiles)`` -- H-bond donor count."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "num_h_donors"
        description = "Number of hydrogen-bond donors (Lipinski), or NULL if invalid"
        categories = ["chem", "descriptors"]
        examples = [
            FunctionExample(
                sql="SELECT chem.num_h_donors('CCO')",
                description="H-bond donors of ethanol (1)",
            ),
        ]

    @classmethod
    def compute(
        cls, smiles: Annotated[pa.StringArray, Param(doc="SMILES string.")]
    ) -> Annotated[pa.Int32Array, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        return _map_int(smiles, chem.num_h_donors)


class NumHAcceptorsFunction(ScalarFunction):
    """``num_h_acceptors(smiles)`` -- H-bond acceptor count."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "num_h_acceptors"
        description = "Number of hydrogen-bond acceptors (Lipinski), or NULL if invalid"
        categories = ["chem", "descriptors"]
        examples = [
            FunctionExample(
                sql="SELECT chem.num_h_acceptors('CCO')",
                description="H-bond acceptors of ethanol",
            ),
        ]

    @classmethod
    def compute(
        cls, smiles: Annotated[pa.StringArray, Param(doc="SMILES string.")]
    ) -> Annotated[pa.Int32Array, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        return _map_int(smiles, chem.num_h_acceptors)


# ===========================================================================
# Fingerprints + similarity -- arity overloads for optional radius / nbits.
# ===========================================================================


class MorganFingerprintFunction(ScalarFunction):
    """``morgan_fingerprint(smiles)`` -- hex fingerprint (radius=2, nbits=2048)."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "morgan_fingerprint"
        description = (
            "Morgan (ECFP-like) fingerprint as a hex bit-string (defaults radius=2, nbits=2048); NULL if invalid"
        )
        categories = ["chem", "fingerprint"]
        examples = [
            FunctionExample(
                sql="SELECT chem.morgan_fingerprint('CCO')",
                description="Default Morgan fingerprint of ethanol",
            ),
        ]

    @classmethod
    def compute(
        cls, smiles: Annotated[pa.StringArray, Param(doc="SMILES string.")]
    ) -> Annotated[pa.StringArray, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        return _map_str(smiles, chem.morgan_fingerprint)


class MorganFingerprintParamsFunction(ScalarFunction):
    """``morgan_fingerprint(smiles, radius, nbits)`` -- explicit parameters."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "morgan_fingerprint"
        description = "Morgan fingerprint as a hex bit-string with explicit radius and nbits; NULL if invalid"
        categories = ["chem", "fingerprint"]
        examples = [
            FunctionExample(
                sql="SELECT chem.morgan_fingerprint('CCO', 3, 1024)",
                description="Morgan fingerprint with radius=3, nbits=1024",
            ),
        ]

    @classmethod
    def compute(
        cls,
        smiles: Annotated[pa.StringArray, Param(doc="SMILES string.")],
        radius: Annotated[int, ConstParam("Morgan radius (e.g. 2).")],
        nbits: Annotated[int, ConstParam("Fingerprint length in bits (e.g. 2048).")],
    ) -> Annotated[pa.StringArray, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        return _map_str(smiles, lambda s: chem.morgan_fingerprint(s, radius, nbits))


class TanimotoFunction(ScalarFunction):
    """``tanimoto(smiles_a, smiles_b)`` -- Morgan/Tanimoto similarity (radius=2)."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "tanimoto"
        description = "Morgan/Tanimoto similarity in [0,1] (radius=2); NULL if either SMILES is invalid"
        categories = ["chem", "similarity"]
        examples = [
            FunctionExample(
                sql="SELECT chem.tanimoto('CCO', 'CCO')",
                description="Self-similarity is 1.0",
            ),
        ]

    @classmethod
    def compute(
        cls,
        smiles_a: Annotated[pa.StringArray, Param(doc="First SMILES string.")],
        smiles_b: Annotated[pa.StringArray, Param(doc="Second SMILES string.")],
    ) -> Annotated[pa.DoubleArray, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        a = smiles_a.to_pylist()
        b = smiles_b.to_pylist()
        out = [None if x is None or y is None else chem.tanimoto(x, y) for x, y in zip(a, b, strict=True)]
        return pa.array(out, type=pa.float64())


class TanimotoRadiusFunction(ScalarFunction):
    """``tanimoto(smiles_a, smiles_b, radius)`` -- explicit Morgan radius."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "tanimoto"
        description = "Morgan/Tanimoto similarity in [0,1] with an explicit radius; NULL if either is invalid"
        categories = ["chem", "similarity"]
        examples = [
            FunctionExample(
                sql="SELECT chem.tanimoto('CCO', 'CCC', 2)",
                description="Tanimoto similarity with radius=2",
            ),
        ]

    @classmethod
    def compute(
        cls,
        smiles_a: Annotated[pa.StringArray, Param(doc="First SMILES string.")],
        smiles_b: Annotated[pa.StringArray, Param(doc="Second SMILES string.")],
        radius: Annotated[int, ConstParam("Morgan radius (e.g. 2).")],
    ) -> Annotated[pa.DoubleArray, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        a = smiles_a.to_pylist()
        b = smiles_b.to_pylist()
        out = [None if x is None or y is None else chem.tanimoto(x, y, radius) for x, y in zip(a, b, strict=True)]
        return pa.array(out, type=pa.float64())


# ===========================================================================
# Substructure search
# ===========================================================================


class SubstructureMatchFunction(ScalarFunction):
    """``substructure_match(smiles, smarts)`` -- does the molecule contain the pattern."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "substructure_match"
        description = (
            "True if the molecule contains the SMARTS pattern. "
            "NULL if the SMILES is invalid OR the SMARTS pattern is invalid."
        )
        categories = ["chem", "substructure"]
        examples = [
            FunctionExample(
                sql="SELECT chem.substructure_match('c1ccccc1O', 'c1ccccc1')",
                description="Phenol contains a benzene ring -> true",
            ),
        ]

    @classmethod
    def compute(
        cls,
        smiles: Annotated[pa.StringArray, Param(doc="SMILES string of the molecule.")],
        smarts: Annotated[pa.StringArray, Param(doc="SMARTS query pattern.")],
    ) -> Annotated[pa.BooleanArray, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        mols = smiles.to_pylist()
        pats = smarts.to_pylist()
        out = [
            None if s is None or p is None else chem.substructure_match(s, p) for s, p in zip(mols, pats, strict=True)
        ]
        return pa.array(out, type=pa.bool_())


SCALAR_FUNCTIONS: list[type] = [
    # validity + identity
    IsValidSmilesFunction,
    CanonicalSmilesFunction,
    MolFormulaFunction,
    InchiFunction,
    InchiKeyFunction,
    # numeric descriptors
    MolWeightFunction,
    ExactMassFunction,
    LogPFunction,
    TpsaFunction,
    # count descriptors
    NumAtomsFunction,
    NumRingsFunction,
    NumRotatableBondsFunction,
    NumHDonorsFunction,
    NumHAcceptorsFunction,
    # fingerprints + similarity
    MorganFingerprintFunction,
    MorganFingerprintParamsFunction,
    TanimotoFunction,
    TanimotoRadiusFunction,
    # substructure
    SubstructureMatchFunction,
]
