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

import json
from collections.abc import Callable
from typing import Annotated

import pyarrow as pa
from vgi.arguments import ConstParam, Param, Returns
from vgi.metadata import FunctionExample
from vgi.scalar_function import ScalarFunction

from . import chem
from .meta import attach_example_queries, object_tags

_SRC = "vgi_chem/scalars.py"

# ``morgan_fingerprint`` has two arity overloads that share one name, so the SDK
# merges their metadata into a single catalog object. To keep that merged view
# self-consistent (VGI180), BOTH overloads carry the SAME doc, which describes the
# defaults-only and explicit-parameter signatures together rather than titling
# itself as one specific arity.
_MORGAN_DOC_LLM = (
    "## morgan_fingerprint\n\n"
    "Builds a **Morgan (ECFP-like) circular fingerprint** for a molecule from its SMILES and "
    "returns it as a hexadecimal bit-string.\n\n"
    "Two arities share this name:\n\n"
    "- `morgan_fingerprint(smiles)` -- uses the fixed defaults `radius=2` and `nbits=2048`.\n"
    "- `morgan_fingerprint(smiles, radius, nbits)` -- set the atom-neighborhood `radius` and the "
    "bit-vector length `nbits` explicitly.\n\n"
    "**Use it** to vectorize molecules for similarity search, clustering, or as machine-learning "
    "features. Two molecules' fingerprints are compared with `tanimoto`.\n\n"
    "- **Input**: a SMILES string (`VARCHAR`); the three-argument form additionally takes "
    "`radius` (`BIGINT`) and `nbits` (`BIGINT`).\n"
    "- **Output**: `VARCHAR` hex-encoded bit vector, or `NULL` if invalid.\n\n"
    "**Edge cases**: a larger `radius` captures bigger atom environments; a smaller `nbits` "
    "increases bit collisions. Only fingerprints built with identical `radius`/`nbits` are "
    "comparable. `NULL`/invalid input returns `NULL`."
)
_MORGAN_DOC_MD = (
    "# morgan_fingerprint\n\n"
    "Morgan/ECFP circular fingerprint as a hex bit-string from a SMILES string.\n\n"
    "## Signatures\n\n"
    "- `morgan_fingerprint(smiles)` -- defaults `radius=2`, `nbits=2048`.\n"
    "- `morgan_fingerprint(smiles, radius, nbits)` -- explicit radius and bit length.\n\n"
    "## Usage\n\n"
    "```sql\n"
    "SELECT chem.main.morgan_fingerprint('CCO');            -- radius=2, nbits=2048\n"
    "SELECT chem.main.morgan_fingerprint('CCO', 3, 1024);   -- explicit\n"
    "```\n\n"
    "## Notes\n\n"
    "Compare two fingerprints with `tanimoto`. Only fingerprints built with the same "
    "`radius`/`nbits` are comparable. Returns `NULL` on invalid input."
)

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
        tags = object_tags(
            title="Is Valid SMILES String",
            description_llm=(
                "## is_valid_smiles\n\n"
                "Tests whether a string is a syntactically and chemically valid "
                "[SMILES](https://en.wikipedia.org/wiki/SMILES) molecule, using RDKit's parser.\n\n"
                "**Use it** as a cheap guard before any other `chem` function, or to filter a "
                "column of user-supplied structures.\n\n"
                "- **Input**: a SMILES string (`VARCHAR`).\n"
                "- **Output**: `BOOLEAN` -- `true` if RDKit parses it to a molecule, `false` "
                "for any non-NULL string it cannot parse.\n\n"
                "**Edge cases**: a `NULL` input returns `NULL` (not `false`); unlike every other "
                "function here, an unparseable string returns `false` rather than `NULL`, so it is "
                "safe to use directly in a `WHERE` clause."
            ),
            description_md=(
                "# is_valid_smiles\n\n"
                "Return `true` when a string parses to a valid molecule via RDKit, `false` "
                "otherwise.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT chem.main.is_valid_smiles('CCO');   -- true (ethanol)\n"
                "SELECT chem.main.is_valid_smiles('xyz');   -- false\n"
                "```\n\n"
                "## Notes\n\n"
                "`NULL` in yields `NULL` out; any other unparseable input yields `false`, so this "
                "is the safe predicate to gate the other descriptor/identity functions."
            ),
            keywords=["smiles", "valid", "validate", "validity", "parse", "check molecule", "is valid", "sanitize"],
            relative_path=_SRC,
            category="validity",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.is_valid_smiles('CCO')",
                description="A valid SMILES (ethanol)",
            ),
            FunctionExample(
                sql="SELECT chem.main.is_valid_smiles('xyz')",
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
        tags = object_tags(
            title="Canonical SMILES Form",
            description_llm=(
                "## canonical_smiles\n\n"
                "Rewrites any valid SMILES into RDKit's **canonical** SMILES -- a single, "
                "deterministic spelling for a given molecular graph.\n\n"
                "**Use it** to deduplicate or group molecules that were written differently "
                "(`OCC` and `CCO` are the same ethanol), or to produce a stable join key.\n\n"
                "- **Input**: a SMILES string (`VARCHAR`).\n"
                "- **Output**: `VARCHAR` canonical SMILES, or `NULL` if the input does not parse.\n\n"
                "**Edge cases**: `NULL`/invalid input returns `NULL`. Canonicalization does not "
                "neutralize charges or strip salts -- it only canonicalizes the graph as written."
            ),
            description_md=(
                "# canonical_smiles\n\n"
                "Normalize a SMILES string to RDKit's canonical form for stable comparison and "
                "deduplication.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT chem.main.canonical_smiles('OCC');   -- 'CCO'\n"
                "```\n\n"
                "Apply it over a whole SMILES column with a `DISTINCT` projection to deduplicate "
                "structures written different ways.\n\n"
                "## Notes\n\n"
                "Returns `NULL` on invalid input. Two strings denoting the same molecule "
                "canonicalize to the same value, making this a reliable grouping/join key."
            ),
            keywords=["smiles", "canonical", "canonicalize", "normalize", "dedupe", "deduplicate", "identity", "rdkit"],
            relative_path=_SRC,
            category="identity",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.canonical_smiles('OCC')",
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
        tags = object_tags(
            title="Molecular Formula String",
            description_llm=(
                "## mol_formula\n\n"
                "Computes the **Hill-system molecular formula** of a molecule from its SMILES "
                "(carbon first, then hydrogen, then other elements alphabetically), e.g. `C9H8O4` "
                "for aspirin.\n\n"
                "**Use it** to report or group compounds by composition, or to feed a mass-spec / "
                "elemental-analysis pipeline.\n\n"
                "- **Input**: a SMILES string (`VARCHAR`).\n"
                "- **Output**: `VARCHAR` Hill formula, or `NULL` for invalid input.\n\n"
                "**Edge cases**: implicit hydrogens are counted, so the formula reflects the full "
                "molecule, not just the heavy-atom skeleton. `NULL`/invalid input returns `NULL`."
            ),
            description_md=(
                "# mol_formula\n\n"
                "Molecular formula in Hill notation from a SMILES string.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT chem.main.mol_formula('CC(=O)OC1=CC=CC=C1C(=O)O');  -- 'C9H8O4'\n"
                "```\n\n"
                "## Notes\n\n"
                "Implicit hydrogens are included. Returns `NULL` on invalid input."
            ),
            keywords=["formula", "molecular formula", "hill", "composition", "elements", "C9H8O4", "empirical formula"],
            relative_path=_SRC,
            category="descriptors",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.mol_formula('CC(=O)OC1=CC=CC=C1C(=O)O')",
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
        tags = object_tags(
            title="Standard InChI Identifier",
            description_llm=(
                "## inchi\n\n"
                "Derives the **standard InChI** (IUPAC International Chemical Identifier) string "
                "for a molecule from its SMILES.\n\n"
                "**Use it** when you need a layered, vendor-neutral structural identifier for "
                "cross-database lookup or canonical storage.\n\n"
                "- **Input**: a SMILES string (`VARCHAR`).\n"
                "- **Output**: `VARCHAR` standard InChI (e.g. `InChI=1S/...`), or `NULL` if invalid.\n\n"
                "**Edge cases**: standard InChI normalizes tautomers/stereo per the InChI rules, so "
                "it may merge structures a raw SMILES would keep distinct. Use `inchikey` for a "
                "fixed-length hash of this value. `NULL`/invalid input returns `NULL`."
            ),
            description_md=(
                "# inchi\n\n"
                "Standard IUPAC InChI identifier from a SMILES string.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT chem.main.inchi('CCO');  -- 'InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3'\n"
                "```\n\n"
                "## Notes\n\n"
                "Standard InChI applies the IUPAC normalization layers. Pair with `inchikey` for a "
                "compact hashed key. Returns `NULL` on invalid input."
            ),
            keywords=["inchi", "identifier", "iupac", "structure key", "standard inchi", "cross-reference", "lookup"],
            relative_path=_SRC,
            category="identity",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.inchi('CCO')",
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
        tags = object_tags(
            title="Standard InChIKey Hash",
            description_llm=(
                "## inchikey\n\n"
                "Computes the **standard InChIKey** -- the fixed-length, 27-character hashed form "
                "of the InChI -- for a molecule from its SMILES.\n\n"
                "**Use it** as a compact, collision-resistant primary key for chemical structures, "
                "ideal for indexing, joining and web search.\n\n"
                "- **Input**: a SMILES string (`VARCHAR`).\n"
                "- **Output**: `VARCHAR` InChIKey like `BSYNRYMUTXBXSQ-UHFFFAOYSA-N`, or `NULL` "
                "if invalid.\n\n"
                "**Edge cases**: the key is derived from the standard InChI, so it inherits the "
                "same tautomer/stereo normalization. The first 14 chars encode connectivity; the "
                "next block encodes stereo/isotope/protonation. `NULL`/invalid input returns `NULL`."
            ),
            description_md=(
                "# inchikey\n\n"
                "Standard 27-character InChIKey hash from a SMILES string.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT chem.main.inchikey('CC(=O)OC1=CC=CC=C1C(=O)O');\n"
                "-- 'BSYNRYMUTXBXSQ-UHFFFAOYSA-N'\n"
                "```\n\n"
                "## Notes\n\n"
                "A fixed-length, indexable structure key. The first 14 characters capture skeletal "
                "connectivity. Returns `NULL` on invalid input."
            ),
            keywords=[
                "inchikey",
                "inchi key",
                "hash",
                "structure key",
                "primary key",
                "identifier",
                "lookup",
                "search",
            ],
            relative_path=_SRC,
            category="identity",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.inchikey('CC(=O)OC1=CC=CC=C1C(=O)O')",
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
        tags = object_tags(
            title="Average Molecular Weight",
            description_llm=(
                "## mol_weight\n\n"
                "Computes the **average molecular weight** (g/mol) of a molecule from its SMILES, "
                "using standard atomic weights (isotope-averaged).\n\n"
                "**Use it** for stoichiometry, dosing math, molecular-property filters, or as the "
                "first Lipinski rule (MW <= 500).\n\n"
                "- **Input**: a SMILES string (`VARCHAR`).\n"
                "- **Output**: `DOUBLE` average MW in g/mol (e.g. ~180.16 for aspirin), or `NULL` "
                "if invalid.\n\n"
                "**Edge cases**: this is the *average* (not monoisotopic) mass -- use `exact_mass` "
                "for the monoisotopic value used in mass spectrometry. `NULL`/invalid input "
                "returns `NULL`."
            ),
            description_md=(
                "# mol_weight\n\n"
                "Average molecular weight in g/mol from a SMILES string.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT ROUND(chem.main.mol_weight('CC(=O)OC1=CC=CC=C1C(=O)O'), 2);  -- 180.16\n"
                "```\n\n"
                "## Notes\n\n"
                "Isotope-averaged weight; see `exact_mass` for the monoisotopic mass. Returns "
                "`NULL` on invalid input."
            ),
            keywords=["molecular weight", "mw", "mass", "g/mol", "average mass", "lipinski", "descriptor"],
            relative_path=_SRC,
            category="descriptors",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.mol_weight('CC(=O)OC1=CC=CC=C1C(=O)O')",
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
        tags = object_tags(
            title="Monoisotopic Exact Mass",
            description_llm=(
                "## exact_mass\n\n"
                "Computes the **monoisotopic (exact) mass** of a molecule from its SMILES -- the "
                "sum of the masses of the most abundant isotope of each atom.\n\n"
                "**Use it** for mass-spectrometry workflows, where the exact mass (not the "
                "average MW) is what an instrument measures.\n\n"
                "- **Input**: a SMILES string (`VARCHAR`).\n"
                "- **Output**: `DOUBLE` monoisotopic mass in Daltons, or `NULL` if invalid.\n\n"
                "**Edge cases**: differs from `mol_weight`, which averages over natural isotope "
                "abundances. `NULL`/invalid input returns `NULL`."
            ),
            description_md=(
                "# exact_mass\n\n"
                "Monoisotopic exact mass (Da) from a SMILES string.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT chem.main.exact_mass('CCO');  -- ~46.0419\n"
                "```\n\n"
                "## Notes\n\n"
                "Uses the most-abundant isotope per element; contrast with `mol_weight` (average). "
                "Returns `NULL` on invalid input."
            ),
            keywords=["exact mass", "monoisotopic", "mass spec", "da", "dalton", "m/z", "descriptor", "isotope"],
            relative_path=_SRC,
            category="descriptors",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.exact_mass('CCO')",
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
        tags = object_tags(
            title="Crippen LogP Estimate",
            description_llm=(
                "## logp\n\n"
                "Estimates **logP** -- the octanol-water partition coefficient, a lipophilicity "
                "measure -- using RDKit's Crippen `MolLogP` contribution model, from a SMILES.\n\n"
                "**Use it** for drug-likeness and ADME filtering; logP is the third Lipinski "
                "rule (logP <= 5) and a core permeability/solubility proxy.\n\n"
                "- **Input**: a SMILES string (`VARCHAR`).\n"
                "- **Output**: `DOUBLE` Crippen logP (can be negative for very polar molecules), "
                "or `NULL` if invalid.\n\n"
                "**Edge cases**: this is a *computed estimate* (Crippen atom contributions), not a "
                "measured value, and may differ from experimental logP. `NULL`/invalid input "
                "returns `NULL`."
            ),
            description_md=(
                "# logp\n\n"
                "Crippen-model octanol-water partition coefficient (logP) from a SMILES string.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT chem.main.logp('CCO');  -- small negative/near-zero (hydrophilic)\n"
                "```\n\n"
                "## Notes\n\n"
                "An estimate from Crippen atom contributions, not an experimental measurement. "
                "Higher means more lipophilic. Returns `NULL` on invalid input."
            ),
            keywords=[
                "logp",
                "crippen",
                "lipophilicity",
                "partition coefficient",
                "octanol water",
                "adme",
                "lipinski",
                "druglikeness",
            ],
            relative_path=_SRC,
            category="descriptors",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.logp('CCO')",
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
        tags = object_tags(
            title="Topological Polar Surface Area",
            description_llm=(
                "## tpsa\n\n"
                "Computes the **topological polar surface area (TPSA)** of a molecule, in "
                "square Angstroms, from its SMILES -- the surface area attributable to polar "
                "(N, O and attached H) atoms.\n\n"
                "**Use it** as a predictor of membrane permeability and oral absorption; TPSA "
                "below ~140 Angstrom^2 (and ~90 for CNS penetration) is a common drug-likeness "
                "filter.\n\n"
                "- **Input**: a SMILES string (`VARCHAR`).\n"
                "- **Output**: `DOUBLE` TPSA in Angstrom^2, or `NULL` if invalid.\n\n"
                "**Edge cases**: TPSA is a fast 2D topological estimate (Ertl method), not a 3D "
                "surface calculation. `NULL`/invalid input returns `NULL`."
            ),
            description_md=(
                "# tpsa\n\n"
                "Topological polar surface area (Angstrom^2) from a SMILES string.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT chem.main.tpsa('CC(=O)OC1=CC=CC=C1C(=O)O');  -- ~63.6 for aspirin\n"
                "```\n\n"
                "## Notes\n\n"
                "A 2D topological estimate (Ertl). Lower TPSA generally predicts better "
                "permeability. Returns `NULL` on invalid input."
            ),
            keywords=[
                "tpsa",
                "polar surface area",
                "permeability",
                "absorption",
                "adme",
                "ertl",
                "descriptor",
                "druglikeness",
            ],
            relative_path=_SRC,
            category="descriptors",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.tpsa('CC(=O)OC1=CC=CC=C1C(=O)O')",
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
        tags = object_tags(
            title="Heavy Atom Count",
            description_llm=(
                "## num_atoms\n\n"
                "Counts the **heavy (non-hydrogen) atoms** in a molecule from its SMILES.\n\n"
                "**Use it** as a simple size/complexity measure for filtering fragments vs. "
                "full molecules, or as a denominator for ligand-efficiency metrics.\n\n"
                "- **Input**: a SMILES string (`VARCHAR`).\n"
                "- **Output**: `INTEGER` heavy-atom count, or `NULL` if invalid.\n\n"
                "**Edge cases**: implicit/explicit hydrogens are **not** counted -- only heavy "
                "atoms. `NULL`/invalid input returns `NULL`."
            ),
            description_md=(
                "# num_atoms\n\n"
                "Number of heavy (non-hydrogen) atoms from a SMILES string.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT chem.main.num_atoms('c1ccccc1');  -- 6 (benzene)\n"
                "```\n\n"
                "## Notes\n\n"
                "Hydrogens are excluded. A handy molecular-size filter. Returns `NULL` on invalid "
                "input."
            ),
            keywords=[
                "heavy atom count",
                "num atoms",
                "atom count",
                "size",
                "fragment",
                "ligand efficiency",
                "descriptor",
            ],
            relative_path=_SRC,
            category="descriptors",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.num_atoms('c1ccccc1')",
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
        tags = object_tags(
            title="Ring Count (SSSR)",
            description_llm=(
                "## num_rings\n\n"
                "Counts the number of **rings** in a molecule from its SMILES, using the smallest "
                "set of smallest rings (SSSR).\n\n"
                "**Use it** to gauge molecular rigidity/complexity or to filter for acyclic vs. "
                "polycyclic structures.\n\n"
                "- **Input**: a SMILES string (`VARCHAR`).\n"
                "- **Output**: `INTEGER` SSSR ring count, or `NULL` if invalid.\n\n"
                "**Edge cases**: SSSR is the conventional ring count; fused/bridged systems are "
                "counted by the SSSR convention rather than every possible cycle. `NULL`/invalid "
                "input returns `NULL`."
            ),
            description_md=(
                "# num_rings\n\n"
                "Ring count (smallest set of smallest rings) from a SMILES string.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT chem.main.num_rings('c1ccccc1');  -- 1 (benzene)\n"
                "```\n\n"
                "## Notes\n\n"
                "Uses the SSSR convention. Returns `NULL` on invalid input."
            ),
            keywords=["ring count", "num rings", "sssr", "cycles", "aromatic", "polycyclic", "rigidity", "descriptor"],
            relative_path=_SRC,
            category="descriptors",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.num_rings('c1ccccc1')",
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
        tags = object_tags(
            title="Rotatable Bond Count",
            description_llm=(
                "## num_rotatable_bonds\n\n"
                "Counts the **rotatable bonds** in a molecule from its SMILES -- single, "
                "non-ring bonds between non-terminal heavy atoms.\n\n"
                "**Use it** as a flexibility metric; a low rotatable-bond count (Veber's rule, "
                "<= 10) correlates with good oral bioavailability.\n\n"
                "- **Input**: a SMILES string (`VARCHAR`).\n"
                "- **Output**: `INTEGER` rotatable-bond count, or `NULL` if invalid.\n\n"
                "**Edge cases**: amide C-N bonds and terminal bonds are excluded per the standard "
                "RDKit definition. `NULL`/invalid input returns `NULL`."
            ),
            description_md=(
                "# num_rotatable_bonds\n\n"
                "Number of rotatable bonds from a SMILES string.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT chem.main.num_rotatable_bonds('CC(=O)OC1=CC=CC=C1C(=O)O');\n"
                "```\n\n"
                "## Notes\n\n"
                "A flexibility metric (Veber's rule). Returns `NULL` on invalid input."
            ),
            keywords=[
                "rotatable bonds",
                "flexibility",
                "veber",
                "bioavailability",
                "conformers",
                "descriptor",
                "druglikeness",
            ],
            relative_path=_SRC,
            category="descriptors",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.num_rotatable_bonds('CC(=O)OC1=CC=CC=C1C(=O)O')",
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
        tags = object_tags(
            title="Hydrogen-Bond Donor Count",
            description_llm=(
                "## num_h_donors\n\n"
                "Counts the **hydrogen-bond donors** in a molecule from its SMILES, by the "
                "Lipinski definition (N-H and O-H groups).\n\n"
                "**Use it** for drug-likeness screening; this is the Lipinski rule HBD <= 5.\n\n"
                "- **Input**: a SMILES string (`VARCHAR`).\n"
                "- **Output**: `INTEGER` H-bond donor count, or `NULL` if invalid.\n\n"
                "**Edge cases**: uses the Lipinski (NHOH) count, which counts donor *groups*; "
                "pair with `num_h_acceptors` for the full HBD/HBA picture. `NULL`/invalid input "
                "returns `NULL`."
            ),
            description_md=(
                "# num_h_donors\n\n"
                "Number of hydrogen-bond donors (Lipinski NHOH count) from a SMILES string.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT chem.main.num_h_donors('CCO');  -- 1 (ethanol O-H)\n"
                "```\n\n"
                "## Notes\n\n"
                "Lipinski rule HBD <= 5. Returns `NULL` on invalid input."
            ),
            keywords=["hydrogen bond donors", "hbd", "h donors", "lipinski", "nhoh", "druglikeness", "descriptor"],
            relative_path=_SRC,
            category="descriptors",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.num_h_donors('CCO')",
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
        tags = object_tags(
            title="Hydrogen-Bond Acceptor Count",
            description_llm=(
                "## num_h_acceptors\n\n"
                "Counts the **hydrogen-bond acceptors** in a molecule from its SMILES, by the "
                "Lipinski definition (N and O atoms).\n\n"
                "**Use it** for drug-likeness screening; this is the Lipinski rule HBA <= 10.\n\n"
                "- **Input**: a SMILES string (`VARCHAR`).\n"
                "- **Output**: `INTEGER` H-bond acceptor count, or `NULL` if invalid.\n\n"
                "**Edge cases**: uses the Lipinski (NO) count of nitrogen and oxygen atoms; pair "
                "with `num_h_donors` for the full picture. `NULL`/invalid input returns `NULL`."
            ),
            description_md=(
                "# num_h_acceptors\n\n"
                "Number of hydrogen-bond acceptors (Lipinski NO count) from a SMILES string.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT chem.main.num_h_acceptors('CCO');  -- 1 (ethanol O)\n"
                "```\n\n"
                "## Notes\n\n"
                "Lipinski rule HBA <= 10. Returns `NULL` on invalid input."
            ),
            keywords=["hydrogen bond acceptors", "hba", "h acceptors", "lipinski", "druglikeness", "descriptor"],
            relative_path=_SRC,
            category="descriptors",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.num_h_acceptors('CCO')",
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
        tags = {
            **object_tags(
                title="Morgan Fingerprint (Hex)",
                description_llm=_MORGAN_DOC_LLM,
                description_md=_MORGAN_DOC_MD,
                keywords=[
                    "morgan fingerprint",
                    "ecfp",
                    "fingerprint",
                    "circular fingerprint",
                    "bit vector",
                    "similarity",
                    "features",
                    "embedding",
                    "descriptor",
                ],
                relative_path=_SRC,
                category="fingerprint",
            ),
            # VGI509: guaranteed-runnable, catalog-qualified examples covering BOTH
            # arities of this function (default params and explicit radius/nbits), so
            # the demonstrated calls stay consistent with the documented signatures.
            "vgi.executable_examples": json.dumps(
                [
                    {
                        "description": "Default Morgan fingerprint of ethanol (radius=2, nbits=2048).",
                        "sql": "SELECT chem.main.morgan_fingerprint('CCO') AS fp",
                    },
                    {
                        "description": "Morgan fingerprint with an explicit radius and bit length.",
                        "sql": "SELECT chem.main.morgan_fingerprint('CCO', 3, 1024) AS fp",
                    },
                ]
            ),
        }
        examples = [
            FunctionExample(
                sql="SELECT chem.main.morgan_fingerprint('CCO')",
                description="Default Morgan fingerprint of ethanol (radius=2, nbits=2048)",
            ),
            FunctionExample(
                sql="SELECT chem.main.morgan_fingerprint('CCO', 3, 1024)",
                description="Morgan fingerprint with explicit radius=3 and nbits=1024",
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
        tags = object_tags(
            title="Morgan Fingerprint (Hex)",
            description_llm=_MORGAN_DOC_LLM,
            description_md=_MORGAN_DOC_MD,
            keywords=[
                "morgan fingerprint",
                "ecfp",
                "fingerprint",
                "circular fingerprint",
                "bit vector",
                "radius",
                "nbits",
                "similarity",
                "features",
            ],
            relative_path=_SRC,
            category="fingerprint",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.morgan_fingerprint('CCO', 3, 1024)",
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
        tags = object_tags(
            title="Tanimoto Similarity Score",
            description_llm=(
                "## tanimoto\n\n"
                "Computes the **Tanimoto similarity** between two molecules (from their SMILES) "
                "over Morgan fingerprints -- a value in [0, 1] where 1.0 means identical "
                "fingerprints.\n\n"
                "**Use it** for similarity search, nearest-neighbor lookup, and clustering of "
                "chemical structures.\n\n"
                "Two arities share this name:\n\n"
                "- `tanimoto(a, b)` -- Morgan `radius=2`.\n"
                "- `tanimoto(a, b, radius)` -- explicit radius.\n\n"
                "- **Input**: two SMILES strings (`VARCHAR`); a separate three-argument overload "
                "adds an explicit Morgan `radius` (`BIGINT`).\n"
                "- **Output**: `DOUBLE` similarity in [0, 1], or `NULL` if either SMILES is "
                "invalid.\n\n"
                "**Edge cases**: identical molecules score exactly `1.0`; disjoint structures "
                "approach `0.0`. If either input is `NULL` or invalid, the result is `NULL`."
            ),
            description_md=(
                "# tanimoto\n\n"
                "Tanimoto (Jaccard) similarity over Morgan fingerprints for two SMILES strings.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT chem.main.tanimoto('CCO', 'CCO');     -- 1.0\n"
                "SELECT chem.main.tanimoto('CCO', 'CCC', 2);  -- explicit radius\n"
                "```\n\n"
                "## Notes\n\n"
                "Range [0, 1]; 1.0 is identical. Returns `NULL` if either input is invalid."
            ),
            keywords=[
                "tanimoto",
                "similarity",
                "jaccard",
                "fingerprint similarity",
                "morgan",
                "nearest neighbor",
                "clustering",
                "similarity search",
            ],
            relative_path=_SRC,
            category="similarity",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.tanimoto('CCO', 'CCO')",
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
        tags = object_tags(
            title="Tanimoto Similarity Score",
            description_llm=(
                "## tanimoto(a, b, radius)\n\n"
                "Explicit-radius arity of **Tanimoto similarity**: compare two molecules (from "
                "their SMILES) over Morgan fingerprints built at a caller-chosen `radius`, "
                "returning a value in [0, 1].\n\n"
                "**Use it** when you want similarity at a specific fingerprint resolution.\n\n"
                "- **Input**: two SMILES strings (`VARCHAR`) and a Morgan `radius` (`BIGINT`).\n"
                "- **Output**: `DOUBLE` similarity in [0, 1], or `NULL` if either SMILES is "
                "invalid.\n\n"
                "**Edge cases**: identical molecules score `1.0`; both fingerprints use the same "
                "`radius`. `NULL`/invalid input returns `NULL`."
            ),
            description_md=(
                "# tanimoto (explicit radius)\n\n"
                "Tanimoto similarity over Morgan fingerprints at a chosen `radius`.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT chem.main.tanimoto('CCO', 'CCC', 2);\n"
                "```\n\n"
                "## Notes\n\n"
                "Range [0, 1]. Returns `NULL` if either input is invalid."
            ),
            keywords=[
                "tanimoto",
                "similarity",
                "jaccard",
                "fingerprint similarity",
                "morgan",
                "radius",
                "nearest neighbor",
                "similarity search",
            ],
            relative_path=_SRC,
            category="similarity",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.tanimoto('CCO', 'CCC', 2)",
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
        tags = object_tags(
            title="SMARTS Substructure Match",
            description_llm=(
                "## substructure_match\n\n"
                "Tests whether a molecule (given as SMILES) **contains a substructure** described "
                "by a [SMARTS](https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html) "
                "query pattern.\n\n"
                "**Use it** to filter compound sets by functional group or scaffold -- e.g. find "
                "every molecule containing a benzene ring or a carboxylic acid.\n\n"
                "- **Input**: a SMILES molecule and a SMARTS pattern (both `VARCHAR`).\n"
                "- **Output**: `BOOLEAN` -- `true` if the pattern matches, `false` if not.\n\n"
                "**Edge cases**: returns `NULL` if **either** the SMILES is invalid **or** the "
                "SMARTS pattern is invalid (a bad pattern is reported as `NULL`, never raised). "
                "`NULL` input returns `NULL`."
            ),
            description_md=(
                "# substructure_match\n\n"
                "Boolean SMARTS substructure test over a SMILES molecule.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT chem.main.substructure_match('c1ccccc1O', 'c1ccccc1');  -- true (phenol has benzene)\n"
                "```\n\n"
                "Use it in a `WHERE` predicate to keep only compounds carrying a scaffold or "
                "functional group -- e.g. the carboxylic-acid SMARTS `[CX3](=O)[OX2H1]`.\n\n"
                "## Notes\n\n"
                "Returns `NULL` when the SMILES or the SMARTS is invalid; otherwise `true`/`false`."
            ),
            keywords=[
                "substructure",
                "smarts",
                "pattern match",
                "scaffold",
                "functional group",
                "contains",
                "filter",
                "substructure search",
            ],
            relative_path=_SRC,
            category="substructure",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.substructure_match('c1ccccc1O', 'c1ccccc1')",
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


# ===========================================================================
# Drug-likeness (scalar predicate over the Lipinski rules)
# ===========================================================================


class DrugLikeFunction(ScalarFunction):
    """``drug_like(smiles)`` -- True if the molecule passes all four Lipinski rules."""

    class Meta:
        """VGI scalar function metadata (name, description, categories, examples)."""

        name = "drug_like"
        description = "True if the molecule passes all four Lipinski rule-of-five criteria, NULL if invalid"
        categories = ["chem", "druglikeness"]
        tags = object_tags(
            title="Lipinski Drug-Likeness Flag",
            description_llm=(
                "## drug_like\n\n"
                "A single-value **drug-likeness predicate**: `true` if the molecule (given as "
                "SMILES) passes **all four** Lipinski rule-of-five criteria, `false` if it "
                "violates any of them.\n\n"
                "**Use it** to filter compound libraries inline -- `WHERE drug_like(smiles)` -- "
                "instead of aggregating the per-rule `lipinski` table function in a subquery. "
                "Reach for `lipinski` instead when you need to see *which* rule a compound "
                "violates.\n\n"
                "Rules applied: molecular weight <= 500, Crippen logP <= 5, H-bond donors <= 5, "
                "H-bond acceptors <= 10.\n\n"
                "- **Input**: a SMILES string (`VARCHAR`).\n"
                "- **Output**: `BOOLEAN` -- `true` if all four rules pass, else `false`; `NULL` "
                "for `NULL` or invalid input.\n\n"
                "**Edge cases**: this is the scalar collapse of `bool_and(passes)` over "
                "`lipinski`; `NULL`/invalid input returns `NULL` (not `false`)."
            ),
            description_md=(
                "# drug_like\n\n"
                "Boolean Lipinski rule-of-five drug-likeness flag from a SMILES string.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT chem.main.drug_like('CC(=O)OC1=CC=CC=C1C(=O)O');  -- true (aspirin)\n"
                "```\n\n"
                "Use it directly in a `WHERE` clause to keep only drug-like compounds; use the "
                "`lipinski` table function when you need the per-rule breakdown.\n\n"
                "## Notes\n\n"
                "Passes when MW <= 500, logP <= 5, HBD <= 5 and HBA <= 10. Returns `NULL` on "
                "invalid input."
            ),
            keywords=[
                "drug-like",
                "druglikeness",
                "lipinski",
                "rule of five",
                "ro5",
                "screening",
                "filter",
                "predicate",
            ],
            relative_path=_SRC,
            category="druglikeness",
        )
        examples = [
            FunctionExample(
                sql="SELECT chem.main.drug_like('CC(=O)OC1=CC=CC=C1C(=O)O')",
                description="Aspirin passes all four Lipinski rules -> true",
            ),
        ]

    @classmethod
    def compute(
        cls, smiles: Annotated[pa.StringArray, Param(doc="SMILES string to screen.")]
    ) -> Annotated[pa.BooleanArray, Returns()]:
        """Map the pure chem function across the input Arrow array."""
        return _map_bool(smiles, chem.drug_like)


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
    # drug-likeness
    DrugLikeFunction,
]

# VGI515: give every example a description by mirroring Meta.examples into a
# vgi.example_queries tag (arity overloads aggregated by function name).
attach_example_queries(SCALAR_FUNCTIONS)
