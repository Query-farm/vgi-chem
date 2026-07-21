"""Pure cheminformatics logic over RDKit, with no Arrow or VGI dependency.

Every function here takes plain Python values (SMILES / SMARTS ``str``) and
returns plain Python values, so the whole module is directly unit-testable and
free of any framework coupling. The VGI scalar/table adapters in
:mod:`vgi_chem.scalars` and :mod:`vgi_chem.tables` map these across Arrow arrays.

Design rules (hard-won; see the README "Robustness" section):

- **Import RDKit exactly once, at module load.** The RDKit import is slow; doing
  it here caches it for the whole process lifetime.
- **Never raise on bad input.** Every public function wraps its RDKit calls and
  returns ``None`` / ``False`` for input RDKit can't parse, so a malformed SMILES
  or SMARTS can never crash the worker. ``parse_*`` helpers return ``None``.
- **Never write to stdout.** RDKit logs to stderr (fine -- stderr does not
  corrupt the stdio Arrow stream). We disable RDKit's logger to keep even that
  quiet, but the critical invariant is simply that nothing here prints.

NULL handling (the ``None`` input case) is the adapters' job: they pass ``None``
straight through and only call into this module for non-NULL rows.
"""

from __future__ import annotations

# --- One-time, process-lifetime RDKit import (slow; cached here). ------------
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    Crippen,
    DataStructs,
    Descriptors,
    inchi,
    rdMolDescriptors,
)
from rdkit.Chem import (
    rdFingerprintGenerator as _rdfp,
)

# Silence RDKit's C++ logger. Any residual native warnings go to stderr only,
# which is harmless to the stdio Arrow protocol; stdout stays untouched.
RDLogger.DisableLog("rdApp.*")

# Default Morgan fingerprint parameters (shared by fingerprint + similarity).
DEFAULT_RADIUS = 2
DEFAULT_NBITS = 2048


# ---------------------------------------------------------------------------
# Parsing helpers -- the single choke point through which all SMILES/SMARTS
# parsing flows. They never raise: unparseable input yields ``None``.
# ---------------------------------------------------------------------------


def parse_mol(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES string into an RDKit ``Mol``, or ``None`` if invalid.

    The empty string is treated as invalid (``None``). RDKit happily parses it
    into a 0-atom molecule, but for SQL purposes an empty SMILES is "no molecule"
    -- so empty input behaves like every other unparseable value (NULL output).
    """
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def parse_smarts(smarts: str) -> Chem.Mol | None:
    """Parse a SMARTS pattern into an RDKit query ``Mol``, or ``None`` if invalid."""
    if not smarts:
        return None
    try:
        return Chem.MolFromSmarts(smarts)
    except Exception:
        return None


def _morgan_generator(radius: int, nbits: int) -> _rdfp.FingerprintGenerator64:
    return _rdfp.GetMorganGenerator(radius=radius, fpSize=nbits)


# ---------------------------------------------------------------------------
# Validity + identity
# ---------------------------------------------------------------------------


def is_valid_smiles(smiles: str) -> bool:
    """True if ``smiles`` parses to a molecule."""
    return parse_mol(smiles) is not None


def canonical_smiles(smiles: str) -> str | None:
    """RDKit canonical SMILES, or ``None`` if the input is invalid."""
    mol = parse_mol(smiles)
    if mol is None:
        return None
    try:
        return str(Chem.MolToSmiles(mol))
    except Exception:
        return None


def mol_formula(smiles: str) -> str | None:
    """Hill-system molecular formula (e.g. ``'C9H8O4'``), or ``None`` if invalid."""
    mol = parse_mol(smiles)
    if mol is None:
        return None
    try:
        return str(rdMolDescriptors.CalcMolFormula(mol))
    except Exception:
        return None


def inchi_of(smiles: str) -> str | None:
    """Standard InChI string, or ``None`` if invalid / not derivable."""
    mol = parse_mol(smiles)
    if mol is None:
        return None
    try:
        out = inchi.MolToInchi(mol)
        return out or None
    except Exception:
        return None


def inchikey_of(smiles: str) -> str | None:
    """Standard InChIKey (27-char hashed InChI), or ``None`` if invalid."""
    mol = parse_mol(smiles)
    if mol is None:
        return None
    try:
        out = inchi.MolToInchiKey(mol)
        return out or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Numeric descriptors (each ``None`` if the SMILES is invalid)
# ---------------------------------------------------------------------------


def mol_weight(smiles: str) -> float | None:
    """Average molecular weight (g/mol)."""
    mol = parse_mol(smiles)
    if mol is None:
        return None
    try:
        return float(Descriptors.MolWt(mol))
    except Exception:
        return None


def exact_mass(smiles: str) -> float | None:
    """Monoisotopic (exact) mass."""
    mol = parse_mol(smiles)
    if mol is None:
        return None
    try:
        return float(Descriptors.ExactMolWt(mol))
    except Exception:
        return None


def num_atoms(smiles: str) -> int | None:
    """Number of heavy (non-hydrogen) atoms."""
    mol = parse_mol(smiles)
    if mol is None:
        return None
    try:
        return int(mol.GetNumHeavyAtoms())
    except Exception:
        return None


def num_rings(smiles: str) -> int | None:
    """Number of rings (SSSR ring count)."""
    mol = parse_mol(smiles)
    if mol is None:
        return None
    try:
        return int(rdMolDescriptors.CalcNumRings(mol))
    except Exception:
        return None


def num_rotatable_bonds(smiles: str) -> int | None:
    """Number of rotatable bonds."""
    mol = parse_mol(smiles)
    if mol is None:
        return None
    try:
        return int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    except Exception:
        return None


def num_h_donors(smiles: str) -> int | None:
    """Number of hydrogen-bond donors (Lipinski)."""
    mol = parse_mol(smiles)
    if mol is None:
        return None
    try:
        return int(rdMolDescriptors.CalcNumHBD(mol))
    except Exception:
        return None


def num_h_acceptors(smiles: str) -> int | None:
    """Number of hydrogen-bond acceptors (Lipinski)."""
    mol = parse_mol(smiles)
    if mol is None:
        return None
    try:
        return int(rdMolDescriptors.CalcNumHBA(mol))
    except Exception:
        return None


def logp(smiles: str) -> float | None:
    """Crippen MolLogP (octanol-water partition coefficient estimate)."""
    mol = parse_mol(smiles)
    if mol is None:
        return None
    try:
        return float(Crippen.MolLogP(mol))
    except Exception:
        return None


def tpsa(smiles: str) -> float | None:
    """Topological polar surface area (TPSA), in Angstrom^2."""
    mol = parse_mol(smiles)
    if mol is None:
        return None
    try:
        return float(rdMolDescriptors.CalcTPSA(mol))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fingerprints + similarity
# ---------------------------------------------------------------------------


def morgan_fingerprint(smiles: str, radius: int = DEFAULT_RADIUS, nbits: int = DEFAULT_NBITS) -> str | None:
    """Morgan (ECFP-like) fingerprint as a lower-case hex string of ``nbits`` bits.

    The hex encodes the dense bit vector MSB-first, so the string length is
    ``nbits / 4`` characters. Returns ``None`` for invalid SMILES or bad params.
    """
    mol = parse_mol(smiles)
    if mol is None:
        return None
    try:
        if radius < 0 or nbits <= 0:
            return None
        gen = _morgan_generator(radius, nbits)
        bitvect = gen.GetFingerprint(mol)
        return str(DataStructs.BitVectToFPSText(bitvect))
    except Exception:
        return None


def tanimoto(smiles_a: str, smiles_b: str, radius: int = DEFAULT_RADIUS) -> float | None:
    """Morgan/Tanimoto similarity in ``[0, 1]``, or ``None`` if either is invalid."""
    mol_a = parse_mol(smiles_a)
    mol_b = parse_mol(smiles_b)
    if mol_a is None or mol_b is None:
        return None
    try:
        if radius < 0:
            return None
        gen = _morgan_generator(radius, DEFAULT_NBITS)
        fp_a = gen.GetFingerprint(mol_a)
        fp_b = gen.GetFingerprint(mol_b)
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Substructure search
# ---------------------------------------------------------------------------


def substructure_match(smiles: str, smarts: str) -> bool | None:
    """True if ``smiles`` contains the ``smarts`` pattern.

    Behaviour:

    - Invalid **SMILES** -> ``None`` (treated like every other value function).
    - Invalid **SMARTS** -> ``None`` (documented: an unparseable query pattern is
      reported as NULL rather than raising, so a bad pattern in one row cannot
      abort the whole query).
    """
    mol = parse_mol(smiles)
    if mol is None:
        return None
    pattern = parse_smarts(smarts)
    if pattern is None:
        return None
    try:
        return bool(mol.HasSubstructMatch(pattern))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Lipinski rule-of-five breakdown (table function)
# ---------------------------------------------------------------------------

# (rule label, threshold, comparison) -- value is compared <= threshold to pass.
_LIPINSKI_RULES: tuple[tuple[str, float], ...] = (
    ("molecular_weight", 500.0),
    ("logp", 5.0),
    ("h_bond_donors", 5.0),
    ("h_bond_acceptors", 10.0),
)


def drug_like(smiles: str) -> bool | None:
    """True if the molecule passes all four Lipinski rule-of-five criteria.

    A scalar convenience over :func:`lipinski`: it collapses the four-rule
    breakdown into a single drug-likeness predicate (``bool_and(passes)``), so
    callers can filter inline with ``WHERE drug_like(smiles)`` instead of a
    correlated subquery over the ``lipinski`` table function. Returns ``None``
    for invalid SMILES (never raises).
    """
    breakdown = lipinski(smiles)
    if breakdown is None:
        return None
    return all(rule[2] for rule in breakdown)


def lipinski(smiles: str) -> list[tuple[str, float, bool]] | None:
    """Lipinski rule-of-five breakdown: one ``(rule, value, passes)`` per rule.

    Rules: MW <= 500, logP <= 5, HBD <= 5, HBA <= 10. Returns ``None`` (no rows)
    for invalid SMILES.
    """
    mol = parse_mol(smiles)
    if mol is None:
        return None
    try:
        values = {
            "molecular_weight": float(Descriptors.MolWt(mol)),
            "logp": float(Crippen.MolLogP(mol)),
            "h_bond_donors": float(rdMolDescriptors.CalcNumHBD(mol)),
            "h_bond_acceptors": float(rdMolDescriptors.CalcNumHBA(mol)),
        }
    except Exception:
        return None
    out: list[tuple[str, float, bool]] = []
    for rule, threshold in _LIPINSKI_RULES:
        value = values[rule]
        out.append((rule, value, value <= threshold))
    return out


# ---------------------------------------------------------------------------
# Curated example-molecule registry (browsable discovery table).
# ---------------------------------------------------------------------------

# A small, hand-picked set of well-known molecules. Exposed as the no-argument
# ``example_molecules()`` table function so an agent can browse real SMILES (and
# ready-computed descriptors) without inventing structures first. Every
# descriptor column is computed live from the SMILES via the functions above, so
# the table can never drift from the scalars it advertises.
_EXAMPLE_MOLECULES: tuple[tuple[str, str], ...] = (
    ("water", "O"),
    ("ethanol", "CCO"),
    ("benzene", "c1ccccc1"),
    ("phenol", "c1ccccc1O"),
    ("acetic acid", "CC(=O)O"),
    ("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("glucose", "OCC1OC(O)C(O)C(O)C1O"),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("nicotine", "CN1CCCC1c1cccnc1"),
    ("penicillin g", "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"),
)


def example_molecules() -> list[dict[str, object]]:
    """Return the curated example-molecule registry with live-computed descriptors.

    One dict per molecule: ``name``, ``smiles``, ``formula``, ``mol_weight``,
    ``logp``, ``tpsa``, ``h_bond_donors``, ``h_bond_acceptors``, ``num_rings``
    and ``drug_like`` (whether the molecule passes all four Lipinski rules).
    Every value is derived from the module's own functions, so the registry stays
    consistent with the scalar/table functions it illustrates.
    """
    rows: list[dict[str, object]] = []
    for name, smiles in _EXAMPLE_MOLECULES:
        rows.append(
            {
                "name": name,
                "smiles": smiles,
                "formula": mol_formula(smiles),
                "mol_weight": mol_weight(smiles),
                "logp": logp(smiles),
                "tpsa": tpsa(smiles),
                "h_bond_donors": num_h_donors(smiles),
                "h_bond_acceptors": num_h_acceptors(smiles),
                "num_rings": num_rings(smiles),
                "drug_like": drug_like(smiles) or False,
            }
        )
    return rows
