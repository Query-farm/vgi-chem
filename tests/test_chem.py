"""Unit tests for the pure cheminformatics logic in ``vgi_chem.chem``.

These exercise the framework-free functions directly (no Arrow, no worker), with
known reference values and tolerances for floating-point descriptors.
"""

from __future__ import annotations

import math

import pytest

from vgi_chem import chem

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
BENZENE = "c1ccccc1"
ETHANOL = "CCO"
PHENOL = "c1ccccc1O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


class TestValidity:
    def test_valid(self) -> None:
        assert chem.is_valid_smiles(ASPIRIN) is True
        assert chem.is_valid_smiles(BENZENE) is True

    def test_invalid(self) -> None:
        assert chem.is_valid_smiles("xyz") is False
        assert chem.is_valid_smiles("") is False

    def test_canonical(self) -> None:
        # Two equivalent SMILES canonicalize to the same string.
        assert chem.canonical_smiles("OCC") == chem.canonical_smiles("CCO")

    def test_canonical_invalid(self) -> None:
        assert chem.canonical_smiles("xyz") is None


class TestFormulaAndMass:
    def test_formula_aspirin(self) -> None:
        assert chem.mol_formula(ASPIRIN) == "C9H8O4"

    def test_formula_invalid(self) -> None:
        assert chem.mol_formula("xyz") is None

    def test_weight_aspirin(self) -> None:
        mw = chem.mol_weight(ASPIRIN)
        assert mw is not None and math.isclose(mw, 180.16, abs_tol=0.05)

    def test_exact_mass_aspirin(self) -> None:
        em = chem.exact_mass(ASPIRIN)
        assert em is not None and math.isclose(em, 180.0423, abs_tol=0.01)

    def test_weight_invalid(self) -> None:
        assert chem.mol_weight("xyz") is None


class TestCounts:
    def test_benzene(self) -> None:
        assert chem.num_atoms(BENZENE) == 6
        assert chem.num_rings(BENZENE) == 1

    def test_ethanol_donors(self) -> None:
        assert chem.num_h_donors(ETHANOL) == 1

    def test_acceptors(self) -> None:
        assert chem.num_h_acceptors(ETHANOL) == 1

    def test_rotatable(self) -> None:
        assert chem.num_rotatable_bonds(BENZENE) == 0

    def test_invalid(self) -> None:
        assert chem.num_atoms("xyz") is None
        assert chem.num_rings("xyz") is None


class TestPhyschem:
    def test_logp_ethanol(self) -> None:
        lp = chem.logp(ETHANOL)
        assert lp is not None and lp < 1.0

    def test_tpsa_aspirin(self) -> None:
        t = chem.tpsa(ASPIRIN)
        assert t is not None and t > 0.0

    def test_invalid(self) -> None:
        assert chem.logp("xyz") is None
        assert chem.tpsa("xyz") is None


class TestInchi:
    def test_inchikey_aspirin(self) -> None:
        assert chem.inchikey_of(ASPIRIN) == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"

    def test_inchi_nonempty(self) -> None:
        out = chem.inchi_of(ETHANOL)
        assert out is not None and out.startswith("InChI=")

    def test_invalid(self) -> None:
        assert chem.inchikey_of("xyz") is None
        assert chem.inchi_of("xyz") is None


class TestFingerprintsAndSimilarity:
    def test_fingerprint_hex_length(self) -> None:
        fp = chem.morgan_fingerprint(ETHANOL)
        assert fp is not None and len(fp) == 2048 // 4

    def test_fingerprint_custom_nbits(self) -> None:
        fp = chem.morgan_fingerprint(ETHANOL, 2, 1024)
        assert fp is not None and len(fp) == 1024 // 4

    def test_fingerprint_invalid(self) -> None:
        assert chem.morgan_fingerprint("xyz") is None

    def test_tanimoto_self(self) -> None:
        assert chem.tanimoto(ASPIRIN, ASPIRIN) == 1.0

    def test_tanimoto_distinct_less_than_one(self) -> None:
        sim = chem.tanimoto(ASPIRIN, CAFFEINE)
        assert sim is not None and 0.0 <= sim < 1.0

    def test_tanimoto_invalid(self) -> None:
        assert chem.tanimoto(ASPIRIN, "xyz") is None
        assert chem.tanimoto("xyz", ASPIRIN) is None


class TestSubstructure:
    def test_match(self) -> None:
        assert chem.substructure_match(PHENOL, BENZENE) is True

    def test_no_match(self) -> None:
        assert chem.substructure_match(ETHANOL, BENZENE) is False

    def test_invalid_smiles(self) -> None:
        assert chem.substructure_match("xyz", BENZENE) is None

    def test_invalid_smarts(self) -> None:
        # Documented behaviour: bad SMARTS -> None rather than raising.
        assert chem.substructure_match(BENZENE, "[") is None


class TestLipinski:
    def test_aspirin_rows(self) -> None:
        rows = chem.lipinski(ASPIRIN)
        assert rows is not None
        rules = {r[0] for r in rows}
        assert rules == {"molecular_weight", "logp", "h_bond_donors", "h_bond_acceptors"}
        # Aspirin is drug-like: passes all four rules.
        assert all(r[2] for r in rows)

    def test_invalid(self) -> None:
        assert chem.lipinski("xyz") is None


@pytest.mark.parametrize(
    "fn",
    [
        chem.is_valid_smiles,
        chem.canonical_smiles,
        chem.mol_formula,
        chem.mol_weight,
        chem.num_atoms,
        chem.logp,
        chem.inchikey_of,
        chem.morgan_fingerprint,
    ],
)
def test_empty_string_safe(fn) -> None:  # type: ignore[no-untyped-def]
    # Empty string is invalid input; must never raise.
    result = fn("")
    assert result is None or result is False
