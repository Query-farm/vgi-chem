"""Cheminformatics for DuckDB/SQL via RDKit, as a VGI worker.

The implementation is split so each concern stays focused:

- ``chem``    -- pure cheminformatics logic over RDKit (molecular descriptors,
  fingerprints/similarity, substructure search, InChI). No Arrow or VGI
  dependency, directly unit-testable; imports RDKit once at module load.
- ``scalars`` -- per-row VGI scalar functions (positional-only; the optional
  ``radius`` / ``nbits`` arguments to fingerprint/similarity are exposed as arity
  overloads sharing a function name).
- ``tables``  -- the ``lipinski(smiles)`` rule-of-five breakdown table function.

``chem_worker.py`` at the repo root assembles these into the ``chem`` catalog
and runs the worker over stdio (or HTTP).
"""

from __future__ import annotations

__version__ = "0.1.0"
