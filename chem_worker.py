# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.16.0",
#     "rdkit>=2024.3",
#     "pyarrow",
# ]
# ///
"""Repo-root stdio/HTTP entry point for the vgi-chem worker (thin shim).

The worker itself -- the ``chem`` catalog, :class:`ChemWorker` and :func:`main`
-- lives in the wheel-importable :mod:`vgi_chem.worker` module. This file is a
thin PEP 723 shim that re-exports them, so:

- ``uv run chem_worker.py`` (Makefile, ci/run-integration.sh, tests) resolves the
  inline deps above and runs the worker exactly as before, and
- the installed wheel exposes the same worker at ``vgi_chem.worker:ChemWorker``
  (used by the ``vgi-chem-worker`` console script and the Docker image).

Usage:
    uv run chem_worker.py               # serve over stdio (DuckDB subprocess)

    INSTALL vgi FROM community; LOAD vgi;
    ATTACH 'chem' (TYPE vgi, LOCATION 'uv run chem_worker.py');

    SELECT chem.mol_formula('CC(=O)OC1=CC=CC=C1C(=O)O');  -- 'C9H8O4'
"""

from __future__ import annotations

from vgi_chem.worker import ChemWorker, main

__all__ = ["ChemWorker", "main"]


if __name__ == "__main__":
    main()
