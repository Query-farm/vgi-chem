# Cheminformatics notebooks

[01 · Similarity depends on the fingerprint](01_similarity_depends_on_the_fingerprint.ipynb)
compares a fixed small-molecule library with phenol using Morgan fingerprints,
Tanimoto scores and a separate SMARTS substructure query. Visible SQL validates
SMILES, merges duplicate representations and verifies the worker's similarity
scores using bit-set intersection and union. Two charts show the full similarity
matrix and ranked candidates. The experiment changes only fingerprint radius.

## Run from this directory

Install [uv](https://docs.astral.sh/uv/) and ensure `uvx` is on PATH, then:

```sh
cd notebooks
uv sync --locked
uv run --locked python execute.py
```

In a notebook editor select this directory's `.venv/bin/python` kernel and run
all cells. The complete notebook directory works outside the worker checkout.
The notebook environment supports Python 3.13–3.14; its worker runs in a separate
Python 3.13 environment provisioned by `uvx`. No local RDKit install or worker
checkout is needed. The runner allows up to 600 seconds per cell for a first-run
worker environment installation.

The literal connection is visible:

```sql
INSTALL vgi FROM community;
LOAD vgi;
ATTACH 'chem' (TYPE vgi,
  LOCATION 'uvx --python 3.13 --with rdkit==2026.9.1 --with vgi-python==0.51.0 --with vgi-rpc==0.53.1 --from git+https://github.com/Query-farm/vgi-chem@v0.4.0 vgi-chem-worker');
```

The command fetches the public v0.4.0 Git tag and pins RDKit 2026.9.1, VGI Python
0.51.0 and VGI RPC 0.53.1. Haybarn 2.0.0rc11 and the notebook's visualization
packages are locked separately here. Initial downloads need network access;
subsequent chemistry calculations use local structures without external APIs.
Change the visible release/dependency pins and rerun both settings when upgrading.

Parameters, connection SQL and analytical SQL remain expanded. Imports, styling,
plotting, prose-generation source and cleanup are folded, with useful outputs
visible. Presentation helpers write review charts inside this notebook directory.

## One-parameter experiment

```sh
uv run --locked python execute.py --parameters '{"MORGAN_RADIUS":3}' \
  --output-dir validation/radius-3
```

Accepted radii are 2 (default) and 3 (experiment). Both are calculated as reference
comparisons each run. The parameter selects the matrix, ranking and interpretation.
The query stays phenol, the bit-vector size stays 2,048, and the source structures,
descriptors and SMARTS results stay fixed. Reset to 2 after comparing outputs.

The runner uses fresh kernels, saves only successful executions and requires a
separate destination for overrides. Unknown parameter names and source-directory
overwrites are rejected. Copies include assets, helpers, this guide and the locked
environment. Logs, extracted charts and experiments under `validation/` are
ignored; successful charts remain embedded in the main notebook.

## Provenance and interpretation

All twelve input records are embedded in SQL: eight simple named structures,
an alternate SMILES for phenol, empty text, malformed text and NULL. They form a
small teaching library, not a representative screening collection. Invalid and
missing records remain visible in the input report and do not become zero-score
candidates. Valid strings are grouped by returned canonical SMILES, retaining
their original labels; the phenol alias does not create a duplicate candidate.

The worker computes molecular formula, average molecular weight (g/mol), TPSA
(square angstroms), fingerprints and substructure matches. No extra salt,
protonation or tautomer standardization is applied. The fixed SMARTS expression
`[OX2H]c1ccccc1` tests a hydrogen-bearing oxygen attached to an aromatic six-carbon
ring. Invalid SMARTS returns NULL, distinct from a valid no-match result.

Morgan radius controls the environment-growth iterations. The worker explicitly
uses the requested radius and 2,048-bit vectors; other generator options retain
the pinned implementation's defaults. The notebook compares those returned bit
vectors using Tanimoto (intersection set-bit count divided by union set-bit
count). The [RDKit fingerprint API](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html)
documents generator parameters and defaults; the
[RDKit introduction](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints)
explains circular fingerprints and bit-vector representations.

The checks cover fixed width, nonempty fingerprints, complete pair grids,
bounds, self-similarity, symmetry and agreement between SQL and worker scores.
They do not assert which molecule must rank first. The plots use fixed 0–1
scales. Rankings use unrounded scores and name as a deterministic tie-breaker;
labels are rounded only for presentation. Phenol's self-match is excluded from
the candidate ranking but retained on the matrix diagonal for validation.

A fingerprint score and an explicit motif match answer different questions.
A finite fingerprint can lose structural information, so a high score alone is
not an identity or substructure proof. This notebook evaluates these examples
and settings; it does not establish one radius as universally better.

The official Strata Sun and VGI media-kit assets are unchanged and bundled locally.
Linked Strata Sun branding appears at the top and final bottom of the notebook.

References: [worker documentation](https://github.com/Query-farm/vgi-chem),
[pinned release](https://github.com/Query-farm/vgi-chem/releases/tag/v0.4.0).
