# RDKit Cheminformatics notebooks

[01 · Similarity depends on the fingerprint](01_similarity_depends_on_the_fingerprint.ipynb)
asks whether changing Morgan radius changes the nearest match to phenol. Visible
SQL validates twelve input records, merges duplicate representations, computes
molecular descriptors and verifies worker Tanimoto scores against returned
fingerprint bits. A phenol/cyclohexanol comparison explains why matching ring and
H-donor counts cannot establish an aromatic-OH motif. Two SVG charts show the
similarity matrix and ranked candidates.

## Run locally or in Colab

Install [uv](https://docs.astral.sh/uv/) and Git, then run from this directory:

```sh
uv sync --locked
uv run --locked vgi-notebooks 01_similarity_depends_on_the_fingerprint.ipynb --timeout 600
```

Use this directory's `.venv/bin/python` in a notebook editor. Alternatively, use
the notebook's **Open in Colab** badge and run all cells. Its first setup cell
installs only the shared package's base dependencies, preserving compatible
preloaded pandas/Matplotlib/IPython packages. The local lock adds `[runner]` for
publication; it is not installed into Colab. A fresh preloaded Python 3.12 kernel
was tested with this full notebook; hosted Colab was not manually exercised.

The notebook file runs without this worker checkout or companion Python files.
The shared package supplies unchanged linked branding, an isolated SQL connection,
SVG presentation and local execution guards. The proper catalog title links to
the worker repository, and the footer links there too. Python setup/plotting code
is folded; parameters and analytical SQL stay visible.

## Published worker, separate environment

```sql
INSTALL vgi FROM community;
LOAD vgi;
ATTACH 'chem' (TYPE vgi,
  LOCATION 'uvx --python 3.13 --from git+https://github.com/Query-farm/vgi-chem@v0.4.0 vgi-chem-worker');
```

This fetches public release **v0.4.0** (`54942211e3bb81fd96318b5d1a6539391f527c7c`)
in its own environment. `uv` provisions Python 3.13 if needed. The tested resolver
supplied RDKit **2026.9.2**, VGI Python **0.51.1**, VGI RPC **0.53.1** and PyArrow
**26.0.1**. The release source is pinned; its transitive dependencies follow the
release declarations and can change on a future fresh resolution. No `--with`
overrides or RDKit install in the notebook kernel are needed. The notebook locks
Haybarn **2.0.0rc11** and shared tooling **v0.3.13** separately.

Initial downloads need network access; all chemistry uses the embedded structures
and runs locally afterwards. SQL starts with an empty temporary secret directory
and persistent secrets disabled. No credentials or external account operations
are involved. Change the visible release tag and rerun both settings when upgrading.

## One-parameter experiment

```sh
uv run --locked vgi-notebooks 01_similarity_depends_on_the_fingerprint.ipynb \
  --parameters '{"MORGAN_RADIUS":3}' --output-dir validation/radius-3 --timeout 600
```

Accepted radii are **2** (default) and **3** (experiment). Both are calculated as a
fixed reference every run; the parameter selects the matrix, ranking and prose.
The query stays phenol, the vector size stays 2,048 bits, and inputs, descriptors,
SMARTS results and the full two-radius comparison stay fixed. Predict which
candidate will lead, compare both charts and reset to **2**.

The runner uses fresh kernels and only replaces a notebook after success.
Overrides require a separate output directory; unknown parameters and attempts to
overwrite the source directory are rejected. Detached execution copies contain
SVG assets and a README. Review PNGs, logs and experiment copies are ignored;
linked SVGs under `charts/` and successful notebook outputs are published.

## Provenance and limits

All twelve input records are embedded in SQL: eight simple named structures,
an alternate phenol SMILES, empty text, malformed text and NULL. They form a small
teaching library, not a representative screening collection. Invalid and missing
records stay visible and do not become zero-score candidates. Valid strings are
grouped by returned canonical SMILES, retaining original labels. No additional
salt, protonation or tautomer standardization is applied.

The worker computes formula, average molecular weight (g/mol), topological polar
surface area (Å²), ring/H-donor counts, fingerprints and substructure matches.
The fixed SMARTS expression `[OX2H]c1ccccc1` tests a hydrogen-bearing oxygen
attached to an aromatic six-carbon ring. Invalid SMARTS yields NULL, distinct from
a valid no-match result. Phenol and cyclohexanol share coarse counts while differing
on this connectivity query; the notebook does not infer potency or medical effects.

Morgan radius controls the environment-growth iterations. Other generator options
retain the implementation's defaults. SQL converts returned hexadecimal vectors
into bit sets and independently calculates intersection/union Tanimoto similarity.
Checks cover width, nonempty vectors, complete grids, bounds, self-similarity,
symmetry, SQL/worker agreement and invalid-input semantics. A separate two-structure
control checks the descriptor/motif distinction. Rankings use unrounded scores and
name as a deterministic tie-breaker; labels are rounded only for presentation.
Phenol's self-match is excluded from ranking and retained on the matrix diagonal.

A finite fingerprint can lose structural information. A high score is neither an
identity proof nor a substructure proof, and these examples do not establish one
radius as universally better. [RDKit's fingerprint API](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html)
documents parameters and defaults; its [introduction](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints)
explains circular fingerprints.

## Coverage evidence

[coverage.json](coverage.json) maps **10 of 20** logical public names to visible
executed SQL and reviewed interpretations, meeting the half-surface target.
Complete native `duckdb_functions()`, `duckdb_tables()` and `duckdb_views()` exports
reconcile the denominator; overloads and the same-named `example_molecules`
relation/function alias count once. No capability or authentication exclusions
are applied.

Released v0.4.0 and current public main `8fa8501ff3a61781c132f4ea137b15f2e4b456d0`
have the same 20-name interface. Current main adds composed metadata examples;
the selected analytical implementations are unchanged. The notebook executes the
released worker: exact-runtime coverage is **10 released / 0 current**. Editorial
compatibility review supports **10/20 current-interface** coverage; it is not an
execution claim for current source. The other ten surfaces remain explicitly
uncovered: `drug_like`, `exact_mass`, `example_molecules`, `inchi`, `inchikey`,
`lipinski`, `logp`, `num_atoms`, `num_h_acceptors`, `num_rotatable_bonds`.

This notebook-only update reuses the successful source CI baseline
[43562740983](https://github.com/Query-farm/vgi-chem/actions/runs/43562740983).
Worker sources and root dependencies are unchanged. Final notebook execution,
experiment, fresh/preloaded bootstrap, visual review and hash-bound coverage
receipts are checked separately.
