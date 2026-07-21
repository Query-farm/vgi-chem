<p align="center">
  <img src="https://raw.githubusercontent.com/Query-farm/vgi/main/docs/vgi-logo.png" alt="Vector Gateway Interface (VGI)" width="320">
</p>

<p align="center"><em>A <a href="https://query.farm">Query.Farm</a> VGI worker for DuckDB.</em></p>

# Cheminformatics — Descriptors, Fingerprints & Substructure Search in DuckDB

> **vgi-chem** · a [Query.Farm](https://query.farm) VGI worker · powered by RDKit

[![CI](https://github.com/Query-farm/vgi-chem/actions/workflows/ci.yml/badge.svg)](https://github.com/Query-farm/vgi-chem/actions/workflows/ci.yml)

A [VGI](https://query.farm) worker that brings **cheminformatics** into
DuckDB/SQL. It computes **molecular descriptors, fingerprints + similarity,
substructure search, and InChI** from SMILES strings as plain SQL functions,
backed by [RDKit](https://www.rdkit.org/) (BSD-3-Clause) — the standard
open-source cheminformatics toolkit.

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'chem' (TYPE vgi, LOCATION 'uv run chem_worker.py');

SELECT chem.is_valid_smiles('CCO');                              -- true
SELECT chem.canonical_smiles('OCC');                             -- 'CCO'
SELECT chem.mol_formula('CC(=O)OC1=CC=CC=C1C(=O)O');             -- 'C9H8O4'
SELECT chem.mol_weight('CC(=O)OC1=CC=CC=C1C(=O)O');              -- ~180.16
SELECT chem.logp('CCO'), chem.tpsa('CCO');                       -- Crippen logP, TPSA
SELECT chem.inchikey('CC(=O)OC1=CC=CC=C1C(=O)O');                -- 'BSYNRYMUTXBXSQ-UHFFFAOYSA-N'
SELECT chem.tanimoto('CCO', 'CCO');                              -- 1.0
SELECT chem.substructure_match('c1ccccc1O', 'c1ccccc1');         -- true
SELECT * FROM chem.lipinski('CC(=O)OC1=CC=CC=C1C(=O)O');         -- rule-of-five rows
```

Everything runs **offline and deterministically** — RDKit computes locally with
no network calls, so the same input always gives the same answer.

## Positioning vs. `ducksmiles`

The catalog already ships a `ducksmiles` worker for **basic SMILES** handling
(validate / canonicalize). `vgi-chem` is **scoped beyond that**: it is the
**descriptors / fingerprints / substructure / InChI** worker. It does include
`is_valid_smiles` and `canonical_smiles` for convenience (so you can do
everything from one `ATTACH`), but its reason to exist is the chemistry that
`ducksmiles` does not cover — molecular weight/formula/exact mass, H-bond and
ring/rotatable-bond counts, Crippen logP and TPSA, Morgan fingerprints and
Tanimoto similarity, SMARTS substructure matching, InChI/InChIKey, and the
Lipinski rule-of-five breakdown. Reach for `ducksmiles` for lightweight SMILES
plumbing; reach for `vgi-chem` when you need real cheminformatics.

## Scalars (per-row) vs. table functions

The split follows what the VGI SDK allows for each function shape:

* **Scalars** take **positional** arguments only and resolve overloads by
  *arity* (DuckDB's `name := value` syntax is a table-function/macro feature, not
  a scalar one). Every per-row answer is a **scalar**, so it works inline in any
  projection or predicate. Where a function takes optional parameters, those are
  extra positional **arity overloads**:

  ```sql
  SELECT morgan_fingerprint(smiles)                FROM compounds;  -- radius=2, nbits=2048
  SELECT morgan_fingerprint(smiles, 3, 1024)       FROM compounds;  -- explicit radius/nbits
  SELECT tanimoto(a.smiles, b.smiles)              FROM a, b;       -- radius=2
  SELECT tanimoto(a.smiles, b.smiles, 3)           FROM a, b;       -- explicit radius
  ```

* **Table functions** return *many* rows. `lipinski(smiles)` expands one molecule
  into one row per rule:

  ```sql
  SELECT * FROM chem.lipinski('CC(=O)OC1=CC=CC=C1C(=O)O');
  ```

**NULL / invalid semantics.** A NULL input yields NULL output for every function.
An **invalid SMILES** yields `NULL` for every value function (formula, weight,
fingerprint, …) and `false` for `is_valid_smiles`; `lipinski` returns **no rows**.
For `substructure_match`, an invalid **SMILES** *or* an invalid **SMARTS**
pattern yields `NULL` (a bad pattern is reported as NULL rather than aborting the
query). Nothing here ever raises a SQL error on bad chemical input.

## Function catalog

| Function | Form | Signature | Returns |
| --- | --- | --- | --- |
| `is_valid_smiles` | scalar | `(smiles)` | `BOOLEAN` |
| `canonical_smiles` | scalar | `(smiles)` | `VARCHAR` (NULL if invalid) |
| `mol_formula` | scalar | `(smiles)` | `VARCHAR` (Hill, NULL if invalid) |
| `mol_weight` | scalar | `(smiles)` | `DOUBLE` (avg MW, g/mol) |
| `exact_mass` | scalar | `(smiles)` | `DOUBLE` (monoisotopic mass) |
| `num_atoms` | scalar | `(smiles)` | `INT` (heavy atoms) |
| `num_rings` | scalar | `(smiles)` | `INT` |
| `num_rotatable_bonds` | scalar | `(smiles)` | `INT` |
| `num_h_donors` | scalar | `(smiles)` | `INT` |
| `num_h_acceptors` | scalar | `(smiles)` | `INT` |
| `logp` | scalar | `(smiles)` | `DOUBLE` (Crippen MolLogP) |
| `tpsa` | scalar | `(smiles)` | `DOUBLE` (topological polar surface area) |
| `inchi` | scalar | `(smiles)` | `VARCHAR` (standard InChI) |
| `inchikey` | scalar | `(smiles)` | `VARCHAR` (standard InChIKey) |
| `morgan_fingerprint` | scalar | `(smiles[, radius, nbits])` | `VARCHAR` (hex bit-string) |
| `tanimoto` | scalar | `(smiles_a, smiles_b[, radius])` | `DOUBLE` in `[0, 1]` |
| `substructure_match` | scalar | `(smiles, smarts)` | `BOOLEAN` (NULL on bad SMARTS) |
| `drug_like` | scalar | `(smiles)` | `BOOLEAN` (passes all four Lipinski rules) |
| `lipinski` | table | `(smiles)` | `(rule VARCHAR, value DOUBLE, passes BOOLEAN)` |

### Descriptors

`mol_weight` is the average molecular weight; `exact_mass` is the monoisotopic
mass. `num_atoms` counts **heavy** (non-hydrogen) atoms. `logp` is RDKit's
Crippen `MolLogP`; `tpsa` is the topological polar surface area. All return
`NULL` for an unparseable SMILES.

### Fingerprints + similarity

`morgan_fingerprint(smiles)` returns a Morgan (ECFP-like) fingerprint as a hex
bit-string (`nbits / 4` hex chars; defaults `radius=2`, `nbits=2048`); the
`(smiles, radius, nbits)` overload sets both explicitly. `tanimoto(a, b)` is the
Morgan/Tanimoto similarity in `[0, 1]` (self-similarity is exactly `1.0`); the
`(a, b, radius)` overload sets the Morgan radius.

### Substructure

`substructure_match(smiles, smarts)` is `true` when the molecule contains the
SMARTS pattern. An invalid SMILES or invalid SMARTS yields `NULL` (documented
behaviour) — a malformed pattern in one row never aborts the whole query.

### Lipinski rule-of-five

`lipinski(smiles)` returns one row per rule — `molecular_weight` (≤ 500),
`logp` (≤ 5), `h_bond_donors` (≤ 5), `h_bond_acceptors` (≤ 10) — each with its
computed `value` and whether it `passes`. Roll it up to a single drug-likeness
flag:

```sql
SELECT bool_and(passes) AS drug_like
FROM chem.lipinski('CC(=O)OC1=CC=CC=C1C(=O)O');   -- true
```

For that roll-up as a single inline predicate, use the `drug_like(smiles)` scalar
— it is exactly `bool_and(passes)` over `lipinski`, so you can filter a whole
column of compounds without a subquery:

```sql
SELECT smiles FROM compounds WHERE chem.drug_like(smiles);
```

An invalid SMILES yields **no rows**.

## Dependencies & licensing

| Component | License | Notes |
| --- | --- | --- |
| `vgi-chem` (this worker) | **MIT** | This repository's own code. |
| [`rdkit`](https://pypi.org/project/rdkit/) | **BSD-3-Clause** | Cheminformatics engine (permissive). |
| [`vgi-python`](https://github.com/Query-farm/vgi-python) | Query Farm Source-Available | The VGI SDK. |

RDKit is **BSD-3-Clause** (permissive, commercial-use-friendly) and is used as an
**unmodified, separately-installed pip dependency** — imported, never vendored or
patched. Descriptor / fingerprint definitions are exactly those RDKit computes;
consult the RDKit docs for their precise definitions.

## Local development

```sh
uv sync --all-extras     # create .venv with vgi-python + rdkit + dev tools
make test                # pytest (unit + integration) + SQL end-to-end
make test-unit           # pytest only
make test-sql            # DuckDB sqllogictest files via haybarn-unittest
uv run ruff check .      # lint
uv run mypy vgi_chem/
```

`tests/test_chem.py` covers the pure cheminformatics logic with known reference
values (aspirin formula/MW/InChIKey, benzene ring/atom counts, ethanol H-bond
donors, Tanimoto self-similarity, substructure matching, and a battery of
invalid-input edge cases). `tests/test_tables.py` drives the `lipinski` table
function through the real bind→init→process lifecycle in-process;
`tests/test_scalars.py` spawns `chem_worker.py` over the VGI client/RPC stack
exactly as DuckDB would after `ATTACH`. The `test/sql/*.test` files are DuckDB
sqllogictest cases run by
[`haybarn-unittest`](https://pypi.org/project/haybarn-unittest/)
(`uv tool install haybarn-unittest`) against a real `ATTACH` + `SELECT`.

## Layout

```
chem_worker.py           entry point; assembles the `chem` catalog (inline uv script metadata)
Makefile                 test / test-unit / test-sql targets
vgi_chem/
  chem.py                pure cheminformatics logic over RDKit (no Arrow/VGI); imports RDKit once
  scalars.py             per-row scalars (arity overloads for fingerprint radius/nbits, tanimoto radius)
  tables.py              the lipinski(smiles) rule-of-five table function
  schema_utils.py        Arrow field/comment helper
tests/
  harness.py             in-process bind→init→process driver
  test_chem.py           pure-logic unit + edge tests with known reference values
  test_tables.py         lipinski table-function integration tests
  test_scalars.py        per-row scalar overloads via vgi.client.Client
test/sql/
  *.test                 DuckDB sqllogictest end-to-end cases (haybarn-unittest)
```

---

## Authorship & License

Written by [Query.Farm](https://query.farm).

Copyright 2026 Query Farm LLC - https://query.farm

