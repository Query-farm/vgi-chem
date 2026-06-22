# CLAUDE.md — vgi-chem

Contributor/agent notes. User-facing docs live in `README.md`; this is the
"how it's built and where the sharp edges are" companion.

## What this is

A [VGI](https://query.farm) worker that brings **cheminformatics** into DuckDB —
molecular descriptors, fingerprints + Tanimoto similarity, SMARTS substructure
search, and InChI/InChIKey, all computed from SMILES strings via
[RDKit](https://www.rdkit.org/) (BSD-3-Clause) — as scalar functions plus a
`lipinski(smiles)` rule-of-five table function. `chem_worker.py` assembles every
function into one `chem` catalog (single `main` schema) over stdio. Sibling
style/tooling to `vgi-conform` and `vgi-calendar`.

## Positioning vs. `ducksmiles`

The catalog already has `ducksmiles` for **basic SMILES** (validate/canonicalize).
`vgi-chem` is deliberately scoped **beyond** that — descriptors, fingerprints,
substructure, InChI. It re-exposes `is_valid_smiles` / `canonical_smiles` only
for one-ATTACH convenience; everything else (MW/formula/exact-mass, ring/H-bond/
rotatable counts, Crippen logP, TPSA, Morgan fingerprints, Tanimoto, SMARTS
match, InChI/InChIKey, Lipinski) is net-new chemistry. Don't turn this into a
SMILES-plumbing duplicate of `ducksmiles`.

## Layout

```
chem_worker.py       repo-root stdio entry point; PEP 723 inline deps; main()
vgi_chem/
  chem.py            pure cheminformatics logic over RDKit; no Arrow/VGI; unit-testable
  scalars.py         per-row scalars (arity overloads for fingerprint radius/nbits, tanimoto radius)
  tables.py          the lipinski(smiles) rule-of-five table function
  schema_utils.py    pa.Field comment / column-doc helper
tests/               pytest: test_chem (pure), test_tables (in-proc), test_scalars (Client RPC)
test/sql/*.test      haybarn-unittest sqllogictest — authoritative E2E
Makefile             test / test-unit / test-sql / lint
```

To add a function: implement the logic in `chem.py` (pure, total — never raises
on garbage; returns `None` for "invalid"), wrap it as a scalar or table function
in the matching module, register it in `chem_worker.py`'s `_FUNCTIONS`.

## Scalars vs table functions — THE core convention (read first)

The VGI SDK makes **scalar functions positional-only**: `name := value` named
args are rejected for scalars and only work on table functions.

- **Per-row functions are scalars with arity overloads** so they work inline in a
  projection (`SELECT mol_weight(smiles) FROM compounds`). Where a function has
  optional params, each arity is its own `ScalarFunction` subclass sharing the
  `Meta.name`: `morgan_fingerprint(smiles)` (radius=2, nbits=2048) /
  `morgan_fingerprint(smiles, radius, nbits)`; `tanimoto(a, b)` (radius=2) /
  `tanimoto(a, b, radius)`. Each overload is written out explicitly (a nested
  `class Meta:` body can't close over an enclosing-scope variable, so no factory).
- **Set-returning functions are table functions**: `lipinski(smiles)` — one row
  per rule. It takes its SMILES as a **positional** table-function argument
  (`Arg(0, arrow_type=pa.string())`).

## Sharp edges (learned the hard way)

1. **RDKit is imported once, at `chem.py` module load.** The import is slow; doing
   it at module scope caches it for the whole process lifetime. Don't move RDKit
   imports inside functions.
2. **Nothing may write to stdout.** The stdio Arrow protocol owns stdout. RDKit's
   C++ logger writes to **stderr** (harmless), and we additionally
   `RDLogger.DisableLog("rdApp.*")`. Never `print()` from worker code.
3. **Every RDKit call is wrapped; bad input → NULL/false, never an exception.**
   `parse_mol` / `parse_smarts` are the single choke points and return `None` on
   unparseable input; every public function in `chem.py` guards its RDKit calls in
   `try/except` and returns `None`/`False`. A malformed SMILES or SMARTS can never
   crash the worker or abort a query.
4. **NULL vs invalid — two outcomes.** NULL input → NULL output everywhere
   (the adapters pass `None` straight through). Invalid (non-NULL) SMILES → `false`
   for `is_valid_smiles`, `NULL` for every value function, **no rows** for
   `lipinski`. `substructure_match` returns `NULL` for an invalid SMILES **or** an
   invalid SMARTS (documented — a bad pattern is reported, not raised). There are
   **no** error-raising functions here.
5. **LIST/explicit return types.** Scalars returning non-default types set them via
   the `Returns()` annotation on the typed Arrow return (`pa.DoubleArray`,
   `pa.Int32Array`, `pa.StringArray`, `pa.BooleanArray`). Multi-column tables use a
   `FIXED_SCHEMA` built from `field(...)`.
6. **The unit suite can pass while the RPC path is broken.** `test_chem.py` calls
   pure functions directly; only `test_scalars.py` (real `vgi.client.Client`
   subprocess) and `test/sql/*.test` (real `ATTACH`+`SELECT`) exercise the wire.
   **Run the SQL suite** — it's authoritative.

## Determinism in tests

Numeric descriptors are asserted with `ROUND(...)` / tolerances (e.g.
`ROUND(mol_weight, 2) = 180.16`); string identities are asserted exactly
(`inchikey('CC(=O)OC1=CC=CC=C1C(=O)O') = 'BSYNRYMUTXBXSQ-UHFFFAOYSA-N'`).
`tanimoto(x, x)` is exactly `1.0`. RDKit is fully offline (no network, no model
downloads), so the suite is hermetic — if `make test-sql` flakes once, re-run; only
a consistent failure is real.

## RDKit is BSD-3-Clause (licensing note)

RDKit is **BSD-3-Clause** (permissive, commercial-use-friendly), used as an
**unmodified, separately pip-installed dependency** — imported, never vendored or
patched. `vgi-chem`'s own code is MIT. No copyleft caveats.

## Testing

```sh
uv run pytest -q              # unit: pure logic + in-proc tables + Client RPC scalars
make test-sql                 # E2E: haybarn-unittest over test/sql/*  (authoritative)
make test                     # both
uv run ruff check . && uv run mypy vgi_chem/
```

`make test-sql` sets `VGI_CHEM_WORKER="uv run --python 3.13 chem_worker.py"`, puts
`~/.local/bin` on PATH, and runs `haybarn-unittest --test-dir . "test/sql/*"`.
Install the runner once with `uv tool install haybarn-unittest`. Each `.test` uses
an explicit `statement ok` / `LOAD vgi;` (haybarn **silently skips** `require vgi`)
then `ATTACH 'chem' ... (TYPE vgi, LOCATION '${VGI_CHEM_WORKER}')` and
catalog-qualifies every call (`chem.<fn>`). CI (`.github/workflows/ci.yml`) runs
unit + lint + a gated `e2e` job that installs haybarn-unittest and runs
`make test-sql`.
```
