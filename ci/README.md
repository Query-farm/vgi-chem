# CI: the vgi-chem worker integration suite

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs the unit tests
and this repo's sqllogictest suite (`test/sql/*.test`) against the vgi-chem
VGI worker through the **real DuckDB `vgi` extension** on every push / PR.

## How it works (no C++ build)

Rather than building the vgi DuckDB extension from source, CI drives a
**prebuilt** standalone `haybarn-unittest` and installs the **signed** `vgi`
extension from the Haybarn community channel:

1. **Install the worker** — `uv sync --frozen` into a venv. `chem_worker.py`
   is a self-contained PEP 723 stdio worker spawned via `uv run chem_worker.py`.
2. **Download the runner** — the matching `haybarn_unittest-*` asset per
   platform from the latest Haybarn release.
3. **Preprocess** — [`preprocess-require.awk`](preprocess-require.awk) rewrites
   each `require <ext>` into an explicit signed `INSTALL ... FROM
   {community,core}; LOAD ...;` and injects `INSTALL vgi FROM community;`
   before each bare `LOAD vgi;` (haybarn silently SKIPs `require vgi`).
4. **Run** — [`run-integration.sh`](run-integration.sh) stages the preprocessed
   tree, points `VGI_CHEM_WORKER` at `uv run chem_worker.py`, warms the
   extension cache once, then runs the suite in a single `haybarn-unittest`
   invocation.

## Run it locally

```bash
uv sync --python 3.13
HAYBARN_UNITTEST=/path/to/haybarn-unittest \
VGI_CHEM_WORKER="uv run --python 3.13 chem_worker.py" \
  ci/run-integration.sh
```
