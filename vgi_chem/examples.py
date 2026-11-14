"""Composed example queries shared by the catalog, schema and per-object metadata.

Each query combines two or more of this worker's own objects (the
``example_molecules`` registry table with a scalar, or two scalars). The SQL is
defined once here and referenced from :mod:`vgi_chem.scalars` (the driving
function's ``Meta.examples``) and :mod:`vgi_chem.worker` (schema-level
``vgi.example_queries`` / the registry table's examples), so the copies can
never drift apart. ``test/sql/composed.test`` pins the rows each one returns.
"""

from __future__ import annotations

# Similarity search: rank the registry against a query molecule (salicylic acid).
SIMILARITY_SEARCH_DESCRIPTION = (
    "Similarity search: the three built-in molecules most similar to salicylic acid "
    "(OC(=O)c1ccccc1O) by Tanimoto -- aspirin 0.448, phenol 0.381, paracetamol 0.226"
)
SIMILARITY_SEARCH_SQL = (
    "SELECT name, round(chem.main.tanimoto(smiles, 'OC(=O)c1ccccc1O'), 3) AS similarity "
    "FROM chem.main.example_molecules ORDER BY similarity DESC, name LIMIT 3"
)

# Substructure filter over the registry: keep carboxylic acids.
ACID_FILTER_DESCRIPTION = (
    "Substructure filter over the registry: molecules carrying a carboxylic-acid group "
    "(SMARTS [CX3](=O)[OX2H1]) -- acetic acid, aspirin, ibuprofen, penicillin g -- lightest first"
)
ACID_FILTER_SQL = (
    "SELECT name, formula, round(mol_weight, 2) AS mol_weight FROM chem.main.example_molecules "
    "WHERE chem.main.substructure_match(smiles, '[CX3](=O)[OX2H1]') ORDER BY mol_weight"
)

# Identify arbitrary SMILES spellings by joining on InChIKey to the registry.
INCHIKEY_LOOKUP_DESCRIPTION = (
    "Identify unfamiliar SMILES spellings by joining on InChIKey to the built-in registry: "
    "C(O)C is ethanol, OC(=O)C acetic acid, Oc1ccccc1 phenol; NCCO (ethanolamine) is not "
    "in the registry, so its name is NULL"
)
INCHIKEY_LOOKUP_SQL = (
    "SELECT q.smiles AS query_smiles, m.name "
    "FROM (VALUES ('OC(=O)C'), ('C(O)C'), ('Oc1ccccc1'), ('NCCO')) AS q(smiles) "
    "LEFT JOIN chem.main.example_molecules AS m "
    "ON chem.main.inchikey(q.smiles) = chem.main.inchikey(m.smiles) "
    "ORDER BY query_smiles"
)

# All-pairs similarity within the registry: its nearest-neighbour pairs.
NEAREST_PAIRS_DESCRIPTION = (
    "All-pairs Tanimoto over the registry: its three most similar pairs -- "
    "paracetamol/phenol 0.292, benzene/phenol 0.273, aspirin/phenol 0.25"
)
NEAREST_PAIRS_SQL = (
    "SELECT a.name AS molecule_a, b.name AS molecule_b, "
    "round(chem.main.tanimoto(a.smiles, b.smiles), 3) AS similarity "
    "FROM chem.main.example_molecules AS a JOIN chem.main.example_molecules AS b ON a.name < b.name "
    "ORDER BY similarity DESC, molecule_a, molecule_b LIMIT 3"
)

COMPOSED_EXAMPLES: list[dict[str, str]] = [
    {"description": SIMILARITY_SEARCH_DESCRIPTION, "sql": SIMILARITY_SEARCH_SQL},
    {"description": ACID_FILTER_DESCRIPTION, "sql": ACID_FILTER_SQL},
    {"description": INCHIKEY_LOOKUP_DESCRIPTION, "sql": INCHIKEY_LOOKUP_SQL},
    {"description": NEAREST_PAIRS_DESCRIPTION, "sql": NEAREST_PAIRS_SQL},
]
