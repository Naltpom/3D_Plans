# 3D_Plans

Application Windows : sources hétérogènes (cotes, photos, cadastre, relevé d'altitudes) → modèle 3D paramétrique à l'échelle, chaque cote avec son origine et sa marge σ.

- `app/` : coquille Tauri 2, interface TypeScript + three.js.
- `engine/plans3d/` : moteur Python (solveur, projet, composants, photos, cadastre, terrain), serveur JSON-lines sur stdin/stdout.
- `tests/` : tests du moteur.

## Installeur

Prérequis : Node, Rust, Python 3.12 dans le PATH.

```
cd app
npm install
npx tauri build
```

`scripts/prepare_runtime.py` télécharge un Python autonome, installe `engine/requirements.txt` et copie le moteur dans `src-tauri/runtime/`. Installeur NSIS produit dans `app/src-tauri/target/release/bundle/nsis/`.

## Développement

```
cd app
npx tauri dev
```

## Tests

```
$env:PYTHONPATH = "engine"; .venv\Scripts\python -m pytest -q tests
cd app; npx tsc --noEmit
```

## Règles

- Projet = dossier : `sources/`, `cache/`, `project.db`, `model.json`.
- Toute observation porte une valeur et un σ en unités physiques ; contradiction si résidu normalisé > 3.
- L'IA propose, le solveur tranche, l'utilisateur valide.
- Cache par source : seules les sources modifiées sont ré-extraites ; le solveur est relancé en entier à chaque mise à jour.
- Outils photo : COLMAP, téléchargé au premier usage dans `%LOCALAPPDATA%\3D_Plans\tools`.
- Aucune photo ni donnée de projet dans le dépôt.
- Usage personnel.
