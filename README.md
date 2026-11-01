# 3D_Plans

Application Windows : sources hétérogènes (cotes, photos, cadastre, relevé d'altitudes) → modèle 3D paramétrique à l'échelle, chaque cote avec son origine et sa marge σ.

- `app/` : coquille Tauri 2, interface TypeScript + three.js.
- `engine/plans3d/` : moteur Python (solveur, projet, composants, photos, cadastre, terrain), serveur JSON-lines sur stdin/stdout.
- `tests/` : tests du moteur.

## Fonctions

- Sources : cotes, photos (COLMAP), cadastre, relevé d'altitudes ; contradictions signalées.
- Composants : murs, sols, plafonds, portes, fenêtres, portes-fenêtres, escalier, boîte, zone extérieure, toit ; arc d'ouverture des portes affiché selon le côté et les charnières.
- Terrain : maillage du relevé d'altitudes, parcelle cadastrale.
- Galerie : gabarits intégrés et catalogue IKEA importé en local (`%LOCALAPPDATA%\3D_Plans\catalog`), recherche par nom et type.
- Placement : ajout au centre de la vue, glisser au sol aimanté aux murs, angle libre (champ ou ±15°), hauteur ; alertes de collision, de passage étroit et d'ouverture gênée.
- Agencement automatique : règles (contre un mur, face à, près de, loin de, alignés, au centre, orienté vers, angle imposé), propositions notées, aperçu sans déplacement possible, application en un clic. L'optimiseur tourne par pas de 90° sauf angle imposé.
- Assistant d'agencement : description en français → règles proposées, validées par le moteur (meubles et éléments existants, distances bornées par la pièce), cochées puis ajoutées par l'utilisateur. IA locale par défaut : Ollama + `qwen2.5:7b-instruct`, installables depuis l'appli (winget, ~4,7 Go). Protocole OpenAI (`/v1/chat/completions`) : serveur, modèle et clé réglables, enregistrés dans `%LOCALAPPDATA%\3D_Plans\settings.json`.
- Objets imaginaires : description d'un objet → gabarit existant, dimensions et paramètres validés (bornes du mobilier), aperçu puis ajout par l'utilisateur.
- Lumière réelle : luminaires de la galerie (lumens → candelas, kelvins → teinte, ponctuel ou spot, interrupteur), soleil selon le lieu (origine du cadastre, sinon latitude et longitude du projet), la date et l'heure ; ombres ; nord = −z.
- Rendu réaliste : lancer de rayons progressif (three-gpu-pathtracer, licence MIT) sur la vue courante, arrêt, export PNG ; murs et plafond côté caméra retirés (vue en coupe).

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
- L'IA propose, le solveur tranche, l'utilisateur valide. L'IA ne produit jamais de coordonnées ; aucune règle n'est ajoutée sans validation.
- Cache par source : seules les sources modifiées sont ré-extraites ; le solveur est relancé en entier à chaque mise à jour.
- Outils photo : COLMAP, téléchargé au premier usage dans `%LOCALAPPDATA%\3D_Plans\tools`.
- Aucune photo ni donnée de projet dans le dépôt.
- Usage personnel.
