# 3D_Plans

Application Windows : sources hétérogènes (cotes, photos, cadastre, relevé d'altitudes) → modèle 3D paramétrique à l'échelle, chaque cote avec son origine et sa marge σ.

- `app/` : coquille Tauri 2, interface TypeScript + three.js.
- `engine/plans3d/` : moteur Python (solveur, projet, composants, photos, cadastre, terrain), serveur JSON-lines sur stdin/stdout.
- `tests/` : tests du moteur.

## Fonctions

- Parcours guidé (par défaut) : projet → sources (photos, plan cadastral, croquis à main levée, construction manuelle, combinables) → positions des photos (nuage dense coloré s'il existe, sinon épars, et caméras) → murs proposés à cocher, repris du cadastre ou tracés → une mesure au mètre pour l'échelle → vérification (précision, contradictions, mesure utile) → meubler. Un panneau par étape, état enregistré dans `project.json` ; « Mode expert » affiche l'interface complète. L'atelier photo s'ouvre en mode guidé seulement pour cliquer un angle désigné sur 2 photos.
- Sources : cotes, photos (COLMAP), cadastre, relevé d'altitudes ; contradictions signalées.
- Tracé en vue de dessus : clic angle par angle sur une grille de 5 cm, angle droit, angles et faces des murs existants aimantés (20 cm), longueur (cm) et angle (°) au clavier, fermeture de la pièce ; murs reliés par leurs angles, positions a priori à ± 5 cm, longueurs tapées enregistrées comme cotes. Portes et fenêtres posées d'un clic sur un mur. Croquis à main levée affiché en fond, calé par 2 points et leur distance, opacité réglable et enregistrée.
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
- Fusion photos + tracé : les murs sûrs des photos (6 au plus) sont appariés aux murs tracés, en biais ou du cadastre (orientation quelconque : quart de tour du repère + lacet, échelle, translation ; écart ≤ 15 cm et 5°, recouvrement des emprises), puis associés au trait du mur ; 2 murs non parallèles calent la position, 2 murs parallèles en plus l'échelle. Sens choisi par le score pondéré par la part de caméras dans la pièce tracée ; à 2 % près (pièce symétrique), le mur le plus long des photos va sur le premier mur tracé ; boutons « Tourner » en secours. L'étape Murs affiche 6 pans au plus par défaut.
- Murs proposés : plans verticaux regroupés (écart ≤ 10 cm), d'au moins 0,8 m de haut et atteignant 1,3 m ; incertains si peu de points, courts ou à moins d'1 m devant un autre mur. Cadastre : murs extérieurs du bâtiment principal sur ses sommets (épaisseur vers l'intérieur, hauteur BD TOPO si connue) et parcelle principale. À la validation : murs perpendiculaires reliés par leurs angles (≤ 75 cm), premier mur de chaque direction à 0, sol et plafond associés ; une nouvelle validation remplace les murs proposés et la cote d'échelle.
- Outils photo : COLMAP, téléchargé au premier usage dans `%LOCALAPPDATA%\3D_Plans\tools`.
- Aucune photo ni donnée de projet dans le dépôt.
- Usage personnel.
