# Brief (31/10/2026)

## Décisions
- Usage personnel, licences non commerciales acceptées. IA 100 % locale (Ollama / llama.cpp, protocole OpenAI).
- V1 : intérieur. Modèle de données prévu pour extérieur, terrain, niveaux, Lambert-93.
- Matériel cible : RTX 5080 16 Go ; extracteurs interchangeables derrière une interface commune.
- Desktop : Tauri + TypeScript + three.js. Moteur : Python (scipy, PyTorch, COLMAP). Pas de Rust hors coquille.
- Distribution : un `.exe` d'installation (Program Files) ; PyTorch et modèles téléchargés au premier lancement.
- Projet = dossier : `sources/`, `cache/`, `project.db` (SQLite), modèle exporté en JSON.
- Observations avec σ en unités physiques ; moindres carrés pondérés + perte robuste ; covariance → marges et « prochaine meilleure mesure » ; contradiction si résidu > 3σ ; l'IA propose, le solveur tranche, l'utilisateur valide.
- Cache par source (empreinte) ; solveur recalculé en entier à chaque changement.
- Ordre : banc d'essai → moteur sans UI → appli → IA → extérieur. Chaque phase attend un feu vert.

## Écarts constatés entre le brief et les fichiers
- Photos : 152 originales, 140 après tri, 130 positionnées par A → entrée commune = ces 130.
- Extrémités de `05_measures.py` : points sur des murs unis, non cliquables dans plusieurs photos. Erreur de géométrie mesurée sur distances plan-à-plan de grandes surfaces ; erreur d'interprétation sur les bords d'ouvertures.
- Cotes sommées (531 = 5 tronçons, 194,5 = 3, 280 = 4) : σ = 1 cm × √n.
- Le salon est en cm ; le moteur sera en m (conversion à l'import de `room.json`).
