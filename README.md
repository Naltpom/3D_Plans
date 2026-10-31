# 3D_Plans

Outil desktop (Windows) : sources hétérogènes (photos, cotes, plans, cadastre) → modèle 3D paramétrique à l'échelle, chaque cote avec origine et marge σ. Usage personnel.

État : phase 0 terminée — `docs/phase0.md`. Décisions : `docs/brief.md`.

## Banc d'essai

Compare des méthodes de reconstruction rapides à la méthode actuelle (COLMAP + nuage dense) sur un projet avec cotes au mètre.

```
bench/colmap_sfm.sh   COLMAP sans nuage dense (incrémental + global), chronométré
bench/ff_run.py       VGGT / MapAnything → nuage + normales (.npz)
bench/align.py        sortie d'une méthode → repère de la référence (similitude RANSAC sur les caméras)
bench/salon_chain.py  chaîne d'extraction du projet, isolée dans un dossier de run
bench/evaluate.py     erreurs cote par cote, échelle refaite sans la cote testée
```

Environnement : Python 3.12, `.venv` (torch cu128 + `third_party/vggt`, `third_party/map-anything`), COLMAP 4.2.1 CUDA. Aucune donnée de projet dans le dépôt.
