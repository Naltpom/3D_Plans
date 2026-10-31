# Phase 0 — résultats (31/10/2026)

130 photos (téléphone, 12 Mpx), 8 cotes de géométrie (plan à plan) + 3 d'interprétation (bords d'ouvertures), échelle refaite sans la cote testée. RTX 5080.

| méthode | temps | caméras ≤ 5 cm de A | géométrie moy. / max | chaîne |
|---|---|---|---|---|
| A : COLMAP incrémental + dense 2000 px | ~10 + 72 min | réf. | 1,3 / 3,6 cm | ok |
| A1000 : idem, dense 1000 px | ~10 + 25 min | réf. | 2,3 / 8,9 cm | ok, 1 bord d'ouverture faux (+67 cm) |
| A' : COLMAP sans dense | ~10 min | réf. | – | échec (points épars insuffisants) |
| D : `global_mapper` sans dense | 4,8 min | 90/130, 3,8 px | – | échec |
| B : VGGT (130 vues, 14,8 Go) | ~2 min | 27/130 (méd. 15 cm) | – | échec |
| C : MapAnything + intrinsèques | 9 min | 12/130 (méd. 57 cm) | – | échec |
| Ep : MapAnything reposé sur poses COLMAP | 10 + 8 min | 121/130 | – | échec : mur épais de ±15 cm entre vues |

Temps COLMAP (130 photos) : points clés 28 s, appariement exhaustif 195 s, global 66 s, incrémental 363 s (1,17 px).

## Conclusions
- Les modèles feed-forward (VGGT, MapAnything, 518 px) ne tiennent pas le centimètre : poses à 15–57 cm, profondeurs incohérentes de ±15 cm entre vues.
- Les poses doivent venir de COLMAP incrémental (5–10 min). Appariement séquentiel à tester pour réduire les 195 s.
- Le nuage dense reste le poste lent (25–72 min) ; la chaîne d'extraction actuelle dépend de sa densité.
- L'erreur d'interprétation domine quand elle survient (−7,7 et +67 cm contre 1–4 cm de géométrie).

Décision (critères du brief) : photos = forme, ouvertures, poses ; cotes au mètre ou sur plans ; prochaine meilleure mesure centrale ; correction sur photo et aide IA à l'interprétation avant toute optimisation de la reconstruction. Dense en tâche de fond.

Piste non testée : profondeur par vue recalée vue par vue sur les points épars COLMAP (échelle par vue).

# Phase 1 — moteur sans interface (31/10/2026)

`engine/plans3d` : observations = expressions des paramètres ± σ ; moindres carrés robustes (soft_l1) ; covariance → marges ; résidus normalisés (seuil 3) regroupés en conflits ; facteurs σ réel/σ déclaré par groupe ; prochaine meilleure mesure ; projet SQLite avec cache par empreinte ; export `model.json`. Tests : `tests/test_engine.py` (15).

Salon (`adapters/salon.py`, 23 cotes au mètre + 17 mesures photo de A, 26 paramètres dont l'échelle) :
- mise à jour 88–103 ms ; sans changement : rien de ré-extrait.
- validation croisée (14 cotes) : toutes à ≤ 1,6 cm ; RMS z = 0,31 (σ déclarés, prudents), 0,62 (σ recalés).
- précision estimée : mètre ~0,35 cm, photo plans ~0,8 cm, photo bords ~4 cm.
- contradiction injectée (+10 cm sur une cote) : détectée, conflit nommé avec les 2 mesures photo opposées ; le reste du modèle bouge ≤ 0,7 cm.
- limite : la cote fausse (σ 1 cm) l'emporte sur les photos (σ 1,5 cm) — le moteur signale, l'utilisateur tranche.
