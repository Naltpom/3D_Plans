# 3D_Plans

Application Windows : sources hétérogènes (cotes, photos, cadastre, relevé d'altitudes) → modèle 3D paramétrique à l'échelle, chaque cote avec son origine et sa marge σ.

- `app/` : coquille Tauri 2, interface TypeScript + three.js.
- `engine/plans3d/` : moteur Python (solveur, projet, composants, photos, cadastre, terrain), serveur JSON-lines sur stdin/stdout.
- `tests/` : tests du moteur.

## Fonctions

- Parcours guidé (par défaut) : projet → sources (photos, plan cadastral, croquis à main levée, construction manuelle, combinables) → positions des photos (nuage coloré, dense s'il existe, sinon épars, et caméras) → murs proposés à cocher, repris du cadastre ou tracés → une mesure au mètre pour l'échelle → vérification (précision, contradictions, mesure utile) → meubler. Un panneau par étape, état enregistré dans `project.json` ; « Mode expert » affiche l'interface complète. L'atelier photo s'ouvre en mode guidé seulement pour cliquer un angle désigné sur 2 photos.
- Accueil : au lancement sans projet ouvert, puis à tout moment par « Projets » ou « 3D_Plans » dans l'en-tête, en mode guidé comme en mode expert ; 12 projets récents (nom, chemin, type, étape, dernière ouverture), « Nouveau projet », « Ouvrir un dossier… ». Le projet courant reste ouvert tel quel (« Revenir à … », son nom dans l'en-tête ou Échap) ; un autre projet reprend à son étape enregistrée ; changement refusé pendant un calcul de photos ; × retire de la liste sans rien supprimer.
- Sources : cotes, photos (COLMAP), cadastre, relevé d'altitudes ; contradictions signalées.
- Positions des photos : calcul lancé seul en passant à l'étape (depuis Sources) si des photos sont là et que leurs positions manquent ou sont périmées (photos ajoutées), jamais relancé si elles sont à jour ; suivi en direct dans la vue : vignette de chaque photo dès que ses détails sont repérés, nombre de paires de photos à comparer, puis nuage coloré et caméras repris des instantanés de COLMAP (toutes les 2 s au plus, 50 000 points au plus, repère provisoire : vertical des photos, caméras à 1,4 m), photos placées cerclées en bleu, les 3 dernières en jaune ; vue recadrée à chaque instantané tant qu'on ne la touche pas, vue finale à la fin. « Arrêter le calcul » coupe COLMAP (positions ou densification) sans rien garder.
- Parcours par type de projet : pièce (sources, photos, murs, échelle, vérifier) ; maison et terrain (cadastre → murs extérieurs et parcelle, relief optionnel avec maison posée au point bas de l'emprise, toit à deux pans sur l'emprise à 35° avec hauteur BD TOPO ou à mesurer, façades photo recalées, débordements de parcelle) ; véhicule (gabarit de caisse depuis la fiche constructeur : parois, plancher, passages de roue, porte latérale, portes arrière, cotes « lu » ; puis photos recalées et cotes au mètre ; intérieur aménageable).
- Étape Murs avec un plan dessiné (pièce) : 4.1 lecture du dessin (couleurs détectées : noir murs, gris escalier, rouge ou orange portes, bleu portes-fenêtres, vert ou jaune fenêtres ; légende réglable, lue par l'IA vision si elle est installée ; contour redressé, angles à moins de 20° de l'équerre mis à 90°, traits qui dépassent ou intérieurs ignorés et signalés ; fusionner, couper, supprimer, renommer) → 4.2 calage sur les photos : chaque mur droit du dessin apparié à une surface verticale des photos (quart de tour, échelle de 0,6 à 1,6, décalage par vote sur les positions, caméras dans la pièce, échelle proche de l'estimation ; puis affectation globale qui respecte l'ordre des murs sur chaque axe, écart ≤ 60 cm et emprises qui se recouvrent ; un mur sans surface photo garde la forme du dessin à ± 30 cm), écart par mur → 4.3 coins à préciser, un à la fois (coin estimé = coin dessiné ramené sur les murs appariés en photo à 15 cm près) : la photo occupe la zone principale ; au-dessus, ce que l'on désigne (coin bas au sol, coin haut au plafond, arête n'importe où sur la verticale) ; dessous, photo précédente / suivante parmi celles qui voient le coin (arête entière dans l'image, à peu près verticale, sans points du nuage devant, tube de 12 cm, les plus centrées d'abord), « Choisir une autre photo » (galerie de toutes les photos positionnées dans la zone principale, vignettes au format de la photo, badge coin visible / caché / hors cadre / refusée / cliquée, filtres « voit le mur X ») et « Ce coin n'est pas visible ici » (photo plus jamais reproposée pour ce coin) ; loupe au survol, réticule et étiquette au clic ; plan du coin en médaillon flottant en haut à droite de la photo, redimensionnable par sa poignée en bas à gauche (coin opposé à l'ancrage), taille mémorisée ; à droite la consigne, l'avancement, les 2 clics avec leur vignette et Passer / Point suivant. 1 clic par photo sur 2 photos : arête = plan vertical du rayon, 2 arêtes donnent l'intersection vue de dessus (refus sous 10° d'écart), bas = rayon ∩ sol, haut = rayon ∩ plafond (hauteur du calage, sinon refus) ; point validé si les clics se recoupent à 12 cm près ; avertissement sans blocage à plus de 10 cm d'un mur apparié en photo ou à plus d'1 m du coin dessiné ; dans le solveur, clic bas ou haut = point 3D (2 observations en pixels), arête = droite verticale (1 observation) → 4.4 portes et fenêtres (type, largeur, suppression ; origine et marge de chaque largeur affichées : dessin ± 30 cm, photos (IA) ± 10 cm ; « même taille que les autres » : les ouvertures d'un même type partagent largeur et hauteur, estimées par le solveur avec toutes leurs mesures, coché par défaut si la largeur est à moins de 25 % de la médiane du type ; vérification par l'IA vision sur une photo qui voit l'ouverture : présence, type et bords ; ouverture vue ± 10 cm, sinon d'après le dessin ± 30 cm) → 4.5 récapitulatif et validation. Escalier généré à partir des points denses : marches = paliers de points horizontaux (normale verticale) à pas régulier, chaque giron = enveloppe de ses points à sa hauteur, contremarche sur le bord tourné vers la marche précédente ; gabarit droit en secours s'il y a moins de 4 marches mesurées. Murs du dessin qui en touchent un autre (bout contre face, à 4 % de la taille du dessin près) : extrémité recollée et contrainte sur la face du mur voisin à ± 1 cm. Rien n'est appliqué au modèle avant 4.5 ; de 4.2 à 4.5 le nuage est affiché dans le repère du calage ; murs nommés A, B, C… dans l'ordre du dessin (sens horaire depuis le mur du haut), mêmes noms partout. Sans dessin : murs des photos ou tracé.
- Annuler / Rétablir : Ctrl+Z, Ctrl+Y ou boutons ; instantané du projet (`project.db`, `model.json`, `project.json`, cotes) avant chaque action qui modifie, 50 derniers.
- Suppression : mur, porte, fenêtre, porte-fenêtre, escalier, objet sélectionné dans la vue, touche Suppr ou bouton ; les ouvertures du mur partent avec lui, ses cotes et clics aussi.
- Vue 3D au clavier : ZQSD ou WASD pour avancer, reculer, se décaler, Espace pour monter, C pour descendre, Maj pour aller 3 fois plus vite (2 m/s sinon) ; souris pour regarder ; inactif dans les champs de saisie.
- Étape Murs : cocher ou décocher un mur l'affiche ou l'estompe aussitôt ; survoler une ligne surligne le mur. Ouvertures proposées : trous rectangulaires (maille de 10 cm) dans les points à moins de 6 cm de la face des murs ; porte si bas au sol, haute de 1,6 m ou plus et large de 1,2 m au plus, porte-fenêtre si plus large, fenêtre si bas au-dessus de 0,5 m ; largeur et hauteur à ± 5 cm.
- Tracé approximatif (« Je ne suis pas sûr de mes murs ») : murs a priori à ± 30 cm, chacun calé sur la surface photo parallèle la plus proche (≤ 60 cm, ≤ 10°), déplacement affiché ; le nuage vu de dessus sert de fond pour décalquer.
- Tracé en vue de dessus : clic angle par angle sur une grille de 5 cm, angle droit, angles et faces des murs existants aimantés (20 cm), longueur (cm) et angle (°) au clavier, fermeture de la pièce ; murs reliés par leurs angles, positions a priori à ± 5 cm, longueurs tapées enregistrées comme cotes. Portes et fenêtres posées d'un clic sur un mur. Croquis à main levée affiché en fond, calé par 2 points et leur distance, opacité réglable et enregistrée.
- Composants : murs, sols, plafonds, portes, fenêtres, portes-fenêtres, escalier, boîte, zone extérieure, toit ; arc d'ouverture des portes affiché selon le côté et les charnières.
- Terrain : maillage du relevé d'altitudes, parcelle cadastrale.
- Galerie : gabarits intégrés et catalogue IKEA importé en local (`%LOCALAPPDATA%\3D_Plans\catalog`), recherche par nom et type.
- Placement : ajout au centre de la vue, glisser au sol aimanté aux murs, angle libre (champ ou ±15°), hauteur ; alertes de collision, de passage étroit et d'ouverture gênée.
- Agencement automatique : règles (contre un mur, face à, près de, loin de, alignés, au centre, orienté vers, angle imposé), propositions notées, aperçu sans déplacement possible, application en un clic. L'optimiseur tourne par pas de 90° sauf angle imposé.
- Assistant d'agencement : description en français → règles proposées, validées par le moteur (meubles et éléments existants, distances bornées par la pièce), cochées puis ajoutées par l'utilisateur. IA locale par défaut : Ollama + `qwen2.5:7b-instruct`, installables depuis l'appli (winget, ~4,7 Go). Protocole OpenAI (`/v1/chat/completions`) : serveur, modèle et clé réglables, enregistrés dans `%LOCALAPPDATA%\3D_Plans\settings.json`.
- Lecture du croquis par IA locale : modèle vision `qwen2.5vl:7b` (Ollama, téléchargé à la demande) interrogé en 3 questions (coins des murs, cotes écrites, portes et fenêtres), coordonnées converties en mètres par le calage ; cote gardée si elle colle à ± 20 % au mur dessiné le plus proche, sinon signalée ; murs cochés créés avec les cotes « lu » (σ 2 cm), positions dessinées a priori à ± 30 cm.
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
- Murs proposés : plans verticaux regroupés (écart ≤ 10 cm), d'au moins 0,8 m de haut et atteignant 1,3 m ; incertains si peu de points, courts ou à moins d'1 m devant un autre mur. Cadastre : murs extérieurs du bâtiment principal sur ses sommets (épaisseur vers l'intérieur, hauteur BD TOPO si connue) et parcelle principale. À la validation : murs perpendiculaires reliés par leurs angles (≤ 75 cm), premier mur de chaque direction à 0, sol et plafond associés ; toute validation (murs des photos ou plan dessiné) remplace les murs de l'étape 4, leurs ouvertures et la cote d'échelle, jamais de double. Plan dessiné : murs a priori à ± 30 cm, plans photo appariés par le calage, clics de coins au sol, échelle des photos libre jusqu'à la mesure.
- Mises à jour : vérification silencieuse au lancement sur `releases/latest/download/latest.json` (rien en l'absence de réseau), si une version plus récente existe, fenêtre centrée sur fond assombri (version, notes, « Mettre à jour maintenant » : téléchargement avec progression, installation et relance ; « Plus tard », Échap ou croix pour fermer), reproposée à chaque lancement tant que la mise à jour n'est pas faite ; paquets signés (clé publique dans `tauri.conf.json`, clé privée hors dépôt, `TAURI_SIGNING_PRIVATE_KEY` au build) ; `PLANS3D_UPDATE_URL` remplace l'adresse pour les essais (en http : `npx tauri dev --config src-tauri/tauri.dev.conf.json`). Version dans la barre d'état et « À propos ».
- Outils photo : COLMAP, téléchargé au premier usage dans `%LOCALAPPDATA%\3D_Plans\tools`.
- Aucune photo ni donnée de projet dans le dépôt.
- Usage personnel.
