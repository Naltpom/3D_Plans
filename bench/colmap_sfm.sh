#!/usr/bin/env bash
# Calcul de base COLMAP (sans nuage dense), chronométré étape par étape.
#   bash colmap_sfm.sh COLMAP_EXE IMAGES_DIR IMAGE_LIST OUT_DIR
# Produit OUT_DIR/{database.db, incremental/0, global/0, timings.json}
# Les deux assemblages (incrémental = méthode A sans nuage dense, global = méthode D / GLOMAP)
# partent des mêmes points clés et du même appariement.
set -euo pipefail
C="$1"; IMG="$2"; LIST="$3"; OUT="$4"
mkdir -p "$OUT/incremental" "$OUT/global"
rm -f "$OUT/database.db"
# focale initiale : 23 mm éq. 24x36 rapportée à la diagonale (43,27 mm) d'une image 4000x3000 (5000 px)
F=$(python -c "print(round(23*5000/43.267))")
t() { date +%s.%N; }
T0=$(t)
"$C" feature_extractor --database_path "$OUT/database.db" --image_path "$IMG" --image_list_path "$LIST" \
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
  --ImageReader.camera_params "$F,$F,2000,1500,0,0,0,0" \
  --FeatureExtraction.use_gpu 1 --SiftExtraction.max_num_features 16384 > "$OUT/features.log" 2>&1
T1=$(t)
"$C" exhaustive_matcher --database_path "$OUT/database.db" \
  --FeatureMatching.use_gpu 1 --FeatureMatching.guided_matching 1 > "$OUT/match.log" 2>&1
T2=$(t)
"$C" global_mapper --database_path "$OUT/database.db" --image_path "$IMG" --output_path "$OUT/global" \
  --GlobalMapper.random_seed 0 > "$OUT/global.log" 2>&1
T3=$(t)
"$C" mapper --database_path "$OUT/database.db" --image_path "$IMG" --output_path "$OUT/incremental" > "$OUT/incremental.log" 2>&1
T4=$(t)
python - "$OUT" "$T0" "$T1" "$T2" "$T3" "$T4" <<'EOF'
import json, sys
out, t0, t1, t2, t3, t4 = sys.argv[1], *map(float, sys.argv[2:])
d = {'features_s': t1 - t0, 'matching_s': t2 - t1, 'global_mapper_s': t3 - t2, 'incremental_mapper_s': t4 - t3}
json.dump(d, open(f'{out}/timings.json', 'w'), indent=1)
print({k: round(v, 1) for k, v in d.items()})
EOF
