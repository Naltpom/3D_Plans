"""Fait tourner la chaîne d'extraction du salon (étapes 03a → 04) sur la sortie d'une méthode de reconstruction,
dans un dossier à part, sans toucher aux résultats du salon.

    python salon_chain.py --salon-work C:/.../salon/work --run RUN_DIR

RUN_DIR doit contenir, dans le repère monde de la méthode A (voir align.py) :
    colmap/sparse/0/      reconstruction pycolmap (caméras + points épars éventuels)
    colmap/dense/fused.ply nuage avec normales (dense, ou épars + normales estimées)
Les scripts du salon sont copiés dans RUN_DIR/scripts et ne lisent/écrivent que RUN_DIR.
Sortie : RUN_DIR/chain.json (temps par étape, étape en échec éventuelle, valeurs en unités u).
"""
import argparse, json, os, shutil, subprocess, sys, time

STEPS = ['03a_frame', '03e_bend_correction', '03b_measure', '03c_measure2', '03d_scale', '03f_export_cm', '04_model']
PATCH_FROM = "W = os.path.join(ROOT, 'work')"
PATCH_TO = "W = os.environ['BENCH_W']"
SP_FROM = "SP = world_to_cm(Fr, np.array([p.xyz for p in _sp])[_ok])"
SP_TO = SP_FROM + " if _ok.any() else np.zeros((0, 3))"


def prepare(salon_work, run):
    sd = os.path.join(run, 'scripts'); os.makedirs(sd, exist_ok=True)
    for name in STEPS + ['frame_io']:
        src = open(os.path.join(salon_work, 'scripts', name + '.py'), encoding='utf-8').read()
        assert PATCH_FROM in src, f'{name}.py : ligne du dossier de travail introuvable'
        src = src.replace(PATCH_FROM, PATCH_TO).replace(SP_FROM, SP_TO)   # 04 : tolère l'absence de points épars
        open(os.path.join(sd, name + '.py'), 'w', encoding='utf-8').write(src)
    os.makedirs(os.path.join(run, 'viz'), exist_ok=True)
    os.makedirs(os.path.join(run, 'model'), exist_ok=True)
    # repère provisoire de A (le nuage de la méthode est déjà exprimé dans le repère monde de A)
    shutil.copy(os.path.join(salon_work, 'prov_frame.npz'), run)
    shutil.copy(os.path.join(salon_work, 'model', 'mesures_utilisateur.json'), os.path.join(run, 'model'))


def extract(run):
    """Longueurs reconstruites en unités u (avant toute réinjection des mesures)."""
    S = json.load(open(os.path.join(run, 'scale.json')))
    s = S['scale_cm_per_u']
    vals = {u['code']: u['m_u'] for u in S['used']}
    room = json.load(open(os.path.join(run, 'model', 'room.json'), encoding='utf-8'))
    ctrl = next(v for v in room.values() if isinstance(v, list) and v and isinstance(v[0], dict) and 'reconstruit' in v[0])
    for c in ctrl:
        vals['room:' + c['cote']] = c['reconstruit'] / s
    return {'scale_chain_cm_per_u': s, 'values_u': vals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--salon-work', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--python', default=sys.executable)
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    prepare(a.salon_work, run)
    env = dict(os.environ, BENCH_W=run, PYTHONIOENCODING='utf-8')
    res = {'steps': [], 'failed': None}
    for st in STEPS:
        t0 = time.time()
        p = subprocess.run([a.python, os.path.join(run, 'scripts', st + '.py')], env=env, cwd=os.path.join(run, 'scripts'),
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        open(os.path.join(run, f'log_{st}.txt'), 'w', encoding='utf-8').write(p.stdout + '\n--- stderr ---\n' + p.stderr)
        res['steps'].append({'step': st, 'seconds': round(time.time() - t0, 2), 'ok': p.returncode == 0})
        if p.returncode != 0:
            res['failed'] = {'step': st, 'error': p.stderr.strip().splitlines()[-1] if p.stderr.strip() else 'code %d' % p.returncode}
            break
    if not res['failed']:
        res.update(extract(run))
    json.dump(res, open(os.path.join(run, 'chain.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != 'values_u'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
