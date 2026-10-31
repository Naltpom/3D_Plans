"""Erreurs cote par cote de chaque méthode (échelle refaite sans la cote testée).

    python evaluate.py RUNS_DIR A Ap D B C E ...
"""
import json, os, sys
import numpy as np

# (libellé, clé ou fragment de clé dans chain.json, valeur au mètre (cm), σ (cm), catégorie)
COTES = [
    ('531 mur → mur PF', '531', 531.0, 5 ** .5, 'geo'),
    ('304 escalier → mur PF', '304', 304.0, 1.0, 'geo'),
    ('209 mur d’angle → cloison', '209', 209.0, 2 ** .5, 'geo'),
    ('194,5 mur → toilettes', 'room:mur sous l’escalier → mur des toilettes', 194.5, 3 ** .5, 'geo'),
    ('280 fond du placard', 'room:fond du placard', 280.0, 2.0, 'geo'),
    ('137 sol → bas du poteau', '137', 137.0, 1.0, 'geo'),
    ('108,5 toilettes → limon', 'room:mur des toilettes → escalier', 108.5, 1.0, 'geo'),
    ('85,5 mur → limon', '85.5', 85.5, 1.0, 'geo'),
    ('68 mur → PF1', 'room:mur sous l’escalier → porte-fenêtre 1', 68.0, 1.0, 'int'),
    ('41,5 contremarche → porte', 'room:m3', 41.5, 1.0, 'int'),
    ('15 porte → mur d’angle', 'room:encadrement porte', 15.0, 1.0, 'int'),
]


def find(vals, key):
    if key in vals: return vals[key]
    k = [v for n, v in vals.items() if n.startswith(key)]
    return k[0] if len(k) == 1 else None


def evaluate(run):
    ch = json.load(open(os.path.join(run, 'chain.json'), encoding='utf-8'))
    if ch.get('failed'): return {'failed': ch['failed']}
    m = {c[0]: find(ch['values_u'], c[1]) for c in COTES}
    geo = [c for c in COTES if c[4] == 'geo' and m[c[0]] is not None]
    scale = lambda cs: sum(m[c[0]] * c[2] / c[3] ** 2 for c in cs) / sum(m[c[0]] ** 2 / c[3] ** 2 for c in cs)
    err = {}
    for c in COTES:
        if m[c[0]] is None: continue
        ref = [g for g in geo if g is not c]
        err[c[0]] = scale(ref) * m[c[0]] - c[2]
    single = [abs(m[i[0]] * k[2] / m[k[0]] - i[2]) for k in geo for i in geo if i is not k]
    g = np.array([abs(err[c[0]]) for c in geo])
    out = {'err_cm': err, 'geo_mean': float(g.mean()), 'geo_max': float(g.max()), 'single_ref_mean': float(np.mean(single)),
           'scale_cm_per_u': scale(geo)}
    al = os.path.join(run, 'align.json')
    if os.path.exists(al):   # échelle propre de la méthode (MapAnything : mètres) ramenée en cm par u de A
        out['own_scale_err_pct'] = 100 * (100 / json.load(open(al))['sim3_scale'] / out['scale_cm_per_u'] - 1)
    return out


def main():
    runs, names = sys.argv[1], sys.argv[2:]
    R = {n: evaluate(os.path.join(runs, n)) for n in names}
    json.dump(R, open(os.path.join(runs, 'results.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    ok = [n for n in names if 'failed' not in R[n]]
    print('| cote | ' + ' | '.join(names) + ' |\n|---|' + '---|' * len(names))
    for c in COTES:
        print(f'| {c[0]} | ' + ' | '.join(f"{R[n]['err_cm'][c[0]]:+.1f}" if n in ok and c[0] in R[n]['err_cm'] else '–' for n in names) + ' |')
    for k, lab in [('geo_mean', 'moy. |géo|'), ('geo_max', 'max |géo|'), ('single_ref_mean', 'moy. 1 seule cote')]:
        print(f'| **{lab}** | ' + ' | '.join(f'{R[n][k]:.1f}' if n in ok else '–' for n in names) + ' |')
    for n in names:
        if n not in ok: print(f"{n} : échec à {R[n]['failed']['step']} — {R[n]['failed']['error']}")


if __name__ == '__main__':
    main()
