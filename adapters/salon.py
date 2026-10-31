"""Projet de test « salon » : cotes au mètre (mesures_utilisateur.json) + mesures photo de la méthode A (run du banc).

    python adapters/salon.py --salon-work .../salon/work --run .../work/bench/runs/A --project .../work/projet_moteur
"""
import argparse, csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'engine'))
from plans3d import Obs, Project, extractor, next_best, solve
from plans3d.core import R_MIN, calibrate, variance_factors

# paramètre: (axe, libellé) — repère du salon : X depuis la 1re contremarche, Z depuis le mur sous l'escalier, Y depuis le sol
PARAMS = {
    'G': ('escalier', 'giron'), 'R': ('escalier', 'hauteur de marche'), 's': ('échelle', 'm par unité photo'),
    'w3_x': ('x', 'mur d’angle de l’entrée'), 'door_a': ('x', 'encadrement porte, côté mur d’angle'), 'door_b': ('x', 'encadrement porte, côté escalier'),
    'pot_lo': ('x', 'poteau, face côté volée'), 'pot_hi': ('x', 'poteau, face côté quart tournant'), 'end_x': ('x', 'bout de l’escalier'),
    'w2_x': ('x', 'mur des portes-fenêtres'), 'wcd_x': ('x', 'porte des toilettes, bord'), 'clo_x0': ('x', 'placard, bord côté entrée'),
    'clo_x1': ('x', 'cloison du placard, face côté placard'), 'part_x': ('x', 'cloison du placard, face côté fauteuil'),
    'limon_z': ('z', 'limon côté mur, face'), 'rampe_z': ('z', 'limon de rampe, face ext.'), 'kit_a': ('z', 'porte cuisine, bord 1'),
    'kit_b': ('z', 'porte cuisine, bord 2'), 'wc_z': ('z', 'mur des toilettes'), 'clo_z': ('z', 'fond du placard'),
    'pf1_a': ('z', 'porte-fenêtre 1, bord 1'), 'pf1_b': ('z', 'porte-fenêtre 1, bord 2'), 'pf2_a': ('z', 'porte-fenêtre 2, bord 1'),
    'pf2_b': ('z', 'porte-fenêtre 2, bord 2'), 'w4_z': ('z', 'mur du fond (portes-fenêtres)'), 'pot_y': ('y', 'bas du poteau'),
}
# cote au mètre : (id, expression, clé dans mesures_utilisateur.json ou (clé, indice))
TAPE = [('G', 'G', 'G'), ('R', 'R', 'R'), ('4', 'limon_z', '4'), ('85.5', 'rampe_z', '85.5'), ('137', 'pot_y', '137'),
        ('7', 'pot_hi - pot_lo', '7'), ('m1', 'w2_x - end_x', 'm1_bout_escalier_mur_PF'), ('m3', '-door_b', 'm3_contremarche_encadrement_porte'),
        ('porte_angle', 'door_a - w3_x', 'encadrement_porte_entree_mur_angle'), ('cuisine_a', 'kit_a', 'mur_entree_porte_cuisine'),
        ('cuisine_l', 'kit_b - kit_a', 'cuisine_passage'), ('cuisine_c', 'wc_z - kit_b', 'cuisine_cadre_coin'),
        ('wc_porte', 'wcd_x - w3_x', 'coin_porte_toilettes'), ('placard_x', 'clo_x0 - w3_x', 'coin_placard'),
        ('placard_l', 'clo_x1 - clo_x0', 'placard_largeur'), ('placard_p', 'clo_z - wc_z', 'placard_profondeur'),
        ('cloison', 'part_x - clo_x1', 'cloison_placard_epaisseur'), ('wc_rampe', 'wc_z - rampe_z', 'mur_toilettes_escalier'),
        ('pf_1', 'pf1_a', ('mur_PF', 0)), ('pf_2', 'pf1_b - pf1_a', ('mur_PF', 1)), ('pf_3', 'pf2_a - pf1_b', ('mur_PF', 2)),
        ('pf_4', 'pf2_b - pf2_a', ('mur_PF', 3)), ('pf_5', 'w4_z - pf2_b', ('mur_PF', 4))]
SIG = {'plan': 0.015, 'escalier': 0.005, 'bord': 0.03}   # σ photo (m), fixés avant résolution d'après le banc (A : 1,3 cm moy.)


@extractor('salon_photos')
def salon_photos(path, sid):
    """path = chain.json d'un run du banc ; lit measures_u.json et scale.json voisins (unités u)."""
    d = os.path.dirname(path); M = json.load(open(os.path.join(d, 'measures_u.json')))
    S = json.load(open(os.path.join(d, 'scale.json'))); V = json.load(open(path, encoding='utf-8'))['values_u']
    x0 = S['x0_u']; room = lambda k: next(v for n, v in V.items() if n.startswith('room:' + k))
    end = max(M['corner_post_+X'], key=lambda c: c['n'])['x']; lo, hi = M['poteau_bottom_x_extent']
    rows = [('G', 'G', M['G']['u'], 'escalier'), ('R', 'R', M['R']['u'], 'escalier'),
            ('limon', 'limon_z', M['wall_stringer_face'][0], 'plan'), ('rampe', 'rampe_z', M['rampe_outer'][0], 'plan'),
            ('pot_y', 'pot_y', M['poteau_bottom_y'][0] - S['floor_local_u']['poteau'], 'plan'),
            ('pot_lo', 'pot_lo', lo - x0, 'bord'), ('pot_hi', 'pot_hi', hi - x0, 'bord'), ('end', 'end_x', end - x0, 'plan'),
            ('w2', 'w2_x', M['w2_x'][0] - x0, 'plan'), ('w3', 'w3_x', M['w3_x'][0] - x0, 'plan'),
            ('cloison', 'clo_x1', M['part_x'][0] - x0, 'plan'), ('w4', 'w4_z', M['w4_z'][0], 'plan'),
            ('wc', 'wc_z', room('mur sous l’escalier → mur des toilettes'), 'plan'), ('placard', 'clo_z', room('fond du placard'), 'plan'),
            ('porte_b', '-door_b', room('m3'), 'bord'), ('porte_a', 'door_a - w3_x', room('encadrement porte'), 'bord'),
            ('pf1', 'pf1_a', room('mur sous l’escalier → porte-fenêtre 1'), 'bord')]
    s0 = S['scale_cm_per_u'] / 100
    return [Obs(f'{sid}:{i}', f'({e}) / s', v, SIG[c] / s0, sid, 'photo', c) for i, e, v, c in rows]


def write_tape_csv(salon_work, path, override=None):
    MU = {}
    for sess in json.load(open(os.path.join(salon_work, 'model', 'mesures_utilisateur.json'), encoding='utf-8'))['session']: MU.update(sess['mesures'])
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter=';'); w.writerow(['id', 'expr', 'value', 'sigma', 'unit', 'status', 'label'])
        for i, e, k in TAPE:
            v = MU[k[0]][k[1]] if isinstance(k, tuple) else MU[k]
            w.writerow([i, e, (override or {}).get(i, v), 1.0, 'cm', 'mesuré', k if isinstance(k, str) else k[0]])


def candidates(P, sigma=0.01):
    """Mesures au mètre possibles : chaque position seule (depuis l'origine de son axe) et chaque différence sur un même axe."""
    by = {}
    for n, (_, a, _) in P.items():
        if a in 'xyz': by.setdefault(a, []).append(n)
    out = [(f'{P[n][2]}', n, sigma) for ns in by.values() for n in ns]
    out += [(f'{P[a][2]} → {P[b][2]}', f'{a} - {b}', sigma) for ns in by.values() for i, a in enumerate(ns) for b in ns[i + 1:]]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--salon-work', required=True); ap.add_argument('--run', required=True)
    ap.add_argument('--project', required=True); a = ap.parse_args()
    os.makedirs(os.path.join(a.project, 'sources'), exist_ok=True); tape = os.path.join(a.project, 'sources', 'cotes.csv')
    pr = Project(a.project)
    for n, (ax, lab) in PARAMS.items(): pr.add_param(n, 0.54 if n == 's' else 0.0, ax, lab)
    write_tape_csv(a.salon_work, tape)
    pr.add_source('cotes', tape, 'cotes_csv'); pr.add_source('photos', os.path.join(a.run, 'chain.json'), 'salon_photos')

    r = pr.update(); cm = lambda v: 100 * v
    print(f"1re mise à jour : extraction {r.timing['extract_s'] * 1e3:.0f} ms {r.timing['reextracted']}, résolution {r.timing['solve_s'] * 1e3:.0f} ms, "
          f"χ²/ddl = {r.chi2:.1f}/{r.dof}")
    for n, x, s in zip(r.names, r.x, r.sigma):
        print(f"  {n:7s} {x:.5f} m/u ± {100 * s / x:.2f} %" if n == 's' else f"  {n:7s} {cm(x):9.2f} ± {cm(s):.2f} cm")
    conf = lambda res: [' ↔ '.join(f'{o.id} ({w:+.1f})' for o, w in g) for g in res.conflicts()] or 'aucun'
    print('conflits :', conf(r))
    print('σ réel / σ déclaré par groupe :', {f'{k[0]}/{k[1]}': (round(f, 2) if f else None, round(q, 1)) for k, (f, q, _) in
          variance_factors(r, lambda o: (o.source, o.label if o.source == 'photos' else 'mètre')).items()})
    r2 = pr.update(); print(f"2e mise à jour (rien de changé) : ré-extraits {r2.timing['reextracted']}, total {(r2.timing['extract_s'] + r2.timing['solve_s']) * 1e3:.0f} ms")

    # cohérence des marges : chaque cote au mètre contrôlable, retirée puis prédite par le reste
    key = lambda o: (o.source, o.label if o.source == 'photos' else 'mètre')
    x0 = {n: v for n, v in zip(r.names, r.x)}
    for lab, obs in [('σ déclarés', r.obs), ('σ recalés', calibrate(r, key))]:
        z = []
        for o in [o for o, q in zip(obs, r.redundancy) if o.source == 'cotes' and q > R_MIN]:
            rr = solve(x0, [p for p in obs if p.id != o.id])
            if not np.isfinite(rr.predict(o.expr)[1]): continue
            v, sp = rr.predict(o.expr); z.append((v - o.value) / np.hypot(sp, o.sigma))
        z = np.array(z); print(f"marges, {lab} ({len(z)} cotes) : RMS z = {np.sqrt((z ** 2).mean()):.2f}, |z| < 1 : {(abs(z) < 1).mean():.0%}, |z| < 2 : {(abs(z) < 2).mean():.0%}")

    # contradiction injectée : m1 faux de +10 cm
    write_tape_csv(a.salon_work, tape, {'m1': 314.0}); ri = pr.update()
    sh = max(abs(ri.value(n)[0] - r.value(n)[0]) for n in r.names if n not in ('s', 'end_x', 'w2_x'))
    print(f"injection m1 +10 cm : ré-extraits {ri.timing['reextracted']}, conflits {conf(ri)}, "
          f"w2_x bouge de {cm(ri.value('w2_x')[0] - r.value('w2_x')[0]):+.1f} cm, end_x de {cm(ri.value('end_x')[0] - r.value('end_x')[0]):+.1f} cm, autres ≤ {cm(sh):.2f} cm")
    write_tape_csv(a.salon_work, tape); pr.update()

    print('prochaine meilleure mesure :')
    for g, lab, _, s in next_best(r, candidates(pr.params()), targets=[n for n in r.names if n != 's'], k=3):
        print(f'  {lab} (marge actuelle {cm(s):.1f} cm, gain total {cm(g):.1f} cm)')


if __name__ == '__main__':
    main()
