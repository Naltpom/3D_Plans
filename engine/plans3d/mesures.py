import math, re
from dataclasses import replace
import numpy as np
from . import components as C
from .core import batch, jacobian, next_best
from .plan import PLAN_SCALE, drawing_scale
from .rpc import RpcError, read_cotes, require_result, rpc, update

SIGMA, PRECIS, UNKNOWN, NEAR, PARALLEL = 0.01, 0.02, 0.3, 0.05, math.sin(math.radians(5))
PLACED, TWIN = ('x', 'z', 'dim', 'escalier'), 0.98
OPENINGS = ('porte', 'fenetre', 'porte_fenetre')
NOUNS = {'mur': 'le mur', 'porte': 'la porte', 'fenetre': 'la fenêtre', 'porte_fenetre': 'la porte-fenêtre', 'escalier': "l'escalier"}
GROUPS = {'murs': 'Murs', 'ouvertures': 'Portes et fenêtres', 'escalier': 'Escalier', 'plafond': 'Hauteur sous plafond'}


def au(noun): return 'au ' + noun[3:] if noun.startswith('le ') else 'à ' + noun


def de(noun): return 'du ' + noun[3:] if noun.startswith('le ') else 'de ' + noun


def title(noun): text = noun[2:] if noun.startswith("l'") else noun.split(' ', 1)[1]; return text[0].upper() + text[1:]


def elements(project, values):
    found = {}
    for c in C.load(project).values():
        a, t = C.anchors(c), c['type']
        if t == 'mur' and c['options'].get('nature') != 'contremarche': pts = [a['a'], a['b']]
        elif t in OPENINGS: pts = [a['g'], a['d']]
        elif t == 'escalier': pts = [{**r, **s} for r, s in ((a['depart'], a['cote']), (a['depart'], a['cote_ext']), (a['arrivee'], a['cote_ext']), (a['arrivee'], a['cote']))]
        else: continue
        suffix = c['label'].removeprefix('Mur ') if t == 'mur' else re.sub(r'^\D*', '', c['id'])
        noun = NOUNS[t] + (f' {suffix}' if t != 'escalier' and suffix else '')
        found[c['id']] = {'id': c['id'], 'type': t, 'comp': c, 'noun': noun, 'label': title(noun), 'pts': pts,
                          'xy': [[C.evaluate(p['x'], values), C.evaluate(p['z'], values)] for p in pts]}
    return found


def between(e1, e2, square=False):
    best = None
    for p, q in ((e1, e2), (e2, e1)):
        if q['type'] != 'mur': continue
        (A, B), ((ax, az), (bx, bz)) = q['pts'], q['xy']; dx, dz = bx - ax, bz - az; L = math.hypot(dx, dz)
        for P, (px, pz) in zip(p['pts'], p['xy']):
            t, d = ((px - ax) * dx + (pz - az) * dz) / L ** 2, (dx * (pz - az) - dz * (px - ax)) / L
            if -0.02 <= t <= 1.02 and abs(d) > NEAR and (best is None or abs(d) < best[0]):
                expr = math.copysign(1, d) * ((B['x'] - A['x']) * (P['z'] - A['z']) - (B['z'] - A['z']) * (P['x'] - A['x'])) / C.call('hypot', B['x'] - A['x'], B['z'] - A['z'])
                best = abs(d), expr, [[px, pz], [ax + t * dx, az + t * dz]], f"perpendiculairement {au(q['noun'])}"
    if best or square: return best and best[1:]
    d, P, Q, a, b = min(((math.dist(a, b), P, Q, a, b) for P, a in zip(e1['pts'], e1['xy']) for Q, b in zip(e2['pts'], e2['xy'])), key=lambda r: r[0])
    return (C.call('hypot', Q['x'] - P['x'], Q['z'] - P['z']), [a, b], 'au plus court') if d > NEAR else None


def item(key, group, label, hint, expr, ends, on, covers=()):
    return {'key': key, 'group': group, 'label': label, 'hint': hint, 'expr': str(expr), 'ends': ends, 'on': list(on), 'covers': list(covers)}


def pair(e1, e2, square=False):
    found = between(e1, e2, square)
    if not found: return None
    group = 'escalier' if 'escalier' in (e1['type'], e2['type']) else 'murs' if e1['type'] == e2['type'] == 'mur' else 'ouvertures'
    return item('entre.' + '.'.join(sorted((e1['id'], e2['id']))), group, f"Entre {e1['noun']} et {e2['noun']}", found[2], found[0], found[1], (e1['id'], e2['id']))


def centre(xy): return [sum(p[i] for p in xy) / len(xy) for i in (0, 1)]


def candidates(els):
    walls, opens, stairs = ([e for e in els.values() if e['type'] in types] for types in (('mur',), OPENINGS, ('escalier',)))
    out = [item(f"longueur.{w['id']}", 'murs', f"Longueur {de(w['noun'])}", "d'un angle à l'autre, au ras du sol", C.measure_expr((w['comp'], 'a'), (w['comp'], 'b')), w['xy'], [w['id']]) for w in walls]
    angle = lambda e: math.atan2(e['xy'][1][1] - e['xy'][0][1], e['xy'][1][0] - e['xy'][0][0])
    out += [pair(u, w, True) for i, u in enumerate(walls) for w in walls[i + 1:] if abs(math.sin(angle(u) - angle(w))) < PARALLEL]
    for field, word, hint in (('l', 'Largeur', "dans l'encadrement, d'un tableau à l'autre"), ('h', 'Hauteur', "du sol ou de l'appui au haut de l'encadrement")):
        same = {}
        for o in opens: same.setdefault(str(o['comp']['expressions'][field]), []).append(o)
        out += [item(f"{word.lower()}.{g[0]['id']}", 'ouvertures', f"{word} {de(g[0]['noun'])}", hint, e, g[0]['xy'] if field == 'l' else [centre(g[0]['xy'])] * 2,
                     [o['id'] for o in g], [o['noun'] for o in g[1:]]) for e, g in same.items()]
    for o in opens:
        w = els.get(o['comp']['options'].get('mur'))
        d, i = min((math.dist(w['xy'][i], o['xy'][i]), i) for i in (0, 1)) if w else (0, 0)
        if d > NEAR: out.append(item(f"position.{o['id']}", 'ouvertures', f"De l'angle {de(w['noun'])} à {o['noun']}", "le long du mur, jusqu'au bord de l'encadrement",
                                     C.measure_expr((w['comp'], 'ab'[i]), (o['comp'], 'gd'[i])), [w['xy'][i], o['xy'][i]], [o['id'], w['id']]))
    for s in stairs:
        ex, c = s['comp']['expressions'], centre(s['xy'])
        out += [item(f"marche_h.{s['id']}", 'escalier', "Hauteur d'une marche", "du dessus d'une marche au dessus de la suivante", ex['r'], [c, c], [s['id']]),
                item(f"marche_p.{s['id']}", 'escalier', "Profondeur d'une marche", "du nez d'une marche au nez de la suivante", ex['g'], [c, c], [s['id']]),
                item(f"largeur.{s['id']}", 'escalier', "Largeur de l'escalier", "d'un bord à l'autre des marches", ex['l'], s['xy'][:2], [s['id']])]
        out += [pair(s, w, True) for w in walls]
    heights = {}
    for w in walls: heights.setdefault(str(w['comp']['expressions']['h']), []).append(w)
    out += [item(f"plafond.{g[0]['id']}", 'plafond', 'Hauteur sous plafond' if len(heights) == 1 else f"Hauteur {de(g[0]['noun'])}", 'du sol au plafond, contre le mur', e,
                 [centre(g[0]['xy'])] * 2, [w['id'] for w in g]) for e, g in heights.items()]
    return [c for c in out if c and C.NAME.search(c['expr'])]


def assess(res, items, values, P):
    base = replace(res, sigma=getattr(res, 'raw', res.sigma)); free = np.isinf(base.sigma)
    rows = [(c['key'], c['expr'], SIGMA) for c in items]
    try: G = jacobian(batch([e for _, e, _ in rows], base._idx)[0], base.x, np.ones(len(rows)))
    except KeyError as e: raise RpcError(e.args[0]) from None
    v, unknown = np.asarray(G.multiply(G @ base.C).sum(1)).ravel(), (abs(G) > 0).astype(float) @ free
    gain = {k: g for g, k, _, _ in next_best(base, rows, [e for _, e, _ in rows], len(rows))}
    disputed = {n for g in res.conflicts() for o, _ in g for n in C.NAME.findall(o.expr) if P.get(n, ('', ''))[1] in PLACED}
    contested = [e for _, e, _ in rows if disputed & set(C.NAME.findall(e))]
    lift = {k: g for g, k, _, _ in next_best(base, rows, contested, len(rows))} if contested else {}
    for c, vi, u in zip(items, v, unknown):
        c.update(value=C.evaluate(c['expr'], values), sigma=None if u else float(np.sqrt(max(vi, 0))), gain=gain.get(c['key'], 0.0), conflict=lift.get(c['key'], 0.0) > 1e-3,
                 score=gain.get(c['key'], 0.0) + lift.get(c['key'], 0.0) + UNKNOWN * u)
    order = np.argsort([-c['score'] for c in items], kind='stable'); M = (G @ base.C @ G.T)[np.ix_(order, order)]; d = np.sqrt(np.clip(np.diag(M), 1e-30, None))
    R, hit = np.abs(M) / np.outer(d, d), [frozenset(np.flatnonzero(row & free)) for row in (abs(G) > 0).toarray()[order]]
    for i, j in enumerate(order): items[j]['twin'] = next((items[order[k]]['key'] for k in range(i) if not items[order[k]].get('twin') and (hit[i] == hit[k] if hit[i] else R[i, k] > TWIN)), None)
    return [items[j] for j in order]


def status(res, items, P):
    groups = {}
    for c in items:
        g = groups.setdefault(c['group'], {'group': c['group'], 'name': GROUPS[c['group']], 'worst': 0.0, 'label': '', 'unknown': 0, 'ok': 0, 'n': 0})
        g['n'] += 1; g['ok'] += c['sigma'] is not None and c['sigma'] <= PRECIS
        if c['sigma'] is None: g['unknown'] += 1
        elif c['sigma'] > g['worst']: g['worst'], g['label'] = c['sigma'], c['label']
    sigma = getattr(res, 'raw', res.sigma)
    scales = [{'name': n, 'axis': P[n][1], 'value': float(res.x[i]), 'sigma': float(sigma[i]) if np.isfinite(sigma[i]) else None} for i, n in enumerate(res.names) if P.get(n, ('', ''))[1] in ('échelle', 'dessin')]
    return list(groups.values()), scales


def conflicts(res, els, P):
    resid, out = {o.id: r for o, r in zip(res.obs, res.resid)}, []
    scale = float(res.x[res._idx[PLAN_SCALE]]) if PLAN_SCALE in res._idx else 1.0
    for g in res.conflicts():
        mine, (top, _) = [o for o, _ in g if o.source == 'cotes'], g[0]
        used = {n for o, _ in g for n in C.NAME.findall(o.expr)}
        nouns = [e['noun'] for e in els.values() if used & {n for n in e['comp']['params'] if e['type'] != 'mur' or P[n][1] in ('x', 'z')}][:3]; where = ' et '.join(', '.join(nouns).rsplit(', ', 1)) or 'une partie du plan'
        if mine: text = f"« {mine[0].label or mine[0].id} » ne colle pas avec le reste du plan : écart de {abs(resid[mine[0].id]) * 100:.0f} cm. Vérifie-la au mètre ou retire-la."
        elif ':clic' in top.id: text = f"Un coin cliqué sur les photos ({where}) ne colle pas avec le reste du plan : refais ce clic à l'étape 4.3, ou mesure ce mur au mètre."
        elif top.status == C.PRIOR_STATUS: text = f"Le dessin place {where} à {abs(resid[top.id]) * (scale if PLAN_SCALE in top.expr else 1) * 100:.0f} cm de ce que donnent les photos et tes mesures : si tes mesures sont justes, c'est le dessin qui se trompe."
        else: text = f"Les photos et le dessin ne s'accordent pas sur {where} : une mesure au mètre tranchera."
        out.append({'text': text, 'ids': [o.id for o, _ in g], 'cotes': [o.id.removeprefix('cotes:') for o in mine]})
    return list({c['text']: c for c in out}.values())


def measured(project, res, items):
    obs = {o.id: (r, w) for o, r, w in zip(res.obs, res.resid, res.w)}; flagged = {o.id for g in res.conflicts() for o, _ in g}
    by = {**{c['expr']: c for c in items}, **{c['key']: c for c in items}}
    return [{'id': r['id'], 'label': r['label'] or r['id'], 'value': r['value'], 'unit': r['unit'], 'resid': obs.get(f"cotes:{r['id']}", (None,))[0],
             'conflict': f"cotes:{r['id']}" in flagged, 'key': (by.get(r['id']) or by.get(r['expr']) or {}).get('key'), 'ends': (by.get(r['id']) or by.get(r['expr']) or {}).get('ends')}
            for r in read_cotes(project) if r['status'] == 'mesuré']


def context(ctx):
    if drawing_scale(ctx.require_project()): update(ctx)
    res = require_result(ctx); project = ctx.project
    values = {n: v for n, (v, _) in C.param_values(project, res).items()}
    return res, project, values, elements(project, values)


@rpc('mesures.list')
def rpc_list(ctx):
    res, project, values, els = context(ctx)
    P = project.params(); items = assess(res, candidates(els), values, P)
    groups, scales = status(res, items, P); found = conflicts(res, els, P); mine = measured(project, res, items)
    return {'candidates': items, 'measures': mine, 'groups': groups, 'scales': scales, 'conflicts': found, 'threshold': PRECIS,
            'supposed': any(w['code'] == 'echelle_supposee' for w in getattr(res, 'warnings', [])) or PLAN_SCALE in P and not mine,
            'precise': not found and all(g['unknown'] == 0 and g['worst'] <= PRECIS for g in groups),
            'elements': [{k: e[k] for k in ('id', 'type', 'label', 'noun', 'xy')} for e in els.values()]}


@rpc('mesures.between')
def rpc_between(ctx, a, b):
    res, project, values, els = context(ctx)
    if a not in els or b not in els or a == b: raise RpcError('choisis deux éléments différents : murs, portes, fenêtres ou escalier')
    found = pair(els[a], els[b])
    if not found: raise RpcError(f"{els[a]['label']} et {els[b]['label']} se touchent : choisis un autre élément")
    return assess(res, [found], values, project.params())[0]
