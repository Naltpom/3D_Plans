import json, math, os
import numpy as np
from . import components
from .lecture import identities, reattached
from .plan import place, trace
from .rpc import RpcError, cotes_add, read_cotes, rpc

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
OPENINGS = {'porte': (83, 204, 0), 'fenetre': (120, 125, 90), 'porte_fenetre': (140, 215, 0)}


def path(project): return os.path.join(project.folder, 'sources', 'construction.json')


def load(project):
    if not os.path.exists(path(project)): return {'traces': [], 'openings': [], 'height': 250, 'open': None}
    with open(path(project), encoding='utf-8') as fh: return json.load(fh)


def save(project, doc):
    os.makedirs(os.path.dirname(path(project)), exist_ok=True)
    with open(path(project), 'w', encoding='utf-8') as fh: json.dump(doc, fh, ensure_ascii=False, indent=1)


def segments(t): return [(k, (k + 1) % len(t['points'])) for k in range(len(t['names']))]


def walls(doc): return {name: (t, k) for t in doc['traces'] for k, name in enumerate(t['names'])}


def fresh(doc):
    used = set(walls(doc))
    return next(n for n in (*LETTERS, *(a + b for a in LETTERS for b in LETTERS)) if n not in used)


def point(p):
    x, z = (float(v) for v in p)
    if not (math.isfinite(x) and math.isfinite(z) and abs(x) < 1e4 and abs(z) < 1e4): raise RpcError('point hors du plan')
    return [round(x, 4), round(z, 4)]


def size(v, low=1, high=10000):
    if v is None or v == '': return None
    v = float(v)
    if not low <= v <= high: raise RpcError(f'valeur hors limites : {v:g} cm')
    return round(v, 1)


def find(doc, name):
    if name not in walls(doc): raise RpcError(f'mur inconnu : {name}')
    return walls(doc)[name]


def current(doc):
    t = next((t for t in doc['traces'] if t['id'] == doc.get('open')), None)
    if not t: raise RpcError('aucun tracé en cours')
    return t


def follow(t, k, move):
    for i in range(k + 1, len(t['points'])): t['points'][i] = point(move(np.array(t['points'][i])))


def stretch(t, k, shift):
    old = np.array(t['points'], float)
    for i in range(k, len(old)):
        d = old[i] - old[i - 1]
        if i > k and not t['typed'][i - 1] and abs(d[0] * shift[1] - d[1] * shift[0]) < 1e-6 * np.linalg.norm(d) * np.linalg.norm(shift): break
        t['points'][i] = point(old[i] + shift)


def apply_op(doc, o):
    kind = o.get('op')
    if kind == 'start':
        if doc.get('open'): raise RpcError('un tracé est déjà en cours')
        t = {'id': f"t{1 + max([int(t['id'][1:]) for t in doc['traces']], default=0)}", 'closed': False, 'cloison': bool(o.get('cloison')), 'points': [point(o['point'])], 'names': [], 'typed': []}
        doc['traces'].append(t); doc['open'] = t['id']
    elif kind in ('add', 'close'):
        t = current(doc)
        if kind == 'close':
            if len(t['points']) < 3 or t['cloison']: raise RpcError('il faut au moins 3 angles pour fermer la pièce')
            t['closed'], doc['open'] = True, None
        else: t['points'].append(point(o['point']))
        t['names'].append(fresh(doc)); t['typed'].append(size(o.get('length')))
    elif kind == 'back':
        t = current(doc)
        if t['names']: t['points'].pop(); t['names'].pop(); t['typed'].pop()
        else: doc['traces'].remove(t); doc['open'] = None
    elif kind == 'finish':
        t = current(doc); doc['open'] = None
        if not t['names']: doc['traces'].remove(t)
    elif kind == 'length':
        t, k = find(doc, o['wall']); value = size(o['value']); a, b = (np.array(t['points'][i]) for i in segments(t)[k])
        t['typed'][k] = value
        if value and (not t['closed'] or k < len(t['names']) - 1):
            stretch(t, k + 1, (b - a) / max(np.linalg.norm(b - a), 1e-9) * value / 100 + a - b)
    elif kind == 'angle':
        t, k = find(doc, o['wall']); value = float(o['value'])
        if not 1 <= value <= 359: raise RpcError('angle entre 1 et 359°')
        if k == 0 or t['closed'] and k == len(t['names']) - 1: raise RpcError('cet angle se règle depuis le coin suivant')
        i, j = segments(t)[k]; before = np.array(t['points'][i]) - np.array(t['points'][(i - 1) % len(t['points'])]); here = np.array(t['points'][j]) - np.array(t['points'][i])
        turn = math.atan2(before[0] * here[1] - before[1] * here[0], before @ here); side = 1 if turn >= 0 else -1
        delta = side * math.radians(180 - value) - turn; c, s = math.cos(delta), math.sin(delta); o_ = np.array(t['points'][i])
        follow(t, i, lambda p: o_ + np.array([[c, -s], [s, c]]) @ (p - o_))
    elif kind == 'remove' and o.get('opening'):
        doc['openings'] = [x for x in doc['openings'] if x['id'] != o['opening']]
    elif kind == 'remove':
        t, k = find(doc, o['wall']); doc['openings'] = [x for x in doc['openings'] if x['wall'] != o['wall']]; n = len(t['names'])
        if doc.get('open') == t['id']: doc['open'] = None
        if t['closed']:
            order = list(range(k + 1, len(t['points']))) + list(range(0, k + 1))
            t.update(closed=False, points=[t['points'][i] for i in order], names=[t['names'][i] for i in order[:-1]], typed=[t['typed'][i] for i in order[:-1]])
        elif k in (0, n - 1):
            cut = slice(1, None) if k == 0 else slice(None, -1)
            t.update(points=t['points'][cut], names=t['names'][cut], typed=t['typed'][cut])
        else:
            rest = {'id': f"t{1 + max(int(x['id'][1:]) for x in doc['traces'])}", 'closed': False, 'cloison': t['cloison'], 'points': t['points'][k + 1:], 'names': t['names'][k + 1:], 'typed': t['typed'][k + 1:]}
            t.update(points=t['points'][:k + 1], names=t['names'][:k], typed=t['typed'][:k]); doc['traces'].append(rest)
        doc['traces'] = [x for x in doc['traces'] if x['names']]
    elif kind == 'opening':
        t, k = find(doc, o['wall']); a, b = (np.array(t['points'][i]) for i in segments(t)[k]); length = np.linalg.norm(b - a) * 100
        if o.get('type') not in OPENINGS: raise RpcError(f"ouverture inconnue : {o.get('type')}")
        w, h, sill = OPENINGS[o['type']]; old = next((x for x in doc['openings'] if x['id'] == o.get('id')), None)
        item = {**(old or {}), 'type': o['type'], 'wall': o['wall'], 'width': size(o.get('width')) or (old or {}).get('width') or w, 'height': size(o.get('height')) or (old or {}).get('height') or h,
                'sill': size(o.get('sill'), 0) if o.get('sill') not in (None, '') else (old or {}).get('sill', sill) if o['type'] == 'fenetre' else 0}
        item['pos'] = min(max(size(o.get('pos'), 0) or 0, 0), max(length - item['width'], 0))
        if item['width'] > length: raise RpcError(f"mur {o['wall']} trop court pour cette ouverture")
        if old: old.update(item)
        else: doc['openings'].append({'id': f"o{1 + max([int(x['id'][1:]) for x in doc['openings']], default=0)}", **item})
    elif kind == 'height': doc['height'] = size(o['value'], 100, 1000)
    else: raise RpcError(f'opération inconnue : {kind}')
    return doc


@rpc('construction.get')
def get(ctx): return load(ctx.require_project())


@rpc('construction.op')
def op(ctx, o):
    project = ctx.require_project(); doc = apply_op(load(project), o); save(project, doc)
    return doc


@rpc('construction.apply')
def build(ctx):
    project = ctx.require_project(); doc = load(project); loaded = components.load(project)
    measured, before = read_cotes(project), identities(project)
    old = [c['id'] for c in loaded.values() if c['options'].get('source') == 'construction']
    if old: components.remove(project, *old)
    made, first, placed, skipped = [], None, [], []
    for t in (t for t in doc['traces'] if t['names'] and (len(t['points']) > 2 or not t['closed'])):
        pts = t['points'] if t['closed'] else t['points'][:len(t['names']) + 1]
        made += trace(ctx, pts, t['closed'], [v and v / 100 for v in t['typed']], source='construction', names=t['names'], height=f'{first}_h' if first else doc['height'] / 100)['created']
        first = first or made[0]
    if first: cotes_add(ctx, 'construction_hauteur', f'{first}_h', doc['height'], 1.0, label='Hauteur sous plafond saisie')
    by = walls(doc)
    for o in doc['openings']:
        if o['wall'] not in by: continue
        t, k = by[o['wall']]; a, b = (np.array(t['points'][i]) for i in segments(t)[k]); centre = a + (o['pos'] + o['width'] / 2) / 100 / max(np.linalg.norm(b - a), 1e-9) * (b - a)
        try: placed.append(place(ctx, o['type'], centre.tolist(), o['width'] / 100, o['height'] / 100, o['sill'] / 100 if o['type'] == 'fenetre' else None, 0.01)['id'])
        except ValueError as e: skipped.append(f"{o['type']} du mur {o['wall']} : {e}")
    kept, dropped = reattached(project, measured, before)
    return {'walls': made, 'openings': placed, 'skipped': skipped, 'kept': [r['id'] for r in kept], 'dropped': dropped}
