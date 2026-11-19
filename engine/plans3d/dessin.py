import base64, io, json, os, re, time
from collections import Counter
import numpy as np
from . import lecture, mesures
from .lecture import COLORS, OPENINGS, ROLES, STRUCTURE, cross
from .rpc import project_meta, read_cotes, rpc, write_cotes

FILE, IMAGE, TRAITS, CURVE, SNAP, TEXT_SPAN = 'dessin.json', 'dessin.png', 'dessin-traits.png', 16, 0.015, 0.06
INKED = STRUCTURE + ('escalier',)


def where(project, name): return os.path.join(project.folder, 'sources', name)


def load(project):
    path = where(project, FILE)
    return json.load(open(path, encoding='utf-8')) if os.path.exists(path) else None


def roles(legend=None): return {**ROLES, **{item['color']: item['role'] for item in legend or []}}


def inked(doc, role): return any(i['kind'] == 'pinceau' and role.get(i.get('color')) in INKED for i in doc['items'])


def arc(p, m, q):
    b, c = m - p, q - p; d = 2 * cross(b, c)
    if abs(d) < 1e-6: return np.array([p, q])
    centre = p + np.array([c[1] * (b @ b) - b[1] * (c @ c), b[0] * (c @ c) - c[0] * (b @ b)]) / d
    a0, a1, a2 = (np.arctan2(v[1] - centre[1], v[0] - centre[0]) for v in (p, m, q)); sweep = (a2 - a0) % (2 * np.pi)
    if (a1 - a0) % (2 * np.pi) > sweep: sweep -= 2 * np.pi
    t = a0 + np.linspace(0, sweep, max(2, int(np.ceil(CURVE * abs(sweep) / (2 * np.pi)))) + 1)
    return centre + np.hypot(*(p - centre)) * np.stack([np.cos(t), np.sin(t)], 1)


def outline(item):
    p, kind = np.array(item['points'], float), item['kind']
    if kind == 'rect': (x0, y0), (x1, y1) = p[0], p[-1]; return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
    if kind == 'cercle':
        t = np.linspace(0, 2 * np.pi, CURVE + 1)
        return (p[0] + p[-1]) / 2 + np.abs(p[-1] - p[0]) / 2 * np.stack([np.cos(t), np.sin(t)], 1)
    return arc(*p) if kind == 'arc' and len(p) == 3 else p


def ring_of(segments, tol):
    cuts = [[0.0, 1.0] for _ in segments]
    for i, (a, b) in enumerate(segments):
        d = b - a; n = np.hypot(*d)
        for j, (p, q) in enumerate(segments):
            e = q - p; den = cross(d, e)
            if i == j: continue
            if abs(den) > 1e-9:
                t, u = cross(p - a, e) / den, cross(p - a, d) / den
                if -tol / n <= t <= 1 + tol / n and -tol / np.hypot(*e) <= u <= 1 + tol / np.hypot(*e): cuts[i].append(min(max(t, 0.0), 1.0))
            cuts[i] += [t for r in (p, q) for t in [(r - a) @ d / n ** 2] if 0 < t < 1 and abs(cross(d, r - a)) < tol * n]
    verts, chains = [], []

    def vertex(x):
        k = next((k for k, v in enumerate(verts) if np.hypot(*(v - x)) < tol), None)
        if k is None: verts.append(x); k = len(verts) - 1
        return k
    for (a, b), ts in zip(segments, cuts):
        ids = [vertex(a + t * (b - a)) for t in sorted(set(ts))]
        chains.append([k for i, k in enumerate(ids) if not i or k != ids[i - 1]])
    live = {tuple(sorted(e)) for c in chains for e in zip(c, c[1:])}
    while True:
        degree = Counter(k for e in live for k in e); dead = {e for e in live if min(degree[e[0]], degree[e[1]]) < 2}
        if not dead: break
        live -= dead
    around = {}
    for u, v in live: around.setdefault(u, []).append(v); around.setdefault(v, []).append(u)
    for u in around: around[u].sort(key=lambda w: np.arctan2(*(verts[w] - verts[u])[::-1]))
    seen, best = set(), (0.0, [])
    for start in [h for e in live for h in (e, e[::-1])]:
        u, v, cycle = *start, []
        while (u, v) not in seen:
            seen.add((u, v)); cycle.append(u); u, v = v, around[v][around[v].index(u) - 1]
        pts = np.array([verts[k] for k in cycle]) if cycle else np.zeros((0, 2))
        area = abs(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])) / 2
        if area > best[0]: best = (area, cycle)
    ring = best[1]; on = {tuple(sorted(e)) for e in zip(ring, ring[1:] + ring[:1])}
    left = [[(verts[run[0]], verts[run[-1]]) for run in runs if len(run) > 1] for c in chains
            for runs in [[c[i:j + 1] for i, j in spans(c, on)]]]
    return np.array([verts[k] for k in ring]) if len(ring) >= 3 else None, left


def spans(chain, on):
    out, start = [], None
    for i in range(len(chain) - 1):
        off = tuple(sorted(chain[i:i + 2])) not in on
        if off and start is None: start = i
        if not off and start is not None: out.append((start, i)); start = None
    return out + ([(start, len(chain) - 1)] if start is not None else [])


def shapes(doc, role):
    items = [i for i in doc['items'] if i['kind'] not in ('texte', 'gomme', 'pinceau') and role.get(i.get('color')) in INKED]
    segs = [(a, b, role[i['color']]) for i in items if role[i['color']] in STRUCTURE for line in [outline(i)] for a, b in zip(line, line[1:]) if np.hypot(*(b - a)) > 1e-6]
    if not any(r == 'mur' for *_, r in segs): raise ValueError('lecture du dessin : aucun mur, dessine les murs avec la couleur des murs')
    size = max(float(np.ptp([p for a, b, _ in segs for p in (a, b)], 0).max()), 100.0)
    ring, left = ring_of([(a, b) for a, b, _ in segs], SNAP * size)
    if ring is None: raise ValueError('lecture du dessin : le contour de la pièce n’est pas fermé, rapproche les bouts des murs')
    d = np.roll(ring, -1, 0) - ring; angles = np.arctan2(d[:, 1], d[:, 0]); base = float(np.angle(np.sum(np.hypot(*d.T) * np.exp(4j * angles))) / 4)
    points = ring[np.abs(np.angle(np.exp(1j * (angles - np.roll(angles, 1))))) > np.radians(2)]
    lines = [piece for (*_, r), pieces in zip(segs, left) if r == 'mur' for piece in pieces]
    holes = [(role[i['color']], *lecture.segment(*outline(i)[:, ::-1].T)) for i in items if role[i['color']] in OPENINGS]
    stairs = []
    if any(role[i['color']] == 'escalier' for i in items):
        from PIL import Image, ImageDraw
        masks = {r: Image.new('1', (int(doc['w']), int(doc['h']))) for r in ('escalier', 'mur')}
        for i in items:
            if role[i['color']] in masks: ImageDraw.Draw(masks[role[i['color']]]).line([tuple(p) for p in outline(i)], fill=1, width=max(2, int(i.get('width', 4))))
        stairs = lecture.stairs_of(np.asarray(masks['escalier'], bool), np.asarray(masks['mur'], bool), size)
    used = Counter(i['color'] for i in doc['items'] if i['kind'] not in ('texte', 'gomme') and i.get('color') in COLORS)
    return {'image': [doc['w'], doc['h']], **lecture.assemble(points, base, lines, holes, size), 'stairs': stairs, 'ai': False,
            'legend': [{'color': c, 'hex': COLORS[c], 'count': n, 'role': role[c]} for c, n in used.items()]}


def length_cm(text, drawn_m):
    t = str(text).strip().lower().replace(',', '.')
    if m := re.fullmatch(r'(\d+)\s*m\s*(\d{1,2})', t): return int(m[1]) * 100 + int(m[2]) * (10 if len(m[2]) == 1 else 1)
    if not (m := re.fullmatch(r'(\d+(?:\.\d+)?)\s*(cm|m)?', t)) or not float(m[1]): return None
    value = float(m[1])
    if m[2]: return value * (100 if m[2] == 'm' else 1)
    return min((value, value * 100), key=lambda v: abs(np.log(v / max(drawn_m * 100, 1e-3))))


def attach(plan, doc):
    walls = [(w['name'], np.array(w['a'], float), np.array(w['b'], float)) for w in plan['walls'] + [e for e in plan.get('extra', []) if e.get('name')]]
    size, grid, found, used = max(float(np.ptp([p for _, a, b in walls for p in (a, b)], 0).max()), 100.0), doc['cm'] / 100 / doc['cell'], [], set()
    for item in (i for i in doc['items'] if i['kind'] == 'texte'):
        x, best = np.array(item['points'][0], float), None
        for name, a, b in walls:
            d = b - a; t = (x - a) @ d / (d @ d); gap = np.hypot(*(x - a - np.clip(t, 0, 1) * d))
            if -0.1 < t < 1.1 and gap < max(2.5 * item.get('size', 24), TEXT_SPAN * size) and (best is None or gap < best[0]): best = (gap, name, float(np.hypot(*d)))
        value = best and best[1] not in used and length_cm(item.get('text', ''), best[2] * grid)
        if value: used.add(best[1]); found.append({'id': item['id'], 'wall': best[1], 'value_cm': round(float(value), 1), 'texte': str(item['text'])[:40], 'px': best[2]})
    return found


def scale(doc, cotes): return float(np.median([c['value_cm'] / 100 / c['px'] for c in cotes])) if cotes else doc['cm'] / 100 / doc['cell']


def read(project, legend=None, doc=None, traits=None):
    doc, role = doc or load(project), roles(legend)
    if not doc: raise ValueError('aucun dessin : dessine d’abord le plan')
    if inked(doc, role): plan = lecture.read(io.BytesIO(base64.b64decode(traits.split(',')[-1])) if traits else where(project, TRAITS), [{'color': c, 'role': role[c]} for c in COLORS], ai=False)
    else: plan = shapes(doc, role)
    cotes = attach(plan, doc)
    return {**plan, 'k': scale(doc, cotes), 'cotes': cotes, 'formes': not inked(doc, role), 'dessin': True}


def place(ctx, plan):
    project = ctx.require_project(); found = attach(plan, load(project)); rows = read_cotes(project); before = project_meta(project).get('dessin_cotes') or []
    trace = lambda c: [{'wall': c['wall'], 's': 0}, {'wall': c['wall'], 's': 1}]
    rest = [r for r in rows if not any(mesures.same(r['trace'], b['trace']) and abs(r['value'] - b['value']) < 0.05 for b in before)]
    kept = [c for c in found if not any(mesures.same(r['trace'], trace(c)) for r in rest)]
    if len(rest) < len(rows): write_cotes(project, rest)
    for c in kept: mesures.place(ctx, c['value_cm'], trace(c), label=f"Mur {c['wall']} (écrit sur le dessin)")
    project_meta(project, dessin_cotes=[{'trace': trace(c), 'value': c['value_cm']} for c in kept])
    return kept


def reading(project, found):
    from .plan import calibration
    plan, to_m, (x0, y0) = read(project), calibration(found), found.get('crop', (0, 0)); m = lambda p: to_m(p[0] - x0, p[1] - y0)
    extra, names = [e for e in plan['extra'] if e.get('name')], [w['name'] for w in plan['walls']]
    ring = [m(w['a']) for w in plan['walls']]; ends = {w['name']: (np.array(w['a'], float), np.array(w['b'], float)) for w in plan['walls']}
    spot = lambda name: (0, names.index(name)) if name in names else (1 + [e['name'] for e in extra].index(name), 0)
    return {'walls': [{'points': [p.round(3).tolist() for p in ring], 'closed': True, 'length': float(sum(np.hypot(*(ring[i - 1] - ring[i])) for i in range(len(ring))))}]
            + [{'points': [m(e['a']).round(3).tolist(), m(e['b']).round(3).tolist()], 'closed': False, 'length': float(np.hypot(*(m(e['b']) - m(e['a']))))} for e in extra],
            'cotes': [{'texte': c['texte'], 'value_cm': c['value_cm'], 'drawn_m': c['px'] * found['length'] / 100, 'ok': True, 'message': '', 'wall': spot(c['wall'])[0], 'edge': spot(c['wall'])[1]} for c in plan['cotes']],
            'openings': [{'type': o['type'], 'point': m(a + (o['t0'] + o['t1']) / 2 * (b - a)).round(3).tolist()} for o in plan['openings'] for a, b in [ends[o['wall']]]], 'model': 'dessin'}


@rpc('dessin.get')
def get(ctx): return load(ctx.require_project())


@rpc('dessin.save')
def save(ctx, doc, png, traits=None, crop=(0, 0)):
    project = ctx.require_project(); os.makedirs(where(project, ''), exist_ok=True)
    with open(where(project, FILE), 'w', encoding='utf-8') as fh: json.dump(doc, fh, ensure_ascii=False)
    for name, data in ((IMAGE, png), (TRAITS, traits)):
        if data: open(where(project, name), 'wb').write(base64.b64decode(data.split(',')[-1]))
        elif os.path.exists(where(project, name)): os.remove(where(project, name))
    try: k = read(project)['k']
    except (ValueError, OSError): k = doc['cm'] / 100 / doc['cell']
    old = project_meta(project).get('croquis') or {}
    from PIL import Image
    with Image.open(where(project, IMAGE)) as image: size = list(image.size)
    found = project_meta(project, croquis={'file': IMAGE, 'dessin': True, 'points': [[0, 0], [100, 0]], 'length': 100 * k, 'size': size, 'crop': [float(v) for v in crop], 'v': time.time_ns() // 10 ** 6,
                                           **{key: old[key] for key in ('opacity',) if key in old}})['croquis']
    return {**found, 'path': where(project, IMAGE)}


@rpc('dessin.read')
def preview(ctx, doc, traits=None):
    ink = inked(doc, ROLES)
    try: plan = read(None, doc=doc, traits=traits) if ink and traits else shapes(doc, ROLES)
    except ValueError as error: return {'error': str(error).removeprefix('lecture du dessin : '), 'image': ink}
    cotes = attach(plan, doc)
    return {'walls': plan['walls'], 'extra': plan['extra'], 'openings': plan['openings'], 'stairs': len(plan['stairs']), 'cotes': cotes, 'k': scale(doc, cotes), 'image': ink}
