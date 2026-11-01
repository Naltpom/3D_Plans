import json, math, os, re, unicodedata, urllib.request
from . import components as C
from .rpc import RpcError, rpc

IKEA_URL = 'https://raw.githubusercontent.com/Nutlope/bygga/main/src/data/ikea-rows.ts'
FOLDER = os.path.join(os.environ.get('LOCALAPPDATA') or os.path.expanduser('~'), '3D_Plans', 'catalog')
SCHEMA = 'create table if not exists furniture(id integer primary key autoincrement, item text, x real, z real, rot real, y real, options text)'
COLOR, LIGHT = '#8c949c', {'puissance': 60, 'teinte': '#ffd8a8'}
SNAP, NARROW, PASSAGE, WALKING, TOLERANCE = 0.10, 0.10, 0.80, 1.0, 1e-4
FIXED, DOORS, LIGHTS = ('mur', 'escalier', 'boite'), ('porte', 'porte_fenetre'), ('suspension', 'lampadaire', 'lampe')


def part(kind, x, y0, z, lx, ly, lz, light=False): return kind, (x, y0 + ly / 2, z), (lx, ly, lz), light
def box(*args, **light): return part('box', *args, **light)
def cyl(*args, **light): return part('cyl', *args, **light)
def legs(w, d, h, t=0.04): return [box(i * (w - t) / 2, 0, j * (d - t) / 2, t, h, t) for i in (-1, 1) for j in (-1, 1)]
def seat(h, p): return p['seatH'] / 100 if 'seatH' in p else min(0.45, 0.55 * h)


def chair(w, h, d, p):
    s = seat(h, p)
    return legs(w, d, s - 0.04) + [box(0, s - 0.04, 0, w, 0.04, d), box(0, s, 0.02 - d / 2, w, h - s, 0.04)]


def sofa(w, h, d, p):
    s, a, b = seat(h, p), min(p.get('aw', 15) / 100, 0.2 * w), min(0.25, 0.3 * d)
    return [box(0, 0, b / 2, w - 2 * a, s, d - b), box(0, 0, (b - d) / 2, w, h, b)] + [box(i * (w - a) / 2, 0, 0, a, min(h, s + 0.2), d) for i in (-1, 1)]


def table(w, h, d, p): return [cyl(0, 0, 0, w, h, d)] if p.get('round') else legs(w, d, h - 0.04, 0.05) + [box(0, h - 0.04, 0, w, 0.04, d)]


def fronts(w, h, d, nx, ny):
    nx, ny = max(1, int(nx)), max(1, int(ny)); fw, fh = w / nx, h / ny
    return [box(0, 0, -0.01, w, h, d - 0.02)] + [box((i + 0.5) * fw - w / 2, j * fh + 0.005, d / 2 - 0.01, fw - 0.01, fh - 0.01, 0.02) for i in range(nx) for j in range(ny)]


def shelves(w, h, d, p):
    n = max(2, round(h / 0.35)) + 1
    return [box(i * (w - 0.02) / 2, 0, 0, 0.02, h, d) for i in (-1, 1)] + [box(0, k * (h - 0.02) / (n - 1), 0, w - 0.04, 0.02, d) for k in range(n)]


def bed(w, h, d, p): return [box(0, 0, 0, w, 0.3, d), box(0, 0.3, 0.03, w - 0.04, 0.2, d - 0.1)] + ([box(0, 0, 0.03 - d / 2, w, h, 0.06)] if h > 0.6 else [])


def lamp(w, h, d, base, pole, shade):
    return [cyl(0, 0, 0, 0.6 * w, base, 0.6 * d), cyl(0, base, 0, 0.03, pole, 0.03), cyl(0, base + pole, 0, w, shade, d, light=True)]


GABARITS = {
    'chaise': ('Chaise', (0.45, 0.9, 0.5), chair), 'fauteuil': ('Fauteuil', (0.8, 0.85, 0.85), sofa), 'canape': ('Canapé', (2.0, 0.85, 0.9), sofa),
    'table': ('Table', (1.4, 0.75, 0.8), table), 'bureau': ('Bureau', (1.2, 0.75, 0.6), table),
    'commode': ('Commode / tiroirs', (0.8, 0.9, 0.48), lambda w, h, d, p: fronts(w, h, d, 1 + (w > 1.2), p.get('rows') or round(h / 0.25))),
    'armoire': ('Armoire', (1.0, 2.0, 0.6), lambda w, h, d, p: fronts(w, h, d, p.get('doors') or round(w / 0.5), 1)),
    'etagere': ('Étagère / bibliothèque', (0.8, 2.0, 0.3), shelves), 'lit': ('Lit', (1.6, 0.9, 2.1), bed),
    'meuble_tv': ('Meuble TV', (1.6, 0.5, 0.4), lambda w, h, d, p: fronts(w, h, d, p.get('doors') or round(w / 0.6), 1)),
    'tabouret': ('Tabouret', (0.4, 0.45, 0.4), table), 'banc': ('Banc', (1.2, 0.45, 0.35), table),
    'suspension': ('Luminaire suspendu', (0.4, 2.5, 0.4), lambda w, h, d, p: [cyl(0, h - 0.6, 0, 0.01, 0.6, 0.01), cyl(0, h - 0.85, 0, w, 0.25, d, light=True)]),
    'lampadaire': ('Lampadaire', (0.4, 1.6, 0.4), lambda w, h, d, p: lamp(w, h, d, 0.03, h - 0.33, 0.3)),
    'lampe': ('Lampe', (0.3, 0.5, 0.3), lambda w, h, d, p: lamp(w, h, d, 0.03, 0.6 * h - 0.03, 0.4 * h)),
}
KIND_LABELS = {**{k: v[0] for k, v in GABARITS.items()}, 'sur_mesure': 'Sur mesure'}
IKEA_KINDS = {k: kind for kind, names in {
    'chaise': 'chair junior kidschair highchair office conference', 'fauteuil': 'armchair swivel lounge chairbed', 'canape': 'sofa chaise section corner usofa',
    'table': 'dining coffee side console bar kidstable dropleaf gateleg conftable riser trolley', 'bureau': 'desk cornerdesk deskdraw boxdesk workbench',
    'armoire': 'wardrobe sliding cornerrobe cabinet slidecab appliance wallcab', 'lit': 'bed cot loft bunk daybed', 'meuble_tv': 'tv cabrow',
    'etagere': 'open bookcase bookrow mixed wallshelf cube extension shoerack divider clothesrack coatstand hatrack airer',
    'tabouret': 'stool pouffe lidbox', 'banc': 'bench backbench shelfbench lounger'}.items() for k in names.split()}
WALL_MOUNTED = {'wallcab', 'wallshelf'}


def plain(text): return unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode().lower()
def number(text):
    try: return float(text)
    except ValueError: return text


def parse_ikea(text):
    items = []
    for line in (text.split('`')[1] if text.count('`') >= 2 else text).splitlines():
        f = line.split('|')
        try: w, h, d = (float(v) / 100 for v in f[3:6])
        except (ValueError, IndexError): continue
        params = dict(kv.partition('=')[::2] for kv in (f[6] if len(f) > 6 else '').split(',') if '=' in kv)
        items.append({'id': 'ikea:' + (f[8] if len(f) > 8 and f[8] else re.sub(r'[^a-z0-9]+', '-', plain(' '.join(f[:6])))), 'label': f'{f[0]} — {f[2]}',
                      'kind': IKEA_KINDS.get(f[1], 'commode'), 'size': [w, h, d], 'params': {k: number(v) for k, v in params.items()},
                      'keywords': ' '.join([f[1], *f[7:8]]), 'source': 'ikea', **({'y': 1.4} if f[1] in WALL_MOUNTED else {})})
    return items


def read(name):
    try:
        with open(os.path.join(FOLDER, name), encoding='utf-8') as fh: return json.load(fh)
    except (OSError, ValueError): return []


def custom_item(custom, item_id='sur_mesure'):
    boxes = [{k: float(b[k]) for k in ('x', 'y', 'z', 'lx', 'ly', 'lz')} for b in custom.get('boxes') or []]
    if not boxes or any(b[k] <= 0 for b in boxes for k in ('lx', 'ly', 'lz')): raise RpcError('objet sur mesure : boxes = [{x, y, z, lx, ly, lz}] (m, tailles > 0)')
    low = {a: min(b[a] for b in boxes) for a in 'xyz'}; high = {a: max(b[a] + b['l' + a] for b in boxes) for a in 'xyz'}
    shift = {'x': (low['x'] + high['x']) / 2, 'y': 0.0, 'z': (low['z'] + high['z']) / 2}
    return {'id': item_id, 'label': str(custom.get('label') or 'Objet sur mesure'), 'kind': 'sur_mesure', 'source': 'perso', 'params': {},
            'size': [high['x'] - low['x'], high['y'], high['z'] - low['z']], 'boxes': [{**b, **{a: b[a] - shift[a] for a in 'xyz'}} for b in boxes]}


def entries():
    gabarits = [{'id': f'gabarit:{k}', 'label': label, 'kind': k, 'size': list(size), 'params': dict(LIGHT) if k in LIGHTS else {}, 'source': 'gabarit'}
                for k, (label, size, _) in GABARITS.items()]
    perso = [custom_item(c, c.get('id') or f'perso:{n}') for n, c in enumerate(read('perso.json'))]
    return gabarits + perso + read('ikea.json')


@rpc('catalog.import_ikea', long=True)
def import_ikea(ctx, url=IKEA_URL):
    try:
        with urllib.request.urlopen(url, timeout=60) as response: items = parse_ikea(response.read().decode('utf-8'))
    except OSError as e: raise RpcError(f'téléchargement du catalogue impossible : {e}') from None
    if not items: raise RpcError('catalogue IKEA vide ou illisible')
    os.makedirs(FOLDER, exist_ok=True)
    with open(os.path.join(FOLDER, 'ikea.json'), 'w', encoding='utf-8') as fh: json.dump(items, fh, ensure_ascii=False)
    return {'count': len(items), 'folder': FOLDER}


@rpc('catalog.search')
def search(ctx, q='', kind=None, limit=50):
    words = plain(q).split()
    found = [e for e in entries() if (not kind or e['kind'] == kind) and all(w in plain(f"{e['label']} {e.get('keywords', '')} {KIND_LABELS[e['kind']]}") for w in words)]
    return {'total': len(found), 'items': found[:int(limit)]}


@rpc('catalog.kinds')
def kinds(ctx):
    counts = {}
    for e in entries(): counts[e['kind']] = counts.get(e['kind'], 0) + 1
    return [{'kind': k, 'label': label, 'count': counts.get(k, 0)} for k, label in KIND_LABELS.items()]


def ref(obj): return f"meuble-{obj['id']}"


def load(project):
    project.db.execute(SCHEMA)
    return [{'id': i, 'item': item, 'x': x, 'z': z, 'rot': rot, 'y': y, **json.loads(options)} for i, item, x, z, rot, y, options in project.db.execute('select * from furniture order by id')]


def get(project, id):
    obj = next((o for o in load(project) if o['id'] == int(id)), None)
    if obj is None: raise RpcError(f'objet inconnu : {id}')
    return obj


def summary(obj): return {**{k: obj[k] for k in ('id', 'item', 'label', 'kind', 'x', 'z', 'y', 'rot', 'size')}, 'ref': ref(obj)}


def parts(obj):
    if 'boxes' in obj: return [box(b['x'] + b['lx'] / 2, b['y'], b['z'] + b['lz'] / 2, b['lx'], b['ly'], b['lz']) for b in obj['boxes']]
    return GABARITS[obj['kind']][2](*obj['size'], obj['params'])


def placed(obj):
    a = math.radians(obj['rot']); c, s = math.cos(a), math.sin(a)
    for kind, (u, y, v), size, light in parts(obj): yield kind, (obj['x'] + u * c + v * s, obj['y'] + y, obj['z'] - u * s + v * c), size, a, light


def cylinder(center, size, a, n=16):
    (x, y, z), (lx, ly, lz), c, s = center, size, math.cos(a), math.sin(a)
    ring = [(lx / 2 * math.cos(2 * math.pi * k / n), lz / 2 * math.sin(2 * math.pi * k / n)) for k in range(n)]
    vertices = [[x + u * c + v * s, y + h * ly / 2, z - u * s + v * c] for h in (-1, 1) for u, v in ring]
    faces = [f for k in range(n) for f in ([k, (k + 1) % n, n + k], [(k + 1) % n, n + (k + 1) % n, n + k])]
    return {'kind': 'mesh', 'vertices': vertices, 'faces': faces + [[o, o + k, o + k + 1] for o in (0, n) for k in range(1, n - 1)]}


def geometry(project):
    shapes = []
    for obj in load(project):
        common = {'id': ref(obj), 'type': 'meuble', 'label': obj['label'], 'sigma_max': 0.0}
        for kind, center, size, a, light in placed(obj):
            shape = C.box_shape(center, size, a) if kind == 'box' else cylinder(center, size, a)
            lit = {'color': obj['params'].get('teinte', LIGHT['teinte']), 'light': {k: obj['params'].get(k, v) for k, v in LIGHT.items()}} if light else {'color': COLOR}
            shapes.append({**common, **shape, **lit})
    return shapes


def rect(center, size, a): return C.footprint({'kind': 'box', 'center': center, 'size': size, 'rotY': a})
def solids(obj): return [(rect(c, s, a), c[1] - s[1] / 2, c[1] + s[1] / 2) for _, c, s, a, _ in placed(obj)]


def outline(obj):
    found = solids(obj)
    return rect([obj['x'], 0, obj['z']], obj['size'], math.radians(obj['rot'])), min(b for _, b, _ in found), max(t for *_, t in found)


def bounds(points): xs, zs = zip(*points); return min(xs), max(xs), min(zs), max(zs)


def overlap(first, second):
    (a0, a1, b0, b1), (c0, c1, d0, d1) = bounds(first), bounds(second)
    return a0 < c1 and c0 < a1 and b0 < d1 and d0 < b1 and C.overlap_area(first, second) > TOLERANCE


def meets(p, q): return overlap(p[0], q[0]) and p[1] < q[2] - 1e-3 and q[1] < p[2] - 1e-3


def gap(first, second):
    def to_segment(p, q, r):
        dx, dz = q[0] - p[0], q[1] - p[1]; t = max(0.0, min(1.0, ((r[0] - p[0]) * dx + (r[1] - p[1]) * dz) / ((dx * dx + dz * dz) or 1.0)))
        return math.hypot(p[0] + t * dx - r[0], p[1] + t * dz - r[1])
    return min(to_segment(p, q, r) for a, b in ((first, second), (second, first)) for p, q in zip(a, a[1:] + a[:1]) for r in b)


def fixed(shapes):
    groups = {}
    for s in shapes:
        if s['type'] not in FIXED or s['kind'] != 'box': continue
        top, height = s['center'][1] + s['size'][1] / 2, s['size'][1]
        groups.setdefault(s['id'], (s['label'], []))[1].append((C.footprint(s), top - (height / s['part'] if s['type'] == 'escalier' else height), top))
    return list(groups.values())


def door_zones(shape):
    (x, y, z), (l, h, ep), a = shape['center'], shape['size'], shape['rotY']; offset = (ep - 0.02) / 2 + l / 2
    return [(rect([x + side * offset * math.sin(a), 0, z + side * offset * math.cos(a)], [l, 0, l], a), y - h / 2, y + h / 2) for side in (1, -1)]


def warnings(project, result=None):
    objects, shapes, found = load(project), C.geometry(project, result), []
    walls, doors = fixed(shapes), [(s['label'], door_zones(s)) for s in shapes if s['type'] in DOORS]
    hulls = [outline(o) for o in objects]; walking = [h[1] < WALKING and o['kind'] not in LIGHTS for o, h in zip(objects, hulls)]
    for i, obj in enumerate(objects):
        def warn(code, message): found.append({'code': code, 'component': ref(obj), 'meuble': obj['id'], 'message': message})
        mine, hull = solids(obj), hulls[i]
        others = [(o['label'], solids(o), [hulls[j]] if walking[j] else []) for j, o in enumerate(objects) if j > i] + [(label, parts_, parts_) for label, parts_ in walls]
        for label, theirs, near in others:
            if any(meets(p, q) for p in mine for q in theirs): warn('collision', f"Collision : {obj['label']} chevauche {label}"); continue
            near = [q[0] for q in near if not overlap(hull[0], q[0])]
            if walking[i] and near and NARROW < (g := min(gap(hull[0], q) for q in near)) < PASSAGE:
                warn('passage', f"Passage de {round(g * 100)} cm entre {obj['label']} et {label} (minimum {round(PASSAGE * 100)} cm)")
        door = next((label for label, zones in doors if any(meets(hull, zone) for zone in zones)), None)
        if door: warn('porte', f"{obj['label']} gêne l’ouverture de {door}")
    return found


def snap(shapes, obj):
    w, _, d = obj['size']; a = math.radians(obj['rot']); x, z = round(obj['x'], 2), round(obj['z'], 2)
    for s in shapes:
        if s['type'] != 'mur' or s['kind'] != 'box': continue
        r = s['rotY']; nx, nz, ux, uz = math.sin(r), math.cos(r), math.cos(r), -math.sin(r)
        extent = lambda px, pz: w / 2 * abs(math.cos(a) * px - math.sin(a) * pz) + d / 2 * abs(math.sin(a) * px + math.cos(a) * pz)
        px, pz = x - s['center'][0], z - s['center'][2]; along, across = px * ux + pz * uz, px * nx + pz * nz
        g = abs(across) - s['size'][2] / 2 - extent(nx, nz)
        if abs(g) < SNAP and abs(along) < s['size'][0] / 2 + extent(ux, uz):
            side = math.copysign(1.0, across); x, z = x - side * g * nx, z - side * g * nz
    return round(x, 6), round(z, 6)


def report(ctx, id=None):
    project = ctx.require_project()
    return {'object': None if id is None else summary(get(project, id)), 'warnings': warnings(project, ctx.result)}


@rpc('furniture.add')
def add(ctx, item=None, custom=None, x=0.0, z=0.0, rot=0.0, y=None, options=None):
    project = ctx.require_project()
    if custom is not None: entry = custom_item(custom)
    else: entry = next((e for e in entries() if e['id'] == item), None)
    if entry is None: raise RpcError(f'objet de catalogue inconnu : {item}')
    data = {k: v for k, v in entry.items() if k not in ('id', 'source', 'keywords', 'y')}; data['params'] = {**data['params'], **(options or {})}
    load(project); cursor = project.db.execute('insert into furniture(item, x, z, rot, y, options) values (?,?,?,?,?,?)',
                                              (entry['id'], float(x), float(z), float(rot), float(entry.get('y', 0.0) if y is None else y), json.dumps(data, ensure_ascii=False)))
    project.db.commit()
    return report(ctx, cursor.lastrowid)


@rpc('furniture.move')
def move(ctx, id, x, z, y=None, snap_walls=True):
    project = ctx.require_project(); obj = {**get(project, id), 'x': float(x), 'z': float(z)}
    x, z = snap(C.geometry(project, ctx.result), obj) if snap_walls else (obj['x'], obj['z'])
    project.db.execute('update furniture set x=?, z=?, y=coalesce(?, y) where id=?', (x, z, y, int(id))); project.db.commit()
    return report(ctx, id)


@rpc('furniture.rotate')
def rotate(ctx, id, rot):
    project = ctx.require_project(); get(project, id)
    project.db.execute('update furniture set rot=? where id=?', (float(rot) % 360, int(id))); project.db.commit()
    return report(ctx, id)


@rpc('furniture.remove')
def remove(ctx, id):
    project = ctx.require_project(); get(project, id)
    project.db.execute('delete from furniture where id=?', (int(id),)); project.db.commit()
    return report(ctx)


@rpc('furniture.check')
def check(ctx): return report(ctx)


@rpc('furniture.list')
def list_(ctx): return [summary(o) for o in load(ctx.require_project())]


@rpc('furniture.geometry')
def rpc_geometry(ctx): return geometry(ctx.require_project())
