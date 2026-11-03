import io, json, os, re
import numpy as np
from scipy import ndimage
from . import cadastre, components, photos, walls
from .plan import free_ids, place, sketch, trace
from .rpc import rpc

PIXELS, SNAP_DEG, EPS, PRIOR_M, STEP_M, ROOM_M = 1000, 20.0, 0.006, 0.3, 0.25, 6.0
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
COLORS = {'noir': '#1a1a1a', 'gris': '#8a8a8a', 'rouge': '#e0302c', 'orange': '#e8953a', 'jaune': '#e3c13b', 'vert': '#3aa65a', 'bleu': '#2f8fe0', 'violet': '#9a5bd0'}
ROLES = {'noir': 'mur', 'gris': 'escalier', 'rouge': 'porte', 'orange': 'porte', 'jaune': 'fenetre', 'vert': 'fenetre', 'bleu': 'porte_fenetre', 'violet': 'ignorer'}
HUES = [('rouge', 0, 18), ('orange', 18, 45), ('jaune', 45, 70), ('vert', 70, 170), ('bleu', 170, 260), ('violet', 260, 335), ('rouge', 335, 361)]
WORDS = {'black': 'noir', 'gray': 'gris', 'grey': 'gris', 'red': 'rouge', 'orange': 'orange', 'yellow': 'jaune', 'green': 'vert', 'blue': 'bleu', 'cyan': 'bleu', 'purple': 'violet', 'violet': 'violet'}
ASKED = {'walls': 'mur', 'doors': 'porte', 'windows': 'fenetre', 'french windows': 'porte_fenetre', 'stairs': 'escalier'}
LEGEND_ASK = ('This is a hand-drawn floor plan. Which pen color is used for walls, doors, windows, french windows and stairs? '
              'Output JSON: {"walls": "black", "doors": "red", "windows": "green", "french windows": "blue", "stairs": "gray"}')
STRUCTURE, OPENINGS = ('mur', 'porte', 'fenetre', 'porte_fenetre'), ('porte', 'fenetre', 'porte_fenetre')
SQUARE = np.ones((3, 3), bool)
NEIGHBOURS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def colour_masks(rgb):
    high, low = rgb.max(2), rgb.min(2); chroma = high - low; r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        hue = np.nan_to_num(np.where(high == r, ((g - b) / chroma) % 6, np.where(high == g, (b - r) / chroma + 2, (r - g) / chroma + 4)) * 60)
    colored = (chroma / np.maximum(high, 1e-6) > 0.3) & (high > 0.25) & (chroma > 0.15); dark = ~colored & (high < 0.4)
    masks = {'noir': dark, 'gris': ~colored & (high >= 0.4) & (high < 0.8) & ~ndimage.binary_dilation(dark | colored, iterations=2)}
    for name, lo, hi in HUES: masks[name] = masks.get(name, np.zeros_like(dark)) | colored & (hue >= lo) & (hue < hi)
    return masks


def outline(mask):
    padded = np.pad(mask, 1); ys, xs = np.nonzero(padded); first = np.lexsort((xs, ys))[0]; start = (int(ys[first]), int(xs[first]))
    point, look, ring = start, 6, [start]
    for _ in range(8 * int(mask.sum()) + 8):
        for turn in range(8):
            d = (look + turn) % 8; q = (point[0] + NEIGHBOURS[d][0], point[1] + NEIGHBOURS[d][1])
            if padded[q]: break
        else: break
        point, look = q, (d + 6) % 8
        if point == start: break
        ring.append(point)
    return np.array([(x - 1.0, y - 1.0) for y, x in ring])


def peucker(chain, eps):
    keep = np.zeros(len(chain), bool); keep[[0, -1]] = True; stack = [(0, len(chain) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1: continue
        a, d, inner = chain[i], chain[j] - chain[i], chain[i + 1:j]; n = np.hypot(*d)
        dist = np.abs(d[0] * (inner[:, 1] - a[1]) - d[1] * (inner[:, 0] - a[0])) / n if n > 0 else np.hypot(*(inner - a).T)
        k = int(np.argmax(dist))
        if dist[k] > eps: keep[i + 1 + k] = True; stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return chain[keep]


def simplify(points, eps):
    far = int(np.argmax(np.hypot(*(points - points[0]).T)))
    return np.vstack([peucker(points[:far + 1], eps)[:-1], peucker(np.vstack([points[far:], points[:1]]), eps)[:-1]])


def cross(u, v): return u[0] * v[1] - u[1] * v[0]


def corners_of(lines):
    found = []
    for i in range(len(lines)):
        (m1, d1, *_), (m2, d2, *_) = lines[i - 1], lines[i]
        found.append(m1 + np.linalg.solve(np.array([d1, -d2]).T, m2 - m1)[0] * d1)
    return np.array(found)


def merge_parallel(lines, eps):
    changed = True
    while changed and len(lines) > 3:
        changed = False
        for i in range(len(lines)):
            j = (i + 1) % len(lines); (m1, d1, l1, end1, s1), (m2, d2, l2, end2, _) = lines[i], lines[j]
            if abs(cross(d1, d2)) > np.sin(np.radians(3)): continue
            if abs(cross(d1, m2 - m1)) < 2 * eps and d1 @ d2 > 0: lines[i] = [(m1 * l1 + m2 * l2) / (l1 + l2), d1, l1 + l2, end2, s1]; del lines[j]
            else: lines.insert(j, [end1, np.array([-d1[1], d1[0]]), 0.0, end1, s1])
            changed = True; break


def straighten(points, eps, short):
    lines = []
    for i in range(len(points)):
        a, b = points[i], points[(i + 1) % len(points)]; length = np.hypot(*(b - a))
        if length > 1e-9: lines.append([(a + b) / 2, (b - a) / length, length, b, False])
    weights, angles = np.array([l[2] for l in lines]), np.array([np.arctan2(l[1][1], l[1][0]) for l in lines])
    base = float(np.angle(np.sum(weights * np.exp(4j * angles))) / 4)
    for line, angle in zip(lines, angles):
        snapped = base + np.round((angle - base) / (np.pi / 2)) * np.pi / 2
        if abs(angle - snapped) < np.radians(SNAP_DEG): line[1], line[4] = np.array([np.cos(snapped), np.sin(snapped)]), True
    merge_parallel(lines, eps)

    def removable(i, size):
        before, after = lines[i - 1], lines[(i + 1) % len(lines)]
        if size < eps / 2: return True
        if not (before[4] and after[4]): return False
        return abs(cross(before[1], after[1])) > 0.5 or abs(cross(before[1], after[0] - before[0])) < 2 * eps
    while len(lines) > 3:
        corners = corners_of(lines); sizes = np.hypot(*(np.roll(corners, -1, 0) - corners).T)
        found = [i for i in np.argsort(sizes) if sizes[i] < short and removable(i, sizes[i])]
        if not found: return corners, base
        del lines[int(found[0])]; merge_parallel(lines, eps)
    return corners_of(lines), base

def segment(ys, xs):
    pts = np.stack([xs, ys], 1).astype(float); center = pts.mean(0); u = np.linalg.svd(pts - center, full_matrices=False)[2][0]; t = (pts - center) @ u
    return center + t.min() * u, center + t.max() * u


def straight(a, b, base):
    v = b - a; angle = np.arctan2(v[1], v[0]); snapped = base + np.round((angle - base) / (np.pi / 2)) * np.pi / 2
    if abs(angle - snapped) >= np.radians(SNAP_DEG): return a, b
    d = np.array([np.cos(snapped), np.sin(snapped)]); middle = (a + b) / 2; half = (v @ d) / 2
    return middle - half * d, middle + half * d


def letter(i): return LETTERS[i % 26] + (str(i // 26 + 1) if i >= 26 else '')


def components_of(mask, grow=0):
    labels, count = ndimage.label(ndimage.binary_dilation(mask, iterations=grow) if grow else mask, np.ones((3, 3)))
    return [np.nonzero((labels == i + 1) & mask) for i in range(count)]


def nearest_wall(ring, a, b, size):
    best, u = None, (b - a) / max(np.hypot(*(b - a)), 1e-9)
    for i, (p, q) in enumerate(ring):
        d = q - p; length = np.hypot(*d); w = d / length
        if abs(cross(u, w)) > np.sin(np.radians(25)): continue
        distance = abs(cross(w, (a + b) / 2 - p)); t0, t1 = sorted(((a - p) @ w / length, (b - p) @ w / length))
        if distance < 0.04 * size and t1 > 0 and t0 < 1 and (best is None or distance < best[0]): best = (distance, i, max(0.0, t0), min(1.0, t1))
    return best


def vision_roles(png):
    from . import ai
    s = ai.settings(); listed = {m.get('id') for m in ai.request(s, '/models', timeout=2).get('data', [])}
    if s['vision'] not in listed and f"{s['vision']}:latest" not in listed: return {}
    content, _ = ai.ask_image(LEGEND_ASK, png, lambda *a: None); found = re.search(r'\{.*\}', str(content), re.S)
    answer = json.loads(found[0]) if found else {}
    return {WORDS[word]: ASKED[key] for key, value in answer.items() if key in ASKED for word in re.findall(r'[a-z]+', str(value).lower()) if word in WORDS}


def read(path, legend=None, ai=True):
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert('RGB'); k = min(1.0, PIXELS / max(image.size)); size_px = list(image.size)
        small = image.resize((round(image.width * k), round(image.height * k))) if k < 1 else image
        rgb = np.asarray(small, float) / 255; buffer = io.BytesIO(); small.save(buffer, 'PNG')
    masks = colour_masks(rgb); size = max(rgb.shape[:2]); least = max(30, 0.0002 * rgb.shape[0] * rgb.shape[1])
    roles, asked = {item['color']: item['role'] for item in legend or []}, {}
    if legend is None and ai:
        try: asked = vision_roles(buffer.getvalue())
        except Exception: asked = {}
    present = [name for name in COLORS if masks[name].sum() >= least]
    legend = [{'color': name, 'hex': COLORS[name], 'count': int(masks[name].sum()), 'role': roles.get(name) or asked.get(name) or ROLES[name]} for name in present]
    role = lambda kind: np.any([masks[item['color']] for item in legend if item['role'] == kind] or [np.zeros(rgb.shape[:2], bool)], axis=0)
    structure, wall_mask = role('mur') | role('porte') | role('fenetre') | role('porte_fenetre'), role('mur')
    if not wall_mask.any(): raise ValueError('lecture du dessin : aucun trait de mur trouvé, vérifie la légende')
    thick = max(1, int(round(2 * np.percentile(ndimage.distance_transform_edt(wall_mask)[wall_mask], 90))))
    region, grow = None, 0
    for share in (0.004, 0.008, 0.016, 0.03):
        grow = max(1, int(share * size)); filled = ndimage.binary_fill_holes(ndimage.binary_dilation(structure, iterations=grow))
        filled = ndimage.binary_erosion(ndimage.binary_opening(filled, SQUARE, iterations=grow + thick), SQUARE, iterations=grow + thick // 2)
        labels, count = ndimage.label(filled)
        if count:
            sizes = ndimage.sum(filled, labels, range(1, count + 1))
            if sizes.max() > 0.02 * filled.size: region = labels == int(np.argmax(sizes)) + 1; break
    if region is None: raise ValueError('lecture du dessin : le contour de la pièce n’est pas fermé')
    eps = max(3.0, EPS * size); points, base = straighten(simplify(outline(region), eps), eps, 3 * thick)
    if np.sum(points[:, 0] * np.roll(points[:, 1], -1) - np.roll(points[:, 0], -1) * points[:, 1]) < 0: points = points[::-1]
    middles = (points + np.roll(points, -1, 0)) / 2; start = int(np.lexsort((middles[:, 0], np.round(middles[:, 1] / eps)))[0]); points = np.roll(points, -start, 0)
    ring = [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
    from PIL import Image, ImageDraw
    band = Image.new('L', (rgb.shape[1], rgb.shape[0])); ImageDraw.Draw(band).line([tuple(p) for p in points] + [tuple(points[0])], fill=1, width=int(2 * (grow + thick)) + 3)
    extra = []
    for ys, xs in components_of(wall_mask & ~np.asarray(band, bool), 1):
        if len(ys) < 3: continue
        a, b = straight(*segment(ys, xs), base)
        if np.hypot(*(b - a)) >= 0.03 * size: extra.append({'a': (a / k).round(1).tolist(), 'b': (b / k).round(1).tolist(), 'inside': bool(cadastre.inside((a + b) / 2, points))})
    openings = []
    for kind in OPENINGS:
        for ys, xs in components_of(role(kind)):
            if len(ys) < 3: continue
            a, b = segment(ys, xs); found = nearest_wall(ring, a, b, size)
            if np.hypot(*(b - a)) >= 0.015 * size and found and found[3] - found[2] > 0.02:
                openings.append({'type': kind, 'wall': letter(found[1]), 't0': round(float(found[2]), 4), 't1': round(float(found[3]), 4)})
    stairs = []
    for ys, xs in components_of(role('escalier'), max(2, int(0.015 * size))):
        if min(np.ptp(xs), np.ptp(ys)) >= 0.03 * size: stairs.append({'box': [float(xs.min() / k), float(ys.min() / k), float(xs.max() / k), float(ys.max() / k)]})
    return {'image': size_px, 'angle': base, 'walls': [{'name': letter(i), 'a': (p / k).round(1).tolist(), 'b': (q / k).round(1).tolist()} for i, (p, q) in enumerate(ring)],
            'extra': extra, 'openings': openings, 'stairs': stairs, 'legend': legend, 'ai': bool(asked)}


@rpc('plan.read_drawing', long=True)
def read_drawing(ctx, legend=None, progress=None):
    found, progress = sketch(ctx), progress or (lambda *a: None)
    if not found: raise ValueError('aucun dessin : ajoute d’abord l’image du plan')
    progress(0.1, 'Lecture du dessin'); plan = read(found['path'], legend); progress(1.0, 'Dessin lu')
    return plan


def frame(plan, k):
    origin, c, s = np.array(plan['walls'][0]['a'], float), np.cos(plan['angle']), np.sin(plan['angle'])
    forward = lambda p: k * np.array([c * (p[0] - origin[0]) + s * (p[1] - origin[1]), -s * (p[0] - origin[0]) + c * (p[1] - origin[1])])
    back = lambda m: origin + np.array([c * m[0] - s * m[1], s * m[0] + c * m[1]]) / k
    return forward, back


def photo_source(project):
    photos.ensure_schema(project)
    for (source,) in project.db.execute("select id from sources where kind='photos_colmap'").fetchall():
        try: photos.photo_state(project, source, compute=False); return source
        except ValueError: continue
    return None


def default_k(plan):
    ends = np.array([w['a'] for w in plan['walls']], float)
    return ROOM_M / max(np.ptp(ends, 0).max(), 1.0)


@rpc('plan.fit', long=True)
def fit(ctx, plan, turn=None):
    project = ctx.require_project(); source = photo_source(project)
    if not source: raise ValueError('calage : aucune photo positionnée')
    found = walls.proposals(project, source, raw=True); _, state, cache = photos.photo_state(project, source, compute=False); pose = photos.raw_pose(state)
    L0, c0 = np.array(pose['L']), np.array(pose['c']); photo = sorted((w for w in found['walls'] if w['sure']), key=lambda w: -w['length'])[:walls.FIT_WALLS]
    ring = plan['walls']; drawn = np.array([w['a'] for w in ring], float); ends = np.array([p for w in photo for p in (w['a'], w['b'])], float)
    if len(photo) < 2: raise ValueError('calage : moins de 2 murs sûrs sur les photos')
    k0 = float(np.sqrt(np.prod(np.ptp(ends, 0).clip(0.5)) / np.prod(np.ptp(drawn, 0).clip(1.0)))); forward, _ = frame(plan, k0)
    traced = [{'id': w['name'], 'a': forward(w['a']), 'b': forward(w['b'])} for w in ring]
    cameras = (photos.camera_centers(photos.load_model(cache)) @ L0.T + c0)[:, [0, 2]]
    best = walls.best_fit(photo, traced, None if turn is None else int(turn), cameras)
    if not best: raise ValueError('calage impossible : il faut au moins 2 murs non parallèles vus sur les photos et sur le dessin')
    angle, factor, shift, paired = best; k = k0 / factor; T = walls.turn_matrix(angle); forward, back = frame(plan, k)
    placed = lambda p: walls.rotate(np.array(p, float), angle) + shift / factor; pairs, gaps = {}, {}
    for seg, t, _ in paired:
        w = next(w for w in ring if w['name'] == t['id']); original = next(p for p in found['walls'] if p['id'] == seg['id'])
        a, b = forward(w['a']), forward(w['b']); u = (b - a) / np.hypot(*(b - a)); middle = (placed(original['a']) + placed(original['b'])) / 2
        pairs.setdefault(t['id'], []).append(seg['id']); gaps[t['id']] = max(gaps.get(t['id'], 0.0), round(float(abs(cross(u, middle - a))) * 100, 1))
    owner = {p: name for name, ids in pairs.items() for p in ids}
    return {'k': k, 'turn': turn, 'pose': {'L': (T @ L0).tolist(), 'c': (T @ c0 + np.array([shift[0], 0.0, shift[1]]) / factor).tolist()}, 'pairs': pairs, 'gaps': gaps,
            'height': found['ceiling_height'], 'photo': [{'id': w['id'], 'sure': w['sure'], 'wall': owner.get(w['id']), 'a': back(placed(w['a'])).round(1).tolist(),
                                                           'b': back(placed(w['b'])).round(1).tolist()} for w in found['walls']]}


@rpc('plan.check_openings', long=True)
def check_openings(ctx, plan):
    project = ctx.require_project(); source = photo_source(project); fitted = plan.get('fit')
    if not source or not fitted: raise ValueError('vérification : cale d’abord le dessin sur les photos')
    _, _, cache = photos.photo_state(project, source, compute=False); points, _ = photos.cloud_sample(cache, 400000)
    points = points @ np.array(fitted['pose']['L']).T + np.array(fitted['pose']['c']); forward, _ = frame(plan, fitted['k']); found = []
    for w in plan['walls']:
        a, b = forward(w['a']), forward(w['b']); length = np.hypot(*(b - a)); u = (b - a) / length; flat = points[:, [0, 2]] - a
        near = np.abs(flat @ np.array([-u[1], u[0]])) < walls.BAND_M
        for hole in walls.holes(np.stack([flat[near] @ u, points[near, 1]], 1), length, fitted.get('height') or 2.5):
            found.append({'type': hole['type'], 'wall': w['name'], 't0': (hole['along'] - hole['width'] / 2) / length, 't1': (hole['along'] + hole['width'] / 2) / length,
                          'width': hole['width'], 'height': hole['height'], 'sill': hole['sill']})
    return found


def add_stairs(project, stair, forward):
    u0, v0, u1, v1 = stair['box']; corners = np.array([forward(p) for p in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]); low, size = corners.min(0), np.ptp(corners, 0)
    run = 'x' if size[0] >= size[1] else 'z'; side = 'z' if run == 'x' else 'x'; count = max(2, int(round(size.max() / STEP_M)))
    return components.add(project, 'escalier', free_ids(project, 'escalier', 1)[0], {'axe': run, run: float(low['xz'.index(run)]), side: float(low['xz'.index(side)]),
                                                                                    'g': float(size.max() / count), 'l': float(size.min()), 'n': count, 'source': 'plan'}, 'Escalier')['id']


@rpc('plan.apply')
def apply(ctx, plan):
    project = ctx.require_project(); source = photo_source(project); fitted = plan.get('fit') if source else None; db = project.db
    stale = [c['id'] for c in components.load(project).values() if c['type'] == 'mur' and c['options'].get('source') not in walls.KEPT or c['options'].get('source') == 'plan']
    if stale: components.remove(project, *stale)
    if source:
        photos.ensure_params(project, source); _, state, _ = photos.photo_state(project, source, compute=False)
        db.execute('delete from photos_assoc where source=?', (source,)); db.execute('delete from photos_axes where source=?', (source,))
        db.executemany('update params set init=? where name=?', [(state['s_est'], f'{source}_s'), (0.0, f'{source}_yaw'), *((0.0, f'{source}_t{a}') for a in 'xyz')]); db.commit()
        if os.path.exists(model := os.path.join(project.folder, 'model.json')): os.remove(model)
        photos.invalidate(project, source)
    forward, _ = frame(plan, fitted['k'] if fitted else plan.get('k') or default_k(plan)); ring = plan['walls']
    ids = trace(ctx, [forward(w['a']).tolist() for w in ring], True, sigma=PRIOR_M, source='plan', names=[w['name'] for w in ring], height=fitted and fitted.get('height'))['created']
    for extra in (e for e in plan.get('extra', []) if e.get('name')):
        trace(ctx, [forward(extra['a']).tolist(), forward(extra['b']).tolist()], False, sigma=PRIOR_M, source='plan', names=[extra['name']], height=f'{ids[0]}_h')
    pairs, clicks = None, []
    if fitted:
        try: pairs = walls.fit_walls(ctx, source, fitted.get('turn'))['pairs']
        except ValueError: pairs = None
        ceiling = walls.proposals(project, source)['ceiling']
        if ceiling: db.execute('insert or replace into photos_assoc values (?,?,?,?)', (source, ceiling, f'{ids[0]}_h', None))
        clicks = [(source, c['image'], float(c['u']), float(c['v']), f"{ids[p['corner']]}:a_sol") for p in plan.get('points', []) for c in p.get('clicks', [])]
        db.executemany('insert into photos_clicks(source, image, u, v, anchor) values (?,?,?,?,?)', clicks); db.commit(); photos.invalidate(project, source)
    names, placed = {w['name']: w for w in ring}, []
    for opening in plan.get('openings', []):
        w = names.get(opening['wall'])
        if not w: continue
        a, b = forward(w['a']), forward(w['b']); middle = (opening['t0'] + opening['t1']) / 2
        try: placed.append(place(ctx, opening['type'], (a + middle * (b - a)).tolist(), width=max(0.4, (opening['t1'] - opening['t0']) * np.hypot(*(b - a))))['id'])
        except ValueError: continue
    stairs = [add_stairs(project, stair, forward) for stair in plan.get('stairs', []) if stair.get('kept', True)]
    return {'walls': ids, 'pairs': pairs, 'openings': placed, 'stairs': stairs, 'clicks': len(clicks)}
