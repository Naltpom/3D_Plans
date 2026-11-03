import io, json, os, re
import numpy as np
from scipy import ndimage
from . import cadastre, components, photos, walls
from .plan import free_ids, place, sketch, trace
from .rpc import rpc

PIXELS, SNAP_DEG, EPS, PRIOR_M, STEP_M, ROOM_M, MATCH_M, SCALES, MATCH_TOL, COLLINEAR_M = 1000, 20.0, 0.006, 0.3, 0.25, 6.0, 0.12, (0.6, 1.6), 0.6, 0.1
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
COLORS = {'noir': '#1a1a1a', 'gris': '#8a8a8a', 'rouge': '#e0302c', 'orange': '#e8953a', 'jaune': '#e3c13b', 'vert': '#3aa65a', 'bleu': '#2f8fe0', 'violet': '#9a5bd0'}
ROLES = {'noir': 'mur', 'gris': 'escalier', 'rouge': 'porte', 'orange': 'porte', 'jaune': 'fenetre', 'vert': 'fenetre', 'bleu': 'porte_fenetre', 'violet': 'ignorer'}
HUES = [('rouge', 0, 18), ('orange', 18, 45), ('jaune', 45, 70), ('vert', 70, 170), ('bleu', 170, 260), ('violet', 260, 335), ('rouge', 335, 361)]
WORDS = {'black': 'noir', 'gray': 'gris', 'grey': 'gris', 'red': 'rouge', 'orange': 'orange', 'yellow': 'jaune', 'green': 'vert', 'blue': 'bleu', 'cyan': 'bleu', 'purple': 'violet', 'violet': 'violet'}
ASKED = {'walls': 'mur', 'doors': 'porte', 'windows': 'fenetre', 'french windows': 'porte_fenetre', 'stairs': 'escalier'}
LEGEND_ASK = ('This is a hand-drawn floor plan where each kind of element is drawn with its own pen color. Look at the colors actually used in the drawing. '
              'Output only JSON with the keys walls, doors, windows, french windows, stairs; each value is the color name used for it in this drawing, or none.')
OPENING_ASK = ('Is there {kind} in this photo, roughly in the middle? Answer only JSON: {{"present": true or false, "kind": "door" or "window" or "french window", '
               '"left": x, "right": x}} with left and right the horizontal pixel positions of its edges in this {width} px wide image.')
KIND_ASK = {'porte': 'a door', 'fenetre': 'a window', 'porte_fenetre': 'a french window (a glazed door reaching the floor)'}
KIND_WORDS = {'door': 'porte', 'window': 'fenetre', 'french window': 'porte_fenetre', 'french door': 'porte_fenetre', 'glass door': 'porte_fenetre'}
CLICK_GAP_M, CORNER_M = 0.12, 0.6
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
    answer = {str(k).replace('_', ' ').lower(): v for k, v in (json.loads(found[0]) if found else {}).items()}
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
        if not any(role == 'mur' and masks[color].sum() >= least for color, role in asked.items()): asked = {}
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


def vertical_planes(state):
    s, found = state['s_est'], []
    for group in walls.plane_groups(state['planes'], s):
        k = 'xyz'.index(group[0]['axis']); run, offset = 2 - k, group[0]['offset']
        low, high = np.min([p['bbox'][0] for p in group], 0) * s, np.max([p['bbox'][1] for p in group], 0) * s
        if high[1] - low[1] < 0.4 or high[run] - low[run] < 0.3: continue
        a, b = np.zeros(2), np.zeros(2); a[k // 2], b[k // 2], a[run // 2], b[run // 2] = offset * s, offset * s, low[run], high[run]
        found.append({'id': group[0]['id'], 'a': a, 'b': b, 'n': int(sum(p['n'] for p in group)), 'planes': [p['id'] for p in group if abs(p['offset'] - offset) * s <= walls.ASSOC_M]})
    return found


def levels(state):
    height, flat = (lambda p: p['offset'] * state['s_est']), [p for p in state['planes'] if p['axis'] == 'y']
    floor = max((p for p in flat if abs(height(p)) < walls.FLOOR_M), key=lambda p: p['n'], default=None)
    ceiling = max((p for p in flat if height(p) > walls.CEILING_M), key=lambda p: p['n'], default=None)
    return floor and floor['id'], ceiling and ceiling['id'], ceiling and float(height(ceiling))


def lined(a, b, **extra):
    d = np.abs(b - a); axis = 0 if d[0] < 0.05 * d[1] else 1 if d[1] < 0.05 * d[0] else None
    if axis is None: return None
    return {'ax': axis, 'pos': float((a[axis] + b[axis]) / 2), 'lo': float(min(a[1 - axis], b[1 - axis])), 'hi': float(max(a[1 - axis], b[1 - axis])), 'len': float(d[1 - axis]), **extra}


def match(drawn, photo, f, shift):
    pairs = {}
    for ax in (0, 1):
        groups, planes = [], sorted((j for j, p in enumerate(photo) if p['ax'] == ax), key=lambda j: photo[j]['pos'])
        for i in sorted((i for i, w in enumerate(drawn) if w['ax'] == ax), key=lambda i: drawn[i]['pos']):
            if groups and abs(drawn[i]['pos'] - drawn[groups[-1][-1]]['pos']) < COLLINEAR_M: groups[-1].append(i)
            else: groups.append([i])

        def score(group, j):
            p = photo[j]; y, lo, hi, total = f * p['pos'] + shift[ax], f * p['lo'] + shift[1 - ax], f * p['hi'] + shift[1 - ax], 0.0
            for i in group:
                w = drawn[i]; gap = abs(w['pos'] - y)
                if gap >= MATCH_TOL or min(hi, w['hi']) - max(lo, w['lo']) <= -0.3: return 0.0
                total += max(w['len'], 0.1) * (1 - (gap / MATCH_TOL) ** 2)
            return total
        best, move = np.zeros((len(groups) + 1, len(planes) + 1)), {}
        for a in range(1, len(groups) + 1):
            for b in range(1, len(planes) + 1):
                s = score(groups[a - 1], planes[b - 1]); options = [(best[a - 1, b], 'skip'), (best[a, b - 1], 'left')] + [(best[a - 1, b - 1] + s, 'take')] * (s > 0)
                best[a, b], move[a, b] = max(options, key=lambda o: o[0])
        a, b = len(groups), len(planes)
        while a and b:
            step = move[a, b]
            if step == 'take':
                for i in groups[a - 1]: pairs[i] = (abs(drawn[i]['pos'] - f * photo[planes[b - 1]]['pos'] - shift[ax]), planes[b - 1])
            a, b = a - (step != 'left'), b - (step != 'skip')
    return pairs

def search(drawn, planes, turns, cameras, traced):
    found = []
    for turn in turns:
        angle = turn * np.pi / 2; photo = [x for p in planes if (x := lined(walls.rotate(p['a'], angle), walls.rotate(p['b'], angle), plane=p)) is not None]
        X = [np.array([w['pos'] for w in drawn if w['ax'] == ax]) for ax in (0, 1)]; LX = [np.array([w['len'] for w in drawn if w['ax'] == ax]) for ax in (0, 1)]
        Q = [np.array([p['pos'] for p in photo if p['ax'] == ax]) for ax in (0, 1)]; LQ = [np.array([p['len'] for p in photo if p['ax'] == ax]) for ax in (0, 1)]
        scales = {1.0} | {round(float(dx / dq), 3) for ax in (0, 1) for dx in (X[ax][:, None] - X[ax][None, :]).ravel() for dq in (Q[ax][:, None] - Q[ax][None, :]).ravel()
                          if abs(dx) > 0.3 and abs(dq) > 0.3 and SCALES[0] < dx / dq < SCALES[1]}
        turned = np.array([walls.rotate(c, angle) for c in cameras]).reshape(-1, 2)
        for f in sorted(scales):
            total, shift = 0.0, [0.0, 0.0]
            for ax in (0, 1):
                if not len(X[ax]) or not len(Q[ax]): continue
                T = (X[ax][:, None] - f * Q[ax][None, :]).ravel(); R = X[ax][None, :, None] - f * Q[ax][None, None, :] - T[:, None, None]
                S = (np.minimum(LX[ax][:, None], f * LQ[ax][None, :])[None] * np.exp(-0.5 * (R / MATCH_M) ** 2)).max(2).sum(1); b = int(np.argmax(S))
                total, shift[ax] = total + float(S[b]), float(T[b])
            inside = walls.inside_share(f * turned + np.array(shift), traced)
            found.append((total * (0.1 + 0.9 * inside) ** 2 * np.exp(-(np.log(f) / 0.3) ** 2), turn, f, shift, photo))
    return max(found, key=lambda item: item[0], default=None)


def refine(drawn, photo, f, shift):
    for _ in range(3):
        pairs = match(drawn, photo, f, shift)
        if not pairs: break
        rows, rhs = [], []
        for i, (_, j) in pairs.items():
            w = drawn[i]; rows.append(np.array([photo[j]['pos'], float(w['ax'] == 0), float(w['ax'] == 1)]) * w['len']); rhs.append(w['pos'] * w['len'])
        rows, rhs = np.array(rows), np.array(rhs); axes = [ax for ax in (0, 1) if rows[:, 1 + ax].any()]
        if len(axes) == 2 and any(len({round(r[0], 1) for r in rows if r[1 + ax]}) >= 2 for ax in (0, 1)):
            solution = np.linalg.lstsq(rows, rhs, rcond=None)[0]
            if SCALES[0] < solution[0] < SCALES[1]: f, shift = float(solution[0]), [float(solution[1]), float(solution[2])]; continue
        shift = [float(np.sum([rhs[n] - f * rows[n, 0] for n in range(len(rows)) if rows[n, 1 + ax]]) / np.sum(rows[:, 1 + ax])) if rows[:, 1 + ax].any() else shift[ax] for ax in (0, 1)]
    return f, shift, match(drawn, photo, f, shift)


@rpc('plan.fit', long=True)
def fit(ctx, plan, turn=None):
    project = ctx.require_project(); source = photo_source(project)
    if not source: raise ValueError('calage : aucune photo positionnée')
    _, state, cache = photos.photo_state(project, source, compute=False); pose = photos.raw_pose(state); L0, c0 = np.array(pose['L']), np.array(pose['c'])
    planes = vertical_planes(state); ring = plan['walls']; drawn_px = np.array([w['a'] for w in ring], float)
    if len(planes) < 2: raise ValueError('calage : moins de 2 surfaces verticales sur les photos (densifie le nuage)')
    ends = np.array([p for q in planes for p in (q['a'], q['b'])]); k0 = float(np.sqrt(np.prod(np.ptp(ends, 0).clip(0.5)) / np.prod(np.ptp(drawn_px, 0).clip(1.0))))
    forward, _ = frame(plan, k0); traced = [{'id': w['name'], 'a': forward(w['a']), 'b': forward(w['b'])} for w in ring]
    drawn = [x for w in traced if (x := lined(w['a'], w['b'], name=w['id'])) is not None]
    cameras = (photos.camera_centers(photos.load_model(cache)) @ L0.T + c0)[:, [0, 2]]
    best = search(drawn, planes, [int(turn)] if turn is not None else range(4), cameras, traced)
    if not best: raise ValueError('calage impossible : aucun mur du dessin ne correspond aux surfaces des photos')
    _, quarter, f, shift, photo = best; f, shift, paired = refine(drawn, photo, f, shift)
    angle, k = quarter * np.pi / 2, k0 / f; T = walls.turn_matrix(angle); moved = np.array(shift) / f; forward, back = frame(plan, k)
    pairs, gaps, planes_of = {}, {}, {}
    for i, (gap, j) in paired.items():
        name, plane = drawn[i]['name'], photo[j]['plane']
        pairs.setdefault(name, []).append(plane['id']); planes_of.setdefault(name, []).extend(plane['planes']); gaps[name] = round(gap / f * 100, 1)
    owner = {plane: name for name, ids in pairs.items() for plane in ids}; placed = lambda p: walls.rotate(np.array(p, float), angle) + moved
    return {'k': k, 'turn': quarter, 'angle': angle, 'shift': moved.tolist(), 'pose': {'L': (T @ L0).tolist(), 'c': (T @ c0 + np.array([moved[0], 0.0, moved[1]])).tolist()},
            'pairs': pairs, 'planes': planes_of, 'gaps': gaps, 'height': levels(state)[2],
            'photo': [{'id': q['id'], 'sure': q['n'] >= walls.SURE_POINTS, 'wall': owner.get(q['id']), 'a': back(placed(q['a'])).round(1).tolist(), 'b': back(placed(q['b'])).round(1).tolist()} for q in planes]}

@rpc('plan.check_corner')
def check_corner(ctx, clicks, pose, point):
    project = ctx.require_project(); source = photo_source(project); _, _, cache = photos.photo_state(project, source, compute=False)
    model = photos.load_model(cache); turns = photos.rotation_matrices(model['q']); centers = photos.camera_centers(model); group = []
    for n, click in enumerate(clicks):
        i = photos.image_index(model, click['image']); (width, height), orientation = model['wh'][i], int(model['orientation'][i])
        group.append((n, i, *photos.undistort(model['K'][i], *photos.display_to_raw(orientation, width, height, click['u'], click['v'])), None))
    found, _ = photos.triangulate(model, turns, group) if len(group) >= 2 else (None, 0.0)
    if found is None: return {'ok': False, 'gap_cm': None, 'offset_cm': None, 'message': 'les 2 photos sont prises du même endroit : choisis-en une autre'}
    L, c = np.array(pose['L'], float), np.array(pose['c'], float); scale = float(np.linalg.norm(L[0]))
    rays = [photos.unit(turns[i].T @ [x, y, 1.0]) for _, i, x, y, _ in group]
    gap = max(np.linalg.norm((found - centers[i]) - ((found - centers[i]) @ d) * d) for (_, i, *_), d in zip(group, rays)) * scale * 2
    placed = L @ found + c; offset = float(np.hypot(placed[0] - point[0], placed[2] - point[1]))
    ok = gap < CLICK_GAP_M and offset < CORNER_M
    message = '' if ok else 'les 2 clics ne désignent pas le même point : reclique le coin' if gap >= CLICK_GAP_M else 'ce point est loin du coin dessiné : vérifie que c’est le bon coin'
    return {'ok': bool(ok), 'gap_cm': round(float(gap) * 100, 1), 'offset_cm': round(offset * 100, 1), 'message': message}


def vision_ready():
    from . import ai
    s = ai.settings()
    try: listed = {m.get('id') for m in ai.request(s, '/models', timeout=2).get('data', [])}
    except Exception: return False
    return s['vision'] in listed or f"{s['vision']}:latest" in listed


@rpc('plan.check_openings', long=True)
def check_openings(ctx, plan, progress=None):
    from PIL import Image, ImageOps
    from . import ai
    project = ctx.require_project(); source = photo_source(project); fitted = plan.get('fit'); progress = progress or (lambda *a: None)
    if not source or not fitted: raise ValueError('vérification : cale d’abord le dessin sur les photos')
    if not vision_ready(): raise ValueError('IA vision indisponible : lance Ollama et télécharge le modèle vision (qwen2.5vl:7b)')
    meta, _, cache = photos.photo_state(project, source, compute=False); model = photos.load_model(cache)
    L, c = np.array(fitted['pose']['L'], float), np.array(fitted['pose']['c'], float); forward, _ = frame(plan, fitted['k']); names = {w['name']: w for w in plan['walls']}; found = []
    for n, opening in enumerate(plan.get('openings', [])):
        progress(n / max(1, len(plan['openings'])), f"Ouverture {n + 1} sur {len(plan['openings'])}")
        result, w = {'index': n, 'seen': None}, names.get(opening['wall'])
        if not w: found.append(result); continue
        a, b = forward(w['a']), forward(w['b']); at = lambda t, y: [float(a[0] + t * (b[0] - a[0])), y, float(a[1] + t * (b[1] - a[1]))]
        corners = [at(opening['t0'], 0.1), at(opening['t1'], 0.1), at(opening['t0'], 2.0), at(opening['t1'], 2.0)]
        for image in photos.visible(ctx, source, corners, limit=1, pose=fitted['pose']):
            i = photos.image_index(model, image); ts = np.linspace(0, 1, 41)
            uv, size, _ = photos.project_to_display(model, i, L, c, corners + [at(t, 1.0) for t in ts])
            box = np.array(uv[:4]); left, right, top, bottom = box[:, 0].min(), box[:, 0].max(), box[:, 1].min(), box[:, 1].max(); margin = right - left
            crop = (int(max(0, left - margin)), int(max(0, top - 0.3 * (bottom - top))), int(min(size[0], right + margin)), int(min(size[1], bottom + 0.3 * (bottom - top))))
            with Image.open(os.path.join(meta['root'], image)) as picture: piece = ImageOps.exif_transpose(picture).convert('RGB').crop(crop)
            k = min(1.0, 768 / max(piece.size)); piece = piece.resize((max(1, round(piece.width * k)), max(1, round(piece.height * k)))); buffer = io.BytesIO(); piece.save(buffer, 'PNG')
            content, _ = ai.ask_image(OPENING_ASK.format(kind=KIND_ASK[opening['type']], width=piece.width), buffer.getvalue(), lambda *a: None)
            answer = re.search(r'\{.*\}', str(content), re.S); answer = json.loads(answer[0]) if answer else {}
            present = answer.get('present') in (True, 'true', 'yes', 'oui'); kind = KIND_WORDS.get(re.sub(r'[_-]', ' ', str(answer.get('kind', ''))).lower().strip())
            result.update(seen=present, photo=image)
            if present and kind: result['type'] = kind
            line = [p for p in uv[4:]]; us = np.array([p[0] if p else np.nan for p in line])
            try: edges = sorted(np.interp(crop[0] + float(answer[key]) / k, us[np.argsort(us)], ts[np.argsort(us)]) for key in ('left', 'right'))
            except (KeyError, TypeError, ValueError): edges = None
            width = (edges[1] - edges[0]) * np.hypot(*(b - a)) if edges else 0.0
            if present and edges and 0.4 <= width <= 3.0 and min(edges[1], opening['t1']) - max(edges[0], opening['t0']) > 0.3 * (opening['t1'] - opening['t0']):
                result.update(t0=round(float(edges[0]), 4), t1=round(float(edges[1]), 4))
        found.append(result)
    progress(1.0, 'Ouvertures vérifiées')
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
        walls.set_pose(ctx, project, source, fitted['angle'], 1.0, np.array(fitted['shift'])); loaded = components.load(project); named = dict(zip([w['name'] for w in ring], ids))
        line = lambda comp: photos.LINE + ';'.join(str(comp['expressions'][f'{end}_{axis}']) for end in ('start', 'end') for axis in 'xz')
        floor, ceiling, _ = levels(state); pairs = sorted(n for n in fitted.get('planes', {}) if n in named)
        rows = [(source, plane, line(loaded[named[n]]), None) for n in pairs for plane in fitted['planes'][n]] + [(source, floor, '0', None)] * bool(floor) + [(source, ceiling, f'{ids[0]}_h', None)] * bool(ceiling)
        db.executemany('insert or replace into photos_assoc values (?,?,?,?)', rows)
        clicks = [(source, c['image'], float(c['u']), float(c['v']), f"{ids[p['corner']]}:a") for p in plan.get('points', []) for c in p.get('clicks', [])]
        db.executemany('insert into photos_clicks(source, image, u, v, anchor) values (?,?,?,?,?)', clicks); db.commit(); photos.invalidate(project, source)
    names, placed = {w['name']: w for w in ring}, []
    for opening in plan.get('openings', []):
        w = names.get(opening['wall'])
        if not w: continue
        a, b = forward(w['a']), forward(w['b']); middle = (opening['t0'] + opening['t1']) / 2
        try: placed.append(place(ctx, opening['type'], (a + middle * (b - a)).tolist(), width=max(0.4, (opening['t1'] - opening['t0']) * np.hypot(*(b - a))), sigma=0.1 if opening.get('seen') else PRIOR_M)['id'])
        except ValueError: continue
    stairs = [add_stairs(project, stair, forward) for stair in plan.get('stairs', []) if stair.get('kept', True)]
    return {'walls': ids, 'pairs': pairs, 'openings': placed, 'stairs': stairs, 'clicks': len(clicks)}
