import io, json, os, re
import numpy as np
from scipy import ndimage
from . import cadastre, components, photos, walls
from .core import FUNCS
from .plan import current_values, drawing_scale, free_ids, place, sketch, trace, wall_ends
from .rpc import read_cotes, rpc, write_cotes

PIXELS, SNAP_DEG, EPS, PRIOR_M, ROOM_M, MATCH_M, SCALES, MATCH_TOL, COLLINEAR_M = 1000, 20.0, 0.006, 0.3, 6.0, 0.12, (0.6, 1.6), 0.6, 0.1
SNAP_M, WALL_SIGMA_M, POINT_SIGMA_M, PLACE_SIGMA_M, LENGTH_SIGMA_M, LENGTH_SHARE, RULE_SIGMA_M, BAND_M, BIN_M, PEAK_POINTS, TALL_SHARE = 0.25, 0.02, 0.03, 0.5, 0.1, 0.25, 0.01, 0.6, 0.02, 20, 0.6
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
STEP_TOL_M, STEP_POINTS, CONTACT_PX, CONTACT_SIGMA_M = 0.02, 40, 0.04, 0.01
FULL_SHARE, SLAB_M, RISE_M, STAIR_EDGE_M, SLOPES, SLOPE_TOL_M, SLOPE_POINTS, TIE_M, TOUCH_M, SAME_M, GOING_SIGMA_M = 0.85, 0.2, 0.175, 0.1, (0.45, 1.2), 0.06, 40, 1.5, 0.05, 0.6, 0.01
CLICK_GAP_M, PLANE_GAP_M, FAR_M, MIN_ANGLE_DEG, TUBE_M, TUBE_END_M, HIDDEN_POINTS, CORNER_HEIGHTS = 0.12, 0.10, 1.0, 10.0, 0.12, 0.35, 4, (0.15, 1.0, 1.6)
AXIS_TOL = np.tan(np.radians(2))
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


def flights(ys, xs, grow, walls=None):
    grid = np.pad(np.zeros((np.ptp(ys) + 1, np.ptp(xs) + 1), bool), 2 * grow); grid[ys - ys.min() + 2 * grow, xs - xs.min() + 2 * grow] = True
    if walls is not None: grid[grow:-grow, grow:-grow] |= np.pad(walls, grow)[ys.min():ys.min() + grid.shape[0] - 2 * grow, xs.min():xs.min() + grid.shape[1] - 2 * grow]
    grid = ndimage.binary_fill_holes(ndimage.binary_closing(grid, SQUARE, iterations=grow))[2 * grow:-2 * grow, 2 * grow:-2 * grow]
    tall = grid.shape[0] >= grid.shape[1]; labels, count = ndimage.label(grid.mean(0 if tall else 1) >= FULL_SHARE)
    if not count: return [[xs.min(), ys.min(), xs.max(), ys.max()]]
    kept = np.flatnonzero(labels == np.argmax(np.bincount(labels)[1:]) + 1); a, b = kept.min(), kept.max() + 1; rest = grid.copy()
    found = [[a, 0, b, grid.shape[0]] if tall else [0, a, grid.shape[1], b]]
    if tall: rest[:, a:b] = False
    else: rest[a:b] = False
    pieces = components_of(ndimage.binary_opening(rest, SQUARE, iterations=grow))
    if pieces and len((piece := max(pieces, key=lambda q: len(q[0])))[0]) >= 0.15 * (b - a) * grid.shape[0 if tall else 1]:
        found.append([piece[1].min(), piece[0].min(), piece[1].max() + 1, piece[0].max() + 1])
    return [[float(c0 + xs.min()), float(r0 + ys.min()), float(c1 + xs.min()), float(r1 + ys.min())] for c0, r0, c1, r1 in found]


def lines_along(mask, run, grow):
    thick = max(2, grow // 3); mask = ndimage.binary_dilation(mask, np.ones((2 * thick + 1, 1) if run else (1, 2 * thick + 1), bool))
    across = mask.shape[run]; inner = mask[:, int(0.15 * across):int(0.85 * across) + 1] if run else mask[int(0.15 * across):int(0.85 * across) + 1]
    return ndimage.label(inner.mean(run) >= 0.6)[1]


def stair_shape(ys, xs, grow, walls=None):
    found = flights(ys, xs, grow, walls); line = np.zeros((np.ptp(ys) + 1, np.ptp(xs) + 1), bool); line[ys - ys.min(), xs - xs.min()] = True
    (c0, r0, c1, r1), origin = (int(v) for v in np.subtract(found[0], [xs.min(), ys.min()] * 2)), np.array([xs.min(), ys.min()], float)
    run = int(r1 - r0 >= c1 - c0); width = (c1 - c0) if run else (r1 - r0); pad = 2 * grow
    if len(found) < 2:
        part = line[max(0, r0 - pad):r1 + pad, max(0, c0 - pad):c1 + pad]
        return found, [max(1, lines_along(part, run, grow) - 1)], None
    rest = np.subtract(found[1], [xs.min(), ys.min()] * 2); middle = (rest[run] + rest[run + 2]) / 2
    low = middle < ((r0 + r1) / 2 if run else (c0 + c1) / 2); start, stop = ((r0, r1) if run else (c0, c1)); square = (start, start + width) if low else (stop - width, stop)
    straight = (square[1], stop) if low else (start, square[0])
    inner = (c0 if run else r0) if (rest[0] + rest[2] if run else rest[1] + rest[3]) / 2 < ((c0 + c1) / 2 if run else (r0 + r1) / 2) else (c1 if run else r1)
    cut = lambda a, b, s0, s1: line[max(0, a - pad):b + pad, max(0, s0 - pad):s1 + pad] if run else line[max(0, s0 - pad):s1 + pad, max(0, a - pad):b + pad]
    treads = [max(1, lines_along(cut(*straight, *((c0, c1) if run else (r0, r1))), run, grow) - 1)]
    rr = [int(v) for v in rest]; treads.append(max(1, lines_along(line[rr[1]:rr[3], max(0, rr[0] - pad):rr[2] + pad] if run else line[max(0, rr[1] - pad):rr[3] + pad, rr[0]:rr[2]], 1 - run, grow) - 1))
    pivot_run = square[1] if low else square[0]; pivot = np.array([inner, pivot_run] if run else [pivot_run, inner], float)
    box = (min(c0, c1), square[0], max(c0, c1), square[1]) if run else (square[0], min(r0, r1), square[1], max(r0, r1))
    yy, xx = np.nonzero(line[box[1]:box[3], box[0]:box[2]]); points = np.stack([xx + box[0], yy + box[1]], 1) - pivot
    entry = np.eye(2)[1 - run] * np.sign(((c0 + c1) / 2 if run else (r0 + r1) / 2) - inner); into = np.eye(2)[run] * (1 if low else -1) * -1
    keep = np.hypot(*points.T) > 0.2 * width; angle = np.degrees(np.arctan2(points[keep] @ into, points[keep] @ entry))
    counts = np.convolve(np.histogram(angle, bins=np.arange(8, 84, 3))[0], [1, 1, 1], 'same')
    winders = ndimage.label(counts >= max(4, 0.3 * counts.max()))[1] + 1 if counts.max() else 1
    tour = float(np.sign(cross(into, np.array([(rest[0] + rest[2]) / 2, (rest[1] + rest[3]) / 2]) - pivot)) or 1.0)
    flight = [c0, straight[0], c1, straight[1]] if run else [straight[0], r0, straight[1], r1]
    return [list(np.add(flight, list(origin) * 2)), found[1]], treads, {'square': list(np.add(box, list(origin) * 2)), 'pivot': list(pivot + origin), 'winders': int(winders), 'tour': tour}


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
    from PIL import Image, ImageDraw
    band = Image.new('L', (rgb.shape[1], rgb.shape[0])); ImageDraw.Draw(band).line([tuple(p) for p in points] + [tuple(points[0])], fill=1, width=int(2 * (grow + thick)) + 3)
    extra, named = [], []
    for ys, xs in components_of(wall_mask & ~np.asarray(band, bool), 1):
        if len(ys) < 3: continue
        a, b = straight(*segment(ys, xs), base); inside = bool(cadastre.inside((a + b) / 2, points))
        if np.hypot(*(b - a)) < 0.03 * size: continue
        ends = [a, b]
        for end in range(2 * inside):
            for i in range(len(points)):
                p, q = points[i], points[(i + 1) % len(points)]; w = (q - p) / np.hypot(*(q - p)); t = (ends[end] - p) @ w
                if abs(cross(w, (b - a) / np.hypot(*(b - a)))) > 0.9 and abs(cross(w, ends[end] - p)) < CONTACT_PX * size and -CONTACT_PX * size < t < np.hypot(*(q - p)) + CONTACT_PX * size:
                    ends[end] = p + np.clip(t, 0, np.hypot(*(q - p))) * w
                    if 0.05 < t / np.hypot(*(q - p)) < 0.95: points = np.insert(points, i + 1, ends[end], 0)
                    break
        item = {'a': (ends[0] / k).round(1).tolist(), 'b': (ends[1] / k).round(1).tolist(), 'inside': inside}
        extra.append(item); named += [item] * (inside and any(e is not o for e, o in zip(ends, (a, b))))
    ring = [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
    for j, item in enumerate(named): item['name'] = letter(len(ring) + j)
    turn = lambda s: (np.degrees(np.arctan2(*(s[1] - s[0])[::-1]) - base) + 45) % 90 - 45
    length = lambda s: np.hypot(*(s[1] - s[0]))
    rules = [{'kind': 'symetrie', 'walls': [letter(i - 1 if i else len(ring) - 1), letter(i), letter((i + 1) % len(ring))], 'ok': None} for i, s in enumerate(ring)
             if abs(abs(turn(s)) - 45) < 20 and max(abs(turn(ring[i - 1])), abs(turn(ring[(i + 1) % len(ring)]))) < 5 and 0.5 < length(ring[i - 1]) / length(ring[(i + 1) % len(ring)]) < 2]
    openings = []
    for kind in OPENINGS:
        for ys, xs in components_of(role(kind)):
            if len(ys) < 3: continue
            a, b = segment(ys, xs); found = nearest_wall(ring, a, b, size)
            if np.hypot(*(b - a)) >= 0.015 * size and found and found[3] - found[2] > 0.02:
                openings.append({'type': kind, 'wall': letter(found[1]), 't0': round(float(found[2]), 4), 't1': round(float(found[3]), 4)})
    stairs = []
    for ys, xs in components_of(role('escalier'), max(2, int(0.015 * size))):
        if min(np.ptp(xs), np.ptp(ys)) >= 0.03 * size:
            parts, treads, turn = stair_shape(ys, xs, max(2, int(0.015 * size)), wall_mask); scaled = lambda v: (np.array(v, float) / k).round(1).tolist()
            stairs.append({'box': [float(xs.min() / k), float(ys.min() / k), float(xs.max() / k), float(ys.max() / k)], 'flights': scaled(parts), 'treads': treads,
                           **({'turn': {**turn, 'square': scaled(turn['square']), 'pivot': scaled(turn['pivot'])}} if turn else {})})
    return {'image': size_px, 'angle': base, 'walls': [{'name': letter(i), 'a': (p / k).round(1).tolist(), 'b': (q / k).round(1).tolist()} for i, (p, q) in enumerate(ring)],
            'extra': extra, 'openings': openings, 'stairs': stairs, 'rules': rules, 'legend': legend, 'ai': bool(asked)}


def renamed(plan, old):
    named = lambda p: p['walls'] + [e for e in p.get('extra', []) if e.get('name')]; walls, before = named(plan), named(old)
    size = float(np.ptp(np.array([w['a'] for w in plan['walls']], float), 0).max()); unit = lambda w: (np.subtract(w['b'], w['a'])) / max(np.hypot(*np.subtract(w['b'], w['a'])), 1e-9)

    def common(w, o):
        u, a = unit(w), np.array(w['a'], float)
        if abs(cross(u, unit(o))) > 0.1 or max(abs(cross(u, np.subtract(o[e], a))) for e in 'ab') > 0.03 * size: return 0.0
        t = sorted([0.0, float(np.subtract(w['b'], a) @ u)]); q = sorted(float(np.subtract(o[e], a) @ u) for e in 'ab')
        return min(t[1], q[1]) - max(t[0], q[0])
    pairs, names = sorted(((common(w, o), i, o['name']) for i, w in enumerate(walls) for o in before), reverse=True), {}
    for gap, i, name in pairs:
        if gap > 0.3 * min(np.hypot(*np.subtract(walls[i]['b'], walls[i]['a'])), size) * 0.1 and i not in names and name not in names.values(): names[i] = name
    taken = set(names.values()) | {o['name'] for o in before}; fresh = (letter(i) for i in range(10 * len(LETTERS)) if letter(i) not in taken)
    rename = {w['name']: names.get(i) or next(fresh) for i, w in enumerate(walls)}
    near = lambda o: min((np.hypot(*np.subtract(w['a'], o['a'])), j) for j, w in enumerate(plan['walls']))
    corner = {k: near(o)[1] for k, o in enumerate(old['walls']) if near(o)[0] < 0.01 * size}
    return {**plan, 'walls': [{**w, 'name': rename[w['name']]} for w in plan['walls']], 'extra': [{**e, 'name': rename[e['name']]} if e.get('name') else e for e in plan['extra']],
            'openings': [{**o, 'wall': rename[o['wall']]} for o in plan['openings']], 'rules': [{**r, 'walls': [rename[w] for w in r['walls']]} for r in plan.get('rules', [])],
            'points': [{**c, 'corner': corner[c['corner']]} for c in old.get('points') or [] if c['corner'] in corner],
            'refused': {str(corner[int(k)]): v for k, v in (old.get('refused') or {}).items() if int(k) in corner}}


@rpc('plan.read_drawing', long=True)
def read_drawing(ctx, legend=None, previous=None, progress=None):
    found, progress = sketch(ctx), progress or (lambda *a: None)
    if not found: raise ValueError('aucun dessin : ajoute d’abord l’image du plan')
    progress(0.1, 'Lecture du dessin'); plan = read(found['path'], legend); progress(1.0, 'Dessin lu')
    return renamed(plan, previous) if previous else plan


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


def vertical_planes(state, seen=None):
    s, found, height = state['s_est'], [], walls.levels(state)[2]
    for group in walls.plane_groups(state['planes'], s):
        k = 'xyz'.index(group[0]['axis']); run, offset = 2 - k, group[0]['offset']
        low, high = np.min([p['bbox'][0] for p in group], 0) * s, np.max([p['bbox'][1] for p in group], 0) * s
        if high[1] - low[1] < 0.4 or high[run] - low[run] < 0.3: continue
        a, b = np.zeros(2), np.zeros(2); a[k // 2], b[k // 2], a[run // 2], b[run // 2] = offset * s, offset * s, low[run], high[run]
        found.append({'id': group[0]['id'], 'a': a, 'b': b, 'n': int(sum(p['n'] for p in group)), 'planes': [p['id'] for p in group if abs(p['offset'] - offset) * s <= walls.ASSOC_M],
                      'meuble': bool(walls.furniture(seen, k, offset * s, low[run], high[run], height))})
    return found


def lined(a, b, **extra):
    d = np.abs(b - a); axis = 0 if d[0] < 0.05 * d[1] else 1 if d[1] < 0.05 * d[0] else None
    if axis is None: return None
    return {'ax': axis, 'pos': float((a[axis] + b[axis]) / 2), 'lo': float(min(a[1 - axis], b[1 - axis])), 'hi': float(max(a[1 - axis], b[1 - axis])), 'len': float(d[1 - axis]), **extra}


def match(drawn, photo, f, shift, tol=MATCH_TOL):
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
                common = min(hi, w['hi']) - max(lo, w['lo'])
                if gap >= tol or common <= -0.3: return 0.0
                total += max(common, 0.1) * (1 - (gap / tol) ** 2)
            return total
        best, move = np.zeros((len(groups) + 1, len(planes) + 1)), {}
        for a in range(1, len(groups) + 1):
            for b in range(1, len(planes) + 1):
                s = score(groups[a - 1], planes[b - 1]); options = [(best[a - 1, b], 'skip'), (best[a, b - 1], 'left')] + [(best[a - 1, b - 1] + s, 'take')] * (s > 0)
                best[a, b], move[a, b] = max(options, key=lambda o: o[0])
        a, b, chosen = len(groups), len(planes), []
        while a and b:
            if move[a, b] == 'take': chosen.insert(0, [a - 1, b - 1])
            a, b = a - (move[a, b] != 'left'), b - (move[a, b] != 'skip')
        for n, (g, _) in enumerate(chosen):
            low, high, out = chosen[n - 1][1] if n else -1, chosen[n + 1][1] if n + 1 < len(chosen) else len(planes), drawn[groups[g][0]].get('out', 0.0)
            chosen[n][1] = max((r for r in range(low + 1, high) if score(groups[g], planes[r]) > 0), key=lambda r: (out * photo[planes[r]]['pos'], r == chosen[n][1]))
        for g, r in chosen:
            for i in groups[g]: pairs[i] = (abs(drawn[i]['pos'] - f * photo[planes[r]]['pos'] - shift[ax]), planes[r])
    placed = {i: (drawn[i], f * photo[j]['pos'] + shift[drawn[i]['ax']]) for i, (_, j) in pairs.items()}
    beyond = lambda p: any(w['ax'] != p['ax'] and w['lo'] + 0.1 < f * p['pos'] + shift[p['ax']] < w['hi'] - 0.1
                           and min((f * p[end] + shift[w['ax']] - y) * w.get('out', 0.0) for end in ('lo', 'hi')) > 0.05 for w, y in placed.values())
    return {i: pair for i, pair in pairs.items() if not beyond(photo[pair[1]])}

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


def ring_model(plan, forward, rules=()):
    n, extras = len(plan['walls']), [e for e in plan.get('extra', []) if e.get('name')]
    segments = [(forward(w['a']), forward(w['b'])) for w in plan['walls']] + [(forward(e['a']), forward(e['b'])) for e in extras]
    u = [(b - a) / max(np.hypot(*(b - a)), 1e-9) for a, b in segments]; names = [w['name'] for w in plan['walls']]
    for rule in rules:
        if rule['kind'] == 'symetrie' and all(w in names for w in rule['walls']):
            e, f, g = (names.index(w) for w in rule['walls']); u[f] = (u[e] + u[g]) / np.hypot(*(u[e] + u[g]))
    normal = [np.array([-d[1], d[0]]) for d in u]; line = list(range(len(segments)))
    for i in range(1, n): line[i] = line[i - 1] if abs(cross(u[i - 1], u[i])) < np.sin(np.radians(3)) and u[i - 1] @ u[i] > 0 else line[i]
    rank = {l: r for r, l in enumerate(dict.fromkeys(line))}; line = [rank[l] for l in line]; m = len(rank)
    size = max(np.ptp(np.array([p for s in segments for p in s]), 0).max(), 1e-6)

    def meet(i, j):
        inv = np.linalg.inv(np.array([normal[i], normal[j]])); M = np.zeros((2, m)); M[:, line[i]] += inv[:, 0]; M[:, line[j]] += inv[:, 1]
        return M, np.zeros(2)

    def slide(i, p):
        M = np.zeros((2, m)); M[:, line[i]] = normal[i]
        return M, u[i] * (u[i] @ p)

    def wall_at(p):
        found = [(abs(normal[i] @ (p - a)), i) for i, (a, b) in enumerate(segments[:n]) if -0.05 <= u[i] @ (p - a) / np.hypot(*(b - a)) <= 1.05]
        return min(found, default=(np.inf, None))
    ends, attached = [[None, None] for _ in segments], []
    for q, (a, b) in enumerate(segments[n:], n):
        for k, p in enumerate((a, b)):
            gap, i = wall_at(p)
            if gap < CONTACT_PX * size and abs(cross(u[q], u[i])) > 0.5: ends[q][k] = meet(q, i); attached.append((q, i, p))
        for k, p in enumerate((a, b)):
            if ends[q][k] is None:
                other = ends[q][1 - k]; ends[q][k] = (other[0], other[1] + p - (a, b)[1 - k]) if other is not None else slide(q, p)
    corners = []
    for i in range(n):
        p = segments[i][0]
        if line[i - 1] != line[i]: corners.append(meet((i - 1) % n, i)); continue
        near = min(((np.hypot(*(p - at)), q) for q, w, at in attached if line[w] == line[i]), default=(np.inf, None))
        corners.append(meet(i, near[1]) if near[0] < CONTACT_PX * size else slide(i, p))
    ends[:n] = [[corners[i], corners[(i + 1) % n]] for i in range(n)]
    d0, count = np.zeros(m), np.zeros(m)
    for i, (a, _) in enumerate(segments): d0[line[i]] += normal[i] @ a; count[line[i]] += 1
    return {'names': [w['name'] for w in plan['walls']] + [e['name'] for e in extras], 'u': u, 'normal': normal, 'line': line, 'm': m, 'ends': ends,
            'd0': d0 / count, 'L0': [float(np.hypot(*(b - a))) for a, b in segments], 'out': [-1.0] * n + [0.0] * len(extras)}


def ring_ends(ring, d): return [(Ma @ d + ca, Mb @ d + cb) for (Ma, ca), (Mb, cb) in ring['ends']]


def length_row(ring, i): (Ma, ca), (Mb, cb) = ring['ends'][i]; return ring['u'][i] @ (Mb - Ma), float(ring['u'][i] @ (cb - ca))


def wall_points(cloud, a, b, n, out, top, meubles, explained):
    flat, h, top = cloud[:, [0, 2]], cloud[:, 1], top or 2.4; length = np.hypot(*(b - a)); u = (b - a) / max(length, 1e-9); t, s = (flat - a) @ u, (flat - a) @ n; margin = 0.02 if length < 1.0 else min(0.1, 0.25 * length)
    keep = (t > margin) & (t < length - margin) & (np.abs(s) < BAND_M) & (h > 0.1) & (h < top - 0.1)
    for p, q in meubles + explained:
        d = q - p; w = np.clip((flat - p) @ d / max(d @ d, 1e-9), 0, 1); keep &= np.hypot(*(flat - p - w[:, None] * d).T) > 0.06
    if keep.sum() < PEAK_POINTS: return None
    s, h = s[keep], h[keep]; middles = np.arange(-BAND_M, BAND_M, BIN_M) + BIN_M / 2
    counts = np.convolve(np.histogram(s, bins=np.append(middles - BIN_M / 2, BAND_M))[0], [1, 2, 1], 'same'); levels = np.arange(0.2, top - 0.2, 0.2)
    tall = lambda x: np.isin(np.arange(len(levels)), ((h[np.abs(s - x) < 1.5 * BIN_M] - 0.2) // 0.2).astype(int)).mean() >= TALL_SHARE
    parallel = lambda x: any(abs(cross(u, (q - p) / max(np.hypot(*(q - p)), 1e-9))) < 0.17 and abs(((p + q) / 2 - a) @ n - x) < 0.25 and max((p - a) @ u, (q - a) @ u) > 0 and min((p - a) @ u, (q - a) @ u) < length for p, q in meubles)
    peaks = [(counts[i], middles[i]) for i in range(len(counts)) if counts[i] >= PEAK_POINTS and counts[i] == counts[max(0, i - 1):i + 2].max() and tall(middles[i]) and not parallel(middles[i])]
    if not peaks: return None
    best = max(peaks)[1]; near = s[np.abs(s - best) < 1.5 * BIN_M]
    return float(np.median(near)), int(len(near))


def adjust(ring, seen, cloud, top, meubles, rules):
    m, names, extra = ring['m'], ring['names'], {}
    index = {name: i for i, name in enumerate(names)}
    line = lambda name: np.eye(m)[ring['line'][index[name]]]

    def solve():
        rows = [(np.eye(m)[l], ring['d0'][l], PLACE_SIGMA_M) for l in range(m)]
        for i in range(len(names)):
            row, c = length_row(ring, i); rows.append((row, ring['L0'][i] - c, LENGTH_SIGMA_M + LENGTH_SHARE * ring['L0'][i]))
        rows += [(line(name), float(ring['normal'][index[name]] @ (a + b) / 2), WALL_SIGMA_M) for name, (a, b) in seen.items() if name in index]
        rows += [(line(name), value, POINT_SIGMA_M) for name, (value, _) in extra.items()]
        for rule in rules:
            if all(w in index for w in rule['walls']):
                (r1, c1), (r2, c2) = (length_row(ring, index[w]) for w in rule['walls'][::2]); rows.append((r1 - r2, c2 - c1, RULE_SIGMA_M))
        A, b, w = (np.array(x, float) for x in zip(*rows))
        return np.linalg.lstsq(A / w[:, None], b / w, rcond=None)[0]
    d = solve(); ends = ring_ends(ring, d); explained = [ends[index[name]] for name in seen if name in index]
    for l in range(m):
        group = [i for i in range(len(names)) if ring['line'][i] == l]
        if any(names[i] in seen for i in group) or not ring['out'][group[0]]: continue
        found = wall_points(cloud, ends[group[0]][0], ends[group[-1]][1], ring['normal'][group[0]], ring['out'][group[0]], top, meubles, explained)
        if found: extra.update({names[i]: (float(d[l] + found[0]), found[1]) for i in group})
    d = solve() if extra else d
    evidence = {name: ('photo', abs(float(ring['normal'][index[name]] @ (a + b) / 2) - d[ring['line'][index[name]]])) for name, (a, b) in seen.items() if name in index}
    evidence.update({name: ('points', abs(value - d[ring['line'][index[name]]])) for name, (value, _) in extra.items()})
    return d, evidence

@rpc('plan.fit', long=True)
def fit(ctx, plan, turn=None):
    project = ctx.require_project(); source = photo_source(project)
    if not source: raise ValueError('calage : aucune photo positionnée')
    _, state, cache = photos.photo_state(project, source, compute=False); pose = photos.raw_pose(state); L0, c0 = np.array(pose['L']), np.array(pose['c']); model = walls.tracked(cache)
    planes = vertical_planes(state, walls.evidence(project, source, state, model)); ring = plan['walls']; drawn_px = np.array([w['a'] for w in ring], float)
    if sum(not p['meuble'] for p in planes) < 2: raise ValueError('calage : moins de 2 surfaces verticales sur les photos (densifie le nuage)')
    ends = np.array([p for q in planes for p in (q['a'], q['b'])]); k0 = float(np.sqrt(np.prod(np.ptp(ends, 0).clip(0.5)) / np.prod(np.ptp(drawn_px, 0).clip(1.0))))
    forward, _ = frame(plan, k0); traced = [{'id': w['name'], 'a': forward(w['a']), 'b': forward(w['b'])} for w in ring]
    normal = lambda w: np.eye(2)[int(abs(w['b'][0] - w['a'][0]) > abs(w['b'][1] - w['a'][1]))]
    outward = lambda w: 1.0 if walls.inside_share(((w['a'] + w['b']) / 2 + 0.05 * normal(w))[None], traced) < 0.5 else -1.0
    drawn = [x for w in traced if (x := lined(w['a'], w['b'], name=w['id'], out=outward(w))) is not None]
    cameras = (photos.camera_centers(model) @ L0.T + c0)[:, [0, 2]]
    best = search(drawn, [p for p in planes if not p['meuble']], [int(turn)] if turn is not None else range(4), cameras, traced)
    if not best: raise ValueError('calage impossible : aucun mur du dessin ne correspond aux surfaces des photos')
    _, quarter, f, shift, photo = best; f, shift, paired = refine(drawn, photo, f, shift)
    angle, k = quarter * np.pi / 2, k0 / f; T = walls.turn_matrix(angle); moved = np.array(shift) / f; forward, back = frame(plan, k)
    placed = lambda p: walls.rotate(np.array(p, float), angle) + moved; pose = {'L': (T @ L0).tolist(), 'c': (T @ c0 + np.array([moved[0], 0.0, moved[1]])).tolist()}
    cloud, height = photos.load_model(cache)['P'] @ np.array(pose['L']).T + pose['c'], walls.levels(state)[2]
    rules = [r for r in plan.get('rules', []) if r.get('ok')]; ring, outs = ring_model(plan, forward, rules), {d['name']: d['out'] for d in drawn}; meubles = [(placed(q['a']), placed(q['b'])) for q in planes if q['meuble']]
    seen = {drawn[i]['name']: photo[j]['plane'] for i, (_, j) in paired.items()}
    on_photo = lambda: {name: (placed(q['a']), placed(q['b'])) for name, q in seen.items()}
    offsets, _ = adjust(ring, on_photo(), cloud, height, meubles, rules)
    again = [x for name, (a, b) in zip(ring['names'], ring_ends(ring, offsets)) if (x := lined(f * a, f * b, name=name, out=outs.get(name, 0.0))) is not None]
    seen = {again[i]['name']: photo[j]['plane'] for i, (_, j) in match(again, photo, f, shift, SNAP_M).items()}
    offsets, evidence = adjust(ring, on_photo(), cloud, height, meubles, rules)
    pairs, planes_of = {name: [q['id']] for name, q in seen.items()}, {name: list(q['planes']) for name, q in seen.items()}
    owner = {plane: name for name, ids in pairs.items() for plane in ids}; ends = ring_ends(ring, offsets)
    fitted = [{'name': name, 'a': back(a).round(1).tolist(), 'b': back(b).round(1).tolist()} for name, (a, b) in zip(ring['names'], ends)]

    def front(q):
        p = lined(walls.rotate(q['a'], angle), walls.rotate(q['b'], angle))
        if not q['meuble'] or p is None: return None
        y, lo, hi = f * p['pos'] + shift[p['ax']], f * p['lo'] + shift[1 - p['ax']], f * p['hi'] + shift[1 - p['ax']]
        return min(((abs(w['pos'] - y), w['name']) for w in again if w['ax'] == p['ax'] and abs(w['pos'] - y) < walls.FRONT_MAX_M * f and (y - w['pos']) * w['out'] < 0
                    and min(hi, w['hi']) > max(lo, w['lo'])), default=(0, None))[1]
    n = len(plan['walls'])

    def stair_width(stair):
        if len(stair.get('flights') or []) < 2: return None
        (lo, hi), rest = (np.sort([forward(b[:2]), forward(b[2:])], 0) for b in stair['flights'][:2])
        if stair.get('turn'): square = np.sort([forward(stair['turn']['square'][:2]), forward(stair['turn']['square'][2:])], 0); lo, hi = np.minimum(lo, square[0]), np.maximum(hi, square[1])
        run = int(np.argmax(hi - lo)); side = 1 - run; inner = (lo if rest.mean(0)[side] < (lo[side] + hi[side]) / 2 else hi)[side]; wall = lo[side] + hi[side] - inner
        found = [(min(p['hi'], hi[run]) - max(p['lo'], lo[run]), abs(wall - p['pos'])) for q in planes if (p := lined(placed(q['a']), placed(q['b']))) and p['ax'] == side
                 and abs(wall - p['pos']) >= 0.5 and abs(p['pos'] - inner) < abs(wall - inner)]
        return round(float(max(found)[1]), 3) if found and max(found)[0] > 0.3 else None
    return {'k': k, 'turn': quarter, 'angle': angle, 'shift': moved.tolist(), 'pose': pose, 'pairs': pairs, 'planes': planes_of, 'height': height,
            'walls': fitted[:n], 'extra': fitted[n:], 'gaps': {name: round(gap * 100, 1) for name, (_, gap) in evidence.items()}, 'seen': {name: kind for name, (kind, _) in evidence.items()},
            'moved': {name: round(abs(offsets[l] - ring['d0'][l]) * 100, 1) for name, l in zip(ring['names'], ring['line'])}, 'stairs': [stair_width(st) for st in plan.get('stairs', [])],
            'photo': [{'id': q['id'], 'sure': q['n'] >= walls.SURE_POINTS, 'meuble': q['meuble'], 'front': front(q), 'wall': owner.get(q['id']), 'a': back(placed(q['a'])).round(1).tolist(), 'b': back(placed(q['b'])).round(1).tolist()} for q in planes]}

@rpc('plan.cloud')
def top_cloud(ctx, plan, limit=6000):
    project = ctx.require_project(); source = photo_source(project); fitted = plan.get('fit')
    if not source or not fitted: return []
    _, state, cache = photos.photo_state(project, source, compute=False); top = fitted.get('height') or 2.4
    points = photos.load_model(cache)['P'] @ np.array(fitted['pose']['L']).T + fitted['pose']['c']; points = points[(points[:, 1] > 0.9) & (points[:, 1] < top - 0.1)][:, [0, 2]]
    points = points[np.random.default_rng(0).permutation(len(points))[:int(limit)]]; c, s = np.cos(plan['angle']), np.sin(plan['angle'])
    return (np.array(plan['walls'][0]['a'], float) + np.stack([c * points[:, 0] - s * points[:, 1], s * points[:, 0] + c * points[:, 1]], 1) / fitted['k']).round(1).tolist()


ANCHORS = {'bas': 'a_sol', 'haut': 'a_haut', 'arete': 'a'}


def click_ray(model, L, c, click):
    i = photos.image_index(model, click['image']); (width, height), orientation = model['wh'][i], int(model['orientation'][i])
    x, y = photos.undistort(model['K'][i], *photos.display_to_raw(orientation, width, height, click['u'], click['v']))
    return L @ photos.camera_centers(model)[i] + c, photos.unit(L @ (photos.rotation_matrices(model['q'][i:i + 1])[0].T @ [x, y, 1.0]))


def corner_rows(model, L, c, click, height):
    origin, ray = click_ray(model, L, c, click); kind = click.get('kind', 'arete')
    if kind == 'arete':
        across = np.array([-ray[2], ray[0]]) / max(np.hypot(ray[0], ray[2]), 1e-9)
        return [across], [float(across @ origin[[0, 2]])], None
    level = 0.0 if kind == 'bas' else height
    if level is None: return None, None, 'hauteur du plafond inconnue : clique le coin bas ou l’arête'
    t = (level - origin[1]) / ray[1] if abs(ray[1]) > 1e-6 else -1.0
    if t <= 0: return None, None, f"ce clic ne vise pas le {'sol' if kind == 'bas' else 'plafond'} : vérifie le type de clic"
    hit = origin + t * ray
    return [np.array([1.0, 0.0]), np.array([0.0, 1.0])], [float(hit[0]), float(hit[2])], None


@rpc('plan.check_corner')
def check_corner(ctx, clicks, pose, point, lines=(), height=None):
    project = ctx.require_project(); source = photo_source(project); _, _, cache = photos.photo_state(project, source, compute=False)
    model = photos.load_model(cache); L, c = np.array(pose['L'], float), np.array(pose['c'], float); rows, rhs, owners = [], [], []
    refuse = lambda message: {'ok': False, 'gap_cm': None, 'offset_cm': None, 'message': message, 'note': '', 'point': None}
    for n, click in enumerate(clicks):
        found, values, error = corner_rows(model, L, c, click, height)
        if error: return refuse(error)
        rows += found; rhs += values; owners += [n] * len(found)
    A, b = np.array(rows), np.array(rhs)
    if len(clicks) < 2 or np.linalg.svd(A, compute_uv=False)[-1] < np.sqrt(2) * np.sin(np.radians(MIN_ANGLE_DEG) / 2):
        return refuse('les 2 photos voient le coin presque du même côté : choisis une photo prise d’un autre endroit')
    placed = np.linalg.lstsq(A, b, rcond=None)[0]; residual = A @ placed - b
    gap = max(float(np.linalg.norm(residual[np.array(owners) == n])) for n in range(len(clicks)))
    offset = float(np.hypot(placed[0] - point[0], placed[1] - point[1]))
    away, wall = max(((abs(cross(np.subtract(q, a) / np.hypot(*np.subtract(q, a)), placed - a)), name) for name, a, q in lines), default=(0.0, ''))
    ok = gap < CLICK_GAP_M
    notes = [f'à {away * 100:.0f} cm du mur {wall} vu en photo : vérifie que tu as cliqué le coin, pas un meuble devant'] * int(away >= PLANE_GAP_M)
    notes += [f'à {offset:.1f} m du coin dessiné : vérifie que c’est le bon coin'.replace('.', ',')] * int(offset >= FAR_M)
    return {'ok': bool(ok), 'gap_cm': round(gap * 100, 1), 'offset_cm': round(offset * 100, 1), 'point': placed.round(3).tolist(),
            'message': '' if ok else 'les 2 clics ne désignent pas le même coin : reclique', 'note': ' ; '.join(notes) if ok else ''}


def corner_views(project, point, pose, walls=None):
    source = photo_source(project); _, _, cache = photos.photo_state(project, source, compute=False); model = photos.load_model(cache)
    L, c = np.array(pose['L'], float), np.array(pose['c'], float); cloud, _ = photos.cloud_sample(cache, 150000); cloud = cloud @ L.T + c
    centers = photos.camera_centers(model) @ L.T + c; x, z = point; targets = np.array([[x, h, z] for h in CORNER_HEIGHTS]); views = []
    framed = lambda p, width, height: p is not None and 0.05 < p[0] / width < 0.95 and 0.05 < p[1] / height < 0.95
    for i, center in enumerate(centers):
        uv, (width, height), _ = photos.project_to_display(model, i, L, c, targets)
        distance = float(np.linalg.norm(targets[1] - center)); inside = all(framed(p, width, height) for p in uv) and 0.8 < distance < 8.0
        view = {'name': str(model['names'][i]), 'index': i, 'inside': bool(inside), 'hidden': None, 'offset': 9.0, 'visible': False,
                'sees': {name: any(framed(p, width, height) for p in photos.project_to_display(model, i, L, c, [[a[0] + s * (b[0] - a[0]), 1.0, a[1] + s * (b[1] - a[1])] for s in (0.1, 0.3, 0.5, 0.7, 0.9)])[0])
                         for name, (a, b) in (walls or {}).items()}}
        if inside and abs(uv[2][0] - uv[0][0]) <= 0.35 * abs(uv[2][1] - uv[0][1]):
            hidden = 0
            for target in targets[1:]:
                ray = target - center; length = np.linalg.norm(ray); t = (cloud - center) @ ray / length ** 2
                hidden += int(((t > 0.3 / length) & (t < 1 - TUBE_END_M / length) & (np.linalg.norm(cloud - center - t[:, None] * ray, axis=1) < TUBE_M)).sum())
            view.update(hidden=hidden, offset=float(abs(uv[1][0] / width - 0.5) + abs(uv[1][1] / height - 0.5)), visible=hidden <= HIDDEN_POINTS)
        views.append(view)
    return sorted(views, key=lambda v: (not v['visible'], not v['inside'], v['hidden'] if v['hidden'] is not None else 1e9, v['offset'])), centers


@rpc('plan.corner_photos')
def corner_photos(ctx, point, pose, limit=8, refused=()):
    views, centers = corner_views(ctx.require_project(), point, pose); chosen = []
    for view in (v for v in views if v['visible'] and v['name'] not in refused):
        if all(np.linalg.norm(centers[view['index']] - centers[other]) > 0.4 for other in chosen): chosen.append(view['index'])
    return [next(v['name'] for v in views if v['index'] == i) for i in chosen[:int(limit)]]


@rpc('plan.photo_gallery')
def photo_gallery(ctx, point, pose, walls=None):
    views, _ = corner_views(ctx.require_project(), point, pose, walls)
    return [{k: v[k] for k in ('name', 'visible', 'inside', 'hidden', 'sees')} for v in views]


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

def stair_region(box, forward, margin=0.3):
    u0, v0, u1, v1 = box; corners = np.array([forward(p) for p in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))])
    return corners.min(0) - margin, corners.max(0) + margin


def measured_steps(cache, pose, low, high, top):
    dense = os.path.join(cache, 'dense', 'fused.ply')
    if not os.path.exists(dense): return []
    from scipy.spatial import ConvexHull
    points, normals = photos.read_ply(dense)
    if normals is None: return []
    L, c = np.array(pose['L'], float), np.array(pose['c'], float); points = points @ L.T + c; up = normals @ L.T; up = np.abs(up[:, 1]) / np.linalg.norm(up, axis=1).clip(1e-9)
    keep = (points[:, 0] > low[0]) & (points[:, 0] < high[0]) & (points[:, 2] > low[1]) & (points[:, 2] < high[1]) & (points[:, 1] > 0.08) & (points[:, 1] < top + 0.1) & (up > 0.85)
    flat = points[keep]
    if len(flat) < 200: return []
    counts, edges = np.histogram(flat[:, 1], bins=np.arange(0.08, top + 0.1, 0.01)); levels = edges[:-1] + 0.005
    best = max(((sum(counts[np.abs(levels - (rise * n)) < STEP_TOL_M].sum() for n in range(1, int(top / rise) + 1)), rise)
                for rise in np.arange(0.15, 0.23, 0.002)), key=lambda item: item[0])[1]
    steps = []
    for n in range(1, int(round(top / best)) + 1):
        near = flat[np.abs(flat[:, 1] - best * n) < STEP_TOL_M]
        if len(near) < STEP_POINTS: continue
        cells = np.floor((near[:, [0, 2]] - low) / 0.08).astype(int); grid = np.zeros(cells.max(0) + 1, bool); grid[cells[:, 0], cells[:, 1]] = True
        labels, count = ndimage.label(grid, np.ones((3, 3))); owner = labels[cells[:, 0], cells[:, 1]]; groups = [near[owner == k] for k in range(1, count + 1)]
        last = np.mean(steps[-1]['contour'], 0) if steps else None
        near = min(groups, key=lambda g: np.linalg.norm(g[:, [0, 2]].mean(0) - last)) if last is not None else max(groups, key=len)
        if last is not None and np.linalg.norm(near[:, [0, 2]].mean(0) - last) > 0.8: continue
        if len(near) < STEP_POINTS: continue
        hull = ConvexHull(near[:, [0, 2]]); ring = near[hull.vertices][:, [0, 2]]
        steps.append({'n': n, 'y': float(np.median(near[:, 1])), 'contour': simplify(ring, 0.03).round(3).tolist() if len(ring) > 3 else ring.round(3).tolist()})
    return steps


def add_measured_stairs(project, steps):
    created = []
    for k, step in enumerate(steps):
        comp_id = free_ids(project, 'marche', 1)[0]
        components.add(project, 'sol', comp_id, {'contour': [[f'{x:.3f}', f'{z:.3f}'] for x, z in step['contour']], 'y': f"{step['y']:.3f}", 'source': 'plan', 'a_priori': False}, f"Marche {step['n']}")
        created.append(comp_id)
        below = steps[k - 1] if k else None
        if not below or step['n'] != below['n'] + 1: continue
        middle = np.mean(below['contour'], 0); ring = np.array(step['contour']); edges = [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
        a, b = min(edges, key=lambda e: np.linalg.norm((e[0] + e[1]) / 2 - middle))
        riser = free_ids(project, 'contremarche', 1)[0]
        components.add(project, 'mur', riser, {'x0': f'{a[0]:.3f}', 'z0': f'{a[1]:.3f}', 'x1': f'{b[0]:.3f}', 'z1': f'{b[1]:.3f}', 'y0': f"{below['y']:.3f}",
                                                'h': f"{step['y'] - below['y']:.3f}", 'ep': '0.02', 'sens': 1.0, 'source': 'plan', 'nature': 'contremarche', 'a_priori': False}, f"Contremarche {step['n']}")
        created.append(riser)
    return created

def squared(points):
    points = np.array(points, float); n = len(points)
    flat = [next((axis for axis in (0, 1) if abs(d[axis]) < AXIS_TOL * abs(d[1 - axis])), None) for d in (points[(e + 1) % n] - points[e] for e in range(n))]
    for e in range(n):
        if flat[e] is None: continue
        run = [e, (e + 1) % n]
        while flat[run[-1]] == flat[e] and len(run) <= n: run.append((run[-1] + 1) % n)
        points[run, flat[e]] = points[run, flat[e]].mean()
    return points


def touching(plan, point, size):
    p, best = np.array(point, float), None
    for w in plan['walls']:
        a, b = np.array(w['a'], float), np.array(w['b'], float); d = b - a; t = (p - a) @ d / (d @ d); dist = np.linalg.norm(a + np.clip(t, -0.05, 1.05) * d - p)
        if dist < CONTACT_PX * size and (best is None or dist < best[0]): best = (dist, w['name'], a + t * d)
    return best


def drawn_flights(ctx, plan, stair):
    found = sketch(ctx); again = read(found['path'], plan.get('legend'), ai=False)['stairs'] if found and os.path.exists(found['path']) else []
    return min(again, key=lambda s: np.abs(np.subtract(s['box'], stair['box'])).sum(), default={'flights': [stair['box']]})['flights']


def measured_slope(points, low, high, run, top):
    inner = np.all((points[:, [0, 2]] > low + STAIR_EDGE_M) & (points[:, [0, 2]] < high - STAIR_EDGE_M), 1) & (points[:, 1] > 0.1) & (points[:, 1] < top - 0.1)
    q = points[inner][:, [2 * run, 1]]; best, rng = (0, None), np.random.default_rng(0)
    if len(q) < SLOPE_POINTS: return None
    for i, j in rng.integers(0, len(q), (2000, 2)):
        d = q[j] - q[i]
        if abs(d[0]) < 0.2 or not SLOPES[0] < abs(d[1] / d[0]) < SLOPES[1]: continue
        slope = d[1] / d[0]; count = int((np.abs(q[:, 1] - q[i, 1] - slope * (q[:, 0] - q[i, 0])) < SLOPE_TOL_M).sum())
        if count > best[0]: best = (count, (float(slope), float(q[i, 0] - q[i, 1] / slope)))
    return best[1] if best[0] >= max(SLOPE_POINTS, 0.25 * len(q)) else None


def anchors(project, ids):
    values, loaded, found = current_values(project), components.load(project), []
    for comp in (loaded[i] for i in ids):
        a, b = wall_ends(comp, values)
        found += [(axis, a[axis], sorted((a[1 - axis], b[1 - axis])), str(comp['expressions'][f"start_{'xz'[axis]}"])) for axis in (0, 1) if abs(a[axis] - b[axis]) < 1e-6]

    def tied(axis, value, low, high, touching=False):
        gap, at, expr = min(((abs(p - value), p, e) for ax, p, (lo, hi), e in found if ax == axis and min(hi, high) > max(lo, low)), default=(np.inf, 0.0, ''))
        if gap < TOUCH_M: return lambda sid: f"({expr}) {'+' if value >= at else '-'} {abs(value - at):.4f}"
        if touching or gap >= TIE_M: return None if touching else lambda sid: float(value)

        def free(sid):
            name = f"{sid}_d{'xz'[axis]}"; project.add_param(name, float(value - at), 'xz'[axis], f"Escalier {sid} — écart au mur ({'xz'[axis]})")
            return f"({expr}) + {name}"
        return free
    return tied


def add_stairs(project, stair, forward, cloud=None, height=None, tied=lambda axis, value, low, high, touching=False: None if touching else lambda sid: float(value)):
    rect = lambda box: (lambda c: (c.min(0), c.max(0)))(np.array([forward(p) for p in ((box[0], box[1]), (box[2], box[3]))]))
    rects, turn, treads = [rect(b) for b in stair.get('flights') or [stair['box']]], stair.get('turn'), stair.get('treads')
    square = rect(turn['square']) if turn else None
    if turn: rects[0] = (np.minimum(rects[0][0], square[0]), np.maximum(rects[0][1], square[1]))
    (low, high), rest = rects[0], rects[1:2]; run = int(np.argmax(high - low)); side = 1 - run; total = (height or 2.5) + SLAB_M
    if stair.get('width'): low[side] = high[side] - stair['width']
    wind = turn['winders'] if turn else 0; count = sum(treads) + wind + 1 if treads and (turn or not rest) else max(2, round(total / RISE_M)); rise = total / count; middle = (low + high) / 2
    slope = measured_slope(cloud, low, high, run, height or 2.5) if cloud is not None else None
    toward = float(np.sign((np.mean(square, 0) if turn else np.mean(rest[0], 0) if rest else middle + 1)[run] - middle[run]) or 1.0)
    sens = float(np.sign(slope[0])) if slope else toward
    start = float(np.clip(slope[1], low[run] - 0.3, high[run] + 0.3)) if slope else float((low if sens > 0 else high)[run])
    edge_run = float((square[0] if toward > 0 else square[1])[run]) if turn else float((high if sens > 0 else low)[run])
    span = float(stair['width'] if stair.get('width') else np.ptp(square, 0)[run]) if turn else 0.0; sq = edge_run if toward > 0 else edge_run - span
    length = abs(edge_run - start); other = float(np.ptp(rest[0], 0)[side]) if rest else 0.0
    going = rise / abs(slope[0]) if slope else length / treads[0] if turn else (length + other) / count
    first = treads[0] + wind if turn else min(max(2, round(length / going)), count - 2) if rest else count - 1
    axis, created = (lambda i: 'xz'[i]), []
    shift = lambda base, d: f"({base}) {'+' if d >= 0 else '-'} {abs(d):.4f}" if isinstance(base, str) else float(base + d)

    def flight(r, sens, at, edge, width, n, going, y0, **extra):
        made = free_ids(project, 'escalier', 1)[0]; at, edge = at(made), edge(made)
        components.add(project, 'escalier', made, {'axe': axis(r), axis(r): at, axis(1 - r): edge, 'g': float(going), 'l': float(width), 'n': int(n), 'r': float(rise), 'sens': sens,
                                                   'y0': float(y0), 'source': 'plan', **extra}, 'Escalier (suite)' if created else 'Escalier')
        project.db.executemany('update obs set value=?, sigma=coalesce(?, sigma) where id=?', [(float(going), slope and GOING_SIGMA_M, f'comp:{made}:{made}_g'), (float(rise), None, f'comp:{made}:{made}_r')]); project.db.commit()
        created.append(made); return at, edge
    wall = tied(side, high[side], low[run], high[run], True); bend = float(np.sign(np.mean(rest[0], 0)[side] - middle[side]) or 1.0) if rest else 1.0
    band_edge = (lambda sid: f"({wall(sid)}) - {sid}_l") if wall else tied(side, low[side], low[run], high[run])
    if turn and sens != toward:
        far, near = float((rest[0][1] if bend > 0 else rest[0][0])[side]), float((high if bend > 0 else low)[side])
        flight(side, -bend, lambda sid: far, lambda sid: sq, span, treads[1] + wind + 1, abs(far - near) / treads[1], 0.0, tournant=wind, tour=-toward)
        flight(run, -toward, tied(run, edge_run, low[side], high[side]), band_edge, high[side] - low[side], treads[0] + 1, abs(float((high if toward < 0 else low)[run]) - edge_run) / treads[0], (treads[1] + wind) * rise)
        return created
    at, edge = flight(run, sens, tied(run, start, low[side], high[side]), band_edge, high[side] - low[side], first + 1, going, 0.0, **({'tournant': wind, 'tour': bend} if turn else {}))
    if rest:
        (a, b) = rest[0]; n = count - first - 1 if turn else count - first; begin = edge if bend < 0 else f"({edge}) + {created[0]}_l" if isinstance(edge, str) else float(high[side])
        square_at = (lambda sid: f"({at}) + {treads[0]} * {created[0]}_g" if toward > 0 else f"({at}) - {treads[0]} * {created[0]}_g - {created[0]}_l") if turn else lambda sid: shift(at, a[run] - start)
        flight(side, bend, lambda sid: begin, square_at, span if turn else b[run] - a[run], n + 1 if turn else n,
               abs((b if bend > 0 else a)[side] - (high if bend > 0 else low)[side]) / max(1, n), first * rise)
    return created


def identities(project):
    loaded, values = components.load(project), current_values(project)
    hosted = [(c['type'], loaded[c['options']['mur']]['label'], components.evaluate(c['expressions']['pos'], values), c['id']) for c in loaded.values() if c['options'].get('mur') in loaded]
    return hosted, {(c['type'], c['label']): c['id'] for c in loaded.values() if c['options'].get('source') == 'plan'}, {c['id']: c['label'] for c in loaded.values()}


def reattached(project, measured, before):
    from . import mesures
    (hosted, named, labels), (now_hosted, now_named, present) = before, identities(project); moved = {named[k]: now_named[k] for k in named if k in now_named}
    for _, old, new in sorted((abs(p - q), a, b) for t, h, p, a in hosted for u, g, q, b in now_hosted if (t, h) == (u, g) and abs(p - q) < SAME_M):
        if old not in moved and new not in moved.values(): moved[old] = new
    replaced, names, taken = {a for *_, a in hosted} | set(named.values()), set(project.params()), {row['id'] for row in read_cotes(project)}
    fresh = {c['key']: c for c in mesures.candidates(mesures.elements(project, current_values(project)))}
    owner = lambda name: max((cid for cid in labels if name.startswith(f'{cid}_')), key=len, default=None)
    lost = lambda cids: sorted({labels[c] for c in cids if c and c not in moved and (c in replaced or c not in present)})
    kept, dropped = [], []
    for row in (r for r in measured if r['id'] not in taken):
        kind, _, rest = row['id'].partition('.'); parts = [moved.get(p, p) for p in rest.split('.')]; key = f"{kind}.{'.'.join(sorted(parts) if kind == 'entre' else parts)}"
        expr = components.NAME.sub(lambda m: moved[cid] + m[0][len(cid):] if (cid := owner(m[0])) in moved else m[0], row['expr'])
        gone = lost(p for p in rest.split('.') if p in labels) if key in fresh else lost(owner(n) for n in components.NAME.findall(row['expr']))
        if key in fresh and not gone: kept.append({**row, 'id': key, 'expr': fresh[key]['expr'], 'label': fresh[key]['label']})
        elif not gone and set(components.NAME.findall(expr)) - set(FUNCS) <= names: kept.append({**row, 'expr': expr})
        else: dropped.append({'id': row['id'], 'label': row.get('label') or row['id'], 'reason': f"{', '.join(gone)} n'existe plus" if gone else "ses éléments n'existent plus"})
    if kept: write_cotes(project, read_cotes(project) + kept)
    return kept, dropped


@rpc('plan.apply')
def apply(ctx, plan):
    project = ctx.require_project(); source = photo_source(project); fitted = plan.get('fit') if source else None; db = project.db
    stale, measured, before = [c['id'] for c in components.load(project).values() if c['type'] == 'mur' and c['options'].get('source') not in walls.KEPT or c['options'].get('source') == 'plan'], read_cotes(project), identities(project)
    if stale: components.remove(project, *stale)
    if source:
        photos.ensure_params(project, source); _, state, cache = photos.photo_state(project, source, compute=False)
        db.execute('delete from photos_assoc where source=?', (source,)); db.execute('delete from photos_axes where source=?', (source,))
        db.executemany('update params set init=? where name=?', [(state['s_est'], f'{source}_s'), (0.0, f'{source}_yaw'), *((0.0, f'{source}_t{a}') for a in 'xyz')]); db.commit()
        if os.path.exists(model := os.path.join(project.folder, 'model.json')): os.remove(model)
        photos.invalidate(project, source)
    forward, _ = frame(plan, fitted['k'] if fitted else plan.get('k') or default_k(plan)); shape = {w['name']: w for w in (fitted or {}).get('walls', []) + (fitted or {}).get('extra', [])}
    ring = [{**w, 'a': shape[w['name']]['a'], 'b': shape[w['name']]['b']} if w['name'] in shape else w for w in plan['walls']]
    ids = trace(ctx, squared([forward(w['a']) for w in ring]).tolist(), True, sigma=PRIOR_M, source='plan', names=[w['name'] for w in ring], height=fitted and fitted.get('height'))['created']
    named, size, contacts, partitions = dict(zip([w['name'] for w in ring], ids)), float(np.ptp(np.array([w['a'] for w in ring], float), 0).max()), [], []
    for extra in ({**e, **shape.get(e['name'], {})} for e in plan.get('extra', []) if e.get('name')):
        ends = {end: touching({'walls': ring}, extra[end], size) for end in 'ab'}
        eid = trace(ctx, squared([forward(ends[end][2] if ends[end] else extra[end]) for end in 'ab']).tolist(), False, sigma=PRIOR_M, source='plan', names=[extra['name']], height=f'{ids[0]}_h')['created'][0]
        partitions.append(eid); contacts += [(eid, end, found[1]) for end, found in ends.items() if found]
    loaded = components.load(project)
    for eid, end, wall_name in contacts:
        wall, comp = loaded[named[wall_name]]['expressions'], loaded[eid]['expressions']; x0, z0, x1, z1 = (f'({wall[k]})' for k in ('start_x', 'start_z', 'end_x', 'end_z'))
        px, pz = (f"({comp[('start' if end == 'a' else 'end') + '_' + axis]})" for axis in 'xz'); w = next(w for w in ring if w['name'] == wall_name)
        length = float(np.hypot(*(forward(w['b']) - forward(w['a']))))
        db.execute('insert or replace into obs values (?,?,?,?,?,?,?)', (f'comp:{eid}:contact_{end}', f'comp:{eid}', f'(({x1} - {x0}) * ({pz} - {z0}) - ({z1} - {z0}) * ({px} - {x0})) / {length:.6f}',
                                                                    0.0, CONTACT_SIGMA_M, components.PRIOR_STATUS, f"{loaded[eid]['label']} contre le mur {wall_name}"))
    values = current_values(project); length = lambda name: components.measure_expr((loaded[named[name]], 'a'), (loaded[named[name]], 'b'), values=values)
    for rule in (r for r in plan.get('rules', []) if r.get('ok') and r['walls'][0] in named and r['walls'][2] in named):
        first, last = rule['walls'][0], rule['walls'][2]
        db.execute('insert or replace into obs values (?,?,?,?,?,?,?)', (f'comp:{named[first]}:egal_{last}', f'comp:{named[first]}', f'({length(first)}) - ({length(last)})', 0.0, RULE_SIGMA_M,
                                                                    components.PRIOR_STATUS, f"Mur {first} = mur {last} (pan coupé symétrique)"))
    db.commit()
    pairs, clicks = None, []
    if fitted:
        walls.set_pose(ctx, project, source, fitted['angle'], 1.0, np.array(fitted['shift'])); loaded = components.load(project); named = dict(zip([w['name'] for w in ring], ids))
        line = lambda comp: photos.LINE + ';'.join(str(comp['expressions'][f'{end}_{axis}']) for end in ('start', 'end') for axis in 'xz')
        floor, ceiling, _ = walls.levels(state); pairs = sorted(n for n in fitted.get('planes', {}) if n in named)
        rows = [(source, plane, line(loaded[named[n]]), None) for n in pairs for plane in fitted['planes'][n]] + [(source, floor, '0', None)] * bool(floor) + [(source, ceiling, f'{ids[0]}_h', None)] * bool(ceiling)
        db.executemany('insert or replace into photos_assoc values (?,?,?,?)', rows)
        clicks = [(source, c['image'], float(c['u']), float(c['v']), f"{ids[p['corner']]}:{ANCHORS[c.get('kind', 'arete')]}") for p in plan.get('points', []) if not p.get('skipped') and len(p.get('clicks', [])) == 2 for c in p['clicks']]
        db.executemany('insert into photos_clicks(source, image, u, v, anchor) values (?,?,?,?,?)', clicks); db.commit(); photos.invalidate(project, source)
    names, placed, masters = {w['name']: w for w in ring}, [], {}
    for opening in plan.get('openings', []):
        w = names.get(opening['wall'])
        if not w: continue
        a, b = forward(w['a']), forward(w['b']); middle = (opening['t0'] + opening['t1']) / 2; group = opening['type'] if opening.get('same') else None
        try: made = place(ctx, opening['type'], (a + middle * (b - a)).tolist(), width=max(0.4, (opening['t1'] - opening['t0']) * np.hypot(*(b - a))), sigma=0.1 if opening.get('seen') else PRIOR_M, share=masters.get(group))['id']
        except ValueError: continue
        placed.append(made)
        if group: masters.setdefault(group, made)
    stairs = []
    for stair, width in zip(plan.get('stairs', []), (fitted or {}).get('stairs') or [None] * len(plan.get('stairs', []))):
        if not stair.get('kept', True): continue
        stair = {**stair, 'width': width}; steps = measured_steps(cache, fitted['pose'], *stair_region(stair['box'], forward, 0.1), fitted.get('height') or 2.5) if fitted else []
        cloud = photos.load_model(cache)['P'] @ np.array(fitted['pose']['L']).T + fitted['pose']['c'] if fitted else None
        stairs += add_measured_stairs(project, steps) if len(steps) >= 4 else add_stairs(project, {**stair, 'flights': stair.get('flights') or drawn_flights(ctx, plan, stair)}, forward, cloud,
                                                                                         fitted and fitted.get('height'), anchors(project, ids))
    drawing_scale(project); kept, dropped = reattached(project, measured, before)
    return {'walls': ids + partitions, 'pairs': pairs, 'openings': placed, 'stairs': stairs, 'clicks': len(clicks), 'kept': [row['id'] for row in kept], 'dropped': dropped}
