import hashlib, json, os, sys, threading, time, urllib.request, zipfile
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from . import photos
from .rpc import RpcError, rpc

FOLDER = os.environ.get('PLANS3D_IA') or os.path.join(os.environ.get('LOCALAPPDATA') or os.path.expanduser('~'), '3D_Plans', 'ia')
FILES = {
    'onnxruntime': ('https://files.pythonhosted.org/packages/88/ea/33814eb0ec96775eda4c1d30b0d86e91d7d2cd0d84c66d3915aef0e06fa3/onnxruntime_directml-1.24.4-cp312-cp312-win_amd64.whl',
                    'f2ecb68b7b7b259d2ef3112ae760149f9b5a1e7c0fbb73d539da6250a648a614', 25111930),
    'segformer': ('https://huggingface.co/Xenova/segformer-b5-finetuned-ade-640-640/resolve/02e76595880a97cf256ceecb93acbc070e21fe97/onnx/model_fp16.onnx',
                  '19461294738e6d21b7bb64d66b4e1c3c77ec8f74c7e87c42acdb89b27bd19ebd', 171918052),
}
CLASSES = [
    ('inconnu', 'Inconnu', '#5f6670', ()), ('mur', 'Mur', '#b9b3a8', (0, 42)), ('sol', 'Sol', '#8d6b4a', (3, 28)), ('plafond', 'Plafond', '#ece7dc', (5,)),
    ('porte', 'Porte', '#e0892e', (14, 58)), ('fenetre', 'Fenêtre', '#4aa8e8', (8,)), ('rideau', 'Rideau, store', '#9d72c9', (18, 63)),
    ('tableau', 'Tableau, affiche', '#d1a92a', (22, 43, 100, 123, 144, 148)), ('radiateur', 'Radiateur', '#c9534f', (146,)),
    ('lit', 'Lit', '#e85d5d', (7, 57, 117, 131)), ('canape', 'Canapé', '#d86aa5', (23, 39, 97)), ('fauteuil', 'Fauteuil', '#b4568c', (30,)),
    ('chaise', 'Chaise', '#eaa53b', (19, 31, 75, 110)), ('table', 'Table', '#a87b3b', (15, 45, 56, 64, 70, 73, 77)), ('bureau', 'Bureau', '#7d5a2b', (33,)),
    ('ecran', 'Écran, TV', '#3fd3d3', (74, 89, 130, 141, 143)), ('miroir', 'Miroir', '#a6dcec', (27,)), ('lampe', 'Lampe, luminaire', '#ffd84a', (36, 82, 85, 87, 134)),
    ('armoire', 'Armoire, meuble', '#6d8f4b', (10, 35, 44, 55, 99)), ('etagere', 'Étagère', '#4b8f6d', (24, 62)), ('escalier', 'Escalier', '#8e8e8e', (53, 59, 95, 96, 121)),
    ('sanitaire', 'Cuisine, sanitaire', '#7cb4c4', (37, 47, 49, 50, 65, 71, 107, 118, 124, 129, 133, 145)), ('plante', 'Plante', '#4caf50', (17, 66, 125, 135)),
    ('personne', 'Personne, animal', '#ff8a80', (12, 126)), ('objet', 'Objet', '#9c907f', ()), ('ciel', 'Ciel', '#8fd3f4', (2,)), ('arbre', 'Arbre', '#2e7d32', (4, 72)),
    ('herbe', 'Herbe', '#8bc34a', (9, 29)), ('sol_ext', 'Sol extérieur', '#a1887f', (6, 11, 13, 16, 34, 46, 52, 54, 68, 91, 94)),
    ('batiment', 'Bâtiment', '#bcaaa4', (1, 25, 48, 51, 61, 79, 84, 86, 88, 106, 114, 140)), ('banc', 'Banc', '#8d6e63', (69,)),
    ('vehicule', 'Véhicule', '#607d8b', (20, 76, 80, 83, 90, 102, 103, 116, 127)), ('eau', 'Eau', '#1e88e5', (21, 26, 60, 104, 109, 113, 128)),
    ('cloture', 'Clôture, poteau', '#795548', (32, 38, 93, 136))]
KEYS = [c[0] for c in CLASSES]
INDEX = {k: i for i, k in enumerate(KEYS)}
ADE = np.full(150, INDEX['objet'], np.uint8)
for i, (_, _, _, ade) in enumerate(CLASSES): ADE[list(ade)] = i
PALETTE = np.array([[int(c[2][i:i + 2], 16) for i in (1, 3, 5)] for c in CLASSES], np.uint8)
ids = lambda *names: np.array([INDEX[n] for n in names])
WALL_POINTS, FLOOR_EDGE, CEILING_EDGE = ids('mur', 'porte', 'tableau'), ids('mur', 'porte'), ids('mur')
OUTSIDE, FURNITURE = ids('ciel', 'arbre', 'herbe', 'sol_ext'), ids('lit', 'canape', 'fauteuil', 'chaise', 'table', 'bureau', 'ecran', 'armoire', 'etagere', 'sanitaire', 'miroir')
ON_WALL = ids('mur', 'porte', 'fenetre', 'tableau', 'radiateur', 'rideau')
BACKGROUND = ids('mur', 'sol', 'plafond', 'porte', 'fenetre', 'rideau', 'tableau', 'radiateur', 'ciel', 'arbre', 'herbe', 'sol_ext', 'batiment', 'personne')
SIZE, VERSION, MAX_DENSE = 768, 1, 1_500_000
CELL_M, BAND_M, LINE_M, NMS_M, SUPPORT_M, RANGE_M, WALL_CELL_M, REFINE_M, ASSOC_M, MAX_SLOPE = 0.05, 0.04, 0.02, 0.3, 0.4, 8.0, 0.05, 0.4, 0.05, 0.1
LOCK, COMPUTE, SESSION, KEEP_M = threading.Lock(), threading.RLock(), {}, 0.1
PUSH_M, PUSH_MIN, PUSH_SHARE, PARALLEL_SLOPE = 0.5, 20, 0.4, 0.05


def path(name): return os.path.join(FOLDER, name + os.path.splitext(FILES[name][0])[1])
def runtime(): return os.path.join(FOLDER, 'onnxruntime-1.24.4')
def installed(): return os.path.isdir(os.path.join(runtime(), 'onnxruntime')) and os.path.exists(path('segformer'))


def fetch(name, progress, start, end):
    url, sha256, size = FILES[name]; target, digest, done = path(name), hashlib.sha256(), 0
    os.makedirs(FOLDER, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(target + '.part', 'wb') as out:
            while chunk := response.read(1 << 20):
                out.write(chunk); digest.update(chunk); done += len(chunk)
                progress(start + (end - start) * min(done / size, 1), f'téléchargement du modèle : {done >> 20} / {size >> 20} Mo')
    except OSError as error: raise RpcError(f'téléchargement impossible ({name}) : {getattr(error, "reason", error)}') from None
    if digest.hexdigest() != sha256: os.remove(target + '.part'); raise RpcError(f'fichier {name} corrompu (empreinte SHA-256), réessayez')
    os.replace(target + '.part', target)


def install(progress=photos.no_progress):
    with LOCK:
        if not os.path.isdir(os.path.join(runtime(), 'onnxruntime')):
            if not os.path.exists(path('onnxruntime')): fetch('onnxruntime', progress, 0, 0.12)
            with zipfile.ZipFile(path('onnxruntime')) as wheel: wheel.extractall(runtime() + '.tmp')
            os.replace(runtime() + '.tmp', runtime()); os.remove(path('onnxruntime'))
        if not os.path.exists(path('segformer')): fetch('segformer', progress, 0.12, 1.0)


def session():
    if 'seg' not in SESSION:
        if runtime() not in sys.path: sys.path.insert(0, runtime())
        import onnxruntime as ort
        options = ort.SessionOptions(); options.log_severity_level = 3; options.enable_mem_pattern = False
        providers = [p for p in ('DmlExecutionProvider', 'CPUExecutionProvider') if p in ort.get_available_providers()]
        try: SESSION['seg'] = ort.InferenceSession(path('segformer'), options, providers=providers)
        except Exception: SESSION['seg'] = ort.InferenceSession(path('segformer'), options, providers=['CPUExecutionProvider'])
    return SESSION['seg']


def device(): return 'carte graphique' if 'DmlExecutionProvider' in session().get_providers() else 'processeur'


def segment(image_path, orientation=1):
    from PIL import Image
    turn = {3: 2, 6: -1, 8: 1}.get(int(orientation), 0); image = Image.open(image_path).convert('RGB'); w, h = image.size; f = SIZE / max(w, h)
    pixels = np.rot90(np.asarray(image.resize((max(32, round(w * f / 32) * 32), max(32, round(h * f / 32) * 32)), Image.BILINEAR), np.float32), turn)
    x = ((pixels / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
    s = session(); logits = s.run(None, {s.get_inputs()[0].name: x})[0][0]
    return np.rot90(logits.argmax(0).astype(np.uint8), -turn)


def folder(cache, *parts): return os.path.join(cache, 'semantique', *parts)
def map_path(cache, name): return folder(cache, 'cartes', hashlib.sha1(str(name).encode()).hexdigest()[:16] + '.png')


def read_map(cache, name):
    from PIL import Image
    file = map_path(cache, name)
    return ADE[np.asarray(Image.open(file))] if os.path.exists(file) else None


def segment_all(meta, cache, model, progress=photos.no_progress, start=0.0, end=1.0):
    from PIL import Image
    os.makedirs(folder(cache, 'cartes'), exist_ok=True); names = [str(n) for n in model['names']]; todo = [i for i, n in enumerate(names) if not os.path.exists(map_path(cache, n))]
    for k, i in enumerate(todo):
        if cache in photos.STOPPED: raise photos.Stopped('calcul arrêté')
        Image.fromarray(segment(os.path.join(meta['root'], names[i]), model['orientation'][i])).save(map_path(cache, names[i]))
        progress(start + (end - start) * (k + 1) / len(todo), f'photo {k + 1} / {len(todo)} ({device()})')
    return len(todo)


def pixels(model, i, points):
    local = points @ photos.rotation_matrices(model['q'][i:i + 1])[0].T + model['t'][i]; fx, fy, cx, cy, k1, k2, p1, p2 = model['K'][i]
    depth = local[:, 2]; safe = np.where(depth > 1e-9, depth, 1.0); x, y = local[:, 0] / safe, local[:, 1] / safe; r2 = x * x + y * y; radial = 1 + k1 * r2 + k2 * r2 * r2
    return fx * (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + cx, fy * (y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + cy, depth


def lookup_labels(model, i, labels, points):
    u, v, depth = pixels(model, i, points); (width, height), (rows, cols) = model['wh'][i], labels.shape
    c, r = np.floor(u * cols / width).astype(int), np.floor(v * rows / height).astype(int); ok = (depth > 0) & (c >= 0) & (c < cols) & (r >= 0) & (r < rows)
    return np.where(ok, labels[r.clip(0, rows - 1), c.clip(0, cols - 1)], 0)


def vote(model, maps, points, owner, image):
    found = np.zeros(len(owner), np.uint8)
    for i in np.unique(image):
        if maps[i] is None: continue
        at = np.flatnonzero(image == i); found[at] = lookup_labels(model, i, maps[i], points[owner[at]])
    keep = found > 0; key = owner[keep].astype(np.int64) * len(KEYS) + found[keep]; unique, count = np.unique(key, return_counts=True)
    point, label = unique // len(KEYS), unique % len(KEYS); order = np.lexsort((-count, point)); point, label, count = point[order], label[order], count[order]
    first = np.r_[True, point[1:] != point[:-1]]; labels, share = np.zeros(len(points), np.uint8), np.zeros(len(points), np.float32)
    total = np.bincount(owner[keep], minlength=len(points)); labels[point[first]] = label[first]; share[point[first]] = count[first] / total[point[first]]
    return labels, share


def read_visibility(file):
    data = np.fromfile(file, np.uint32); count, at = int(data[0]) | int(data[1]) << 32, 2; sizes = np.empty(count, np.int64)
    for k in range(count): sizes[k] = data[at]; at += int(sizes[k]) + 1
    starts = 3 + np.r_[0, np.cumsum(sizes[:-1] + 1)]; offsets = np.arange(sizes.sum()) - np.repeat(np.r_[0, np.cumsum(sizes[:-1])], sizes)
    return np.repeat(np.arange(count), sizes), data[np.repeat(starts, sizes) + offsets].astype(np.int64)


def dense_cameras(cache):
    sparse = os.path.join(cache, 'dense', 'sparse')
    if not os.path.exists(os.path.join(sparse, 'images.txt')):
        photos.run_colmap(['model_converter', '--input_path', sparse, '--output_path', sparse, '--output_type', 'TXT'], os.path.join(cache, 'convert.log'), photos.no_progress, 0, 0, '')
    lines = [line for line in open(os.path.join(sparse, 'images.txt'), encoding='utf-8') if line[0] != '#']
    return np.array([line.split()[5:8] for line in lines[::2] if line.strip()], float)


def dense_signature(cache):
    file = os.path.join(cache, 'dense', 'fused.ply')
    return f'{os.path.getsize(file)}:{os.path.getmtime(file)}' if os.path.exists(file) and os.path.exists(file + '.vis') else ''


def label_points(meta, cache, model):
    maps = [read_map(cache, n) for n in model['names']]; owner = np.repeat(np.arange(len(model['P'])), np.diff(model['Tptr']))
    labels, share = vote(model, maps, model['P'], owner, model['Timg']); points, parts = [model['P']], [(labels, share)]
    if dense_signature(cache):
        dense, _ = photos.read_ply(os.path.join(cache, 'dense', 'fused.ply')); owner, image = read_visibility(os.path.join(cache, 'dense', 'fused.ply.vis'))
        if len(dense) > MAX_DENSE:
            kept = np.zeros(len(dense), bool); kept[np.random.default_rng(0).permutation(len(dense))[:MAX_DENSE]] = True; use = kept[owner]
            owner, image, dense = (np.cumsum(kept) - 1)[owner[use]], image[use], dense[kept]
        parts.append(vote(model, maps, dense, owner, cKDTree(model['t']).query(dense_cameras(cache))[1][image])); points.append(dense)
    return np.vstack(points).astype(np.float32), np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def corrections(cache):
    file = folder(cache, 'corrections.json')
    return photos.load_json(file) if os.path.exists(file) else []


def grow(points, labels, start, step):
    tree, same, todo, seen = cKDTree(points), labels == labels[start], [start], {start}
    while todo and len(seen) < 200000:
        near = {j for group in tree.query_ball_point(points[todo], step) for j in group if same[j]} - seen; seen |= near; todo = list(near)
    return np.fromiter(seen, int)


def apply_corrections(points, labels, cache, step):
    labels = labels.copy()
    for fix in corrections(cache):
        start = int(cKDTree(points).query(fix['point'])[1])
        if KEYS[labels[start]] == fix['from']: labels[grow(points, labels, start, step)] = INDEX[fix['to']]
    return labels


def points_signature(cache, model):
    count = len(os.listdir(folder(cache, 'cartes'))) if os.path.isdir(folder(cache, 'cartes')) else 0
    return f'{VERSION}:{count}:{len(model["P"])}:{dense_signature(cache)}'


def labelled(project, source):
    meta, frame, cache = photos.photo_state(project, source, compute=False); model = photos.load_model(cache); file = folder(cache, 'points.npz')
    if not os.path.isdir(folder(cache, 'cartes')): return None
    signature, data = points_signature(cache, model), None
    with COMPUTE:
        if os.path.exists(file):
            with np.load(file) as saved:
                if str(saved['signature']) == signature: data = dict(saved)
        if data is None:
            points, labels, share = label_points(meta, cache, model); data = {'P': points, 'L': labels, 'S': share, 'signature': signature}; np.savez(file, **data)
    step = 0.06 / frame['s_est']
    return {'P': data['P'], 'L': apply_corrections(data['P'], data['L'], cache, step), 'S': data['S'], 'frame': frame, 'cache': cache, 'model': model, 'meta': meta}


def classify(project, source, points):
    data = labelled(project, source)
    if data is None: return None
    distance, at = cKDTree(data["P"]).query(np.asarray(points, float), distance_upper_bound=0.08 / data['frame']['s_est'])
    return np.where(np.isfinite(distance), data['L'][np.minimum(at, len(data['L']) - 1)], 0)


def to_frame(frame, points): return (np.asarray(points, float) @ np.array(frame['R']).T + frame['t']) * frame['s_est']


def rays(model, i, frame, rows_cols, mask=None, r=None, c=None):
    if mask is not None: r, c = np.nonzero(mask)
    (width, height), (rows, cols) = model['wh'][i], rows_cols
    x, y = photos.undistort(model['K'][i], (c + 0.5) * width / cols, (r + 0.5) * height / rows)
    rotation = np.array(frame['R']) @ photos.rotation_matrices(model['q'][i:i + 1])[0].T
    directions = np.stack([x, y, np.ones_like(x)], 1) @ rotation.T
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def at_level(origin, directions, plane):
    a, b, c = (0.0, 0.0, plane) if np.isscalar(plane) else plane
    with np.errstate(divide='ignore', invalid='ignore'): t = (a * origin[0] + b * origin[2] + c - origin[1]) / (directions[:, 1] - a * directions[:, 0] - b * directions[:, 2])
    ok = (t > 0) & (t < RANGE_M)
    return origin + t[ok, None] * directions[ok], ok


def level_at(plane, xz): return plane[0] * xz[..., 0] + plane[1] * xz[..., 1] + plane[2]


def fit_plane(p, minimum):
    if len(p) < minimum: return None
    A, w = np.c_[p[:, 0], p[:, 2], np.ones(len(p))], np.ones(len(p))
    for _ in range(10):
        coef = np.linalg.lstsq(A * w[:, None], p[:, 1] * w, rcond=None)[0]; w = 1 / np.maximum(np.abs(p[:, 1] - A @ coef), 0.02)
    if np.abs(coef[:2]).max() > MAX_SLOPE: coef = np.array([0.0, 0.0, np.median(p[:, 1])])
    return coef


def ceiling_samples(data, centers, floor):
    model, frame, found = data['model'], data['frame'], []
    for i, name in enumerate(model['names']):
        raw = read_map(data['cache'], name)
        if raw is None: continue
        turn = {3: 2, 6: -1, 8: 1}.get(int(model['orientation'][i]), 0); labels = np.rot90(raw, turn); rows = np.arange(labels.shape[0])[:, None]
        top = np.where(np.isin(labels, ids('plafond')) & ndimage.binary_dilation(np.isin(labels, CEILING_EDGE)), rows, -1).max(0)
        low = np.where(np.isin(labels, ids('sol')) & ndimage.binary_dilation(np.isin(labels, FLOOR_EDGE)) & (rows > top), rows, labels.shape[0]).min(0)
        wall = np.vstack([np.zeros((1, labels.shape[1])), np.cumsum(np.isin(labels, FLOOR_EDGE), 0)]); cols = np.arange(labels.shape[1])
        ok = (top >= 0) & (low < labels.shape[0]) & (low - top > 2)
        ok[ok] &= (wall[low[ok], cols[ok]] - wall[top[ok] + 1, cols[ok]]) >= 0.9 * (low[ok] - top[ok] - 1)
        if not ok.any(): continue
        index = np.rot90(np.indices(raw.shape), turn, axes=(1, 2))
        pick = lambda r: (index[0][r, cols[ok]], index[1][r, cols[ok]])
        down, up = (rays(model, i, frame, raw.shape, None, *pick(r)) for r in (low[ok], top[ok]))
        hit, kept = at_level(centers[i], down, floor); up = up[kept]; reach = np.linalg.norm(hit[:, [0, 2]] - centers[i, [0, 2]], axis=1)
        flat = np.linalg.norm(up[:, [0, 2]], axis=1); good = flat > 1e-6
        found.append(centers[i] + up[good] * (reach[good] / flat[good])[:, None])
    return np.vstack(found) if found else np.zeros((0, 3))


def levels(data):
    if 'levels' in data: return data['levels']
    frame = data['frame']; points, centers = to_frame(frame, data['P']), to_frame(frame, photos.camera_centers(data['model'])); middle = centers[:, [0, 2]].mean(0)
    floor = fit_plane(points[(data['L'] == INDEX['sol']) & (np.abs(points[:, 1]) < 0.3)], 50)
    floor = np.zeros(3) if floor is None else floor; above = np.vstack([ceiling_samples(data, centers, floor), points[data['L'] == INDEX['plafond']]])
    ceiling = fit_plane(above, 20)
    if ceiling is not None and np.abs(ceiling[:2] - floor[:2]).max() > PARALLEL_SLOPE: ceiling = np.array([*floor[:2], np.median(above[:, 1] - floor[0] * above[:, 0] - floor[1] * above[:, 2])])
    flat = [p for p in frame['planes'] if p['axis'] == 'y' and p['offset'] * frame['s_est'] > 1.8]
    if ceiling is None and flat: ceiling = np.array([0.0, 0.0, max(flat, key=lambda p: p['n'])['offset'] * frame['s_est']])
    data['levels'] = floor, ceiling, None if ceiling is None else float(level_at(ceiling, middle) - level_at(floor, middle))
    return data['levels']


def evidence(data):
    frame, model = data['frame'], data['model']; floor, ceiling, height = levels(data)
    points, centers = to_frame(frame, data['P']), to_frame(frame, photos.camera_centers(model)); above = points[:, 1] - level_at(floor, points[:, [0, 2]])
    wall = np.isin(data['L'], WALL_POINTS) & (above > 0.2) & (above < (height or 2.4) - 0.1)
    samples, owners = [points[wall][:, [0, 2]]], [np.full(wall.sum(), -1)]
    for i, name in enumerate(model['names']):
        labels = read_map(data['cache'], name)
        if labels is None: continue
        edges = [(np.isin(labels, ids('sol')) & ndimage.binary_dilation(np.isin(labels, FLOOR_EDGE)), floor)]
        if height: edges.append((np.isin(labels, ids('plafond')) & ndimage.binary_dilation(np.isin(labels, CEILING_EDGE)), ceiling))
        for mask, plane in edges:
            if mask.any():
                hit, _ = at_level(centers[i], rays(model, i, frame, labels.shape, mask), plane); samples.append(hit[:, [0, 2]]); owners.append(np.full(len(hit), i))
    return np.vstack(samples), np.concatenate(owners), centers, height, points


def carve(samples, centers, owners, low, shape):
    grid = np.zeros(shape, np.int32); rng = np.random.default_rng(0); pick = rng.permutation(len(samples))[:60000]
    sources = np.where(owners[pick, None] >= 0, centers[np.maximum(owners[pick], 0)][:, [0, 2]], np.nan)
    missing = np.isnan(sources[:, 0])
    if missing.any():
        nearest = cKDTree(centers[:, [0, 2]]).query(samples[pick][missing])[1]; sources[missing] = centers[nearest][:, [0, 2]]
    for t in np.linspace(0.02, 0.95, 40):
        cells = np.floor((sources + t * (samples[pick] - sources) - low) / CELL_M).astype(int); ok = np.all((cells >= 0) & (cells < shape), 1)
        np.add.at(grid, (cells[ok, 0], cells[ok, 1]), 1)
    return grid


def peaks(samples, axis, low, high):
    other = 1 - axis; n_pos, n_run = int(np.ceil((high[axis] - low[axis]) / LINE_M)) + 1, int(np.ceil((high[other] - low[other]) / 0.1)) + 1
    grid = np.zeros((n_pos, n_run)); np.add.at(grid, (((samples[:, axis] - low[axis]) / LINE_M).astype(int).clip(0, n_pos - 1), ((samples[:, other] - low[other]) / 0.1).astype(int).clip(0, n_run - 1)), 1)
    window = ndimage.uniform_filter1d(grid, 3, axis=0) * 3; support = (window >= 2).sum(1) * 0.1; found = []
    for k in np.argsort(-support):
        if support[k] < SUPPORT_M: break
        if all(abs(k - j) * LINE_M >= NMS_M for j, _ in found): found.append((k, support[k]))
    lines = []
    for k, strength in found:
        near = np.abs(samples[:, axis] - (low[axis] + (k + 0.5) * LINE_M)) < BAND_M
        lines.append((float(np.median(samples[near, axis])), float(strength)))
    return lines


def loops(inside, xs, zs):
    edges = {}
    for i in range(inside.shape[0]):
        for j in range(inside.shape[1]):
            if not inside[i, j]: continue
            for (di, dj), (a, b) in (((-1, 0), ((i, j + 1), (i, j))), ((1, 0), ((i + 1, j), (i + 1, j + 1))), ((0, -1), ((i, j), (i + 1, j))), ((0, 1), ((i + 1, j + 1), (i, j + 1)))):
                ni, nj = i + di, j + dj
                if not (0 <= ni < inside.shape[0] and 0 <= nj < inside.shape[1] and inside[ni, nj]): edges[a] = b
    if not edges: return []
    start = min(edges); ring, at = [start], edges[start]
    while at != start and len(ring) <= len(edges): ring.append(at); at = edges[at]
    corners = [p for k, p in enumerate(ring) if (ring[k - 1][0] == p[0]) != (p[0] == ring[(k + 1) % len(ring)][0])]
    return [[xs[a], zs[b]] for a, b in corners]


def fractions(table, low, xs, zs):
    at = lambda values, axis: np.clip(np.round((np.asarray(values) - low[axis]) / CELL_M).astype(int), 0, table.shape[axis] - 1)
    i, j = at(xs, 0), at(zs, 1); sums = table[i][:, j]; area = np.maximum(np.outer(np.diff(i), np.diff(j)), 1)
    return (sums[1:, 1:] - sums[:-1, 1:] - sums[1:, :-1] + sums[:-1, :-1]) / area


def mismatch(table, low, xs, zs):
    share, area = fractions(table, low, xs, zs), np.outer(np.diff(xs), np.diff(zs))
    return float((np.where(share > 0.5, 1 - share, share) * area).sum() + table[-1, -1] * CELL_M ** 2 - (share * area).sum())


def prune(table, low, lines, tolerance):
    while True:
        base, best = mismatch(table, low, *lines), None
        for axis in (0, 1):
            for k in range(len(lines[axis])) if len(lines[axis]) > 2 else ():
                trial = [lines[0], lines[1]]; trial[axis] = lines[axis][:k] + lines[axis][k + 1:]; cost = mismatch(table, low, *trial) - base
                if best is None or cost < best[0]: best = (cost, trial)
        if best is None or best[0] > tolerance: return lines
        lines = best[1]


def wall_planes(data, points):
    frame = data['frame']; scale = frame['s_est']; found = []
    from .walls import plane_groups
    for group in plane_groups(frame['planes'], scale):
        k = 'xyz'.index(group[0]['axis']); run, offset = 2 - k, group[0]['offset'] * scale
        low, high = np.min([p['bbox'][0] for p in group], 0) * scale, np.max([p['bbox'][1] for p in group], 0) * scale
        near = (np.abs(points[:, k] - offset) < 0.03) & (points[:, run] >= low[run]) & (points[:, run] <= high[run]) & (points[:, 1] >= low[1]) & (points[:, 1] <= high[1])
        counts = np.bincount(data['L'][near], minlength=len(KEYS)); known = counts[1:].sum()
        if known >= 20: found.append({'axis': k // 2, 'position': float(offset), 'low': float(low[run]), 'high': float(high[run]), 'wall': float(counts[ON_WALL].sum() / known),
                                      'kind': KEYS[int(counts[1:].argmax()) + 1], 'planes': [p['id'] for p in group]})
    return found


def outer_wall(walls, axis, position, out, low, high):
    inside = walls[(walls[:, 1 - axis] >= low + 0.1) & (walls[:, 1 - axis] <= high - 0.1)]; gap = (inside[:, axis] - position) * out
    counts = np.bincount(((gap[(gap > -0.1) & (gap < PUSH_M)] + 0.1) / 0.05).astype(int), minlength=int((PUSH_M + 0.1) / 0.05))
    strong = np.flatnonzero(counts >= max(PUSH_MIN, PUSH_SHARE * counts.max())) if len(inside) else []
    if not len(strong) or strong[-1] * 0.05 - 0.1 <= 0.05: return None
    band = np.abs(gap - (strong[-1] * 0.05 - 0.075)) <= 0.075
    if len(np.unique(((inside[band, 1 - axis] - low) / 0.1).astype(int))) < 0.3 * (high - low - 0.2) / 0.1: return None
    return float(position + out * np.median(gap[band]))


def refine(polygon, candidates, walls):
    corners = np.array(polygon, float); count = len(corners); area = 0.5 * np.sum(corners[:, 0] * np.roll(corners[:, 1], -1) - np.roll(corners[:, 0], -1) * corners[:, 1])
    edges = []
    for k in range(count):
        a, b = corners[k], corners[(k + 1) % count]; axis = 0 if abs(a[0] - b[0]) < 1e-9 else 1; normal = np.array([b[1] - a[1], a[0] - b[0]]) * (1 if area > 0 else -1)
        out = float(np.sign(normal[axis])) or 1.0; low, high = sorted((a[1 - axis], b[1 - axis])); position, kind = float(a[axis]), 'bord'
        here = [c for c in candidates[axis] if abs(c[0] - position) < 1e-6]; kind = here[0][1] if here else kind
        options = [c for c in candidates[axis] if c[1] == 'plan' and 0 < (c[0] - position) * out <= REFINE_M and min(c[2], high) - max(c[3], low) > 0.3 * (high - low)]
        if options: position, kind = max(options, key=lambda c: (c[0] - position) * out)[0], 'plan'
        if (pushed := outer_wall(walls, axis, position, out, low, high)) is not None: position, kind = pushed, 'pic'
        edges.append([axis, position, kind, out])
    rebuilt = [[edges[k][1], edges[k - 1][1]] if edges[k][0] == 0 else [edges[k - 1][1], edges[k][1]] for k in range(count)]
    return rebuilt, edges


def outline(samples, owners, centers, planes):
    low, high = np.percentile(samples, 0.5, 0) - 0.3, np.percentile(samples, 99.5, 0) + 0.3; shape = tuple(np.ceil((high - low) / CELL_M).astype(int) + 1)
    if min(shape) < 4 or max(shape) > 600: return None
    grid = carve(samples, centers, owners, low, shape)
    inside = ndimage.binary_fill_holes(ndimage.binary_closing(grid >= max(3, 0.1 * np.percentile(grid[grid > 0], 90)), iterations=2))
    labels, count = ndimage.label(inside)
    if not count: return None
    cams = np.floor((centers[:, [0, 2]] - low) / CELL_M).astype(int).clip(0, np.array(shape) - 1); best = np.bincount(labels[cams[:, 0], cams[:, 1]], minlength=count + 1)[1:].argmax() + 1
    room = labels == best; cells = np.argwhere(room) * CELL_M + low; table = np.pad(room.astype(float).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    candidates = [[(p['position'], 'plan', p['low'], p['high']) for p in planes if p['axis'] == axis and p['wall'] >= 0.5] for axis in (0, 1)]
    for axis in (0, 1):
        for position, _ in peaks(samples, axis, low, high):
            if all(abs(position - c[0]) > NMS_M for c in candidates[axis]): candidates[axis].append((position, 'pic', -np.inf, np.inf))
        for edge in (cells[:, axis].min(), cells[:, axis].max() + CELL_M):
            if all(abs(edge - c[0]) > NMS_M for c in candidates[axis]): candidates[axis].append((float(edge), 'bord', -np.inf, np.inf))
    lines = [sorted(c[0] for c in candidates[axis]) for axis in (0, 1)]
    xs, zs = prune(table, low, lines, max(0.2, 0.02 * room.sum() * CELL_M ** 2))
    if len(xs) < 2 or len(zs) < 2: return None
    labels, count = ndimage.label(fractions(table, low, xs, zs) > 0.5)
    if not count: return None
    centre = [np.searchsorted(xs, centers[:, 0]) - 1, np.searchsorted(zs, centers[:, 2]) - 1]; ok = (centre[0] >= 0) & (centre[0] < len(xs) - 1) & (centre[1] >= 0) & (centre[1] < len(zs) - 1)
    votes = np.bincount(labels[centre[0][ok], centre[1][ok]], minlength=count + 1)[1:] if ok.any() else np.bincount(labels.ravel(), minlength=count + 1)[1:]
    polygon = loops(ndimage.binary_fill_holes(labels == votes.argmax() + 1), xs, zs)
    return refine(polygon, candidates, samples[owners < 0]) if len(polygon) >= 4 else None


def edge_support(samples, axis, position, low, high):
    other = 1 - axis; near = np.abs(samples[:, axis] - position) < 0.06; bins = max(1, int(np.ceil((high - low) / 0.1)))
    hit = np.unique(((samples[near, other] - low) / 0.1).astype(int)); hit = hit[(hit >= 0) & (hit < bins)]
    return len(hit) / bins, int(near.sum())


def room(data):
    samples, owners, centers, height, points = evidence(data)
    if len(samples) < 50: return None
    planes = wall_planes(data, points); shape = outline(samples, owners, centers, planes)
    if not shape: return None
    polygon, sides = shape; edges = []
    for k, (axis, position, kind, out) in enumerate(sides):
        a, b = np.array(polygon[k]), np.array(polygon[(k + 1) % len(polygon)]); low, high = sorted((a[1 - axis], b[1 - axis])); share, n = edge_support(samples, axis, position, low, high)
        matched = [p for p in planes if p['axis'] == axis and abs(p['position'] - position) <= ASSOC_M and p['wall'] >= 0.5 and min(p['high'], high) > max(p['low'], low)]
        edges.append({'id': f's{k + 1:02d}', 'axis': 'xz'[axis], 'position': float(position), 'a': a.tolist(), 'b': b.tolist(), 'length': float(high - low), 'support': float(share), 'n': n,
                      'kind': kind, 'seen': kind != 'bord' or share >= 0.3, 'sens': out, 'planes': [i for p in matched for i in p['planes']]})
    return {'edges': edges, 'height': height, 'cameras': len(centers), 'samples': len(samples), 'furniture_planes': [p for p in planes if p['wall'] < 0.5]}


def wall_hits(data, edges, height):
    frame, model, floor = data['frame'], data['model'], levels(data)[0]; centers = to_frame(frame, photos.camera_centers(model)); top = height or 2.5
    rise = lambda hit: hit[:, 1] - level_at(floor, hit[:, [0, 2]])
    grids = [np.zeros((int(np.ceil(e['length'] / WALL_CELL_M)) + 1, int(np.ceil(top / WALL_CELL_M)) + 1, len(KEYS)), np.int32) for e in edges]; seen = [set() for _ in edges]
    for i, name in enumerate(model['names']):
        labels = read_map(data['cache'], name)
        if labels is None: continue
        mask = np.isin(labels, ON_WALL); directions = rays(model, i, frame, labels.shape, mask); classes = labels[mask]; best = np.full(len(classes), np.inf); owner = np.full(len(classes), -1)
        for k, e in enumerate(edges):
            axis = 'xz'.index(e['axis']) * 2; other = 2 - axis
            with np.errstate(divide='ignore', invalid='ignore'): t = (e['position'] - centers[i][axis]) / directions[:, axis]
            hit = centers[i] + t[:, None] * directions; low, high = sorted((e['a'][1 - axis // 2], e['b'][1 - axis // 2]))
            ok = (t > 0) & (t < best) & (hit[:, other] >= low) & (hit[:, other] <= high) & (rise(hit) >= 0) & (rise(hit) <= top)
            best[ok], owner[ok] = t[ok], k
        for k, e in enumerate(edges):
            pick = owner == k
            if not pick.any(): continue
            axis = 'xz'.index(e['axis']) * 2; other = 2 - axis; hit = centers[i] + best[pick, None] * directions[pick]; start = min(e['a'][1 - axis // 2], e['b'][1 - axis // 2])
            u, v = ((hit[:, other] - start) / WALL_CELL_M).astype(int).clip(0, grids[k].shape[0] - 1), (rise(hit) / WALL_CELL_M).astype(int).clip(0, grids[k].shape[1] - 1)
            np.add.at(grids[k], (u, v, classes[pick]), 1)
            if pick.sum() >= 20: seen[k].add(i)
    return grids, [len(found) for found in seen]


def openings(grid, along_start=0.0):
    total = grid.sum(2); found = []
    for kind in ('porte', 'fenetre'):
        mask = (grid[:, :, INDEX[kind]] >= np.maximum(2, 0.5 * total)) & (total >= 2); mask = ndimage.binary_opening(ndimage.binary_closing(mask, iterations=2))
        labels, count = ndimage.label(mask)
        for sl in ndimage.find_objects(labels):
            width, tall, bottom = (sl[0].stop - sl[0].start) * WALL_CELL_M, (sl[1].stop - sl[1].start) * WALL_CELL_M, sl[1].start * WALL_CELL_M
            if width < 0.4 or tall < 0.4: continue
            floor_level = bottom < 0.2
            typ = 'porte' if kind == 'porte' else 'porte_fenetre' if floor_level and tall >= 1.6 else 'fenetre'
            if typ == 'porte' and tall < 1.5: continue
            found.append({'type': typ, 'along': along_start + (sl[0].start * WALL_CELL_M + width / 2), 'width': width, 'height': tall, 'sill': 0.0 if floor_level else bottom, 'sigma': 0.05})
    return found


def clusters(points, cell):
    if not len(points): return []
    keys = np.floor(points / cell).astype(np.int64); unique, inverse = np.unique(keys, axis=0, return_inverse=True); tree = cKDTree(unique)
    pairs = tree.query_pairs(1.8); from scipy.sparse import coo_matrix; from scipy.sparse.csgraph import connected_components
    graph = coo_matrix((np.ones(len(pairs)), tuple(np.array(list(pairs)).T)) if pairs else ([], ([], [])), shape=(len(unique),) * 2)
    return [np.flatnonzero(labels == k) for labels in [connected_components(graph, directed=False)[1][inverse.ravel()]] for k in np.unique(labels)]


def objects(data, points, kinds, min_points=15):
    found = []
    for kind in kinds:
        mask = data['L'] == INDEX[kind]
        for group in clusters(points[mask], 0.15):
            if len(group) < min_points: continue
            chunk = points[mask][group]; low, high = np.percentile(chunk, 5, 0), np.percentile(chunk, 95, 0)
            found.append({'kind': kind, 'label': CLASSES[INDEX[kind]][1], 'center': ((low + high) / 2).round(3).tolist(), 'size': (high - low).round(3).tolist(), 'n': len(group)})
    return sorted(found, key=lambda o: -o['n'])


def scene(data):
    shares = np.zeros(len(KEYS))
    for name in data['model']['names']:
        labels = read_map(data['cache'], name)
        if labels is not None: shares += np.bincount(labels.ravel(), minlength=len(KEYS)) / labels.size
    shares /= max(len(data['model']['names']), 1); outside = float(shares[OUTSIDE].sum())
    return {'shares': {KEYS[k]: round(float(s), 4) for k, s in enumerate(shares) if s > 0.002}, 'outside': outside,
            'place': 'dehors' if outside > 0.25 else 'dedans' if outside < 0.05 else 'dedans et dehors'}


def summary_path(cache): return folder(cache, 'piece.json')


def span(e):
    axis = 'xz'.index(e['axis']); return sorted((e['a'][1 - axis], e['b'][1 - axis]))


def named(edges, before, frame):
    top, used, k = max([int(e['id'][1:]) for e in before if e['id'][1:].isdigit()], default=0), set(), lambda e: 'xz'.index(e['axis']) * 2
    for e in edges:
        same = [p for p in before if p['id'] not in used and p['axis'] == e['axis'] and abs(p['position'] - e['position']) <= KEEP_M and min(span(p)[1], span(e)[1]) > max(span(p)[0], span(e)[0])]
        if same: e['id'] = min(same, key=lambda p: abs(p['position'] - e['position']))['id']
        else: top += 1; e['id'] = f's{top:02d}'
        used.add(e['id']); e['raw'] = [*frame['R'][k(e)], e['position'] / frame['s_est'] - frame['t'][k(e)]]
    return [p for p in before if p['id'] not in used]


def understand(project, source, force=False):
    data = labelled(project, source)
    if data is None: return None
    file = summary_path(data['cache']); signature = f"{points_signature(data['cache'], data['model'])}:{len(corrections(data['cache']))}:{os.path.getmtime(folder(data['cache'], 'corrections.json')) if corrections(data['cache']) else 0}"
    with COMPUTE:
        saved = photos.load_json(file) if os.path.exists(file) else {}
        if not force and saved.get('signature') == signature: return {**saved, 'data': data}
        points = to_frame(data['frame'], data['P']); place = scene(data); shape = room(data) if place['place'] != 'dehors' else None
        if shape:
            for edge, grid, count in zip(shape['edges'], *wall_hits(data, shape['edges'], shape['height'])):
                edge.update(wall_share=float(grid[:, :, INDEX['mur']].sum() / max(grid.sum(), 1)), openings=openings(grid), photos=count)
            before = saved.get('room') or {}; shape['retired'] = named(shape['edges'], before.get('edges', []) + before.get('retired', []), data['frame'])
        result = {'signature': signature, 'scene': place, 'room': shape, 'device': None,
                  'lights': objects(data, points, ['lampe'], 8), 'furniture': objects(data, points, KEYS[INDEX['lit']:INDEX['personne']]),
                  'counts': {KEYS[k]: int(c) for k, c in enumerate(np.bincount(data['L'], minlength=len(KEYS))) if c}}
        json.dump(result, open(file + '.tmp', 'w', encoding='utf-8'), ensure_ascii=False); os.replace(file + '.tmp', file)
    return {**result, 'data': data}


def legend(counts):
    return [{'key': k, 'label': CLASSES[INDEX[k]][1], 'color': CLASSES[INDEX[k]][2], 'n': n} for k, n in sorted(counts.items(), key=lambda kv: -kv[1]) if k != 'inconnu']


def public(result):
    if not result: return None
    room = result['room']
    return {**{k: v for k, v in result.items() if k not in ('data', 'signature')}, 'legend': legend(result['counts']),
            'closed': bool(room) and all(e['seen'] for e in room['edges']), 'missing': [k for k, e in enumerate(room['edges']) if not e['seen']] if room else []}


@rpc('semantique.status')
def status(ctx, source='photos'):
    project = ctx.require_project(); meta, _, cache = photos.photo_state(project, source, compute=False); model = photos.load_model(cache)
    done = sum(os.path.exists(map_path(cache, n)) for n in model['names'])
    return {'installed': installed(), 'download_mb': round(sum(size for _, _, size in FILES.values()) / 2 ** 20), 'photos': len(model['names']), 'segmented': done,
            'classes': [{'key': k, 'label': l, 'color': c} for k, l, c, _ in CLASSES],
            'dense': bool(dense_signature(cache)), 'result': public(understand(project, source)) if done else None}


@rpc('semantique.analyse', long=True)
def analyse(ctx, source='photos', progress=None):
    progress = progress or photos.no_progress; project = ctx.require_project(); meta, _, cache = photos.photo_state(project, source, compute=False)
    model = photos.load_model(cache); photos.STOPPED.discard(cache); started = time.perf_counter()
    try:
        if not installed(): install(lambda f, m='': progress(0.3 * f, m))
        segmented = segment_all(meta, cache, model, progress, 0.3 if not installed() else 0.0, 0.9)
        progress(0.92, 'étiquettes des points et murs'); result = public(understand(project, source, force=True))
    except photos.Stopped: return {'stopped': True}
    progress(1.0, 'terminé')
    return {**result, 'segmented': segmented, 'device': device(), 'total_s': round(time.perf_counter() - started, 1)}


@rpc('semantique.dense', long=True)
def dense(ctx, source='photos', progress=None):
    progress = progress or photos.no_progress; project = ctx.require_project(); meta, _, cache = photos.photo_state(project, source, compute=False); started = time.perf_counter()
    try: done = photos.densified(cache, meta, photos.DENSE_SIZE, lambda f, m='': progress(0.9 * f, m), started)
    except photos.Stopped: return {'stopped': True}
    progress(0.92, 'étiquettes des points et murs'); result = public(understand(project, source, force=True)) if os.path.isdir(folder(cache, 'cartes')) else None
    progress(1.0, 'terminé')
    return {**(result or {}), **done, 'total_s': round(time.perf_counter() - started, 1)}


@rpc('semantique.corriger')
def corriger(ctx, source, point_xyz_m, to):
    if to not in INDEX: raise RpcError(f'classe inconnue : {to}')
    project = ctx.require_project(); _, frame, cache = photos.photo_state(project, source, compute=False); linear_map, shift, _ = photos.project_transform(project, source, frame)
    data = labelled(project, source)
    if data is None: raise RpcError('analyse des photos non faite')
    world = np.linalg.solve(linear_map, np.asarray(point_xyz_m, float) - shift); start = int(cKDTree(data['P']).query(world)[1])
    fixes = corrections(cache) + [{'point': data['P'][start].tolist(), 'from': KEYS[data['L'][start]], 'to': to}]
    json.dump(fixes, open(folder(cache, 'corrections.json'), 'w', encoding='utf-8'))
    changed = len(grow(data['P'], data['L'], start, 0.06 / frame['s_est']))
    return {'from': KEYS[data['L'][start]], 'to': to, 'points': changed, 'result': public(understand(project, source))}


@rpc('semantique.annuler_corrections')
def annuler(ctx, source='photos'):
    project = ctx.require_project(); _, _, cache = photos.photo_state(project, source, compute=False); file = folder(cache, 'corrections.json')
    if os.path.exists(file): os.remove(file)
    return public(understand(project, source))


def semantic_planes(cache, frame):
    file = summary_path(cache); room = (photos.load_json(file).get('room') if os.path.exists(file) else None) or {}; planes = {}
    for e in room.get('edges', []) + room.get('retired', []):
        k, m = 'xz'.index(e['axis']) * 2, np.array(frame['R']) @ e['raw'][:3] if 'raw' in e else None
        planes.setdefault(e['id'], {'id': e['id'], 'axis': e['axis'], 'offset': float((e['raw'][3] + m @ frame['t']) / m[k]) if m is not None else e['position'] / frame['s_est']})
    return planes


def projector(project, source, frame):
    linear_map, shift, _ = photos.project_transform(project, source, frame)
    return (lambda point: linear_map @ (np.array(frame['R']).T @ (np.asarray(point, float) / frame['s_est'] - frame['t'])) + shift), np.linalg.norm(linear_map[0]) / frame['s_est']


@rpc('semantique.lumieres')
def lumieres(ctx, source='photos'):
    from . import furniture
    project = ctx.require_project(); found = understand(project, source)
    if not found: raise RpcError('analyse des photos non faite')
    to_project, factor = projector(project, source, found['data']['frame']); top = (found['room'] or {}).get('height') or 2.5; added = []
    for light in found['lights']:
        center, size = np.array(light['center']), np.array(light['size']); bottom = center[1] - size[1] / 2; at = to_project(center)
        item = 'suspension' if center[1] > top - 0.7 else 'lampadaire' if bottom < 0.3 and size[1] > 0.8 else 'lampe'
        added.append(furniture.add(ctx, item=f'gabarit:{item}', x=float(at[0]), z=float(at[2]), y=None if item != 'lampe' else float(max(0.0, bottom * factor)),
                                   size=[0.4, top * factor, 0.4] if item == 'suspension' else None))
    return {'added': len(added)}


def placed_openings(project, source):
    try: found = understand(project, source)
    except (OSError, ValueError, KeyError): return None
    if not found or not found['room']: return None
    to_project, factor = projector(project, source, found['data']['frame']); listed = []
    for e in found['room']['edges']:
        axis = 'xz'.index(e['axis']) * 2; start = min(e['a'][1 - axis // 2], e['b'][1 - axis // 2])
        for o in e.get('openings', []):
            centre = np.zeros(3); centre[axis], centre[2 - axis] = e['position'], start + o['along']
            listed.append({**o, 'xz': to_project(centre)[[0, 2]], 'width': o['width'] * factor, 'height': o['height'] * factor, 'sill': o['sill'] * factor, 'sigma': o['sigma'] * factor})
    return listed


@rpc('semantique.classes')
def classes(ctx): return [{'id': i, 'key': k, 'label': l, 'color': c} for i, (k, l, c, _) in enumerate(CLASSES)]
