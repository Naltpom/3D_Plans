import json, os, shutil, sqlite3, tempfile, threading, time
from collections import Counter
from datetime import datetime
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from . import photos
from .rpc import RpcError, project_meta, rpc

HERE = os.path.dirname(__file__)
EVERY_S, EVERY_N, MIN_S, MIN_IMAGES, PHONE_POINTS, MAX_BYTES, PRE = 30, 20, 5, 5, 3000, 30 << 20, 1024
CSP = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:"
SIDES = ('devant toi', 'à droite', 'derrière toi', 'à gauche')
AXES, LAYERS = ('longueur', 'largeur', 'hauteur'), 6
FAST = ['--Mapper.ba_global_frames_ratio', '1.5', '--Mapper.ba_global_points_ratio', '1.5', '--Mapper.ba_global_max_refinements', '1', '--Mapper.ba_global_max_num_iterations', '30',
        '--Mapper.ba_global_ignore_redundant_points3D', '1', '--Mapper.multiple_models', '0']
PHONE_KEYS = {'passage', 'paused', 'gardees', 'floues', 'doublons', 'freq', 'attente', 'resolution'}


def page():
    read = lambda name: open(os.path.join(HERE, name), encoding='utf-8').read()
    return read('scan.html').replace('/*FILTRE*/', read('scan_filtre.js')).encode()


def dated(data, ms):
    from PIL import Image
    if data[:3] != b'\xff\xd8\xff': raise ValueError('image JPEG attendue')
    i, kept = 2, []
    while i + 4 <= len(data) and data[i] == 0xFF and data[i + 1] not in (0xD9, 0xDA):
        end = i + 2 + int.from_bytes(data[i + 2:i + 4], 'big')
        if not 0xE1 <= data[i + 1] <= 0xEF: kept.append(data[i:end])
        i = end
    when, exif = datetime.fromtimestamp(ms / 1000), Image.Exif()
    exif[0x8769] = {0x9003: when.strftime('%Y:%m:%d %H:%M:%S'), 0x9291: f'{when.microsecond // 1000:03d}'}
    block = exif.tobytes()
    return data[:2] + b'\xff\xe1' + (len(block) + 2).to_bytes(2, 'big') + block + b''.join(kept) + data[i:]


def camera(database, size):
    known = {}
    if os.path.exists(database):
        db = sqlite3.connect(database); known = {(w, h): c for c, w, h in db.execute('select camera_id, width, height from cameras')}; db.close()
    if size in known: return ['--ImageReader.existing_camera_id', str(known[size])]
    focal = 26.0 * np.hypot(*size) / 43.267
    return ['--ImageReader.single_camera', '1', '--ImageReader.camera_params', f'{focal:.1f},{focal:.1f},{size[0] / 2},{size[1] / 2},0,0,0,0']


def flat(vectors, up): return photos.unit_rows(vectors - np.outer(vectors @ up, up))


def coverage(model, mode):
    rotations, centers = photos.rotation_matrices(model['q']), photos.camera_centers(model)
    up = photos.unit(photos.image_ups({**model, 'orientation': np.ones(len(model['q']))}).sum(0))
    last = int(np.argmax([int(str(name)[:5]) for name in model['names']]))
    forward, right = flat(rotations[:, 2], up), flat(rotations[:, 0], up)
    if mode != 'piece':
        axes = np.eye(3) - np.einsum('ni,nj->nij', rotations[:, 2], rotations[:, 2])
        middle = np.linalg.lstsq(axes.sum(0), np.einsum('nij,nj->i', axes, centers), rcond=None)[0]
        around = flat(centers - middle, up); angle = np.arctan2(around @ right[last], around @ around[last])
        seen = np.sort(np.degrees(angle) % 360); gaps = np.diff(np.r_[seen, seen[0] + 360]); widest = int(np.argmax(gaps)); missing = np.clip(gaps - 40, 0, None).sum()
        far = (np.arange(12)[:, None] * 30 - seen[None] + 180) % 360 - 180
        high = np.degrees(np.arcsin(np.clip(photos.unit_rows(centers - middle) @ up, -1, 1))).max()
        gaps = ([f"continue vers la {'droite' if (seen[widest] + gaps[widest] / 2 + 180) % 360 - 180 > 0 else 'gauche'}"] if missing else []) + (['puis un tour plus haut'] if high < 20 and mode == 'objet' else [])
        return {'sectors': (np.abs(far).min(1) <= 20).astype(float).tolist(), 'turn': int(round((360 - missing) / 10) * 10), 'gaps': ', '.join(gaps), 'center': middle}
    angle = np.arctan2(forward @ right[last], forward @ forward[last]); bins = np.bincount(np.round(angle / (np.pi / 4)).astype(int) % 8, minlength=8)
    pitch = rotations[:, 2] @ up; share = lambda mask: np.count_nonzero(mask) / len(pitch)
    gaps = list(dict.fromkeys(SIDES[(k + 1) // 2 % 4] for k in np.flatnonzero(bins < 0.35 * bins.max())))
    gaps += [name for name, mask in (('le plafond', pitch > 0.57), ('le sol', pitch < -0.57)) if share(mask) < 0.04]
    return {'sectors': (bins / bins.max()).round(2).tolist(), 'gaps': ', '.join(gaps)}


def phone_cloud(model, version):
    order = np.argsort([int(str(name)[:5]) for name in model['names']]); recent, last = order[-3:], order[-1]
    rotations, centers = photos.rotation_matrices(model['q']), photos.camera_centers(model)
    up = photos.unit(photos.image_ups({**model, 'orientation': np.ones(len(model['q']))}).sum(0))
    owner = np.repeat(np.arange(len(model['P'])), np.diff(model['Tptr']))
    seen = np.unique(owner[np.isin(model['Timg'], recent)])[:PHONE_POINTS * 4]
    seen = np.random.default_rng(0).permutation(seen)[:PHONE_POINTS]
    back = -flat(rotations[last:last + 1, 2], up)[0]; basis = np.array([np.cross(up, back), up, back])
    local = (model['P'][seen] - centers[last]) @ basis.T
    return {'version': version, 'points': local.round(3).ravel().tolist(), 'colors': model['C'][seen].ravel().tolist()}


def floor(points, up, reach):
    low = points[points @ up <= np.percentile(points @ up, 40)]
    for _ in range(5):
        middle = low.mean(0); normal = np.linalg.svd(low - middle, full_matrices=False)[2][2]; normal *= np.sign(normal @ up); gap = np.abs((low - middle) @ normal)
        low = low[gap < max(0.01 * reach, 2.5 * np.median(gap))]
    return normal, middle


def isolate(model):
    found, centers, up = coverage(model, 'objet'), photos.camera_centers(model), photos.unit(photos.image_ups({**model, 'orientation': np.ones(len(model['q']))}).sum(0))
    rel, around = model['P'][np.diff(model['Tptr']) >= 3] - found['center'], centers - found['center']
    reach = np.median(np.linalg.norm(around - np.outer(around @ up, up), axis=1))
    up, base = floor(rel[np.linalg.norm(rel - np.outer(rel @ up, up), axis=1) < 0.9 * reach], up, reach)
    e1 = photos.unit(np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])); e2 = np.cross(up, e1)
    xy, h, ground = np.stack([rel @ e1, rel @ e2], 1), (rel - base) @ up, 0.0; near = np.linalg.norm(xy, axis=1)
    inside = (near < 0.75 * reach) & (h > ground + 0.02 * reach) & (h < ground + 2.5 * reach)
    if np.count_nonzero(inside) < 30: raise ValueError("objet introuvable : tourne autour en le gardant au centre de l'image")
    points = np.c_[xy, h][inside]; gap = cKDTree(points).query(points, 5)[0][:, -1]; radius = 2.5 * np.median(gap); points = points[gap < radius]
    pairs = cKDTree(points).query_pairs(radius, output_type='ndarray')
    labels = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2), directed=False)[1]
    xy, h = points[labels == np.bincount(labels).argmax(), :2], points[labels == np.bincount(labels).argmax(), 2] - ground
    best = min(range(90), key=lambda a: np.prod(np.ptp(np.percentile(xy @ rotation2(a), [2, 98], axis=0), axis=0)))
    local = xy @ rotation2(best); lo, hi = np.percentile(local, [2, 98], axis=0)
    if hi[0] - lo[0] < hi[1] - lo[1]: local, lo, hi = local[:, ::-1], lo[::-1], hi[::-1]
    top, boxes = np.percentile(h, 98), []
    for a, b in zip(np.linspace(0, top, LAYERS + 1)[:-1], np.linspace(0, top, LAYERS + 1)[1:]):
        part = local[(h >= a) & (h <= b)]
        box = list(np.percentile(part, [2, 98], axis=0).T.ravel()) if len(part) >= 15 else boxes[-1][2:] if boxes else [lo[0], hi[0], lo[1], hi[1]]
        if boxes and max(abs(np.subtract(box, boxes[-1][2:]))) < 0.12 * (hi[0] - lo[0]): boxes[-1] = [boxes[-1][0], b, *np.where([1, 0, 1, 0], np.minimum(box, boxes[-1][2:]), np.maximum(box, boxes[-1][2:]))]
        else: boxes.append([a, b, *box])
    simple = 1 < len(boxes) <= 4 and len(h) >= 200
    shapes = [{'x': x0 - lo[0], 'y': a, 'z': z0 - lo[1], 'lx': x1 - x0, 'ly': b - a, 'lz': z1 - z0} for a, b, x0, x1, z0, z1 in (boxes if simple else [[0, top, lo[0], hi[0], lo[1], hi[1]]])]
    sample = np.random.default_rng(0).permutation(len(h))[:PHONE_POINTS]
    shown = np.stack([local[sample, 0] - lo[0], h[sample], local[sample, 1] - lo[1]], 1)
    return {'dims': [float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(top)], 'boxes': [{k: float(v) for k, v in b.items()} for b in shapes], 'simple': simple, 'points': len(h)}, shown


def rotation2(degrees):
    a = np.radians(degrees); return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def folders(path): return [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def tidy(work, prefix, keep=''):
    for folder in folders(work):
        if os.path.basename(folder).startswith(prefix) and not (keep + os.sep).startswith(folder + os.sep): shutil.rmtree(folder, ignore_errors=True)


def known(database):
    if not os.path.exists(database): return set()
    db = sqlite3.connect(database); names = {name for (name,) in db.execute('select name from images')}; db.close()
    return names


def link(path, target):
    if os.path.exists(target): return
    try: os.link(path, target)
    except OSError: shutil.copy2(path, target)


class Scan:
    def __init__(self, ctx, source, mode):
        self.ctx, self.project, self.source, self.mode = ctx, ctx.require_project(), source, mode
        self.id, self.modes = datetime.now().strftime('%Y%m%d_%H%M%S'), ('facades' if (project_meta(self.project).get('guide') or {}).get('type') == 'maison' else 'piece', 'objet')
        self.mode = mode if mode in self.modes else self.modes[0]
        self.folder, self.work = (os.path.join(self.project.folder, part, 'scan', self.id) for part in ('sources', 'cache'))
        self.lock, self.wake = threading.Lock(), threading.Condition()
        self.paths, self.seen, self.done, self.stopped, self.now, self.want, self.processed = [], set(), False, False, False, False, 0
        self.phone, self.cloud, self.object, self.final, self.model = {}, {}, None, '', ''
        self.info = {'placed': 0, 'sectors': [], 'gaps': '', 'computing': False, 'at': None, 'duration_s': None, 'colmap_mb': 0, 'engine_mb': 0, 'error': ''}
        os.makedirs(os.path.join(self.work, 'images'), exist_ok=True)

    def public(self):
        return {'id': self.id, 'project': os.path.basename(os.path.normpath(self.project.folder)), 'mode': self.mode, 'modes': self.modes, 'active': not self.done, 'passages': dict(Counter(str(p) for p, _ in self.seen)), 'final': self.final, 'received': len(self.paths), 'phone': self.phone, 'cloud': self.cloud.get('version', 0),
                **self.info, 'ago_s': None if self.info['at'] is None else round(time.time() - self.info['at']), 'object': self.object and {**{k: v for k, v in self.object.items() if k != 'boxes'}, 'blocs': len(self.object['boxes'])}}

    def changed(self): self.ctx.send({'event': 'scan', **self.public()})

    def signal(self, **change):
        with self.wake:
            for name, value in change.items(): setattr(self, name, value)
            self.wake.notify_all()

    def add(self, data, passage, index, ms):
        if self.ctx.project is not self.project: raise RpcError('ce projet est fermé sur le PC')
        data = dated(data, ms)
        with self.lock:
            if (passage, index) not in self.seen:
                path = os.path.join(self.folder, f'passage-{passage}', f'{index:05d}.jpg'); os.makedirs(os.path.dirname(path), exist_ok=True)
                open(path, 'wb').write(data); self.seen.add((passage, index)); self.paths.append(path)
                if self.mode == 'objet' and self.final in ('objet', 'échec'): self.final, self.object = '', None
                if self.mode != 'objet':
                    with self.ctx.lock: photos.add(self.ctx, [path], self.source)
            count = len(self.paths)
        self.signal(); self.changed()
        return {'received': count}

    def command(self, op):
        kind = op.get('op')
        if kind == 'mode' and not self.paths and op.get('mode') in self.modes: self.mode = op['mode']
        elif kind == 'finish': self.signal(**{'want': True} if self.mode == 'objet' else {'done': True})
        elif kind == 'objet': return self.keep(op)
        elif kind == 'now': self.signal(now=True)
        elif kind != 'mode': raise RpcError('opération inconnue')
        self.changed()
        return self.public()

    def keep(self, op):
        from .furniture import add_perso
        if not self.object or op.get('axe') not in AXES: raise RpcError("pas d'objet calculé")
        k = float(op['cm']) / 100 / self.object['dims'][AXES.index(op['axe'])]
        if not 0 < k * max(self.object['dims']) <= 5: raise RpcError('cote hors limites')
        item = add_perso(op.get('nom') or 'Objet scanné', [{key: round(v * k, 4) for key, v in box.items()} for box in self.object['boxes']])
        self.object['added'] = {'label': item['label'], 'dims_cm': [round(v * 100 * k, 1) for v in self.object['dims']]}
        self.signal(done=True, final='ajouté'); self.changed()
        return self.public()

    def abort(self):
        self.signal(done=True, stopped=True, final='arrêté'); photos.STOPPED.add(self.work); self.changed()
        if process := photos.PROCESSES.get(self.work): process.kill()

    def run(self, emit, progress):
        last = time.monotonic()
        while not self.done:
            with self.wake:
                self.wake.wait(1)
                fresh, waited = len(self.paths) - self.processed, time.monotonic() - last
                due = len(self.paths) >= MIN_IMAGES and fresh and (self.now or fresh >= EVERY_N and waited >= MIN_S or waited >= EVERY_S)
            if due and not self.done:
                self.compute(emit, progress); last = time.monotonic()
            if self.want and not self.done: self.solve(emit, progress)

    def compute(self, emit, progress):
        work = lambda *parts: os.path.join(self.work, *parts)
        with self.lock: paths = self.paths[:]
        names, before, extend, started = [f'{i:05d}.jpg' for i in range(len(paths))], self.processed, False, time.perf_counter(); fresh = range(before, len(paths))
        live = lambda **event: emit({'event': 'photos.live', 'total': len(paths), **event})
        steps, lap = {}, lambda key: steps.__setitem__(key, round(time.perf_counter() - started - sum(steps.values()), 1))
        self.now = False; self.info.update(computing=True, error=''); self.changed()
        try:
            from PIL import Image
            groups = {}
            for i in fresh:
                link(paths[i], work('images', names[i])); live(index=i, thumb=photos.thumbnail(paths[i]))
                with Image.open(paths[i]) as image: groups.setdefault(image.size, []).append(names[i])
            for db, features, size in (('database.db', photos.FEATURES, photos.IMAGE_SIZE), ('pre.db', PRE, PRE)):
                for shape, group in groups.items():
                    photos.write_lines(work('list.txt'), group)
                    photos.run_colmap(['feature_extractor', '--database_path', work(db), '--image_path', work('images'), '--image_list_path', work('list.txt'), '--ImageReader.camera_model', 'OPENCV',
                                       *camera(work(db), shape), '--FeatureExtraction.use_gpu', '1', '--SiftExtraction.max_num_features', str(features), '--FeatureExtraction.max_image_size', str(size)],
                                      work('features.log'), progress, 0, .2, 'points clés')
                lap(f'points_{db[:3]}_s')
            photos.run_colmap(['exhaustive_matcher', '--database_path', work('pre.db'), '--FeatureMatching.use_gpu', '1', '--TwoViewGeometry.min_num_inliers', str(photos.PAIR_INLIERS)],
                              work('prematching.log'), progress, .2, .3, 'pré-appariement')
            lap('pre_appariement_s'); pairs = sorted(photos.verified_pairs(work('pre.db'), names) | photos.neighbour_pairs(len(names)))
            photos.write_lines(work('pairs.txt'), [f'{names[i]} {names[j]}' for i, j in pairs]); live(pairs=len(pairs))
            photos.run_colmap(['matches_importer', '--database_path', work('database.db'), '--match_list_path', work('pairs.txt'), '--match_type', 'pairs', '--FeatureMatching.use_gpu', '1'],
                              work('matching.log'), progress, .3, .5, 'appariement')
            lap('appariement_s')
            mapper, out = ['mapper', '--database_path', work('database.db'), '--image_path', work('images'), *FAST], tempfile.mkdtemp(prefix='poses-', dir=self.work)
            extend = bool(self.model) and self.info['placed'] >= 0.4 * before
            photos.run_colmap([*mapper, '--input_path', self.model, '--output_path', out] if extend else [*mapper, '--output_path', out], work('mapper.log'), progress, .5, .95, 'poses')
            lap('placement_s')
            models = [out] if extend else folders(out)
            for folder in models: photos.run_colmap(['model_converter', '--input_path', folder, '--output_path', folder, '--output_type', 'TXT'], work('convert.log'), progress, .95, .95, 'lecture')
            read = [(photos.read_text_model(folder)[0], folder) for folder in models if os.path.exists(os.path.join(folder, 'images.txt'))]
            if read:
                model, self.model = max(read, key=lambda item: len(item[0]['names'])); self.show(model, names, live)
            self.processed = len(paths)
        except photos.Stopped: return
        except Exception as error:
            self.info['error'], self.model = f'Calcul sauté, le suivant repart : {error}', '' if extend else self.model
        finally:
            tidy(self.work, 'poses-', self.model)
            self.info.update(computing=False, at=time.time(), steps=steps, images=len(fresh), duration_s=round(time.perf_counter() - started, 1), colmap_mb=round(photos.PEAKS.get(self.work, 0)), engine_mb=round(photos.peak_mb()))
            self.changed()

    def solve(self, emit, progress):
        work = lambda *parts: os.path.join(self.work, *parts)
        self.want, self.final = False, 'calcul'; self.changed()
        try:
            if len(self.paths) > self.processed: self.compute(emit, progress)
            tidy(self.work, 'objet-'); out = tempfile.mkdtemp(prefix='objet-', dir=self.work)
            photos.run_colmap(['global_mapper', '--database_path', work('database.db'), '--image_path', work('images'), '--output_path', out], work('objet.log'), progress, .5, .9, 'poses')
            for folder in folders(out):
                photos.run_colmap(['point_filtering', '--input_path', folder, '--output_path', folder], work('filter.log'), progress, .9, .92, 'lecture')
                photos.run_colmap(['model_converter', '--input_path', folder, '--output_path', folder, '--output_type', 'TXT'], work('convert.log'), progress, .92, .95, 'lecture')
            model = max((photos.read_text_model(folder)[0] for folder in folders(out)), key=lambda m: len(m['names']), default=None)
            if model is None or len(model['names']) < 8: raise ValueError("trop peu d'images placées pour l'objet : fais un tour de plus")
            self.object, shown = isolate(model); self.info.update(placed=len(model['names']), **{k: v for k, v in coverage(model, 'objet').items() if k != 'center'})
            self.cloud = {'version': self.cloud.get('version', 0) + 1, 'points': shown.round(4).ravel().tolist(), 'colors': [], 'box': [self.object['dims'][0], self.object['dims'][2], self.object['dims'][1]]}
            self.final = 'objet'
        except photos.Stopped: pass
        except Exception as error: self.final, self.info['error'] = 'échec', str(error)
        self.changed()

    def show(self, model, names, live):
        position = {name: i for i, name in enumerate(names)}; kept = np.random.default_rng(0).permutation(len(model['P']))[:photos.LIVE_POINTS]
        live(placed=sorted(position[name] for name in model['names']))
        live(**photos.live_view(model['names'], model['q'], model['t'], model['P'][kept], model['C'][kept], position, [1] * len(names)))
        if len(model['names']) >= 8: self.info.update({k: v for k, v in coverage(model, self.mode).items() if k != 'center'})
        self.info['placed'] = len(model['names'])
        self.cloud = phone_cloud(model, self.cloud.get('version', 0) + 1)

    def finish(self, emit, progress):
        result = {'id': self.id, 'received': len(self.paths), 'stopped': self.stopped}
        if self.stopped or len(self.paths) < MIN_IMAGES or self.mode == 'objet': return {**result, 'objet': self.object and self.object.get('added')}
        self.final = 'calcul'; self.changed()
        try:
            names = known(os.path.join(self.work, 'database.db')); seed = {path: f'{i:05d}.jpg' for i, path in enumerate(self.paths) if f'{i:05d}.jpg' in names}
            meta, frame, _ = photos.photo_state(self.project, self.source, progress, emit=emit, seed=(self.work, seed) if seed else None)
            self.project.db.execute('update params set init=? where name=?', (frame['s_est'], f'{self.source}_s')); self.project.db.commit()
            self.info['placed'], self.final = meta['registered'], 'terminé'
            return {**result, 'registered': meta['registered'], 'photos': meta['photos']}
        except photos.Stopped:
            self.final = 'arrêté'; return {**result, 'stopped': True}
        except Exception as error:
            self.final, self.info['error'] = 'échec', str(error); raise
        finally:
            self.changed()


def current(handler):
    phone = handler.server.phone; scan = getattr(phone.ctx, 'scan', None)
    return scan if scan and scan.project is phone.project else None


def get(handler, path):
    if path == '/scan': return handler.reply(200, page(), 'text/html; charset=utf-8', [('Content-Security-Policy', CSP)])
    if path != '/scan/cloud': return handler.reply(404)
    if not handler.allowed(): return handler.reply(403)
    scan = current(handler)
    handler.reply(200, json.dumps(scan.cloud if scan else {}).encode())


def post(handler, path, size):
    if path not in ('/scan/state', '/scan/image', '/scan/cmd'): return handler.reply(404)
    if not handler.allowed(): return handler.reply(403)
    if not 0 < size <= (MAX_BYTES if path == '/scan/image' else 65536): return handler.reply(413)
    body, scan, answer = handler.rfile.read(size), current(handler), lambda code, value: handler.reply(code, json.dumps(value, ensure_ascii=False).encode())
    try:
        if path == '/scan/state':
            if not scan: return answer(200, {'active': False})
            scan.phone = {k: v for k, v in json.loads(body).items() if k in PHONE_KEYS}; scan.changed()
            return answer(200, scan.public())
        if not scan or scan.done: return answer(409, {'error': "Le PC n'attend pas de scan : ouvre « Scanner au téléphone » sur le PC."})
        if path == '/scan/cmd': return answer(200, scan.command(json.loads(body)))
        if handler.headers.get('X-Scan') != scan.id: return answer(409, {'error': "Image d'un ancien scan.", 'old': True})
        return answer(200, scan.add(body, int(handler.headers['X-Passage']), int(handler.headers['X-Index']), float(handler.headers['X-Time'])))
    except (ValueError, KeyError, TypeError) as error:
        return answer(400, {'error': str(error) if isinstance(error, ValueError) else f'requête refusée : {error!r}'})


@rpc('scan.live', long=True)
def live(ctx, source='photos', mode='piece', progress=photos.no_progress):
    old = getattr(ctx, 'scan', None)
    if old and not old.done: raise RpcError('un scan est déjà en cours')
    scan = ctx.scan = Scan(ctx, source, mode); photos.STOPPED.discard(scan.work)
    scan.changed()
    try: scan.run(ctx.emit, progress)
    finally: scan.done = True
    return scan.finish(ctx.emit, progress)


@rpc('scan.stop')
def stop(ctx, source='photos'):
    scan = getattr(ctx, 'scan', None)
    if scan: scan.abort()
    if ctx.project and ctx.project.db.execute('select 1 from sources where id=?', (source,)).fetchone(): photos.stop(ctx, source)


@rpc('scan.now')
def now(ctx):
    scan = getattr(ctx, 'scan', None)
    if scan and not scan.done: scan.signal(now=True)


@rpc('scan.state')
def state(ctx):
    scan = getattr(ctx, 'scan', None)
    return scan.public() if scan and scan.project is ctx.project else None
