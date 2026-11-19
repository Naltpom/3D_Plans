import json, os, shutil, sqlite3, threading, time
from collections import Counter
from datetime import datetime
import numpy as np
from . import photos
from .rpc import RpcError, rpc

HERE = os.path.dirname(__file__)
EVERY_S, EVERY_N, MIN_S, MIN_IMAGES, PHONE_POINTS, MAX_BYTES, PRE = 30, 20, 5, 5, 3000, 30 << 20, 1024
CSP = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:"
SIDES = ('devant toi', 'à droite', 'derrière toi', 'à gauche')
MODES = ('piece',)
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
    if mode == 'objet':
        axes = np.eye(3) - np.einsum('ni,nj->nij', rotations[:, 2], rotations[:, 2])
        middle = np.linalg.lstsq(axes.sum(0), np.einsum('nij,nj->i', axes, centers), rcond=None)[0]
        around = flat(centers - middle, up); angle = np.arctan2(around @ right[last], around @ around[last])
        bins = np.bincount(np.round(angle / (np.pi / 6)).astype(int) % 12, minlength=12)
        missing = [k if k <= 6 else k - 12 for k in np.flatnonzero(bins == 0)]
        high = np.degrees(np.arcsin(np.clip(photos.unit_rows(centers - middle) @ up, -1, 1))).max()
        gaps = ([f"continue vers la {'droite' if min(missing, key=abs) > 0 else 'gauche'}"] if missing else []) + (['puis un tour plus haut'] if high < 20 else [])
        return {'sectors': (bins > 0).astype(float).tolist(), 'turn': int(30 * np.count_nonzero(bins)), 'gaps': ', '.join(gaps), 'center': middle}
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


def link(path, target):
    if os.path.exists(target): return
    try: os.link(path, target)
    except OSError: shutil.copy2(path, target)


class Scan:
    def __init__(self, ctx, source, mode):
        self.ctx, self.project, self.source, self.mode = ctx, ctx.require_project(), source, mode
        self.id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.folder, self.work = (os.path.join(self.project.folder, part, 'scan', self.id) for part in ('sources', 'cache'))
        self.lock, self.wake = threading.Lock(), threading.Condition()
        self.paths, self.seen, self.done, self.stopped, self.now, self.processed = [], set(), False, False, False, 0
        self.phone, self.cloud, self.object, self.final = {}, {}, None, ''
        self.info = {'placed': 0, 'sectors': [], 'gaps': '', 'computing': False, 'at': None, 'duration_s': None, 'colmap_mb': 0, 'engine_mb': 0, 'error': ''}
        os.makedirs(os.path.join(self.work, 'images'), exist_ok=True)

    def public(self):
        return {'id': self.id, 'project': os.path.basename(os.path.normpath(self.project.folder)), 'mode': self.mode, 'modes': MODES, 'active': not self.done, 'passages': dict(Counter(str(p) for p, _ in self.seen)), 'final': self.final, 'received': len(self.paths), 'phone': self.phone, 'cloud': self.cloud.get('version', 0),
                **self.info, 'ago_s': None if self.info['at'] is None else round(time.time() - self.info['at']), 'object': self.object}

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
                if self.mode == 'piece':
                    with self.ctx.lock: photos.add(self.ctx, [path], self.source)
            count = len(self.paths)
        self.signal(); self.changed()
        return {'received': count}

    def command(self, op):
        kind = op.get('op')
        if kind == 'mode' and not self.paths and op.get('mode') in MODES: self.mode = op['mode']
        elif kind == 'finish': self.signal(done=True)
        elif kind == 'now': self.signal(now=True)
        elif kind != 'mode': raise RpcError('opération inconnue')
        self.changed()
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

    def compute(self, emit, progress):
        work = lambda *parts: os.path.join(self.work, *parts)
        with self.lock: paths = self.paths[:]
        names, fresh, started = [f'{i:05d}.jpg' for i in range(len(paths))], range(self.processed, len(paths)), time.perf_counter()
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
            lap('appariement_s'); before, self.processed = self.processed, len(paths)
            mapper = ['mapper', '--database_path', work('database.db'), '--image_path', work('images'), *FAST]
            extend = os.path.isdir(work('model')) and self.info['placed'] >= 0.4 * before
            for folder in ('next', 'sparse'): shutil.rmtree(work(folder), ignore_errors=True); os.makedirs(work(folder))
            photos.run_colmap([*mapper, '--input_path', work('model'), '--output_path', work('next')] if extend else [*mapper, '--output_path', work('sparse')], work('mapper.log'), progress, .5, .95, 'poses')
            lap('placement_s')
            models = [work('next')] if extend else [work('sparse', name) for name in os.listdir(work('sparse')) if os.path.isdir(work('sparse', name))]
            for folder in models: photos.run_colmap(['model_converter', '--input_path', folder, '--output_path', folder, '--output_type', 'TXT'], work('convert.log'), progress, .95, .95, 'lecture')
            read = [(photos.read_text_model(folder)[0], folder) for folder in models if os.path.exists(os.path.join(folder, 'images.txt'))]
            if read:
                model, folder = max(read, key=lambda item: len(item[0]['names']))
                shutil.rmtree(work('model'), ignore_errors=True); os.replace(folder, work('model')); self.show(model, names, live)
        except photos.Stopped: return
        except ValueError as error: self.info['error'] = str(error)
        finally:
            self.info.update(computing=False, at=time.time(), steps=steps, images=len(fresh), duration_s=round(time.perf_counter() - started, 1), colmap_mb=round(photos.PEAKS.get(self.work, 0)), engine_mb=round(photos.peak_mb()))
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
        if self.stopped or len(self.paths) < MIN_IMAGES: return result
        self.final = 'calcul'; self.changed()
        try:
            seed = (self.work, {path: f'{i:05d}.jpg' for i, path in enumerate(self.paths[:self.processed])})
            meta, frame, _ = photos.photo_state(self.project, self.source, progress, emit=emit, seed=seed)
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
    scan.changed(); scan.run(ctx.emit, progress)
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
