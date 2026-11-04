import datetime, hashlib, os, re
import numpy as np
from scipy.spatial import cKDTree
from . import photos
from .rpc import rpc

GAP_MIN, CELL_M, MIN_FREE, FREE_RATIO, MIN_POINTS, MIN_COS, GROW, MAX_RAYS, MARGIN, VERSION = 5.0, 0.1, 3, 2, 2, 0.5, 1, 600000, 4.0, 2
NAME_TIME = re.compile(r'(20\d\d)(\d\d)(\d\d)\D?(\d\d)(\d\d)(\d\d)')
PALETTE = np.array([(63, 182, 217), (232, 149, 58), (180, 140, 240), (108, 197, 138), (240, 122, 174), (245, 196, 81), (90, 200, 190), (220, 90, 80)], np.uint8)
FIXED, MOBILE, SHARED, UNSEEN = (139, 149, 163), (232, 149, 58), (236, 238, 241), (70, 76, 86)
OFFSET = 1 << 20
NEIGHBOURS = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if a or b or c])
TIMES, THUMBS = {}, {}


def read_time(path):
    from PIL import Image
    try:
        with Image.open(path) as image: value = image.getexif().get_ifd(0x8769).get(0x9003)
        if value: return datetime.datetime.strptime(str(value).strip()[:19], '%Y:%m:%d %H:%M:%S')
    except (OSError, ValueError): pass
    if found := NAME_TIME.search(os.path.basename(path)):
        try: return datetime.datetime(*map(int, found.groups()))
        except ValueError: pass
    return datetime.datetime.fromtimestamp(os.path.getmtime(path)) if os.path.exists(path) else datetime.datetime(1970, 1, 1)


def capture_time(path):
    key = (path, os.path.getmtime(path) if os.path.exists(path) else None)
    if key not in TIMES: TIMES[key] = read_time(path)
    return TIMES[key]


def split(times, gap_s):
    sessions = []
    for i in sorted(range(len(times)), key=lambda i: times[i]):
        if not sessions or (times[i] - times[sessions[-1][-1]]).total_seconds() > gap_s: sessions.append([])
        sessions[-1].append(i)
    return sessions


def gap_minutes(project, source):
    photos.ensure_schema(project); row = project.db.execute('select minutes from photos_gap where source=?', (source,)).fetchone()
    return row[0] if row else GAP_MIN


def sessions(project, source):
    manifest, _ = photos.source_cache(project, source); paths = photos.read_manifest(manifest); times = [capture_time(path) for path in paths]
    names = dict(project.db.execute('select first, name from photos_groups where source=?', (source,))); excluded = photos.excluded_photos(project, source)
    return [{'id': paths[members[0]], 'name': names.get(paths[members[0]]) or f'Séance {k + 1}', 'start': times[members[0]].isoformat(), 'end': times[members[-1]].isoformat(),
             'photos': [paths[i] for i in members], 'excluded': all(paths[i] in excluded for i in members), 'color': '#%02x%02x%02x' % tuple(PALETTE[k % len(PALETTE)])}
            for k, members in enumerate(split(times, 60 * gap_minutes(project, source)))]


def key(path): return os.path.normcase(os.path.abspath(path))
def encode(cells): return ((cells[:, 0] + OFFSET) << 42) | ((cells[:, 1] + OFFSET) << 21) | (cells[:, 2] + OFFSET)


def image_groups(project, source, meta, model):
    index = {key(path): k for k, session in enumerate(sessions(project, source)) for path in session['photos']}
    return np.array([index.get(key(os.path.join(meta['root'], str(name))), 0) for name in model['names']], np.int64)


def tracks(cache):
    model = photos.load_model(cache)
    if 'Tptr' not in model:
        folder = os.path.join(cache, 'sparse', photos.load_json(os.path.join(cache, 'meta.json'))['model'])
        read, _ = photos.read_text_model(folder); model.update(Tptr=read['Tptr'], Timg=read['Timg']); np.savez(os.path.join(cache, 'model.npz'), **model)
    return model


def free_space(cells_of, keys, starts, ends, ray_group, groups, cell, chunk=20000):
    free = np.zeros(groups * len(keys), np.int64); step = cell / 2
    for first in range(0, len(starts), chunk):
        a, vector = starts[first:first + chunk], ends[first:first + chunk] - starts[first:first + chunk]; length = np.linalg.norm(vector, axis=1)
        n = np.maximum(((length - MARGIN * cell) / step).astype(np.int64), 0); ray = np.repeat(np.arange(len(n)), n); along = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        code = encode(cells_of(a[ray] + vector[ray] * ((along + 0.5) * step / length[ray])[:, None]))
        at = np.searchsorted(keys, code).clip(max=len(keys) - 1); hit = keys[at] == code; pairs = np.unique(ray[hit] * len(keys) + at[hit])
        free += np.bincount(ray_group[first + pairs // len(keys)] * len(keys) + pairs % len(keys), minlength=len(free))
    return free.reshape(groups, len(keys))


def around(keys, cells, values):
    total = values.copy()
    for offset in NEIGHBOURS:
        code = encode(cells + offset); at = np.searchsorted(keys, code).clip(max=len(keys) - 1); hit = keys[at] == code; total[:, hit] += values[:, at[hit]]
    return total


def analyse(points, obs_point, obs_image, centers, image_group, groups, cell, seed=0):
    cells_of = lambda p: np.floor(p / cell).astype(np.int64)
    codes = encode(cells_of(points)); keys, inverse = np.unique(codes, return_inverse=True); count = len(keys)
    cells = np.stack([(keys >> 42) - OFFSET, ((keys >> 21) & (OFFSET * 2 - 1)) - OFFSET, (keys & (OFFSET * 2 - 1)) - OFFSET], 1)
    obs_group = image_group[obs_image]; pairs = np.unique(obs_point * groups + obs_group)
    occupied = np.bincount((pairs % groups) * count + inverse[pairs // groups], minlength=groups * count).reshape(groups, count)
    facing = np.abs(np.einsum('ij,ij->i', photos.point_normals(points)[obs_point], photos.unit_rows(points[obs_point] - centers[obs_image]))) >= MIN_COS
    rays = np.random.default_rng(seed).permutation(np.flatnonzero(facing))[:MAX_RAYS]
    free = free_space(cells_of, keys, centers[obs_image[rays]], points[obs_point[rays]], obs_group[rays], groups, cell)
    absent = (free >= MIN_FREE) & (free > FREE_RATIO * around(keys, cells, occupied)) & (occupied.sum(0) - occupied >= MIN_POINTS)
    for _ in range(GROW):
        grown = absent.copy(); source = np.flatnonzero(absent.any(0))
        for offset in NEIGHBOURS:
            code = encode(cells[source] + offset); at = np.searchsorted(keys, code).clip(max=count - 1); hit = keys[at] == code
            for g in range(groups): np.logical_or.at(grown[g], at[hit], absent[g, source[hit]] & (occupied[g, at[hit]] == 0))
        absent = grown
    seen, voxel_groups = np.zeros(len(points), np.int64), np.zeros(count, np.int64); np.bitwise_or.at(seen, obs_point, 1 << obs_group); np.bitwise_or.at(voxel_groups, inverse, seen)
    return {'keys': keys, 'mobile': absent.any(0), 'groups': voxel_groups, 'cells': cells, 'cell': cell, 'point_groups': seen, 'point_mobile': absent.any(0)[inverse]}


def computed(project, source):
    _, cache = photos.source_cache(project, source)
    if not os.path.exists(os.path.join(cache, 'meta.json')): raise ValueError(f'source {source} : positions des photos non calculées')
    return photos.load_json(os.path.join(cache, 'meta.json')), cache


def mobility(project, source):
    meta, cache = computed(project, source); model = tracks(cache); image_group = image_groups(project, source, meta, model)
    centers = photos.camera_centers(model); up = photos.unit(photos.image_ups(model).sum(0)); cell = CELL_M * (np.median(centers @ up) - np.percentile(model['P'] @ up, 2)) / photos.CAMERA_HEIGHT_M
    signature = hashlib.sha256(image_group.tobytes() + f'{VERSION, CELL_M, MIN_FREE, FREE_RATIO, MIN_POINTS, MIN_COS, GROW, MAX_RAYS, MARGIN, cell, len(model["P"])}'.encode()).hexdigest()
    path = os.path.join(cache, 'groupes.npz')
    if os.path.exists(path):
        with np.load(path) as saved:
            if str(saved['signature']) == signature: return dict(saved)
    obs_point = np.repeat(np.arange(len(model['P'])), np.diff(model['Tptr']))
    result = {**analyse(model['P'], obs_point, model['Timg'], centers, image_group, int(image_group.max()) + 1, cell), 'signature': signature}
    np.savez(path, **result); return result


def lookup(result, points):
    tree = cKDTree((result['cells'] + 0.5) * float(result['cell'])); distance, at = tree.query(np.asarray(points, float), distance_upper_bound=1.5 * float(result['cell']))
    found = np.isfinite(distance); at = np.minimum(at, len(result['keys']) - 1)
    return found, np.where(found, result['groups'][at], 0), found & result['mobile'][at]


def fixed(project, source, points=None):
    result = mobility(project, source)
    return ~result['point_mobile'] if points is None else ~lookup(result, points)[2]


def colours(project, source, points, mode):
    found, seen, mobile = lookup(mobility(project, source), points)
    if mode == 'mobilite': return np.where(mobile[:, None], MOBILE, FIXED).astype(np.uint8)
    single = (seen & (seen - 1)) == 0; first = np.log2(np.maximum(seen, 1)).astype(int)
    return np.where((found & single)[:, None], PALETTE[first % len(PALETTE)], np.where(found[:, None], SHARED, UNSEEN)).astype(np.uint8)


def thumbs(paths, limit):
    chosen = [paths[int(i)] for i in np.unique(np.linspace(0, len(paths) - 1, min(limit, len(paths))).round())]
    for path in chosen:
        if path not in THUMBS: THUMBS[path] = photos.thumbnail(path, 64)
    return [THUMBS[path] for path in chosen if THUMBS[path]]


@rpc('photos.groups')
def groups(ctx, source='photos', thumbs_per_group=8):
    project = ctx.require_project(); listed = sessions(project, source)
    return {'gap_min': gap_minutes(project, source), 'groups': [{**{k: v for k, v in s.items() if k != 'photos'}, 'count': len(s['photos']), 'thumbs': thumbs(s['photos'], int(thumbs_per_group))}
                                                              for s in listed]}


@rpc('photos.set_gap')
def set_gap(ctx, source, minutes):
    if not float(minutes) > 0: raise ValueError(f'écart invalide : {minutes} min')
    project = ctx.require_project(); photos.ensure_schema(project)
    project.db.execute('insert or replace into photos_gap values (?,?)', (source, float(minutes))); project.db.commit()
    return {'source': source, 'gap_min': float(minutes)}


@rpc('photos.set_group')
def set_group(ctx, source, id, name=None, excluded=None):
    project = ctx.require_project(); session = next((s for s in sessions(project, source) if s['id'] == id), None)
    if not session: raise ValueError(f'séance inconnue : {id}')
    if name is not None: project.db.execute('insert or replace into photos_groups values (?,?,?)', (source, id, name.strip() or None))
    if excluded is not None:
        project.db.executemany('delete from photos_excluded where source=? and path=?', [(source, path) for path in session['photos']])
        if excluded: project.db.executemany('insert into photos_excluded values (?,?)', [(source, path) for path in session['photos']])
        photos.invalidate(project, source)
    project.db.commit(); return {'source': source, 'id': id}


@rpc('photos.mobility')
def mobility_rpc(ctx, source='photos'):
    project = ctx.require_project(); result = mobility(project, source); seen = result['point_groups']; total = max(len(seen), 1)
    listed = sessions(project, source); meta, cache = computed(project, source); model = photos.load_model(cache)
    used = np.bincount(image_groups(project, source, meta, model), minlength=len(listed))
    return {'points': len(seen), 'mobile': int(result['point_mobile'].sum()), 'mobile_share': float(result['point_mobile'].sum() / total),
            'shared_share': float(((seen & (seen - 1)) != 0).sum() / total),
            'groups': [{'id': s['id'], 'name': s['name'], 'photos': len(s['photos']), 'registered': int(used[k]), 'points': int(((seen >> k) & 1).sum())} for k, s in enumerate(listed)]}


@rpc('photos.fixed')
def fixed_rpc(ctx, source, points_xyz_m, pose=None):
    project = ctx.require_project(); _, frame, _ = photos.photo_state(project, source, compute=False)
    linear_map, shift, _ = photos.project_transform(project, source, frame, pose)
    raw = np.linalg.solve(linear_map, (np.atleast_2d(np.asarray(points_xyz_m, float)) - shift).T).T
    return fixed(project, source, raw).tolist()
