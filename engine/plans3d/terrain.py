import itertools, json, math, os, shutil
import numpy as np
from . import cadastre, components as C
from .core import Obs
from .project import Project, extractor
from .rpc import rpc

SIGMA, MAX_NODES, EDGE = 0.2, 2500, 0.5


def make_grid(values, east, north, dx, dy, epsg=2154, nodata=None, sigma=SIGMA):
    y = np.array(values, float)
    if nodata is not None: y[y == float(nodata)] = np.nan
    if y.ndim != 2 or min(y.shape) < 2 or not np.isfinite(y).any(): raise ValueError('grille d’altitudes : au moins 2 × 2 nœuds renseignés')
    return {'y': y, 'E': float(east), 'N': float(north), 'dx': float(dx), 'dy': float(dy), 'epsg': int(epsg), 'sigma': float(sigma)}


def read_asc(path):
    with open(path, encoding='ascii', errors='replace') as f: tokens = f.read().split()
    head, k = {}, 0
    while tokens[k][0].isalpha(): head[tokens[k].lower()] = float(tokens[k + 1]); k += 2
    nx, nz, step, corner = int(head['ncols']), int(head['nrows']), head['cellsize'], 'xllcorner' in head
    east, south = (head[f'{a}ll{"corner" if corner else "center"}'] + (step / 2 if corner else 0) for a in 'xy')
    return make_grid(np.array(tokens[k:k + nx * nz], float).reshape(nz, nx), east, south + (nz - 1) * step, step, step, nodata=head.get('nodata_value'))


def read_tiff(path):
    from PIL import Image
    with Image.open(path) as image: values, tags = np.asarray(image, float), dict(image.tag_v2)
    if not {33550, 33922} <= tags.keys(): raise ValueError(f'{os.path.basename(path)} : TIFF sans géoréférencement')
    (sx, sy), (i0, j0, _, east, north), keys = tags[33550][:2], tags[33922][:5], tags.get(34735, ())
    geo = dict(zip(keys[4::4], keys[7::4])); half = 0.5 if geo.get(1025, 1) == 1 else 0.0
    return make_grid(values, east + (half - i0) * sx, north - (half - j0) * sy, sx, sy, geo.get(3072, 2154), tags.get(42113))


def read_json(path):
    with open(path, encoding='utf-8') as f: data = json.load(f)
    origin, (px, pz) = data['origine'], np.broadcast_to(np.asarray(data['pas'], float), 2)
    values, rest = np.array(data['valeurs'], float).reshape(int(data['nz']), int(data['nx'])), (data.get('nodata'), data.get('sigma', SIGMA))
    if 'lon' not in origin: return make_grid(values, origin['E'], origin['N'], px, pz, origin.get('epsg', data.get('epsg', 2154)), *rest)
    forward, _, factor = cadastre.lambert_conic(2154); east, north = forward(origin['lon'], origin['lat']); k = factor(east, north)
    return make_grid(values, east, north, px * k, pz * k, 2154, *rest)


def read_grid(path): return {'.asc': read_asc, '.tif': read_tiff, '.tiff': read_tiff}.get(os.path.splitext(path)[1].lower(), read_json)(path)


def frame(grid, origin):
    nz, nx = grid['y'].shape
    corners = [(grid['E'] + i * grid['dx'], grid['N'] - j * grid['dy']) for i, j in ((0, 0), (nx - 1, 0), (0, nz - 1))]
    p0, p1, p2 = (np.array([e - origin['E'], origin['N'] - n]) / origin['k'] for e, n in cadastre.reproject(corners, grid['epsg'], origin['epsg']))
    return p0, np.column_stack([(p1 - p0) / (nx - 1), (p2 - p0) / (nz - 1)])


def sample(grid, origin, x, z):
    (x0, z0), A = frame(grid, origin); values = grid['y']; nz, nx = values.shape
    fi, fj = np.linalg.solve(A, np.stack([np.ravel(x) - x0, np.ravel(z) - z0]))
    i, j = np.clip(np.floor(fi), 0, nx - 2).astype(int), np.clip(np.floor(fj), 0, nz - 2).astype(int); u, v = fi - i, fj - j
    out = (values[j, i] * (1 - u) + values[j, i + 1] * u) * (1 - v) + (values[j + 1, i] * (1 - u) + values[j + 1, i + 1] * u) * v
    return np.where((fi > -EDGE) & (fi < nx - 1 + EDGE) & (fj > -EDGE) & (fj < nz - 1 + EDGE), out, np.nan).reshape(np.shape(x))


def nodes(o):
    j, i = np.mgrid[0:o['nz'], 0:o['nx']]
    return o['x0'] + i * o['pas_x'], o['z0'] + j * o['pas_z']


def build(comp, quantity, initial, components):
    o = comp['options']; o['nx'], o['nz'] = int(o.get('nx', 2)), int(o.get('nz', 2))
    o.update({k: float(o.get(k, default)) for k, default in (('x0', 0), ('z0', 0), ('pas_x', 1), ('pas_z', o.get('pas_x', 1)), ('alt0', 0))})
    if min(o['nx'], o['nz']) < 2 or min(o['pas_x'], o['pas_z']) <= 0: raise ValueError('terrain : nx, nz ≥ 2 et pas > 0')
    for (j, i), value in np.ndenumerate(np.asarray(o.pop('altitudes', 0.0), float) * np.ones((o['nz'], o['nx']))):
        quantity(f'y_{i}_{j}', 'y', f'altitude du nœud ({i}, {j})', float(value))


def altitude(comp, x, z):
    o, e = comp['options'], comp['expressions']
    fi, fj = (min(max((v - o[a + '0']) / o['pas_' + a], 0.0), o['n' + a] - 1.0) for v, a in ((x, 'x'), (z, 'z')))
    i, j = min(int(fi), o['nx'] - 2), min(int(fj), o['nz'] - 2); u, v = round(fi - i, 9), round(fj - j, 9)
    weights = ((0, 0, (1 - u) * (1 - v)), (1, 0, u * (1 - v)), (0, 1, (1 - u) * v), (1, 1, u * v))
    return sum((w * e[f'y_{i + di}_{j + dj}'] for di, dj, w in weights if w), C.Expr())


def point(comp, x, z): return {'x': C.Expr.of(float(x)), 'y': altitude(comp, x, z), 'z': C.Expr.of(float(z))}


def point_name(name):
    try: x, z = map(float, str(name).split(','))
    except ValueError: return None
    return (x, z) if math.isfinite(x + z) else None


class Anchors(dict):
    def __init__(self, comp, named): super().__init__(named); self.comp = comp
    def __contains__(self, name): return dict.__contains__(self, name) or point_name(name) is not None
    def __missing__(self, name): return point(self.comp, *point_name(name))


def anchors(comp):
    o = comp['options']; x, z = (o[a + '0'] + np.array([0, 0.5, 1]) * (o['n' + a] - 1) * o['pas_' + a] for a in 'xz')
    return Anchors(comp, {**{f'coin_{a}{b}': point(comp, x[2 * a], z[2 * b]) for a in (0, 1) for b in (0, 1)}, 'centre': point(comp, x[1], z[1])})


def geometry(comp, evaluate_expr, walls):
    o, e = comp['options'], comp['expressions']; (x, z), nx = nodes(o), o['nx']
    y = [evaluate_expr(e[f'y_{i}_{j}']) for j in range(o['nz']) for i in range(nx)]
    k = (np.arange(o['nz'] - 1)[:, None] * nx + np.arange(nx - 1)).ravel()
    faces = np.column_stack([k, k + nx, k + 1, k + 1, k + nx, k + nx + 1]).reshape(-1, 3)
    return [{'kind': 'mesh', 'vertices': np.column_stack([x.ravel(), y, z.ravel()]).round(4).tolist(), 'faces': faces.tolist(), 'alt0': o['alt0']}]


C.TYPES['terrain'] = C.ComponentType('Terrain', build, anchors, geometry, '#8b6d4c', {
    'x0': 'nœud nord-ouest (x)', 'z0': 'nœud nord-ouest (z)', 'pas_x': 'pas est (m)', 'pas_z': 'pas sud (m)', 'nx': 'nœuds est', 'nz': 'nœuds sud',
    'alt0': 'altitude du niveau y = 0', 'altitudes': 'altitudes initiales [[…]] (nz × nx)'})


def layout(grid, origin, max_nodes, alt0):
    (x0, z0), A = frame(grid, origin); nz, nx = grid['y'].shape; counts = lambda s: (math.ceil((nx - 1) / s) + 1, math.ceil((nz - 1) / s) + 1)
    cx, cz = counts(next(s for s in itertools.count(1) if math.prod(counts(s)) <= max(max_nodes, 4)))
    o = {'x0': x0, 'z0': z0, 'nx': cx, 'nz': cz, 'pas_x': A[0, 0] * (nx - 1) / (cx - 1), 'pas_z': A[1, 1] * (nz - 1) / (cz - 1)}
    here = sample(grid, origin, 0.0, 0.0)
    o['alt0'] = round(float(here if np.isfinite(here) else np.nanmin(grid['y'])), 2) if alt0 is None else float(alt0)
    heights = sample(grid, origin, *nodes(o))
    return {**o, 'altitudes': (np.where(np.isfinite(heights), heights, np.nanmean(heights)) - o['alt0']).round(3).tolist()}


def add(project, path, id='terrain', max_nodes=MAX_NODES, alt0=None):
    path, folder = os.path.abspath(path), project.folder; grid = read_grid(path)
    sid = project.free_source_id(cadastre.param_prefix(os.path.splitext(os.path.basename(path))[0]))
    stored = os.path.join(folder, 'sources', sid + os.path.splitext(path)[1].lower()); os.makedirs(os.path.dirname(stored), exist_ok=True)
    if os.path.normcase(stored) != os.path.normcase(path): shutil.copy2(path, stored)
    origin = cadastre.load_cache(folder, 'origine')
    if origin is None:
        origin = {'E': grid['E'], 'N': grid['N'], 'epsg': grid['epsg'], 'source': sid, 'k': cadastre.lambert_conic(grid['epsg'])[2](grid['E'], grid['N'])}
        cadastre.save_cache(folder, 'origine', origin)
    comp = C.load(project).get(id) or C.add(project, 'terrain', id, layout(grid, origin, int(max_nodes), alt0))
    if comp['type'] != 'terrain': raise ValueError(f'{id} n’est pas un terrain')
    project.add_source(sid, stored, 'grille_alti'); project.db.execute("update sources set hash='' where kind='grille_alti'"); project.db.commit()
    o = comp['options']
    return {'source': sid, 'terrain': id, 'nx': o['nx'], 'nz': o['nz'], 'noeuds': o['nx'] * o['nz'], 'pas': [o['pas_x'], o['pas_z']], 'alt0': o['alt0'],
            'altitudes': [float(np.nanmin(grid['y'])), float(np.nanmax(grid['y']))]}


@extractor('grille_alti')
def grille_alti(path, sid):
    folder = os.path.dirname(os.path.dirname(os.path.abspath(path))); origin = cadastre.load_cache(folder, 'origine')
    if origin is None or not os.path.exists(os.path.join(folder, 'project.db')): raise ValueError(f'{sid} : grille hors projet ou sans origine (terrain.add)')
    grid, project = read_grid(path), Project(folder)
    try: terrains = [c for c in C.load(project).values() if c['type'] == 'terrain']
    finally: project.db.close()
    observations = []
    for comp in terrains:
        values = sample(grid, origin, *nodes(comp['options'])) - comp['options']['alt0']
        observations += [Obs(f"{sid}:{comp['id']}_{i}_{j}", str(comp['expressions'][f'y_{i}_{j}']), float(values[j, i]), grid['sigma'], sid, 'lu',
                             f"{comp['label']} — altitude du nœud ({i}, {j})") for j, i in np.argwhere(np.isfinite(values))]
    return observations


@rpc('terrain.add')
def rpc_add(ctx, path, id='terrain', max_noeuds=MAX_NODES, alt0=None): return add(ctx.require_project(), path, id, max_noeuds, alt0)


@rpc('terrain.geometry')
def rpc_geometry(ctx): return [s for s in C.geometry(ctx.require_project(), ctx.result) if s['type'] == 'terrain']
