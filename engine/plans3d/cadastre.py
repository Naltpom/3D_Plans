import json, math, os, re, shutil
from functools import lru_cache
import numpy as np
from scipy.spatial import cKDTree
from .core import Obs
from .project import extractor, file_hash
from .rpc import rpc

SIGMA_COORDINATE, SIGMA_HEIGHT = 0.3, 0.5
PDF_POINT_M = 0.0254 / 72
CROSS_ARM_M, GRID_M, TOUCH_M, SHARED_M = 10.0, 100.0, 0.1, 0.05
KIND_BY_EXTENSION = {'.pdf': 'cadastre_pdf', '.geojson': 'cadastre_geojson', '.json': 'cadastre_geojson'}
ECCENTRICITY = math.sqrt(2 / 298.257222101 - 1 / 298.257222101 ** 2)


@lru_cache(None)
def lambert_conic(epsg):
    if epsg == 2154: lat_origin, lat_1, lat_2, false_east, false_north = 46.5, 44, 49, 7e5, 6.6e6
    elif 3942 <= epsg <= 3950:
        zone = epsg - 3900; lat_origin, lat_1, lat_2, false_east, false_north = zone, zone - .75, zone + .75, 1.7e6, (zone - 41) * 1e6 + 2e5
    else: raise ValueError(f'EPSG {epsg} non géré (2154, 3942–3950, 4326)')
    e = ECCENTRICITY
    m = lambda phi: math.cos(phi) / math.sqrt(1 - (e * math.sin(phi)) ** 2)
    t = lambda phi: math.tan(math.pi / 4 - phi / 2) / ((1 - e * math.sin(phi)) / (1 + e * math.sin(phi))) ** (e / 2)
    lat_origin, lat_1, lat_2, lon_origin = map(math.radians, (lat_origin, lat_1, lat_2, 3))
    n = math.log(m(lat_1) / m(lat_2)) / math.log(t(lat_1) / t(lat_2))
    scale = 6378137.0 * m(lat_1) / (n * t(lat_1) ** n); rho_origin = scale * t(lat_origin) ** n

    def forward(lon, lat):
        rho, theta = scale * t(math.radians(lat)) ** n, n * (math.radians(lon) - lon_origin)
        return false_east + rho * math.sin(theta), false_north + rho_origin - rho * math.cos(theta)

    def inverse(x, y):
        dx, dy = x - false_east, rho_origin - (y - false_north)
        tt = (math.hypot(dx, dy) / scale) ** (1 / n); phi = math.pi / 2 - 2 * math.atan(tt)
        for _ in range(8): phi = math.pi / 2 - 2 * math.atan(tt * ((1 - e * math.sin(phi)) / (1 + e * math.sin(phi))) ** (e / 2))
        return math.degrees(math.atan2(dx, dy) / n + lon_origin), math.degrees(phi)

    def factor(x, y): return n * math.hypot(x - false_east, rho_origin - y + false_north) / (6378137.0 * m(math.radians(inverse(x, y)[1])))
    return forward, inverse, factor


def reproject(points, source_epsg, target_epsg):
    if source_epsg == target_epsg: return [tuple(p[:2]) for p in points]
    to_lonlat = lambert_conic(source_epsg)[1] if source_epsg != 4326 else (lambda x, y: (x, y))
    from_lonlat = lambert_conic(target_epsg)[0] if target_epsg != 4326 else (lambda x, y: (x, y))
    return [from_lonlat(*to_lonlat(*p[:2])) for p in points]


def signed_area(points):
    x, y = np.asarray(points, float).reshape(-1, 2).T
    return 0.5 * float(x @ np.roll(y, -1) - y @ np.roll(x, -1))


def area(points): return abs(signed_area(points))


def inside(point, polygon):
    x, y = np.asarray(polygon, float).T; x2, y2 = np.roll(x, -1), np.roll(y, -1)
    crossing = ((y > point[1]) != (y2 > point[1])) & (point[0] < (x2 - x) * (point[1] - y) / np.where(y2 == y, 1e-300, y2 - y) + x)
    return bool(crossing.sum() % 2)


def polygons_gap(first, second):
    a, b = np.asarray(first, float), np.asarray(second, float)
    if any(inside(p, b) for p in a) or any(inside(p, a) for p in b): return 0.0
    cross = lambda o, p, q: (p[..., 0] - o[..., 0]) * (q[..., 1] - o[..., 1]) - (p[..., 1] - o[..., 1]) * (q[..., 0] - o[..., 0])
    a1, a2, b1, b2 = a[:, None], np.roll(a, -1, 0)[:, None], b[None], np.roll(b, -1, 0)[None]
    if np.any((cross(a1, a2, b1) * cross(a1, a2, b2) < 0) & (cross(b1, b2, a1) * cross(b1, b2, a2) < 0)): return 0.0

    def point_to_edges(points, polygon):
        start, step = polygon, np.roll(polygon, -1, 0) - polygon
        t = np.clip(((points[:, None] - start) * step).sum(-1) / np.maximum((step * step).sum(-1), 1e-300), 0, 1)
        return np.linalg.norm(points[:, None] - (start + t[..., None] * step), axis=-1).min()
    return float(min(point_to_edges(a, b), point_to_edges(b, a)))


def clean_ring(ring, tolerance):
    out = []
    for p in ring:
        p = (float(p[0]), float(p[1]))
        if not out or math.dist(p, out[-1]) > tolerance: out.append(p)
    if len(out) > 1 and math.dist(out[0], out[-1]) <= tolerance: out.pop()
    return out


def entity(kind, nature, points, label='', observed=None, height=None, sigma=SIGMA_COORDINATE, contenance=None):
    observed = observed or [True] * len(points)
    return {'type': kind, 'nature': nature, 'label': label, 'pts': points, 'observed': observed, 'h': height, 'sigma': sigma,
            'partiel': not all(observed), 'principal': False, 'contenance': contenance}


def read_geojson(path):
    data = json.load(open(path, encoding='utf-8'))
    match = re.search(r'EPSG:+(\d+)', json.dumps(data.get('crs') or {})); epsg = int(match.group(1)) if match else 4326
    entities_found = []
    for feature in data.get('features', []):
        geometry_, props = feature.get('geometry') or {}, feature.get('properties') or {}
        kind = ('parcelle' if any(k in props for k in ('idu', 'contenance', 'numero', 'section')) else
                'bati' if any(k in props for k in ('hauteur', 'construction_legere', 'usage_1')) or props.get('type') in ('01', '02') else None)
        polygons = {'Polygon': [geometry_.get('coordinates')], 'MultiPolygon': geometry_.get('coordinates')}.get(geometry_.get('type'))
        if not kind or not polygons: continue
        idu = str(props.get('idu') or '')
        label = f"{props['section']}{props['numero']}" if props.get('section') and props.get('numero') else \
            idu[-6:] if len(idu) == 14 else str(props.get('numero') or feature.get('id') or idu)
        for polygon in polygons:
            ring = clean_ring(polygon[0], 1e-9 if epsg == 4326 else 1e-4)
            if len(ring) >= 3:
                entities_found.append(entity(kind, ('léger' if props.get('construction_legere') or props.get('type') == '02' else 'dur') if kind == 'bati' else '',
                                             ring, label, height=props.get('hauteur'),
                                             sigma=float(props.get('precision_planimetrique') or SIGMA_COORDINATE),
                                             contenance=(props.get('contenance') or props.get('surface')) if kind == 'parcelle' and len(polygons) == len(polygon) == 1 else None))
    return entities_found, epsg


def path_rings(items):
    rings, current = [], []
    for item in items:
        if item[0] in ('re', 'qu'):
            quad = item[1].quad if item[0] == 're' else item[1]
            rings.append([(p.x, p.y) for p in (quad.ul, quad.ur, quad.lr, quad.ll, quad.ul)]); continue
        start, end = (item[1].x, item[1].y), (item[-1].x, item[-1].y)
        if current and math.dist(current[-1], start) < 1e-3: current.append(end)
        else: current and rings.append(current); current = [start, end]
    current and rings.append(current)
    return [ring for raw in rings if len(raw) >= 4 and math.dist(raw[0], raw[-1]) < 1e-3 and len(ring := clean_ring(raw, 1e-3)) >= 3]


def read_pdf(path):
    import pymupdf
    with pymupdf.open(path) as document:
        page = document[0]; text = page.get_text(); drawings = page.get_drawings(); page_rect = +page.rect
    match = re.search(r'dition\s*:\s*1\s*/\s*(\d{3,6})', text) or re.search(r'\b1\s*/\s*(\d{3,6})\b', text)
    stated_scale = float(match.group(1)) if match else None
    match = re.search(r'CC\s*(4[2-9]|50)\b', text)
    epsg = 3900 + int(match.group(1)) if match else 2154 if re.search(r'lambert[\s-]*93', text, re.I) else None
    rects = [i[1] if i[0] == 're' else i[1].rect for d in drawings if d.get('color') and 's' in d['type'] for i in d['items'] if i[0] in ('re', 'qu')]
    shapes, lines = [], []
    for drawing in drawings:
        fill, color, items = drawing.get('fill'), drawing.get('color'), drawing['items']
        if len(items) == 1 and items[0][0] == 'l' and color: lines.append((items[0][1], items[0][2])); continue
        lightness = sum(fill) / 3 if fill else None
        if fill and .05 < lightness < .97: kind, nature = 'bati', ('dur' if lightness < .75 else 'léger')
        elif not fill and color and max(color) < .2 and (drawing.get('width') or 0) <= 1.5 and not re.search(r'\[\s*[\d.]', drawing.get('dashes') or ''):
            kind, nature = 'parcelle', ''
        else: continue
        rings = path_rings(items)
        shapes += [(kind, nature, ring) for ring in ([max(rings, key=area)] if kind == 'bati' and rings else rings)]
    count_inside = lambda r: sum(r.x0 < x < r.x1 and r.y0 < y < r.y1 for _, _, ring in shapes for x, y in ring)
    candidates = [(r, count_inside(r)) for r in rects if r.width * r.height > .2 * page_rect.width * page_rect.height]
    most = max((c for _, c in candidates), default=0)
    frame = min((r for r, c in candidates if c and c >= .5 * most), key=lambda r: r.width * r.height, default=page_rect)
    found = []
    for kind, nature, ring in shapes:
        observed = [frame.x0 + .5 < x < frame.x1 - .5 and frame.y0 + .5 < y < frame.y1 - .5 for x, y in ring]
        if any(observed): found.append(entity(kind, nature, ring, observed=observed))
    horizontal = [(a, b) for a, b in lines if abs(a.y - b.y) < 1e-3 * abs(a.x - b.x)]
    vertical = [(a, b) for a, b in lines if abs(a.x - b.x) < 1e-3 * abs(a.y - b.y)]
    crosses = [((h[0].x + h[1].x) / 2, h[0].y, abs(h[1].x - h[0].x), abs(v[1].y - v[0].y)) for h in horizontal for v in vertical
               if math.dist(((h[0].x + h[1].x) / 2, h[0].y), (v[0].x, (v[0].y + v[1].y) / 2)) < 2]
    stated_m_per_point = stated_scale * PDF_POINT_M if stated_scale else None
    if crosses:
        m_per_point_x = float(np.median([CROSS_ARM_M / c[2] for c in crosses]))
        m_per_point_y = float(np.median([CROSS_ARM_M / c[3] for c in crosses]))
    elif stated_m_per_point: m_per_point_x = m_per_point_y = stated_m_per_point
    else: raise ValueError('échelle introuvable (ni « 1/N » dans le texte, ni croix de quadrillage)')
    warnings = []
    if crosses and stated_m_per_point and max(abs(m_per_point_x / stated_m_per_point - 1), abs(m_per_point_y / stated_m_per_point - 1)) > .05:
        warnings.append(f"échelle d'édition 1/{stated_scale:.0f} contredite par la croix de quadrillage : 1/{m_per_point_x / PDF_POINT_M:.0f} (E-O) "
                        f"× 1/{m_per_point_y / PDF_POINT_M:.0f} (N-S), image étirée — échelle de la croix retenue")
    center = ((frame.x0 + frame.x1) / 2, (frame.y0 + frame.y1) / 2); parcels = [e for e in found if e['type'] == 'parcelle']
    main = next((e for e in parcels if inside(center, e['pts'])), None) or \
        min(parcels, key=lambda e: math.dist(center, np.mean(e['pts'], 0)), default=None)
    if main: main['principal'] = True
    for e in found: e['pts'] = [(x * m_per_point_x, -y * m_per_point_y) for x, y in e['pts']]
    found = [e for e in found if e['type'] == 'bati' or e['principal'] or area(e['pts']) > 1]
    info = {'epsg': epsg, 'echelle_texte': stated_scale, 'echelle': [m_per_point_x / PDF_POINT_M, m_per_point_y / PDF_POINT_M],
            'echelle_source': 'croix' if crosses else 'texte', 'croix': [(c[0] * m_per_point_x, -c[1] * m_per_point_y) for c in crosses],
            'avertissements': warnings}
    return found, info


def cache_path(folder, name=None):
    directory = os.path.join(folder, 'cache', 'cadastre')
    return directory if name is None else os.path.join(directory, name + '.json')


def load_cache(folder, name):
    path = folder and cache_path(folder, name)
    return json.load(open(path, encoding='utf-8')) if path and os.path.exists(path) else None


def save_cache(folder, name, data):
    if folder:
        os.makedirs(cache_path(folder), exist_ok=True)
        json.dump(data, open(cache_path(folder, name), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def cached_sources(folder):
    directory = folder and cache_path(folder)
    if not directory or not os.path.isdir(directory): return []
    return [load_cache(folder, f[:-5]) for f in sorted(os.listdir(directory)) if f.endswith('.json') and f != 'origine.json']


def param_prefix(source_id): return re.sub(r'\W', '_', source_id)


def extract(path, source_id, kind=None, folder=None):
    kind = kind or KIND_BY_EXTENSION.get(os.path.splitext(path)[1].lower(), 'cadastre_geojson')
    origin, info, warnings = load_cache(folder, 'origine'), {}, []
    if kind == 'cadastre_pdf':
        found, info = read_pdf(path); epsg, warnings = info['epsg'], info['avertissements']
        main = next((e for e in found if e['principal']), found[0] if found else None)
        georeferenced = bool(origin and info['croix'] and epsg and main)
        if georeferenced:
            (origin_e, origin_n), (cross_e, cross_n), first = reproject([(origin['E'], origin['N'])], origin['epsg'], epsg)[0], info['croix'][0], main['pts'][0]
            shift_e = round((origin_e + cross_e - first[0]) / GRID_M) * GRID_M - cross_e
            shift_n = round((origin_n + cross_n - first[1]) / GRID_M) * GRID_M - cross_n
            for e in found: e['pts'] = [(x + shift_e, y + shift_n) for x, y in e['pts']]
            gap = math.dist((first[0] + shift_e, first[1] + shift_n), (origin_e, origin_n))
            if gap > GRID_M / 3: warnings.append(f"géoréférencement douteux : parcelle principale à {gap:.0f} m de l'origine du projet")
    else:
        found, source_epsg = read_geojson(path); epsg = origin['epsg'] if origin else (2154 if source_epsg == 4326 else source_epsg)
        for e in found: e['pts'] = reproject(e['pts'], source_epsg, epsg)
        georeferenced = True
        main = next((e for e in found if e['type'] == 'parcelle'), found[0] if found else None)
        if main and main['type'] == 'parcelle': main['principal'] = True
    if not found: raise ValueError(f'{os.path.basename(path)} : aucune parcelle ni aucun bâtiment trouvé')
    if georeferenced and 'k' not in (origin or {}):
        origin = origin or {'E': main['pts'][0][0], 'N': main['pts'][0][1], 'epsg': epsg, 'source': source_id}
        origin['k'] = lambert_conic(origin['epsg'])[2](origin['E'], origin['N']); save_cache(folder, 'origine', origin)
    (origin_e, origin_n), scale = ((origin['E'], origin['N']), origin['k']) if georeferenced else (main['pts'][0], 1)
    for e in found:
        absolute = reproject(e['pts'], epsg, origin['epsg']) if georeferenced else e['pts']
        e['pts'] = [(round((east - origin_e) / scale, 4), round((origin_n - north) / scale, 4)) for east, north in absolute]
    main_parcel = next((e['pts'] for e in found if e['type'] == 'parcelle' and e['principal']), None) or next(
        (e['pts'] for s in cached_sources(folder) if s['georeference'] and georeferenced for e in s['entites'] if e['type'] == 'parcelle' and e['principal']), None)
    buildings = [e for e in found if e['type'] == 'bati' and main_parcel and inside(np.mean(e['pts'], 0), main_parcel)]
    if buildings: max(buildings, key=lambda e: area(e['pts']))['principal'] = True
    prefix, observations, params, counters = param_prefix(source_id), [], {}, {'parcelle': 0, 'bati': 0}
    for e in found:
        e['id'] = ('p' if e['type'] == 'parcelle' else 'b') + str(counters[e['type']]); counters[e['type']] += 1
        e['observe'] = e['type'] == 'parcelle' or (main_parcel is not None and polygons_gap(e['pts'], main_parcel) <= TOUCH_M)
        e['refs'] = [f"{e['id']}_{i}" for i in range(len(e['pts']))]
    vertices = [(e, i) for e in found if e['observe'] for i, observed in enumerate(e['observed']) if observed]
    points = np.array([e['pts'][i] for e, i in vertices]).reshape(-1, 2)
    for (e, i), near in zip(vertices, cKDTree(points).query_ball_point(points, SHARED_M)):
        first, j = vertices[min(near)]; e['refs'][i] = first['refs'][j]
    for e in found:
        if not e['observe']: continue
        is_parcel = e['type'] == 'parcelle'
        name = (f"parcelle {e['label'] or e['id']}" if is_parcel else f"bâtiment {e['nature']} {e['label'] or e['id']}") + \
            ((' (principale)' if is_parcel else ' (principal)') if e['principal'] else '')
        for i, ((x, z), observed, ref) in enumerate(zip(e['pts'], e['observed'], e['refs'])):
            for axis, value in (('x', x), ('z', z)) if observed and ref == f"{e['id']}_{i}" else ():
                param = f"{prefix}_{ref}_{axis}"; params[param] = (value, axis, f'{name} · sommet {i} · {axis.upper()}')
                observations.append(Obs(f"{source_id}:{ref}_{axis}", param, value, e['sigma'], source_id, 'lu', params[param][2]))
        if e['contenance'] and not e['partiel']:
            p, c = [f"{prefix}_{ref}_" for ref in e['refs']], float(e['contenance'])
            shoelace = ' + '.join(f'{a}x*{b}z - {b}x*{a}z' for a, b in zip(p, p[1:] + p[:1]))
            observations.append(Obs(f"{source_id}:{e['id']}_aire", f'abs({shoelace}) / 2', c, max(.01 * c, 1.0), source_id, 'lu', f'{name} · contenance {c:g} m²'))
        if e['h'] is not None:
            param = f"{prefix}_{e['id']}_h"; params[param] = (float(e['h']), 'y', f'{name} · hauteur')
            observations.append(Obs(f"{source_id}:{e['id']}_h", param, float(e['h']), SIGMA_HEIGHT, source_id, 'lu', f'{name} · hauteur (BD TOPO)'))
    keys = ('id', 'type', 'nature', 'label', 'pts', 'refs', 'h', 'principal', 'partiel', 'observe', 'contenance')
    data = {'source': source_id, 'kind': kind, 'georeference': georeferenced, 'origine': origin if georeferenced else None, **info,
            'avertissements': warnings, 'entites': [{k: e[k] for k in keys} for e in found]}
    save_cache(folder, source_id, data)
    return observations, params, data


def run_extractor(path, source_id, kind):
    folder = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    return extract(path, source_id, kind, folder if os.path.exists(os.path.join(folder, 'project.db')) else None)[0]


@extractor('cadastre_pdf')
def cadastre_pdf(path, source_id): return run_extractor(path, source_id, 'cadastre_pdf')


@extractor('cadastre_geojson')
def cadastre_geojson(path, source_id): return run_extractor(path, source_id, 'cadastre_geojson')


def sync_source(project, source_id, path, kind):
    observations, params, data = extract(path, source_id, kind, project.folder)
    own = re.compile(rf'{re.escape(param_prefix(source_id))}_[pb]\d+_(\d+_[xz]|h)')
    stale = [(name,) for (name,) in project.db.execute('select name from params') if own.fullmatch(name) and name not in params]
    project.db.executemany('delete from params where name=?', stale)
    project.db.executemany('insert or replace into params values (?,?,?,?)', [(name, *v) for name, v in params.items()])
    project.db.execute("update sources set hash='' where id=?", (source_id,))
    return observations, params, data


def add_source(project, path, source_id=None):
    path = os.path.abspath(path); kind = KIND_BY_EXTENSION.get(os.path.splitext(path)[1].lower())
    if not kind: raise ValueError(f'{path} : extension non reconnue (.pdf, .geojson, .json)')
    sources_dir = os.path.join(project.folder, 'sources'); os.makedirs(sources_dir, exist_ok=True)
    source_id = project.free_source_id(param_prefix(source_id or os.path.splitext(os.path.basename(path))[0]))
    stored = os.path.join(sources_dir, os.path.basename(path))
    if os.path.normcase(os.path.dirname(path)) != os.path.normcase(sources_dir):
        if os.path.exists(stored) and file_hash(stored) != file_hash(path): stored = os.path.join(sources_dir, f'{source_id}_{os.path.basename(path)}')
        shutil.copy2(path, stored)
    project.add_source(source_id, stored, kind)
    observations, params, data = sync_source(project, source_id, stored, kind)
    others = project.db.execute("select id, path, kind from sources where kind in ('cadastre_pdf', 'cadastre_geojson') and id!=?", (source_id,))
    for other in others.fetchall(): sync_source(project, *other)
    project.db.commit()
    return {'id': source_id, 'kind': kind, 'entites': len(data['entites']), 'parametres': len(params), 'observations': len(observations),
            'georeference': data['georeference'], 'avertissements': data['avertissements']}


def entities(folder, source_id):
    data = load_cache(folder, source_id)
    if data is None: raise ValueError(f'source cadastre inconnue : {source_id}')
    return data


def geometry(folder, result=None):
    index = result._idx if result is not None else {}
    solved = lambda name, value: float(result.x[index[name]]) if name in index else value
    out = []
    for source in cached_sources(folder):
        prefix = param_prefix(source['source'])
        for e in source['entites']:
            refs = e.get('refs') or [f"{e['id']}_{i}" for i in range(len(e['pts']))]
            points = [[solved(f'{prefix}_{ref}_x', x), solved(f'{prefix}_{ref}_z', z)] for ref, (x, z) in zip(refs, e['pts'])]
            height = solved(f"{prefix}_{e['id']}_h", e['h']) if e['h'] is not None else None
            out.append({'source': source['source'], 'georeference': source['georeference'],
                        **{k: e[k] for k in ('id', 'type', 'nature', 'label', 'principal', 'partiel', 'observe')}, 'pts': points, 'h': height})
    return {'origine': load_cache(folder, 'origine'), 'entites': out}


@rpc('cadastre.add')
def rpc_add(ctx, path, id=None): return add_source(ctx.require_project(), path, id)


@rpc('cadastre.entities')
def rpc_entities(ctx, source): return entities(ctx.require_project().folder, source)


@rpc('cadastre.geometry')
def rpc_geometry(ctx): return geometry(ctx.require_project().folder, ctx.result)
