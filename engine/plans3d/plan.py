import io, json, os, re, shutil
import numpy as np
from . import cadastre, components
from .rpc import cotes_add, project_meta, rpc

CLICK_SIGMA_M, CADASTRE_SIGMA_M, OPENING_SIGMA_M, TYPED_SIGMA_CM, NEAR_WALL_M, MARGIN_M = 0.05, 0.2, 0.1, 1.0, 0.6, 0.05
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def free_ids(project, base, count):
    taken, ids, k = set(components.load(project)), [], 1
    while len(ids) < count:
        if f'{base}{k}' not in taken: ids.append(f'{base}{k}')
        k += 1
    return ids


def prior(project, owner, name, value, sigma, label):
    project.db.execute('insert or replace into obs values (?,?,?,?,?,?,?)',
                       (f'comp:{owner}:{name}', f'comp:{owner}', name, float(value), sigma, components.PRIOR_STATUS, label))


def current_values(project): return {name: value for name, (value, _) in components.param_values(project).items()}


def wall_ends(comp, values):
    return [np.array([components.evaluate(comp['expressions'][f'{end}_{axis}'], values) for axis in 'xz']) for end in ('start', 'end')]


def trace(ctx, points, closed=True, lengths=None, refs=None, height=None, inward=False, sigma=CLICK_SIGMA_M, source='tracé', cote_status='mesuré', cote_sigma=TYPED_SIGMA_CM, names=None):
    project = ctx.require_project(); points = np.asarray(points, float).reshape(-1, 2); n = len(points)
    edges = list(range(n if closed else n - 1)); after = lambda e: (e + 1) % n
    if len(edges) < (3 if closed else 1): raise ValueError('tracé : au moins 2 points, 3 pour une pièce fermée')
    if any(np.linalg.norm(points[after(e)] - points[e]) < 0.01 for e in edges): raise ValueError('tracé : deux points confondus')
    taken = set(components.load(project)); ids = [f'mur{n}' if f'mur{n}' not in taken else free_ids(project, f'mur{n}_', 1)[0] for n in names] if names else free_ids(project, 'mur', len(edges))
    labels = [f'Mur {n}' for n in names] if names else [f'Mur {i[3:]}' for i in ids]; created = {}
    runs = ['' if refs else 'x' if abs(points[after(e)][1] - points[e][1]) < 1e-6 else 'z' if abs(points[after(e)][0] - points[e][0]) < 1e-6 else '' for e in edges]

    def coordinate(i, axis):
        if refs and refs[i].get(axis): return refs[i][axis]
        for e in (i, i - 1 if i else n - 1 if closed else None):
            if e in edges and runs[e] == ('z' if axis == 'x' else 'x'):
                created.setdefault(f'{ids[e]}_{axis}', (points[i]['xz'.index(axis)], axis, f'{labels[e]} — position ({axis})', ids[e]))
                return f'{ids[e]}_{axis}'
        e, end = (i, 0) if i in edges else (i - 1, 1)
        created.setdefault(f'{ids[e]}_{axis}{end}', (points[i]['xz'.index(axis)], axis, f"{labels[e]} — {'début' if end == 0 else 'fin'} ({axis})", ids[e]))
        return f'{ids[e]}_{axis}{end}'

    def sens(e):
        a, b = points[e], points[after(e)]; direction = (b - a) / np.linalg.norm(b - a); left = np.array([-direction[1], direction[0]])
        side = (1.0 if cadastre.inside((a + b) / 2 + 0.01 * left, points) else -1.0) * (1 if inward else -1) if closed else 1.0
        normal = side * left
        return float(np.sign(normal[1])) if runs[e] == 'x' else float(np.sign(normal[0])) if runs[e] == 'z' else side

    options = []
    for e in edges:
        run, start, stop = runs[e], e, after(e)
        if run: geometry = {'axe': run, ('z' if run == 'x' else 'x'): coordinate(start, 'z' if run == 'x' else 'x'), f'{run}0': coordinate(start, run), f'{run}1': coordinate(stop, run)}
        else: geometry = {'x0': coordinate(start, 'x'), 'z0': coordinate(start, 'z'), 'x1': coordinate(stop, 'x'), 'z1': coordinate(stop, 'z')}
        options.append({**geometry, 'sens': sens(e), 'source': source, 'h': height if isinstance(height, str) else (height or 2.5) if e == 0 else f'{ids[0]}_h'})
    for name, (value, axis, label, _) in created.items(): project.add_param(name, float(value), axis, label)
    for e, opts in zip(edges, options): components.add(project, 'mur', ids[e], opts, labels[e])
    for name, (value, _, label, owner) in created.items(): prior(project, owner, name, value, sigma, f'{label} (tracé)')
    project.db.commit(); values = current_values(project); loaded = components.load(project); os.makedirs(os.path.join(project.folder, 'sources'), exist_ok=True)
    for e, length in zip(edges, lengths or []):
        if length: cotes_add(ctx, f'trace_{ids[e]}', components.measure_expr((loaded[ids[e]], 'a'), (loaded[ids[e]], 'b'), values=values), float(length) * 100,
                             float(cote_sigma), label=f"{labels[e]} — longueur {'lue sur le croquis' if cote_status == 'lu' else 'saisie'}", status=cote_status)
    return {'created': ids}


@rpc('plan.trace')
def rpc_trace(ctx, points, closed=True, lengths=None, cote_status='mesuré', cote_sigma=TYPED_SIGMA_CM, prior=CLICK_SIGMA_M):
    return trace(ctx, points, closed, lengths, sigma=float(prior), cote_status=cote_status, cote_sigma=cote_sigma)


@rpc('plan.place')
def place(ctx, type, point, width=None, height=None, sill=None, sigma=None, share=None):
    project = ctx.require_project(); point = np.asarray(point, float); values = current_values(project); best = None
    if type not in components.OPENING_DEFAULTS: raise ValueError(f'ouverture inconnue : {type}')
    for comp in components.load(project).values():
        if comp['type'] != 'mur': continue
        a, b = wall_ends(comp, values); length = np.linalg.norm(b - a); direction = (b - a) / length; along = (point - a) @ direction
        distance = abs(direction[0] * (point - a)[1] - direction[1] * (point - a)[0])
        if -MARGIN_M <= along <= length + MARGIN_M and distance < NEAR_WALL_M and (best is None or distance < best[0]): best = (distance, comp, along, length)
    if best is None: raise ValueError('clique sur un mur (à moins de 60 cm)')
    _, wall, along, length = best; width = float(width or components.OPENING_DEFAULTS[type]['l'])
    if length < width + 2 * MARGIN_M: raise ValueError(f"{wall['label']} trop court pour cette ouverture")
    comp_id = free_ids(project, type, 1)[0]; position = min(max(along - width / 2, MARGIN_M), length - width - MARGIN_M)
    shared = {'l': f'{share}_l', 'h': f'{share}_h', **({'al': f'{share}_al'} if type == 'fenetre' else {})} if share else {}
    comp = components.add(project, type, comp_id, {'mur': wall['id'], 'pos': position, 'l': width, **({'h': float(height)} if height else {}), **({'al': float(sill)} if sill is not None else {}), **shared})
    covered, params = {o.expr for o in project.observations() if o.source == f'comp:{comp_id}'}, project.params()
    for name in comp['params']:
        if name not in covered: prior(project, comp_id, name, params[name][0], float(sigma or OPENING_SIGMA_M), f"{params[name][2]} (a priori)")
    if height: prior(project, comp_id, f'{comp_id}_h', float(height), MARGIN_M / 5, f"{comp['label']} — hauteur (fiche)")
    if share: project.db.execute('insert or replace into obs values (?,?,?,?,?,?,?)', (f'comp:{comp_id}:{share}_l', f'comp:{comp_id}', f'{share}_l', width, float(sigma or OPENING_SIGMA_M), components.PRIOR_STATUS, f"{comp['label']} — largeur, même taille que {share}"))
    project.db.commit()
    return components.summary(comp)


@rpc('plan.walls')
def walls(ctx):
    project = ctx.require_project(); values = current_values(project)
    return [{'id': c['id'], 'label': c['label'], 'source': c['options'].get('source', ''), 'params': c['params'],
             'uses': sorted({n for key in ('start_x', 'start_z', 'end_x', 'end_z') for n in components.NAME.findall(str(c['expressions'][key]))}),
             'a': a.tolist(), 'b': b.tolist(), 'height': components.evaluate(c['expressions']['h'], values)}
            for c in components.load(project).values() if c['type'] == 'mur' and c['options'].get('nature') != 'contremarche' for a, b in [wall_ends(c, values)]]


@rpc('plan.sketch')
def sketch(ctx, path=None, points=None, length=None, size=None, opacity=None):
    project = ctx.require_project(); found = project_meta(project).get('croquis')
    if path:
        extension = os.path.splitext(path)[1].lower()
        if extension not in IMAGE_EXTENSIONS: raise ValueError('croquis : image JPEG ou PNG')
        os.makedirs(os.path.join(project.folder, 'sources'), exist_ok=True); target = os.path.join(project.folder, 'sources', f'croquis{extension}')
        shutil.copy2(path, target); found = project_meta(project, croquis={'file': os.path.basename(target)})['croquis']
    if points is not None:
        if not found: raise ValueError('aucun croquis')
        (u0, v0), (u1, v1) = points
        if not float(length) > 0 or np.hypot(u1 - u0, v1 - v0) < 1: raise ValueError('calage : deux points distincts et une longueur positive')
        found = project_meta(project, croquis={**found, 'points': [[float(u0), float(v0)], [float(u1), float(v1)]], 'length': float(length),
                                               'size': [float(v) for v in size]})['croquis']
    if opacity is not None:
        if not found or not 0.05 <= float(opacity) <= 1: raise ValueError('opacité du croquis : entre 0,05 et 1')
        found = project_meta(project, croquis={**found, 'opacity': float(opacity)})['croquis']
    return found and {**found, 'path': os.path.join(project.folder, 'sources', found['file'])}


def main_building(project):
    sources = cadastre.cached_sources(project.folder); sources = [s for s in sources if s['georeference']] or sources
    for source in sources:
        house = next((e for e in source['entites'] if e['type'] == 'bati' and e['principal']), None)
        parcel = next((e for e in source['entites'] if e['type'] == 'parcelle' and e['principal']), None)
        if house or parcel: return cadastre.param_prefix(source['source']), house, parcel
    return None, None, None


@rpc('plan.cadastre')
def from_cadastre(ctx, create=False):
    project = ctx.require_project(); prefix, house, parcel = main_building(project); params = project.params()
    if not house and not parcel: return None
    refs = lambda e: [{axis: f'{prefix}_{ref}_{axis}' for axis in 'xz' if f'{prefix}_{ref}_{axis}' in params} for ref in e['refs']]
    built = [c for c in components.load(project).values() if c['options'].get('source') == 'cadastre']
    summary = {'house': house and {'label': house['label'] or house['id'], 'points': house['pts'], 'walls': len(house['pts']),
                                   'perimeter': float(sum(np.linalg.norm(np.subtract(a, b)) for a, b in zip(house['pts'], house['pts'][1:] + house['pts'][:1])))},
               'parcel': parcel and {'label': parcel['label'] or parcel['id'], 'points': parcel['pts'], 'area': cadastre.area(parcel['pts'])},
               'created': [c['id'] for c in built]}
    if not create or built: return summary
    if house:
        height = f"{prefix}_{house['id']}_h" if f"{prefix}_{house['id']}_h" in params else None
        summary['created'] = trace(ctx, house['pts'], True, refs=refs(house), height=height, inward=True, sigma=CADASTRE_SIGMA_M, source='cadastre')['created']
    if parcel:
        comp_id = free_ids(project, 'parcelle', 1)[0]; sommets = [[r.get('x', x), r.get('z', z)] for r, (x, z) in zip(refs(parcel), parcel['pts'])]
        comp = components.add(project, 'parcelle', comp_id, {'sommets': sommets, 'source': 'cadastre'}, f"Parcelle {summary['parcel']['label']}")
        for name in comp['params']: prior(project, comp_id, name, project.params()[name][0], CADASTRE_SIGMA_M, f'{name} (cadastre)')
        project.db.commit(); summary['created'].append(comp_id)
    return summary


SKETCH_PIXELS, SKETCH_TOLERANCE, STRAIGHT_DEG, TEXT_M = 1280, 0.2, 6.0, 1.5
SKETCH_ASK = {
    'corners': 'Locate every corner of the drawn walls, following the outline in order. Output JSON: [{"point_2d": [x, y], "label": "corner"}]',
    'texts': 'Locate the handwritten dimension texts (lengths such as 4 m or 350). Output JSON: [{"bbox_2d": [x1, y1, x2, y2], "text_content": "..."}]',
    'openings': 'Locate the doors and windows drawn on the plan. Output JSON: [{"bbox_2d": [x1, y1, x2, y2], "label": "door"}] with label door, window or french window.'}
OPENING_LABELS = {'door': 'porte', 'window': 'fenetre', 'french window': 'porte_fenetre'}


def calibration(found):
    (u0, v0), (u1, v1) = found['points']; k = found['length'] / np.hypot(u1 - u0, v1 - v0); angle = np.arctan2(v1 - v0, u1 - u0); c, s = np.cos(angle), np.sin(angle)
    return lambda u, v: np.array([k * (c * (u - u0) + s * (v - v0)), k * (c * (v - v0) - s * (u - u0))])


def listed(content):
    found = re.search(r'\[.*\]', str(content), re.S)
    try: data = json.loads(found[0]) if found else []
    except ValueError: data = []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def numbers(value, count):
    try: values = [float(v) for v in value]
    except (TypeError, ValueError): return None
    return values if len(values) == count else None


def straighten(points):
    points = [p.copy() for p in points]; n = len(points)
    for e in range(n):
        a, b = points[e], points[(e + 1) % n]; angle = abs(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))) % 180
        if min(angle, 180 - angle) < STRAIGHT_DEG: a[1] = b[1] = (a[1] + b[1]) / 2
        elif abs(angle - 90) < STRAIGHT_DEG: a[0] = b[0] = (a[0] + b[0]) / 2
    return points


def written(text):
    found = re.search(r'(\d+(?:[.,]\d+)?)\s*(cm|m)?', str(text))
    if not found: return None
    number = float(found[1].replace(',', '.'))
    return number * 100 if found[2] == 'm' or (found[2] is None and number < 30) else number


def nearest_edge(points, point):
    best = None
    for e in range(len(points)):
        p, q = points[e], points[(e + 1) % len(points)]; t = np.clip((point - p) @ (q - p) / ((q - p) @ (q - p)), 0, 1)
        distance = np.linalg.norm(point - p - t * (q - p))
        if distance < TEXT_M and (best is None or distance < best[0]): best = (distance, e, float(np.linalg.norm(q - p)))
    return best


@rpc('plan.read_sketch', long=True)
def read_sketch(ctx, progress=None):
    from PIL import Image
    from . import ai
    found, progress = sketch(ctx), progress or (lambda *a: None)
    if not found or not found.get('points'): raise ValueError('croquis : cale-le d’abord (2 points et leur distance)')
    with Image.open(found['path']) as image:
        image = image.convert('RGB'); k = min(1.0, SKETCH_PIXELS / max(image.size))
        small = image.resize((round(image.width * k), round(image.height * k))) if k < 1 else image
        buffer = io.BytesIO(); small.save(buffer, 'PNG')
    to_m = calibration(found); metres = lambda u, v: to_m(u / k, v / k); answers, model = {}, ''
    for step, (key, question) in enumerate(SKETCH_ASK.items()):
        progress(step / 3, ('Lecture des murs', 'Lecture des cotes', 'Lecture des portes et fenêtres')[step])
        content, model = ai.ask_image(question, buffer.getvalue(), lambda *a: None); answers[key] = listed(content)
    corners = [metres(*xy) for xy in (numbers(item.get('point_2d'), 2) for item in answers['corners']) if xy]
    corners = [p for i, p in enumerate(corners) if all(np.linalg.norm(p - q) >= 0.05 for q in corners[:i])]
    walls, cotes, openings = [], [], []
    if len(corners) >= 3:
        middle = np.mean(corners, 0); corners = straighten(sorted(corners, key=lambda p: np.arctan2(p[1] - middle[1], p[0] - middle[0])))
        walls.append({'points': [p.round(3).tolist() for p in corners], 'closed': True,
                      'length': float(sum(np.linalg.norm(corners[(e + 1) % len(corners)] - corners[e]) for e in range(len(corners))))})
    for item in answers['texts']:
        box, value = numbers(item.get('bbox_2d'), 4), written(item.get('text_content', ''))
        if not box or not value: continue
        edge = nearest_edge(corners, metres((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)) if walls else None
        gap = abs(value / 100 - edge[2]) / (value / 100) if edge else 1.0; ok = bool(edge) and gap <= SKETCH_TOLERANCE
        message = '' if ok else f"{value:g} cm écrits pour {edge[2]:.2f} m dessinés ({gap:.0%} d'écart)".replace('.', ',') if edge else 'aucun mur dessiné à côté'
        cotes.append({'texte': str(item.get('text_content', ''))[:40], 'value_cm': float(value), 'drawn_m': edge and edge[2], 'ok': ok, 'message': message,
                      'wall': 0 if edge else None, 'edge': edge and edge[1]})
    for item in answers['openings']:
        box, kind = numbers(item.get('bbox_2d'), 4), OPENING_LABELS.get(str(item.get('label', '')).lower())
        if box and kind: openings.append({'type': kind, 'point': metres((box[0] + box[2]) / 2, (box[1] + box[3]) / 2).round(3).tolist()})
    progress(1.0, 'Croquis lu')
    return {'walls': walls, 'cotes': cotes, 'openings': openings, 'model': model}


ROOF_SLOPE_DEG, BUILT = 35.0, ('mur', 'porte', 'fenetre', 'porte_fenetre')


def footprint(project):
    prefix, house, _ = main_building(project)
    if house: return np.array(house['pts'], float), (f"{prefix}_{house['id']}_h" if f"{prefix}_{house['id']}_h" in project.params() else None)
    values = current_values(project); ends = [p for c in components.load(project).values() if c['type'] == 'mur' for p in wall_ends(c, values)]
    return (np.array(ends), None) if ends else (None, None)


@rpc('plan.roof')
def roof(ctx):
    project = ctx.require_project(); points, height = footprint(project)
    if points is None: raise ValueError('toit : il faut une emprise (cadastre) ou des murs')
    existing = [c for c in components.load(project).values() if c['type'] == 'toit']
    if existing: return {'id': existing[0]['id'], 'measured': bool(height), 'created': False}
    edges = [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]; p, q = max(edges, key=lambda e: np.linalg.norm(e[1] - e[0]))
    d = (q - p) / np.linalg.norm(q - p); angle = float(np.degrees(np.arctan2(-d[1], d[0]))); across = np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))])
    u, v = (points - p) @ d, (points - p) @ across; origin = p + u.min() * d + v.min() * across; length, width = u.max() - u.min(), v.max() - v.min()
    rise, base = 0.5 * width * np.tan(np.radians(ROOF_SLOPE_DEG)), str(walls_base(project))
    gutter, ridge = (f'({base}) + {height}', f'({base}) + {height} + {rise:.3f}') if height else (3.0, 3.0 + rise)
    comp_id = free_ids(project, 'toit', 1)[0]
    comp = components.add(project, 'toit', comp_id, {'x0': f'{origin[0]:.4f}', 'z0': f'{origin[1]:.4f}', 'long': f'{length:.4f}', 'larg': f'{width:.4f}',
                                                     'rot': f'{angle:.4f}', 'hg': gutter, 'hf': ridge})
    return {'id': comp['id'], 'measured': bool(height), 'created': True}


def walls_base(project):
    walls = [c for c in components.load(project).values() if c['type'] == 'mur']
    return walls[0]['expressions']['y0'] if walls else components.Expr()


@rpc('plan.settle')
def settle(ctx):
    project = ctx.require_project(); loaded = components.load(project)
    ground = next((c for c in loaded.values() if c['type'] == 'terrain'), None)
    if ground is None: raise ValueError('relief : importe d’abord une grille d’altitudes')
    points, _ = footprint(project)
    if points is None: raise ValueError('relief : il faut des murs ou une emprise')
    from . import terrain
    lowest = min(points, key=lambda p: components.evaluate(terrain.altitude(ground, *p), current_values(project)))
    base = terrain.altitude(ground, *lowest); moved = []
    for comp in loaded.values():
        if comp['type'] in BUILT and 'y0' in comp['expressions']:
            comp['expressions']['y0'] = base; moved.append(comp['id'])
            project.db.execute('update components set json=? where id=?', (json.dumps(comp, ensure_ascii=False), comp['id']))
    project.db.commit()
    return {'moved': moved, 'base': str(base)}


@rpc('plan.vehicle')
def vehicle(ctx, longueur, largeur, hauteur, roue_longueur=0, roue_largeur=0, roue_hauteur=0, roue_position=0, porte_cote='droite',
            porte_largeur=0, porte_hauteur=0, porte_position=0, arriere_largeur=0, arriere_hauteur=0):
    project = ctx.require_project(); cm = lambda v: float(v) / 100
    L, W, H = cm(longueur), cm(largeur), cm(hauteur)
    if min(L, W, H) <= 0: raise ValueError('gabarit : longueur, largeur et hauteur utiles en cm, positives')
    if any(c['type'] == 'mur' for c in components.load(project).values()): raise ValueError('gabarit : le projet a déjà des murs')
    created = trace(ctx, [[0, 0], [L, 0], [L, W], [0, W]], True, [L, W, None, None], height=f'{H:.4f}', source='gabarit', cote_status='lu', cote_sigma=0.5)['created']
    components.add(project, 'sol', 'sol1', {'source': 'gabarit'}, 'Plancher'); created.append('sol1')
    if min(cm(roue_longueur), cm(roue_largeur), cm(roue_hauteur)) > 0:
        for side, z in (('gauche', 0.0), ('droite', W - cm(roue_largeur))):
            comp_id = free_ids(project, 'roue', 1)[0]
            components.add(project, 'boite', comp_id, {'nature': 'passage de roue', 'x0': f'{cm(roue_position):.4f}', 'z0': f'{z:.4f}', 'lx': f'{cm(roue_longueur):.4f}',
                                                       'lz': f'{cm(roue_largeur):.4f}', 'h': f'{cm(roue_hauteur):.4f}', 'rot': '0', 'source': 'gabarit'}, f'Passage de roue {side}')
            created.append(comp_id)
    if cm(porte_largeur) > 0:
        z = W if porte_cote == 'droite' else 0.0
        created.append(place(ctx, 'porte', [cm(porte_position) + cm(porte_largeur) / 2, z], cm(porte_largeur), cm(porte_hauteur) or None)['id'])
    if cm(arriere_largeur) > 0: created.append(place(ctx, 'porte', [0.0, W / 2], cm(arriere_largeur), cm(arriere_hauteur) or None)['id'])
    return {'created': created}
