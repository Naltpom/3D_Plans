import numpy as np
from . import components, photos
from .rpc import read_cotes, rpc, write_cotes

MERGE_M, ASSOC_M, SPAN_M, TOP_M, LENGTH_M, SURE_LENGTH_M, CORNER_M, FRONT_M, FLOOR_M, CEILING_M = 0.1, 0.03, 0.8, 1.3, 0.5, 1.0, 0.75, 0.1, 0.2, 1.8
SURE_POINTS, SURE_SHARE, DENSIFY_S_PER_PHOTO, SCALE_COTE, FRONT_MAX_M = 200, 0.01, 13, 'echelle', 1.0
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def plane_groups(planes, scale):
    groups = []
    for plane in sorted((p for p in planes if p['axis'] in ('x', 'z')), key=lambda p: -p['n']):
        group = next((g for g in groups if g[0]['axis'] == plane['axis'] and abs(g[0]['offset'] - plane['offset']) * scale <= MERGE_M), None)
        if group: group.append(plane)
        else: groups.append([plane])
    return groups


def overlap(wall, other):
    run = wall['run']
    return min(wall['high'][run], other['high'][run]) - max(wall['low'][run], other['low'][run])


def candidates(frame, model):
    scale, rotation, shift = frame['s_est'], np.array(frame['R']), np.array(frame['t'])
    centers = photos.camera_centers(model) @ rotation.T + shift; forwards = photos.rotation_matrices(model['q'])[:, 2] @ rotation.T; found = []
    for group in plane_groups(frame['planes'], scale):
        k = 'xyz'.index(group[0]['axis']); run = 2 - k; offset = group[0]['offset']
        low, high = np.min([p['bbox'][0] for p in group], 0), np.max([p['bbox'][1] for p in group], 0)
        if (high[1] - low[1]) * scale < SPAN_M or high[1] * scale < TOP_M or (high[run] - low[run]) * scale < LENGTH_M: continue
        side = np.sign(np.median(centers[:, k]) - offset) or 1.0; middle = (low + high) / 2; middle[k] = offset; towards = middle - centers
        seen = (np.einsum('ij,ij->i', forwards, towards) > np.cos(np.radians(40)) * np.linalg.norm(towards, axis=1)) & ((centers[:, k] - offset) * side > 0)
        found.append({'k': k, 'run': run, 'offset': offset, 'low': low, 'high': high, 'side': side, 'n': sum(p['n'] for p in group), 'photos': int(seen.sum()),
                      'planes': [p['id'] for p in group if abs(p['offset'] - offset) * scale <= ASSOC_M],
                      'names': {name for p in group for name in [p['id'], *p.get('aliases', [])]}})
    found = sorted(found, key=lambda wall: -wall['n'])[:len(LETTERS)]; threshold = max(SURE_POINTS, SURE_SHARE * sum(wall['n'] for wall in found))
    for letter, wall in zip(LETTERS, found): wall['id'] = letter
    for wall in found:
        behind = next((other for other in found if other['k'] == wall['k'] and other['side'] == wall['side'] and FRONT_M < (wall['offset'] - other['offset']) * wall['side'] * scale <= FRONT_MAX_M
                       and overlap(wall, other) > 0.3 * (wall['high'][wall['run']] - wall['low'][wall['run']])), None)
        length = (wall['high'][wall['run']] - wall['low'][wall['run']]) * scale
        wall['reason'] = f"devant le mur {behind['id']} : meuble ?" if behind else 'peu de points' if wall['n'] < threshold else 'court' if length < SURE_LENGTH_M else ''
    return found


def proposals(project, source):
    meta, frame, cache = photos.photo_state(project, source, compute=False); model = photos.load_model(cache)
    linear_map, shift, _ = photos.project_transform(project, source, frame); rotation, offset = np.array(frame['R']), np.array(frame['t'])
    to_project = lambda point: ((np.asarray(point, float) - offset) @ rotation) @ linear_map.T + shift
    mapping, cameras = photos.axis_mapping(project, source), photos.camera_centers(model) @ linear_map.T + shift
    guided = [c for c in components.load(project).values() if c['type'] == 'mur' and c['options'].get('source') == source]
    values = {name: value for name, (value, _) in components.param_values(project).items()}
    corner = lambda comp, end: np.array([components.evaluate(comp['expressions'][f'{end}_x'], values), 0.0, components.evaluate(comp['expressions'][f'{end}_z'], values)])
    clicked = {}
    for image, anchor in project.db.execute('select image, anchor from photos_clicks where source=?', (source,)): clicked.setdefault(anchor, set()).add(image)
    walls = []
    for wall in candidates(frame, model):
        axis = mapping['xyz'[wall['k']]][0]
        if axis == 'y': continue
        def at(value, height=0.0):
            point = wall['low'].copy(); point[wall['k']], point[wall['run']], point[1] = wall['offset'], value, height
            return to_project(point)
        run = 'z' if axis == 'x' else 'x'; points = sorted((at(wall['low'][wall['run']]), at(wall['high'][wall['run']])), key=lambda p: p['xyz'.index(run)])
        position = float(np.mean([p['xyz'.index(axis)] for p in points])); comp = next((c for c in guided if wall['names'] & set(c['options'].get('plans', []))), None)
        angle = lambda end: comp['options'].get('angles', {}).get(end) or ('clics' if len(clicked.get(f"{comp['id']}:{end}", ())) >= 2 else None)
        ends = [corner(comp, name) if comp and angle(end) else point for name, end, point in (('start', 'a', points[0]), ('end', 'b', points[1]))]
        walls.append({'id': wall['id'], 'label': f"Mur {wall['id']}", 'axis': axis, 'position': position, 'a': [float(ends[0][0]), float(ends[0][2])],
                      'b': [float(ends[1][0]), float(ends[1][2])], 'length': float(np.linalg.norm(ends[1] - ends[0])),
                      'height': float(at(wall['low'][wall['run']], wall['high'][1])[1] - points[0][1]), 'n': int(wall['n']), 'photos': wall['photos'],
                      'sure': not wall['reason'], 'reason': wall['reason'], 'planes': wall['planes'],
                      'sens': float(np.sign(position - np.median(cameras[:, 'xyz'.index(axis)])) or 1.0),
                      'component': comp and comp['id'], 'angles': {end: comp and angle(end) for end in 'ab'}})
    levels = [p for p in frame['planes'] if p['axis'] == 'y']; height = lambda plane: plane['offset'] * frame['s_est']
    floor = max((p for p in levels if abs(height(p)) < FLOOR_M), key=lambda p: p['n'], default=None)
    ceiling = max((p for p in levels if height(p) > CEILING_M), key=lambda p: p['n'], default=None)
    closed = [wall for wall in walls if wall['component'] and all(wall['angles'].values())]
    return {'walls': walls, 'floor': floor and floor['id'], 'ceiling': ceiling and ceiling['id'],
            'ceiling_height': ceiling and float(to_project([0, ceiling['offset'], 0])[1] - to_project([0, 0, 0])[1]),
            'measure': max(closed, key=lambda wall: wall['length'])['component'] if closed else None,
            'dense': not frame['cloud'].endswith('épars'), 'densify_min': max(1, round(meta['registered'] * DENSIFY_S_PER_PHOTO / 60))}


def corners(wall, chosen):
    run = 'z' if wall['axis'] == 'x' else 'x'; r, p = 'xz'.index(run), 'xz'.index(wall['axis']); links = {}
    for end in 'ab':
        near = [(abs(other['position'] - wall[end][r]), other) for other in chosen if other['axis'] == run and other is not links.get('a')
                and min(other['a'][p], other['b'][p]) - CORNER_M <= wall['position'] <= max(other['a'][p], other['b'][p]) + CORNER_M]
        distance, other = min(near, key=lambda item: item[0], default=(np.inf, None))
        links[end] = other if distance <= CORNER_M else None
    return links


def clear(project, source):
    guided = [c for c in components.load(project).values() if c['type'] == 'mur' and c['options'].get('source') == source]
    if not guided: return
    owned, planes = {n for c in guided for n in c['params']}, {p for c in guided for p in c['options'].get('plans', [])}
    stale = [(source, plane) for plane, expr in project.db.execute('select plane_id, expr from photos_assoc where source=?', (source,)).fetchall()
             if plane in planes or owned & set(components.NAME.findall(expr))]
    project.db.executemany('delete from photos_assoc where source=? and plane_id=?', stale)
    project.db.execute('delete from obs where source=? or id=?', (source, f'cotes:{SCALE_COTE}'))
    try: components.remove(project, *(c['id'] for c in guided))
    except ValueError: project.db.rollback(); raise
    rows = read_cotes(project)
    if any(row['id'] == SCALE_COTE for row in rows): write_cotes(project, [row for row in rows if row['id'] != SCALE_COTE])


def free_id(project, base):
    existing, name, k = components.load(project), base, 2
    while name in existing: name, k = f'{base}{k}', k + 1
    return name


@rpc('photos.propose_walls')
def propose_walls(ctx, source='photos'):
    return proposals(ctx.require_project(), source)


@rpc('photos.create_walls')
def create_walls(ctx, source, walls):
    project = ctx.require_project(); found = proposals(project, source); chosen = [wall for wall in found['walls'] if wall['id'] in walls]
    if {wall['axis'] for wall in chosen} != {'x', 'z'}: raise ValueError('il faut au moins un mur dans chaque direction')
    photos.ensure_schema(project); clear(project, source)
    origin = {axis: min((w for w in chosen if w['axis'] == axis), key=lambda w: w['position']) for axis in 'xz'}
    ids = {w['id']: free_id(project, f"mur{w['id']}") for w in chosen}; first = ids[chosen[0]['id']]
    position = {w['id']: '0' if origin[w['axis']] is w else f"{ids[w['id']]}_{w['axis']}" for w in chosen}
    for w in chosen:
        if position[w['id']] != '0': project.add_param(position[w['id']], w['position'] - origin[w['axis']]['position'], w['axis'], f"{w['label']} — position ({w['axis']})")
    for w in chosen:
        run = 'z' if w['axis'] == 'x' else 'x'; links = corners(w, chosen)
        end = lambda e: position[links[e]['id']] if links[e] else w[e]['xz'.index(run)] - origin[run]['position']
        components.add(project, 'mur', ids[w['id']], {
            'axe': run, w['axis']: position[w['id']], f'{run}0': end('a'), f'{run}1': end('b'), 'sens': w['sens'],
            'h': (found['ceiling_height'] or 2.5) if ids[w['id']] == first else f'{first}_h',
            'source': source, 'plans': w['planes'], 'angles': {e: links[e] and ids[links[e]['id']] for e in 'ab'}}, w['label'])
    levels = [(found['floor'], '0')] * bool(found['floor']) + [(found['ceiling'], f'{first}_h')] * bool(found['ceiling'])
    project.db.executemany('insert or replace into photos_assoc values (?,?,?,?)',
                           [(source, plane, position[w['id']], None) for w in chosen for plane in w['planes']] + [(source, plane, expr, None) for plane, expr in levels])
    photos.ensure_params(project, source); photos.invalidate(project, source)
    return {'created': list(ids.values())}
