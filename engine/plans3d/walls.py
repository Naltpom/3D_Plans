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


FIT_M, FIT_SCALE, FIT_WALLS, FIT_DEG, FIT_TIE = 0.15, (0.5, 2.0), 6, 5.0, 0.02


def turn_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def matrix_text(matrix): return ''.join(f"{'+' if matrix[np.abs(matrix[:, i]).argmax(), i] > 0 else '-'}{'xyz'[np.abs(matrix[:, i]).argmax()]}" for i in range(3))
def rotate(point, angle): c, s = np.cos(angle), np.sin(angle); return np.array([c * point[0] + s * point[1], -s * point[0] + c * point[1]])
def direction(a, b): return np.arctan2(b[1] - a[1], b[0] - a[0])
def wrap(angle, period=2 * np.pi): return (angle + period / 2) % period - period / 2


def traced_walls(project, source):
    values = {name: value for name, (value, _) in components.param_values(project).items()}; found = []
    for comp in components.load(project).values():
        if comp['type'] != 'mur' or comp['options'].get('source') == source: continue
        a, b = (np.array([components.evaluate(comp['expressions'][f'{end}_{axis}'], values) for axis in 'xz']) for end in ('start', 'end'))
        if np.linalg.norm(b - a) > 0.5: found.append({'id': comp['id'], 'a': a, 'b': b, 'expr': photos.LINE + ';'.join(str(comp['expressions'][f'{end}_{axis}']) for end in ('start', 'end') for axis in 'xz')})
    return found


def matches(segments, traced, scale, shift):
    found = []
    for seg in segments:
        a, b = scale * seg['a'] + shift, scale * seg['b'] + shift; best = None
        for t in traced:
            u = (t['b'] - t['a']) / np.linalg.norm(t['b'] - t['a']); normal = np.array([-u[1], u[0]])
            if abs(wrap(direction(a, b) - direction(t['a'], t['b']), np.pi)) > np.radians(FIT_DEG) or abs(normal @ ((a + b) / 2 - t['a'])) > FIT_M: continue
            low, high = sorted(((a - t['a']) @ u, (b - t['a']) @ u)); length = np.linalg.norm(t['b'] - t['a'])
            fit = min(high, length) - max(low, 0) - 2 * (max(0, -low) + max(0, high - length))
            if min(high, length) > max(low, 0) and (best is None or fit > best[2]): best = (seg, t, fit)
        if best: found.append(best)
    return found


def inside_share(points, traced):
    if not len(points): return 1.0
    inside = np.zeros(len(points), bool)
    for t in traced:
        (ax, az), (bx, bz) = t['a'], t['b']; x, z = points[:, 0], points[:, 1]
        crossing = ((az > z) != (bz > z)) & (x < (bx - ax) * (z - az) / np.where(bz == az, 1e-300, bz - az) + ax)
        inside ^= crossing
    return float(inside.mean())


def best_fit(photo, traced, turn, cameras=np.zeros((0, 2))):
    angles = {round(float(wrap(direction(s['a'], s['b']) - direction(t['a'], t['b']) + flip)), 3) for s in photo for t in traced for flip in (0, np.pi)}
    if turn is not None: angles = {a for a in angles if abs(wrap(a - turn * np.pi / 2)) < np.radians(20)}
    best, rank = (0.0, None), (False, 0.0)
    for angle in sorted(angles, key=abs):
        segments = [{**w, 'a': rotate(np.array(w['a']), angle), 'b': rotate(np.array(w['b']), angle)} for w in photo]
        pairs = [(s, t) for s in segments for t in traced if abs(wrap(direction(s['a'], s['b']) - direction(t['a'], t['b']), np.pi)) < np.radians(FIT_DEG)]
        normal = lambda t: np.array([-(t['b'] - t['a'])[1], (t['b'] - t['a'])[0]]) / np.linalg.norm(t['b'] - t['a'])
        scales = {1.0} | {round(float(normal(t) @ (t['a'] - t2['a']) / (normal(t) @ (s['a'] - s2['a']))), 4) for s, t in pairs for s2, t2 in pairs
                          if s is not s2 and t is not t2 and abs(wrap(direction(t['a'], t['b']) - direction(t2['a'], t2['b']), np.pi)) < np.radians(FIT_DEG)
                          and abs(normal(t) @ (s['a'] - s2['a'])) > 0.3}
        for scale in (f for f in scales if FIT_SCALE[0] < f < FIT_SCALE[1]):
            for s, t in pairs:
                for s3, t3 in pairs:
                    n1, n3 = normal(t), normal(t3)
                    if abs(n1[0] * n3[1] - n1[1] * n3[0]) < 0.3: continue
                    shift = np.linalg.solve(np.array([n1, n3]), [n1 @ t['a'] - scale * n1 @ s['a'], n3 @ t3['a'] - scale * n3 @ s3['a']])
                    found = matches(segments, traced, scale, shift)
                    if not found: continue
                    score = sum(fit for *_, fit in found) * (0.2 + 0.8 * inside_share(scale * np.array([rotate(c, angle) for c in cameras]).reshape(-1, 2) + shift, traced))
                    order = (any(seg['id'] == photo[0]['id'] and t is traced[0] for seg, t, _ in found), -abs(angle))
                    if score > best[0] * (1 + FIT_TIE) or score >= best[0] * (1 - FIT_TIE) and order > rank: best, rank = (score, (angle, scale, shift, found)), order
    return best[1]


@rpc('photos.fit_walls')
def fit_walls(ctx, source='photos', turn=None):
    project = ctx.require_project(); found = proposals(project, source); traced = traced_walls(project, source)
    photo = sorted((w for w in found['walls'] if w['sure'] and not w['component']), key=lambda w: -w['length'])[:FIT_WALLS]
    _, frame, cache = photos.photo_state(project, source, compute=False); linear_map, offset, _ = photos.project_transform(project, source, frame)
    cameras = (photos.camera_centers(photos.load_model(cache)) @ linear_map.T + offset)[:, [0, 2]]
    fit = best_fit(photo, traced, None if turn is None else int(turn), cameras) if photo and traced else None
    if not fit or not any(abs(np.sin(direction(t['a'], t['b']) - direction(u['a'], u['b']))) > 0.5 for _, t, _ in fit[3] for _, u, _ in fit[3]):
        raise ValueError('appariement impossible : il faut au moins 2 murs non parallèles vus sur les photos et tracés')
    angle, factor, shift, paired = fit
    scale, translation, current, _ = photos.current_values(project, source); total = float(np.arctan2(current[0, 2], current[0, 0])) + angle; quarter = round(total / (np.pi / 2))
    mapping = photos.mapping_matrix(photos.axis_mapping(project, source)); photos.axes(ctx, source, matrix_text(np.rint(turn_matrix(quarter * np.pi / 2)) @ mapping))
    moved = factor * turn_matrix(angle) @ translation + np.array([shift[0], 0.0, shift[1]])
    photos.ensure_params(project, source)
    project.db.executemany('update params set init=? where name=?', [(factor * scale, f'{source}_s'), (float(total - quarter * np.pi / 2), f'{source}_yaw'),
                                                                      *((float(v), f'{source}_t{a}') for a, v in zip('xyz', moved))])
    stale = {plane for w in photo for plane in w['planes']}
    project.db.executemany('delete from photos_assoc where source=? and plane_id=?', [(source, plane) for plane in stale])
    rows = [(source, plane, t['expr'], None) for seg, t, _ in paired for plane in seg['planes']] + [(source, found['floor'], '0', None)] * bool(found['floor'])
    project.db.executemany('insert or replace into photos_assoc values (?,?,?,?)', rows); photos.invalidate(project, source)
    parallel = any(abs(np.sin(direction(t['a'], t['b']) - direction(u['a'], u['b']))) < 0.1 and abs((t['b'] - t['a'])[0] * (u['a'] - t['a'])[1] - (t['b'] - t['a'])[1] * (u['a'] - t['a'])[0]) > 0.3 * np.linalg.norm(t['b'] - t['a'])
                   for _, t, _ in paired for _, u, _ in paired if t is not u)
    return {'pairs': [[seg['id'], t['id']] for seg, t, _ in paired], 'scale_known': parallel}


SNAP_M, SNAP_DEG, HOLE_CELL, BAND_M = 0.6, 10.0, 0.1, 0.06


@rpc('photos.snap_walls')
def snap_walls(ctx, source, walls):
    project = ctx.require_project(); found = proposals(project, source); paired = []
    for t in (t for t in traced_walls(project, source) if t['id'] in walls):
        u = (t['b'] - t['a']) / np.linalg.norm(t['b'] - t['a']); normal = np.array([-u[1], u[0]]); best = None
        for w in found['walls']:
            a, b = np.array(w['a']), np.array(w['b']); low, high = sorted(((a - t['a']) @ u, (b - t['a']) @ u))
            if abs(wrap(direction(a, b) - direction(t['a'], t['b']), np.pi)) > np.radians(SNAP_DEG) or abs(normal @ ((a + b) / 2 - t['a'])) > SNAP_M: continue
            if min(high, np.linalg.norm(t['b'] - t['a'])) > max(low, 0) and (best is None or w['n'] > best['n']): best = w
        if not best: continue
        project.db.executemany('insert or replace into photos_assoc values (?,?,?,?)', [(source, plane, t['expr'], None) for plane in best['planes']])
        comp = components.load(project)[t['id']]; run = comp['options'].get('axe'); name = str(comp['expressions'][f"start_{'z' if run == 'x' else 'x'}"]) if run else ''
        moved = None
        if name in project.params() and best['axis'] == ('z' if run == 'x' else 'x'):
            moved = abs(best['position'] - project.params()[name][0]) * 100
            project.db.execute('update params set init=? where name=?', (best['position'], name))
            project.db.execute("update obs set value=? where expr=? and source like 'comp:%'", (best['position'], name))
        paired.append({'photo': best['id'], 'wall': t['id'], 'moved_cm': moved})
    if paired: photos.ensure_params(project, source); photos.invalidate(project, source)
    return {'pairs': paired}


def holes(points, length, height, cell=HOLE_CELL):
    from scipy import ndimage
    nu, nv = int(length / cell), int(height / cell)
    if nu < 3 or nv < 3: return []
    grid = np.zeros((nv, nu), bool); inside = (points[:, 0] >= 0) & (points[:, 0] < nu * cell) & (points[:, 1] >= 0) & (points[:, 1] < nv * cell)
    grid[(points[inside, 1] / cell).astype(int), (points[inside, 0] / cell).astype(int)] = True
    labels, count = ndimage.label(~grid); found = []
    for rows, cols in ndimage.find_objects(labels):
        r0, r1, c0, c1 = rows.start, rows.stop - 1, cols.start, cols.stop - 1
        if r1 >= nv - 1 or c0 == 0 or c1 >= nu - 1 or (c1 - c0 + 1) * cell < 0.4 or (r1 - r0 + 1) * cell < 0.4: continue
        ring = np.concatenate([grid[r0:r1 + 1, c0 - 1], grid[r0:r1 + 1, c1 + 1], grid[r1 + 1, c0:c1 + 1]])
        if ring.mean() < 0.5 or grid[r0:r1 + 1, c0:c1 + 1].mean() > 0.05: continue
        width, tall, bottom = (c1 - c0 + 1) * cell, (r1 - r0 + 1) * cell, r0 * cell
        kind = ('porte' if width <= 1.2 else 'porte_fenetre') if bottom < 0.15 and tall >= 1.6 else 'fenetre' if bottom > 0.5 and tall <= 2.2 else None
        if kind: found.append({'type': kind, 'along': c0 * cell + width / 2, 'width': width, 'height': tall, 'sill': bottom, 'sigma': cell / 2})
    return found


@rpc('photos.propose_openings')
def propose_openings(ctx, source='photos'):
    from .plan import current_values, wall_ends
    project = ctx.require_project(); _, frame, cache = photos.photo_state(project, source, compute=False)
    points, _ = photos.cloud_sample(cache, 400000); linear_map, shift, _ = photos.project_transform(project, source, frame); points = points @ linear_map.T + shift
    values, existing = current_values(project), components.load(project); found = []
    for comp in existing.values():
        if comp['type'] != 'mur': continue
        a, b = wall_ends(comp, values); length = np.linalg.norm(b - a); u = (b - a) / length; normal = np.array([-u[1], u[0]])
        base, height = components.evaluate(comp['expressions']['y0'], values), components.evaluate(comp['expressions']['h'], values)
        flat = points[:, [0, 2]] - a; near = np.abs(flat @ normal) < BAND_M
        local = np.stack([flat[near] @ u, points[near, 1] - base], 1)
        taken = [o for o in existing.values() if o['options'].get('mur') == comp['id']]
        for hole in holes(local, length, height):
            if any(abs(components.evaluate(o['expressions']['pos'], values) + components.evaluate(o['expressions']['l'], values) / 2 - hole['along']) < hole['width'] for o in taken): continue
            found.append({**hole, 'wall': comp['id'], 'label': comp['label'], 'point': (a + hole['along'] * u).round(3).tolist()})
    return found
