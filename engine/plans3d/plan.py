import os, shutil
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


def trace(ctx, points, closed=True, lengths=None, refs=None, height=None, inward=False, sigma=CLICK_SIGMA_M, source='tracé'):
    project = ctx.require_project(); points = np.asarray(points, float).reshape(-1, 2); n = len(points)
    edges = list(range(n if closed else n - 1)); after = lambda e: (e + 1) % n
    if len(edges) < (3 if closed else 1): raise ValueError('tracé : au moins 2 points, 3 pour une pièce fermée')
    if any(np.linalg.norm(points[after(e)] - points[e]) < 0.01 for e in edges): raise ValueError('tracé : deux points confondus')
    ids = free_ids(project, 'mur', len(edges)); labels = [f'Mur {i[3:]}' for i in ids]; created = {}
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
        options.append({**geometry, 'sens': sens(e), 'source': source, 'h': height if height is not None else 2.5 if e == 0 else f'{ids[0]}_h'})
    for name, (value, axis, label, _) in created.items(): project.add_param(name, float(value), axis, label)
    for e, opts in zip(edges, options): components.add(project, 'mur', ids[e], opts, labels[e])
    for name, (value, _, label, owner) in created.items(): prior(project, owner, name, value, sigma, f'{label} (tracé)')
    project.db.commit(); values = current_values(project); loaded = components.load(project); os.makedirs(os.path.join(project.folder, 'sources'), exist_ok=True)
    for e, length in zip(edges, lengths or []):
        if length: cotes_add(ctx, f'trace_{ids[e]}', components.measure_expr((loaded[ids[e]], 'a'), (loaded[ids[e]], 'b'), values=values), float(length) * 100,
                             TYPED_SIGMA_CM, label=f'{labels[e]} — longueur saisie')
    return {'created': ids}


@rpc('plan.trace')
def rpc_trace(ctx, points, closed=True, lengths=None):
    return trace(ctx, points, closed, lengths)


@rpc('plan.place')
def place(ctx, type, point, width=None):
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
    comp = components.add(project, type, comp_id, {'mur': wall['id'], 'pos': position, 'l': width})
    covered, params = {o.expr for o in project.observations() if o.source == f'comp:{comp_id}'}, project.params()
    for name in comp['params']:
        if name not in covered: prior(project, comp_id, name, params[name][0], OPENING_SIGMA_M, f"{params[name][2]} (a priori)")
    project.db.commit()
    return components.summary(comp)


@rpc('plan.walls')
def walls(ctx):
    project = ctx.require_project(); values = current_values(project)
    return [{'id': c['id'], 'label': c['label'], 'source': c['options'].get('source', ''), 'params': c['params'],
             'uses': sorted({n for key in ('start_x', 'start_z', 'end_x', 'end_z') for n in components.NAME.findall(str(c['expressions'][key]))}),
             'a': a.tolist(), 'b': b.tolist(), 'height': components.evaluate(c['expressions']['h'], values)}
            for c in components.load(project).values() if c['type'] == 'mur' for a, b in [wall_ends(c, values)]]


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
