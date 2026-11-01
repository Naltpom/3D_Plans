import csv, json, math, os
import pytest
from plans3d import Project, cadastre
from plans3d import components as C
from plans3d.core import compile_expr
from plans3d.rpc import Ctx, call

TRUE_VALUES = {'m1_x1': 4.80, 'm2_z1': 3.60, 'm1_ep': 0.18, 'p1_pos': 0.40, 'p1_l': 0.90, 'p1_h': 2.10, 'f1_pos': 1.00, 'f1_l': 1.40, 'f1_h': 1.15,
               'f1_al': 0.95, 'c1_y': 2.55, 'e1_x': 1.00, 'e1_g': 0.27, 'e1_r': 0.18, 'e1_l': 0.85}
ROOM_MEASURES = [
    ('largeur', 'm4:face', 'm2:face', 'x'), ('profondeur', 'm1:face', 'm3:face', 'z'), ('ep_m1', 'm1:face', 'm1:face_ext', 'z'),
    ('porte_x', 'm4:face', 'p1:g', 'x'), ('porte_l', 'p1:g', 'p1:d', 'x'), ('porte_reste', 'p1:d', 'm2:face', 'x'), ('porte_h', 's0:niveau', 'p1:haut', 'y'),
    ('fen_z', 'm1:face', 'f1:g', 'z'), ('fen_l', 'f1:g', 'f1:d', 'dist'), ('fen_al', 's0:niveau', 'f1:bas', 'y'), ('fen_h', 'f1:bas', 'f1:haut', 'y'),
    ('plafond', 's0:niveau', 'c1:niveau', 'y'), ('esc_x', 'm4:face', 'e1:depart', 'x'), ('esc_run', 'e1:depart', 'e1:arrivee', 'x'),
    ('esc_reste', 'e1:arrivee', 'm2:face', 'x'), ('esc_h', 'e1:bas', 'e1:haut', 'y'), ('esc_l', 'e1:cote', 'e1:cote_ext', 'z')]


def build_room(folder):
    project = Project(folder)
    def add(type, comp_id, **options): C.add(project, type, comp_id, options)
    add('mur', 'm1', axe='x', z='0', x0='0', x1=4.85, sens=-1)
    add('mur', 'm2', axe='z', x='m1_x1', z0='0', z1=3.65)
    add('mur', 'm3', axe='x', z='m2_z1', x0='0', x1='m1_x1')
    add('mur', 'm4', axe='z', x='0', z0='0', z1='m2_z1', sens=-1)
    add('sol', 's0'); add('plafond', 'c1', y=2.6)
    add('porte', 'p1', mur='m1', pos=0.45, l=0.95)
    add('fenetre', 'f1', mur='m2', pos=1.05, l=1.45, h=1.2, al=0.9)
    add('escalier', 'e1', axe='x', n=13, x=1.05, z='m2_z1', cote=-1, g=0.25, r=0.175, l=0.9)
    return project


def add_tape_measures(project, measures, true_values):
    ctx = Ctx(project=project); expressions = {}
    values = {n: v for n, (v, _) in C.param_values(project).items()} | true_values; index = {n: i for i, n in enumerate(values)}
    path = os.path.join(project.folder, 'cotes.csv')
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';'); writer.writerow(['id', 'expr', 'value', 'sigma', 'unit', 'status', 'label'])
        for measure_id, start, end, axis in measures:
            expr = expressions[measure_id] = C.rpc_measure(ctx, start, end, axis)['expr']
            writer.writerow([measure_id, expr, compile_expr(expr, index)[0](list(values.values())), 0.01, 'm', 'mesuré', f'{start} → {end}'])
    project.add_source('cotes', path, 'cotes_csv')
    return expressions


def assert_recovered(result, true_values, max_sigma=0.02):
    assert not result.contradictions
    for name, true in true_values.items():
        value, sigma = result.value(name); assert sigma < max_sigma and abs(value - true) < 3 * sigma, (name, value, sigma, true)


def close(a, b): return a == pytest.approx(b, abs=1e-6)


def test_room(tmp_path):
    project = build_room(str(tmp_path)); expr = add_tape_measures(project, ROOM_MEASURES, TRUE_VALUES)
    assert expr['largeur'] == 'm1_x1' and expr['ep_m1'] == 'm1_ep' and expr['porte_reste'] == 'm1_x1 - p1_pos - p1_l' and expr['fen_l'] == 'f1_l'
    assert expr['esc_run'] == '12 * e1_g' and expr['esc_h'] == '13 * e1_r' and expr['esc_reste'] == 'm1_x1 - e1_x - 12 * e1_g'
    assert {o.id for o in project.observations() if o.status == C.PRIOR_STATUS} >= {'comp:p1:p1_h', 'comp:e1:e1_r', 'comp:m2:m2_ep'}
    result = project.update(); assert_recovered(result, TRUE_VALUES)
    shapes = C.geometry(project, result); by_id = {}
    for shape in shapes: by_id.setdefault(shape['id'], []).append(shape)
    assert len(shapes) == 4 + 2 + 2 + 12 and [len(by_id[i]) for i in ('m1', 'p1', 'f1', 'e1', 's0')] == [1, 1, 1, 12, 1]
    assert all(s['sigma_max'] is not None and s['sigma_max'] < 0.21 for s in shapes)
    v = lambda name: result.value(name)[0]
    m1, m2, p1, f1, top = by_id['m1'][0], by_id['m2'][0], by_id['p1'][0], by_id['f1'][0], by_id['e1'][-1]
    assert close(m1['size'], [v('m1_x1'), 2.5, v('m1_ep')]) and close(m1['center'], [v('m1_x1') / 2, 1.25, -v('m1_ep') / 2]) and close(m1['rotY'], 0)
    assert close(m2['size'], [v('m2_z1'), 2.5, v('m2_ep')]) and close(m2['center'][0], v('m1_x1') + v('m2_ep') / 2) and close(m2['rotY'], -math.pi / 2)
    assert close(p1['size'], [v('p1_l'), v('p1_h'), v('m1_ep') + 0.02]) and close(p1['center'], [v('p1_pos') + v('p1_l') / 2, v('p1_h') / 2, -v('m1_ep') / 2])
    assert close(f1['center'], [v('m1_x1') + v('m2_ep') / 2, v('f1_al') + v('f1_h') / 2, v('f1_pos') + v('f1_l') / 2]) and close(f1['rotY'], -math.pi / 2)
    assert close(top['size'], [v('e1_g'), 12 * v('e1_r'), v('e1_l')]) and close(top['center'], [v('e1_x') + 11.5 * v('e1_g'), 6 * v('e1_r'), v('m2_z1') - v('e1_l') / 2])
    assert close(sum(by_id['s0'][0]['vertices'], []), [0, 0, v('m1_x1'), 0, v('m1_x1'), v('m2_z1'), 0, v('m2_z1')]) and close(by_id['c1'][0]['y'], v('c1_y'))
    assert close(C.geometry(project)[0]['size'], m1['size'])


def test_undetermined_wall_does_not_drift(tmp_path):
    project = build_room(str(tmp_path)); C.add(project, 'mur', 'm5', {'axe': 'x', 'z': 6.0, 'x0': 1.0, 'x1': 3.0})
    add_tape_measures(project, ROOM_MEASURES + [('m5_l', 'm5:a', 'm5:b', 'dist')], {**TRUE_VALUES, 'm5_x1': 3.5})
    for _ in range(3):
        result = project.update(); v = lambda name: result.value(name)[0]
        assert close(v('m5_z'), 6.0) and close(v('m5_x0') + v('m5_x1'), 4.0) and close(v('m5_x1') - v('m5_x0'), 2.5)
        assert_recovered(result, TRUE_VALUES)
    model = project.model(); model['params']['m5_z']['value'] = 868.0
    with open(os.path.join(project.folder, 'model.json'), 'w', encoding='utf-8') as f: json.dump(model, f)
    assert close(project.update().value('m5_z')[0], 6.0)

def test_oblique_wall(tmp_path):
    project = build_room(str(tmp_path)); C.add(project, 'mur', 'mb', {'x0': 2.1, 'z0': 0.9, 'x1': 3.4, 'z1': 2.3})
    true_values = {**TRUE_VALUES, 'mb_x0': 2.0, 'mb_z0': 1.0, 'mb_x1': 3.5, 'mb_z1': 2.2}
    expr = add_tape_measures(project, ROOM_MEASURES + [('mb_ax', 'm4:face', 'mb:a', 'x'), ('mb_az', 'm1:face', 'mb:a', 'z'), ('mb_L', 'mb:a', 'mb:b', 'dist'),
                                                       ('mb_diag', 'm1:a', 'mb:b', 'dist'), ('mb_bx', 'mb:b', 'm2:face', 'x')], true_values)
    assert expr['mb_L'] == 'hypot(mb_x1 - mb_x0, mb_z1 - mb_z0)' and expr['mb_diag'] == 'hypot(mb_x1, mb_z1)' and expr['mb_bx'] == 'm1_x1 - mb_x1'
    result = project.update(); assert_recovered(result, true_values, max_sigma=0.03)
    shape = next(s for s in C.geometry(project, result) if s['id'] == 'mb')
    assert shape['size'][0] == pytest.approx(math.hypot(1.5, 1.2), abs=0.02) and shape['rotY'] == pytest.approx(math.atan2(-1.2, 1.5), abs=0.02)
    wall = C.load(project)['mb']; values = {n: v for n, (v, _) in C.param_values(project, result).items()}
    inner, outer = ({axis: C.evaluate(e, values) for axis, e in C.anchor(wall, name).items()} for name in ('a', 'a_ext'))
    assert math.hypot(outer['x'] - inner['x'], outer['z'] - inner['z']) == pytest.approx(values['mb_ep'])
    with pytest.raises(ValueError): C.anchor(wall, 'face')
    with pytest.raises(ValueError): C.measure_expr((wall, 'a'), (C.load(project)['m1'], 'face'), 'x')


def test_store_remove_rpc(tmp_path):
    project = build_room(str(tmp_path)); ctx = Ctx(project=project)
    expected = {'mur', 'porte', 'fenetre', 'porte_fenetre', 'sol', 'plafond', 'escalier', 'boite', 'parcelle'}
    assert expected <= {t['type'] for t in C.rpc_types(ctx)} and [c['id'] for c in C.rpc_list(ctx)][:4] == ['m1', 'm2', 'm3', 'm4']
    with pytest.raises(ValueError, match='m2'): C.rpc_remove(ctx, 'm1')
    with pytest.raises(ValueError): C.rpc_add(ctx, 'mur', 'm1')
    with pytest.raises(KeyError): C.rpc_add(ctx, 'mur', 'mx', {'axe': 'x', 'z': 'zz_z'})
    added = C.rpc_add(ctx, 'boite', 'b1', {'nature': 'placard', 'x0': 1.0, 'y0': 0.1})
    assert added['params'] == ['b1_x0', 'b1_z0', 'b1_lx', 'b1_lz', 'b1_h', 'b1_y0', 'b1_rot'] and 'coin_11' in added['anchors']
    anchors = {a['name']: a for a in C.rpc_anchors(ctx, 'b1')}
    assert anchors['coin_00']['axes'] == {'x': 'b1_x0', 'z': 'b1_z0'} and anchors['centre']['axes']['y'] == 'b1_y0 + 0.5 * b1_h'
    C.rpc_add(ctx, 'parcelle', 'pa', {'sommets': [[0, 0], [30, 0], [30, 20], [10, 25], [0, 20]]})
    box, parcel = [s for s in C.rpc_geometry(ctx) if s['id'] in ('b1', 'pa')]
    assert box['color'] == C.BOX_NATURES['placard'][1] and box['sigma_max'] is None and len(parcel['vertices']) == 5
    assert C.rpc_measure(ctx, 'pa:s3', 'pa:s4', 'dist')['value'] == pytest.approx(math.hypot(20, 5))
    for comp_id in ('b1', 'pa', 'p1'): C.rpc_remove(ctx, comp_id)
    assert not any(n.startswith(('b1_', 'pa_', 'p1_')) for n in project.params()) and not [o for o in project.observations() if o.source == 'comp:p1']
    assert call(ctx, 'components.measure_expr', {'a': 'm4:face', 'b': 'm2:face', 'axis': 'x'})['expr'] == 'm1_x1'


def test_rotated_box(tmp_path):
    project = Project(str(tmp_path)); ctx = Ctx(project=project)
    C.add(project, 'boite', 'b2', {'nature': 'lit', 'x0': 1.0, 'z0': 2.0, 'lx': 2.0, 'lz': 1.0, 'h': 0.5, 'rot': 30})
    C.add(project, 'boite', 'b3', {'rot': '90'}); C.add(project, 'boite', 'b4', {'rot': 'b2_rot + 60'})
    assert [o.value for o in project.observations() if o.id == 'comp:b2:b2_rot'] == [30] and C.fields('boite')[-1]['unit'] == '°'
    assert {a['name']: a['axes'] for a in C.rpc_anchors(ctx, 'b3')}['coin_10'] == {'x': 'b3_x0', 'z': 'b3_z0 - b3_lx'}
    values = {n: v for n, (v, _) in C.param_values(project).items()}; c, s = math.cos(math.pi / 6), math.sin(math.pi / 6)
    point = lambda comp_id, name: [C.evaluate(e, values) for e in C.anchor(C.load(project)[comp_id], name).values()]
    assert close(point('b2', 'coin_10'), [1 + 2 * c, 2 - 2 * s]) and close(point('b2', 'coin_01'), [1 + s, 2 + c])
    assert close(point('b2', 'x1'), [1 + 2 * c + s / 2, 2 - 2 * s + c / 2]) and close(point('b2', 'centre'), [1 + c + s / 2, 2 - s + c / 2, 0.25])
    assert close(point('b4', 'coin_10'), point('b3', 'coin_10'))
    shapes = {shape['id']: shape for shape in C.geometry(project)}; box = shapes['b2']
    assert box['color'] == C.BOX_NATURES['lit'][1] and close(box['rotY'], math.pi / 6) and close(box['size'], [2, 0.5, 1]) and close(shapes['b4']['rotY'], math.pi / 2)
    assert close(sum(map(list, C.footprint(box)), []), sum((point('b2', f'coin_{k}') for k in ('00', '10', '11', '01')), []))
    expr = add_tape_measures(project, [('dx', 'b2:coin_00', 'b2:coin_10', 'x'), ('dz', 'b2:coin_00', 'b2:coin_10', 'z')], {'b2_lx': 2.0, 'b2_rot': 31.0})
    result = project.update(); rot, sigma = result.value('b2_rot')
    assert 'cos' in expr['dx'] and abs(rot - 31) < 0.1 and 0 < sigma < 0.5


def test_roof(tmp_path):
    project = Project(str(tmp_path)); ctx = Ctx(project=project)
    C.add(project, 'toit', 't1', {'x0': 1.0, 'z0': 2.0, 'long': 10.0, 'larg': 8.0, 'hg': 3.0, 'hf': 6.0, 'deb': 0.5}); C.add(project, 'toit', 't2', {'rot': 90.0})
    assert [f['name'] for f in next(t for t in C.rpc_types(ctx) if t['type'] == 'toit')['fields']] == ['x0', 'z0', 'long', 'larg', 'rot', 'hg', 'hf', 'deb']
    priors = {o.id: (o.expr, o.value, o.sigma) for o in project.observations() if o.source == 'comp:t1'}
    assert set(priors) == {'comp:t1:t1_pente', 'comp:t1:t1_deb', 'comp:t1:t1_rot'} and priors['comp:t1:t1_pente'][1:] == (35.0, 15.0)
    assert {o.id: C.obs_unit(o) for o in project.observations() if o.source == 'comp:t1'} == {'comp:t1:t1_pente': '°', 'comp:t1:t1_deb': 'm', 'comp:t1:t1_rot': '°'}
    values = {n: v for n, (v, _) in C.param_values(project).items()}; roof = C.load(project)['t1']
    point = lambda comp, name: {a: round(C.evaluate(e, values), 9) for a, e in C.anchor(comp, name).items()}
    assert point(roof, 'faitage_a') == {'x': 0.5, 'z': 6.0, 'y': 6.0} and point(roof, 'faitage_b') == {'x': 11.5, 'z': 6.0, 'y': 6.0}
    assert point(roof, 'gouttiere_1') == {'x': 6.0, 'z': 1.5, 'y': 3.0} and point(roof, 'gouttiere_2') == {'x': 6.0, 'z': 10.5, 'y': 3.0}
    assert point(roof, 'pignon_b') == {'x': 11.0, 'z': 6.0, 'y': 6.0} and point(roof, 'coin_11') == {'x': 11.0, 'z': 10.0}
    assert point(C.load(project)['t2'], 'faitage_b') == {'x': 4.0, 'z': -10.4, 'y': 6.0}
    assert close(C.evaluate(roof['expressions']['pente'], values), math.degrees(math.atan2(3, 4.5)))
    shapes = [s for s in C.geometry(project) if s['id'] == 't1']
    assert [(s['kind'], s['part']) for s in shapes] == [('mesh', 'pan_1'), ('mesh', 'pan_2'), ('mesh', 'pignon_a'), ('mesh', 'pignon_b')]
    assert close(sum(shapes[0]['vertices'], []), [0.5, 3, 1.5, 11.5, 3, 1.5, 11.5, 6, 6, 0.5, 6, 6]) and shapes[1]['vertices'][0] == [0.5, 3, 10.5]
    assert close(sum(shapes[2]['vertices'], []), [1, 6 - 3 * 4 / 4.5, 2, 1, 6 - 3 * 4 / 4.5, 10, 1, 6, 6]) and shapes[2]['color'] == C.TYPES['mur'].color
    measured = C.rpc_measure(ctx, 't1:gouttiere_1', 't1:faitage_a', 'y')
    assert measured['expr'] == 't1_hf - t1_hg' and measured['value'] == pytest.approx(3.0)
    expr = add_tape_measures(project, [('hauteur', 't1:gouttiere_1', 't1:faitage_a', 'y'), ('long', 't1:pignon_a', 't1:pignon_b', 'dist')],
                             {'t1_hf': 6.4, 't1_long': 11.0})
    result = project.update(); value, sigma = result.predict(expr['hauteur'])
    assert abs(value - 3.4) < 0.03 and 0 < sigma < 0.02 and abs(result.value('t1_long')[0] - 11.0) < 0.03

def make_land(tmp_path):
    east, north = 1700112.5, 8200030.0
    ring = lambda x0, x1, n0, n1: [[east + x0, north + n1], [east + x1, north + n1], [east + x1, north + n0], [east + x0, north + n0]]
    feature = lambda props, pts: {'type': 'Feature', 'properties': props, 'geometry': {'type': 'Polygon', 'coordinates': [pts + pts[:1]]}}
    path = tmp_path / 'terrain.geojson'; crs = {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:EPSG::3949'}}
    json.dump({'type': 'FeatureCollection', 'crs': crs, 'features': [feature({'numero': '0001', 'contenance': 200}, ring(0, 20, 0, 10)),
                                                                    feature({'hauteur': 6.0}, ring(4, 12, 1, 7))]}, open(path, 'w', encoding='utf-8'))
    project = Project(str(tmp_path / 'projet')); cadastre.add_source(project, str(path))
    parcel, house = (e['pts'] for e in cadastre.entities(project.folder, 'terrain')['entites'])
    return project, max(x for x, _ in parcel), house


def test_inclusion_warnings(tmp_path):
    project, edge, house = make_land(tmp_path)
    (hx0, hz0), (hx1, _) = map(min, zip(*house)), map(max, zip(*house))
    add = lambda comp_id, **options: C.add(project, 'boite', comp_id, {'nature': 'potager', 'h': 0.3, **options})
    add('dedans', x0=14.0, z0=2.0, lx=2.0, lz=1.0, rot=30); add('meuble', nature='lit', x0=hx0 + 1, z0=hz0 + 1, lx=1.4, lz=1.9)
    assert project.update().warnings == []
    add('diag', x0=edge - math.sqrt(2), z0=5.0, lx=2.0, lz=2.0, rot=45); add('empiete', x0=hx1 - 1, z0=hz0 - 2, lx=2.0, lz=3.0)
    C.add(project, 'parcelle', 'zone', {'sommets': [[15, 6], [25, 6], [25, 8], [17, 8], [17, 12], [15, 12]]})
    warnings = project.update().warnings
    found = {(w['component'], w['code']): w['aire'] for w in warnings}
    assert set(found) == {('diag', 'hors_parcelle'), ('empiete', 'sur_batiment'), ('zone', 'hors_parcelle')}
    assert found[('diag', 'hors_parcelle')] == pytest.approx(2.0, rel=0.01) and found[('empiete', 'sur_batiment')] == pytest.approx(1.0, rel=0.01)
    assert found[('zone', 'hors_parcelle')] == pytest.approx(28 - (5 * 2 + 2 * 2), rel=0.01)
    assert 'Potager diag déborde de 2,00 m² hors de la parcelle principale' in next(w['message'] for w in warnings if w['component'] == 'diag')
    with open(os.path.join(project.folder, 'model.json'), encoding='utf-8') as f: assert json.load(f)['warnings'] == warnings
    assert C.overlap_area([(0, 0), (4, 0), (4, 4), (2, 1), (0, 4)], [(1, -1), (3, -1), (3, 5), (1, 5)]) == pytest.approx(3.5)
