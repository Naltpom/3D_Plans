import numpy as np
import pytest
from PIL import Image, ImageDraw
from plans3d import components, lecture, plan, rpc
from plans3d.rpc import Ctx


def drawing(path):
    image = Image.new('RGB', (800, 700), 'white'); pen = ImageDraw.Draw(image)
    room = [(100, 100), (600, 103), (602, 300), (700, 300), (700, 600), (98, 602)]
    pen.line(room + [room[0]], fill=(0, 0, 0), width=5)
    pen.line([(100, 100), (60, 60)], fill=(0, 0, 0), width=4)
    pen.line([(300, 602), (300, 450)], fill=(0, 0, 0), width=4)
    pen.line([(250, 96), (350, 96)], fill=(230, 40, 30), width=7)
    pen.line([(700, 380), (700, 520)], fill=(30, 140, 230), width=7)
    for k in range(8): pen.line([(450 + 20 * k, 450), (450 + 20 * k, 580)], fill=(128, 128, 128), width=2)
    pen.rectangle([440, 450, 610, 580], outline=(128, 128, 128), width=2)
    image.save(path)


def test_read_drawing(tmp_path):
    drawing(tmp_path / 'plan.png'); read = lecture.read(str(tmp_path / 'plan.png'), ai=False)
    names = [w['name'] for w in read['walls']]
    assert names == list('ABCDEFG') and read['walls'][0]['a'][1] < 110 and read['walls'][0]['b'][0] > 590
    for w in read['walls']:
        d = np.subtract(w['b'], w['a']) @ np.array([[np.cos(read['angle']), -np.sin(read['angle'])], [np.sin(read['angle']), np.cos(read['angle'])]])
        assert min(abs(d)) < 1e-3 * max(abs(d))
    assert {(o['type'], o['wall']) for o in read['openings']} == {('porte', 'A'), ('porte_fenetre', 'D')}
    assert len(read['stairs']) == 1 and {(e['inside'], e.get('name')) for e in read['extra']} == {(True, 'H'), (False, None)}
    assert {item['color']: item['role'] for item in read['legend']} == {'noir': 'mur', 'gris': 'escalier', 'rouge': 'porte', 'bleu': 'porte_fenetre'}
    ignored = lecture.read(str(tmp_path / 'plan.png'), [{**item, 'role': 'ignorer' if item['color'] == 'rouge' else item['role']} for item in read['legend']], ai=False)
    assert {o['type'] for o in ignored['openings']} == {'porte_fenetre'}


def test_apply_replaces_and_undo(tmp_path, monkeypatch):
    monkeypatch.setattr(lecture, 'vision_roles', lambda png: {})
    drawing(tmp_path / 'plan.png'); ctx = Ctx(); rpc.call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'})
    rpc.call(ctx, 'plan.sketch', {'path': str(tmp_path / 'plan.png')}); read = rpc.call(ctx, 'plan.read_drawing', {})
    for _ in range(2):
        done = rpc.call(ctx, 'plan.apply', {'plan': read}); rpc.call(ctx, 'update', {})
        loaded = components.load(ctx.project)
        assert sorted(c['label'] for c in loaded.values() if c['type'] == 'mur') == [f'Mur {n}' for n in 'ABCDEFGH'] and len(done['openings']) == 2 and done['stairs']
        assert len(loaded) == 11 and all(c['options'].get('axe') for c in loaded.values() if c['type'] == 'mur')
    width = np.ptp([p for w in plan.walls(ctx) for p in (w['a'], w['b'])], 0)
    assert width == pytest.approx([6.0, 5.0 / 6.0 * 6.0], rel=0.05)
    assert rpc.call(ctx, 'history.undo', {})['done'] and len(components.load(ctx.project)) == 11
    assert rpc.call(ctx, 'history.undo', {})['done'] and not components.load(ctx.project)
    assert rpc.call(ctx, 'history.redo', {})['done'] and len(components.load(ctx.project)) == 11
    rpc.call(ctx, 'components.remove', {'id': 'murD'})
    assert 'porte_fenetre1' not in components.load(ctx.project) and ctx.project.update().converged
    rpc.call(ctx, 'project.guide', {'guide': {'step': 4}, 'undo': True}); assert rpc.call(ctx, 'history.undo', {})['done']


def test_fit_all_walls_and_corner_check(tmp_path, monkeypatch):
    monkeypatch.setattr(lecture, 'vision_roles', lambda png: {})
    from test_walls import model as photo_model, ROTATION
    from plans3d import Project, photos
    import json, os, time
    folder = str(tmp_path / 'p'); project = Project(folder); ctx = Ctx(project=project)
    photos.add(ctx, [os.path.join(folder, 'a.jpg')]); cache = photos.source_cache(project, 'photos')[1]; os.makedirs(cache)
    np.savez(os.path.join(cache, 'model.npz'), **photo_model(np.random.default_rng(3)))
    with open(os.path.join(cache, 'meta.json'), 'w') as f: json.dump({'root': folder, 'registered': 6, 'model': '0'}, f)
    photos.extract(ctx)
    image = Image.new('RGB', (700, 800), 'white'); ImageDraw.Draw(image).rectangle([100, 100, 500, 600], outline=(0, 0, 0), width=5); image.save(tmp_path / 'plan.png')
    rpc.call(ctx, 'plan.sketch', {'path': str(tmp_path / 'plan.png')}); read = rpc.call(ctx, 'plan.read_drawing', {})
    started = time.perf_counter(); fit = rpc.call(ctx, 'plan.fit', {'plan': read}); read['fit'] = fit
    assert time.perf_counter() - started < 10 and sorted(fit['pairs']) == list('ABCD') and max(fit['gaps'].values()) < 5
    assert set(rpc.call(ctx, 'plan.corner_photos', {'point': [0.0, 0.0], 'pose': fit['pose']})) <= {f'img{i}.jpg' for i in range(6)}
    listed = rpc.call(ctx, 'plan.photo_gallery', {'point': [0.0, 0.0], 'pose': fit['pose'], 'walls': {'A': [[0.0, 0.0], [1.0, 0.0]]}})
    assert len(listed) == 6 and all('A' in v['sees'] for v in listed) and [v['visible'] for v in listed] == sorted((v['visible'] for v in listed), reverse=True)
    for corner in range(4):
        x, z = lecture.frame(read, fit['k'])[0](read['walls'][corner]['a']); seen = rpc.call(ctx, 'photos.visible', {'source': 'photos', 'points_xyz_m': [[x, 1.0, z]], 'pose': fit['pose']})
        if len(seen) >= 2: break
    check = lambda pair: rpc.call(ctx, 'plan.check_corner', {'clicks': pair, 'pose': fit['pose'], 'point': [x, z], 'height': fit['height'], 'lines': [['A', [x - 1.0, z + 0.5], [x + 1.0, z + 0.5]]]})
    at = lambda name, y, kind, du=0.0: {'image': name, 'kind': kind, **dict(zip('uv', np.add(rpc.call(ctx, 'photos.project_points', {'source': 'photos', 'image': name, 'points_xyz_m': [[x, y, z]], 'pose': fit['pose']})['uv'][0], [du, 0.0])))}
    a, b = seen[:2]
    for pair in ([at(a, 1.0, 'arete'), at(b, 1.6, 'arete')], [at(a, 0.0, 'bas'), at(b, 0.0, 'bas')], [at(a, fit['height'], 'haut'), at(b, 0.0, 'bas')], [at(a, 0.0, 'bas'), at(b, 1.2, 'arete')]):
        result = check(pair); assert result['ok'] and result['gap_cm'] < 3 and result['offset_cm'] < 5, (pair, result)
    assert not check([at(a, 0.0, 'bas'), at(b, 0.0, 'bas', 300.0)])['ok'] and not check([at(a, 1.0, 'arete'), at(a, 1.5, 'arete')])['ok']
    assert not rpc.call(ctx, 'plan.check_corner', {'clicks': [at(a, 2.0, 'haut'), at(b, 0.0, 'bas')], 'pose': fit['pose'], 'point': [x, z]})['ok']
    clicks = [at(a, 0.0, 'bas'), at(b, 1.0, 'arete')]
    read['points'] = [{'corner': corner, 'clicks': clicks}]; assert rpc.call(ctx, 'plan.apply', {'plan': read})['clicks'] == 2
    refit = {**read, 'fit': rpc.call(ctx, 'plan.fit', {'plan': read, 'turn': fit['turn']})}; assert rpc.call(ctx, 'plan.apply', {'plan': refit})['clicks'] == 2 and project.update().converged
    state = photos.photo_state(project, 'photos', compute=False)[1]; turned = {**read, 'fit': rpc.call(ctx, 'plan.fit', {'plan': read, 'turn': 2}), 'points': []}
    preview = rpc.call(ctx, 'photos.cloud', {'source': 'photos', 'limit': 300, 'pose': turned['fit']['pose']})['points']; rpc.call(ctx, 'plan.apply', {'plan': turned})
    assert np.allclose(photos.project_transform(project, 'photos', state)[0], turned['fit']['pose']['L']) and np.allclose(rpc.call(ctx, 'photos.cloud', {'source': 'photos', 'limit': 300})['points'], preview, atol=1e-3)
    rpc.call(ctx, 'plan.apply', {'plan': read}); result = project.update()
    lengths = sorted(np.hypot(*np.subtract(w['b'], w['a'])) for w in plan.walls(ctx))
    assert result.converged and not result.conflicts() and lengths[-1] / lengths[0] == pytest.approx(5 / 4, rel=0.03)


def test_order_constrained_match():
    drawn = [lecture.lined(np.array([x, 0.0]), np.array([x, 2.0]), name=n) for x, n in ((0.0, 'A'), (1.0, 'B'), (3.0, 'C'), (4.0, 'D'))]
    photo = [lecture.lined(np.array([x, 0.0]), np.array([x, 2.0])) for x in (0.0, 1.45, 3.0, 4.75)]
    pairs = lecture.match(drawn, photo, 1.0, [0.0, 0.0])
    assert {drawn[i]['name']: j for i, (_, j) in pairs.items()} == {'A': 0, 'B': 1, 'C': 2}
    crossed = [lecture.lined(np.array([x, 0.0]), np.array([x, 2.0])) for x in (0.0, 0.5, 0.6)]
    assert sorted(j for _, j in lecture.match(drawn[:2], crossed, 1.0, [0.0, 0.0]).values()) == [0, 2]


def test_apply_keeps_measures(tmp_path, monkeypatch):
    from plans3d.rpc import cotes_add, read_cotes
    monkeypatch.setattr(lecture, 'vision_roles', lambda png: {})
    drawing(tmp_path / 'plan.png'); ctx = Ctx(); rpc.call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'})
    rpc.call(ctx, 'plan.sketch', {'path': str(tmp_path / 'plan.png')}); read = rpc.call(ctx, 'plan.read_drawing', {})
    door = next(o for o in read['openings'] if o['type'] == 'porte'); read['openings'] = [{**door, 't0': 0.2, 't1': 0.35}, {**door, 't0': 0.62, 't1': 0.8}]
    rpc.call(ctx, 'plan.apply', {'plan': read}); wall = components.load(ctx.project)['murC']
    cotes_add(ctx, 'longueur.murC', components.measure_expr((wall, 'a'), (wall, 'b')), 410)
    for k in (1, 2): cotes_add(ctx, f'largeur.porte{k}', f'porte{k}_l', 80, label=f'Largeur de la porte {k}')
    done = rpc.call(ctx, 'plan.apply', {'plan': read})
    assert sorted(done['kept']) == ['largeur.porte1', 'largeur.porte2', 'longueur.murC'] and not done['dropped'] and len(read_cotes(ctx.project)) == 3
    done = rpc.call(ctx, 'plan.apply', {'plan': {**read, 'openings': read['openings'][1:]}}); rows = {r['id']: r['expr'] for r in read_cotes(ctx.project)}
    assert rows == {'longueur.murC': rows['longueur.murC'], 'largeur.porte1': 'porte1_l'} and done['dropped'] == [{'id': 'largeur.porte1', 'label': 'Largeur de la porte 1', 'reason': "Porte porte1 n'existe plus"}]
    assert next(r for r in read_cotes(ctx.project) if r['id'] == 'largeur.porte1')['label'] == 'Largeur de la porte 1'
    assert ctx.project.update().converged


def test_match_outermost_and_beyond():
    drawn = [lecture.lined(np.array([0.2, 0.0]), np.array([0.2, 3.0]), name='D', out=-1.0), lecture.lined(np.array([0.0, 0.0]), np.array([3.0, 0.0]), name='A', out=-1.0),
             lecture.lined(np.array([2.0, 0.0]), np.array([2.0, 3.0]), name='B', out=1.0)]
    photo = [lecture.lined(np.array([x, 0.0]), np.array([x, 3.0])) for x in (0.0, 0.3, 2.0)] + [lecture.lined(np.array([0.0, 0.0]), np.array([3.0, 0.0])),
                                                                                                lecture.lined(np.array([1.5, -0.9]), np.array([1.5, -0.4]))]
    assert {drawn[i]['name']: j for i, (_, j) in lecture.match(drawn, photo, 1.0, [0.0, 0.0]).items()} == {'D': 0, 'A': 3, 'B': 2}
    inner = [{**drawn[1], 'lo': 0.0, 'hi': 1.0}, lecture.lined(np.array([1.4, 0.0]), np.array([1.4, 1.0]), name='K', out=-1.0)]
    assert {inner[i]['name'] for i in lecture.match(inner, photo, 1.0, [0.0, 0.0])} == {'A'}


def test_fit_skips_furniture(tmp_path, monkeypatch):
    import json, os, test_walls
    from plans3d import Project, photos
    w, h, d = test_walls.ROOM; face = test_walls.face
    room = lambda rng: test_walls.to_photo(np.vstack([face(rng, 2500, [0, 0, 0], [w, 0, d]), face(rng, 2500, [0, h, 0], [w, h, d]), face(rng, 2500, [0, 0, 0], [0, h, d]),
                                                      face(rng, 2500, [w, 0, 0], [w, h, d]), face(rng, 2500, [0, 0, 0], [w, h, 0]), face(rng, 2500, [0, 0, d], [w, h, d]),
                                                      face(rng, 2500, [0.6, 0, 0.4], [0.6, 1.9, 4.6])]))
    monkeypatch.setattr(lecture, 'vision_roles', lambda png: {}); monkeypatch.setattr(test_walls, 'room', room)
    folder = str(tmp_path / 'p'); project = Project(folder); ctx = Ctx(project=project)
    photos.add(ctx, [os.path.join(folder, 'a.jpg')]); cache = photos.source_cache(project, 'photos')[1]; os.makedirs(cache); built = test_walls.model(np.random.default_rng(3))
    np.savez(os.path.join(cache, 'model.npz'), **test_walls.tracked(built))
    with open(os.path.join(cache, 'meta.json'), 'w') as f: json.dump({'root': folder, 'registered': 6, 'model': '0'}, f)
    photos.extract(ctx)
    image = Image.new('RGB', (700, 800), 'white'); ImageDraw.Draw(image).rectangle([160, 100, 500, 600], outline=(0, 0, 0), width=5); image.save(tmp_path / 'plan.png')
    rpc.call(ctx, 'plan.sketch', {'path': str(tmp_path / 'plan.png')}); read = rpc.call(ctx, 'plan.read_drawing', {}); fit = rpc.call(ctx, 'plan.fit', {'plan': read})
    furniture = {q['id'] for q in fit['photo'] if q['meuble']}
    assert len(furniture) == 1 and sorted(fit['pairs']) == list('ABCD') and not furniture & {i for ids in fit['pairs'].values() for i in ids}


def test_stair_flights_and_slope(tmp_path, monkeypatch):
    monkeypatch.setattr(lecture, 'vision_roles', lambda png: {})
    image = Image.new('RGB', (800, 700), 'white'); pen = ImageDraw.Draw(image); pen.rectangle([100, 100, 700, 600], outline=(0, 0, 0), width=5)
    pen.rectangle([600, 120, 690, 420], outline=(128, 128, 128), width=2); pen.rectangle([520, 340, 600, 420], outline=(128, 128, 128), width=2)
    for k in range(1, 9): pen.line([(600, 120 + 30 * k), (690, 120 + 30 * k)], fill=(128, 128, 128), width=2)
    image.save(tmp_path / 'plan.png'); read = lecture.read(str(tmp_path / 'plan.png'), ai=False)
    (main, rest), = [s['flights'] for s in read['stairs']]
    assert main == pytest.approx([600, 120, 690, 330], abs=6) and rest == pytest.approx([520, 340, 600, 420], abs=6) and read['stairs'][0]['turn']['square'] == pytest.approx([600, 330, 690, 420], abs=6)
    rng = np.random.default_rng(1); z = rng.uniform(1.0, 3.4, 400)
    steps = np.stack([rng.uniform(0.25, 0.75, 400), 0.8 * (z - 1.0) + rng.normal(0, 0.01, 400), z], 1); clutter = rng.uniform([0, 0, 0], [1, 2.4, 4], (150, 3))
    slope, start = lecture.measured_slope(np.vstack([steps, clutter]), np.array([0.1, 0.9]), np.array([0.9, 3.6]), 1, 2.5)
    assert slope == pytest.approx(0.8, abs=0.05) and start == pytest.approx(1.0, abs=0.05)
    assert lecture.measured_slope(np.vstack([steps, clutter]) * [1, 1, -1], np.array([0.1, -3.6]), np.array([0.9, -0.9]), 1, 2.5)[0] < 0
    ctx = Ctx(); rpc.call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'}); rpc.call(ctx, 'plan.sketch', {'path': str(tmp_path / 'plan.png')})
    made = rpc.call(ctx, 'plan.apply', {'plan': {**read, 'stairs': [{'box': read['stairs'][0]['box']}]}})['stairs']; loaded = components.load(ctx.project)
    first, second = (loaded[i] for i in made)
    assert first['options']['axe'] == 'z' and second['options']['axe'] == 'x' and second['options']['y0'] > 1.5 and second['options']['sens'] == -1.0
    assert 'murB' in str(first['expressions']['x']) and ctx.project.update().converged


def test_stair_measure_moves_stair(tmp_path, monkeypatch):
    from plans3d import mesures
    from plans3d.rpc import cotes_add
    monkeypatch.setattr(lecture, 'vision_roles', lambda png: {})
    image = Image.new('RGB', (800, 700), 'white'); pen = ImageDraw.Draw(image); pen.rectangle([100, 100, 700, 600], outline=(0, 0, 0), width=5)
    pen.rectangle([450, 150, 540, 450], outline=(128, 128, 128), width=2)
    for k in range(1, 9): pen.line([(450, 150 + 33 * k), (540, 150 + 33 * k)], fill=(128, 128, 128), width=2)
    image.save(tmp_path / 'plan.png'); ctx = Ctx(); rpc.call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'}); rpc.call(ctx, 'plan.sketch', {'path': str(tmp_path / 'plan.png')})
    read = rpc.call(ctx, 'plan.read_drawing', {}); stair, = rpc.call(ctx, 'plan.apply', {'plan': read})['stairs']; project = ctx.project
    assert f'{stair}_dz' in components.load(project)[stair]['params'] and 'murA' in str(components.load(project)[stair]['expressions']['z'])
    values = plan.current_values(project); items = {c['key']: c for c in mesures.candidates(mesures.elements(project, values))}
    for key, extra in (('entre.murA.murC', 0.0), (f'marche_p.{stair}', 0.0), (f'entre.{stair}.murC', 0.5)):
        cotes_add(ctx, key, items[key]['expr'], (components.evaluate(components.Expr.of(items[key]['expr']), values) + extra) * 100)
    before = components.evaluate(components.load(project)[stair]['expressions']['z'], values); result = project.update()
    after = components.evaluate(components.load(project)[stair]['expressions']['z'], plan.current_values(project))
    assert result.converged and not result.contradictions and abs(after - before) > 0.35


def test_identical_openings_share_size(tmp_path, monkeypatch):
    monkeypatch.setattr(lecture, 'vision_roles', lambda png: {})
    drawing(tmp_path / 'plan.png'); ctx = Ctx(); rpc.call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'})
    rpc.call(ctx, 'plan.sketch', {'path': str(tmp_path / 'plan.png')}); read = rpc.call(ctx, 'plan.read_drawing', {})
    door = next(o for o in read['openings'] if o['type'] == 'porte')
    read['openings'] = [{**door, 'same': True}, {**door, 't0': 0.62, 't1': 0.8, 'same': True}]
    placed = rpc.call(ctx, 'plan.apply', {'plan': read})['openings']; loaded = components.load(ctx.project)
    assert str(loaded[placed[1]]['expressions']['l']) == f'{placed[0]}_l' and str(loaded[placed[1]]['expressions']['h']) == f'{placed[0]}_h'
    result = ctx.project.update(); widths = sorted((t1 - t0) * 500 / 100 * 6 / 6.02 for t0, t1 in ((door['t0'], door['t1']), (0.62, 0.8)))
    assert result.converged and widths[0] < result.value(f'{placed[0]}_l')[0] < widths[1] + 0.2


def quarter_turn(path):
    image = Image.new('RGB', (760, 700), 'white'); pen = ImageDraw.Draw(image); grey = (128, 128, 128)
    room = [(100, 100), (600, 100), (600, 600), (200, 600), (200, 570), (130, 500), (100, 500)]
    pen.line(room + [room[0]], fill=(0, 0, 0), width=5); pen.line([(100, 300), (300, 300)], fill=(0, 0, 0), width=5)
    pen.rectangle([520, 150, 597, 450], outline=grey, width=2); pen.rectangle([440, 370, 520, 450], outline=grey, width=2)
    for y in np.linspace(177.5, 370, 8): pen.line([(520, y), (597, y)], fill=grey, width=2)
    for end in ((597, 403), (597, 450), (553, 450)): pen.line([(520, 370), end], fill=grey, width=2)
    for x in (467, 493): pen.line([(x, 370), (x, 450)], fill=grey, width=2)
    image.save(path)


def test_quarter_turn_stair_partition_and_symmetry(tmp_path, monkeypatch):
    monkeypatch.setattr(lecture, 'vision_roles', lambda png: {})
    quarter_turn(tmp_path / 'plan.png'); read = lecture.read(str(tmp_path / 'plan.png'), ai=False)
    assert [w['name'] for w in read['walls']] == list('ABCDEFGH') and read['walls'][6]['b'][1] == pytest.approx(300, abs=4)
    partition, = [e for e in read['extra'] if e.get('name')]
    assert partition['name'] == 'I' and partition['a'] == pytest.approx(read['walls'][7]['a'], abs=1) and partition['b'][0] == pytest.approx(300, abs=8)
    assert read['rules'] == [{'kind': 'symetrie', 'walls': ['D', 'E', 'F'], 'ok': None}]
    stair, = read['stairs']
    assert stair['treads'] == [8, 3] and stair['turn']['winders'] == 4 and stair['turn']['square'] == pytest.approx([520, 370, 597, 450], abs=8)
    ctx = Ctx(); rpc.call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'}); rpc.call(ctx, 'plan.sketch', {'path': str(tmp_path / 'plan.png')})
    made = rpc.call(ctx, 'plan.apply', {'plan': read})['stairs']; loaded = components.load(ctx.project); first, second = (loaded[i] for i in made)
    assert first['options']['tournant'] == 4 and first['options']['n'] == 13 and second['options']['n'] == 4 and first['options']['axe'] != second['options']['axe']
    assert {'Mur I'} <= {c['label'] for c in loaded.values() if c['type'] == 'mur'} and ctx.project.update().converged
    shapes = components.stairs_geometry(first, lambda e: components.evaluate(e, plan.current_values(ctx.project)), [])
    assert [s.get('kind') for s in shapes] == ['box'] * 8 + ['mesh'] * 4


def test_wall_by_wall_solve_and_symmetry():
    walls = [((100, 100), (600, 100)), ((600, 100), (600, 600)), ((600, 600), (200, 600)), ((200, 600), (200, 560)), ((200, 560), (130, 500)), ((130, 500), (100, 500)), ((100, 500), (100, 100))]
    drawn = {'angle': 0.0, 'walls': [{'name': n, 'a': list(a), 'b': list(b)} for n, (a, b) in zip('ABCDEFG', walls)], 'extra': []}
    forward = lecture.frame(drawn, 0.01)[0]; ring = lecture.ring_model(drawn, forward)
    seen = {'C': (np.array([4.0, 5.2]), np.array([1.0, 5.2])), 'G': (np.array([-0.3, 4.0]), np.array([-0.3, 0.0])), 'A': (np.array([0.0, 0.0]), np.array([5.0, 0.0])),
            'B': (np.array([5.0, 0.0]), np.array([5.0, 5.0]))}
    offsets, evidence, _ = lecture.adjust(ring, seen, np.zeros((0, 3)), 2.5, [], [])
    ends = dict(zip(ring['names'], lecture.ring_ends(ring, offsets)))
    assert ends['C'][0][1] == pytest.approx(5.2, abs=0.01) and ends['G'][0][0] == pytest.approx(-0.3, abs=0.01) and max(gap for _, gap in evidence.values()) < 0.01
    rules = [{'kind': 'symetrie', 'walls': ['D', 'E', 'F'], 'ok': True}]; ring = lecture.ring_model(drawn, forward, rules)
    ends = dict(zip(ring['names'], lecture.ring_ends(ring, lecture.adjust(ring, seen, np.zeros((0, 3)), 2.5, [], rules)[0])))
    size = {n: np.hypot(*(b - a)) for n, (a, b) in ends.items()}; d = ends['E'][1] - ends['E'][0]
    assert size['D'] == pytest.approx(size['F'], abs=0.01) and abs(abs(d[0]) - abs(d[1])) < 0.01


def test_short_wall_found_on_corner_edge():
    column = lambda t, s, top, n=60: np.column_stack([np.full(n, t), np.linspace(0.2, top, n), np.full(n, s)])
    found = lambda *columns: lecture.wall_points(np.vstack(columns), np.zeros(2), np.array([0.6, 0.0]), np.array([0.0, 1.0]), -1.0, 2.4, [], [])
    assert found(column(0.04, 0.24, 2.0)) == (pytest.approx(0.24), 60) and found(column(0.04, 0.4, 1.1)) is None
    assert found(column(0.04, 0.24, 2.0, 90), column(0.3, -0.1, 2.0))[0] == pytest.approx(-0.1)


def test_far_wall_matched_by_order_and_extent():
    drawn = [lecture.lined(np.array([x, 0.0]), np.array([x, top]), name=n, out=-1.0 if x < 2 else 1.0) for x, top, n in ((0.0, 4.0, 'A'), (1.4, 2.0, 'B'), (5.0, 4.0, 'C'))]
    drawn += [lecture.lined(np.array([0.0, y]), np.array([5.0, y]), name=n, out=-1.0 if y < 2 else 1.0) for y, n in ((0.0, 'D'), (4.0, 'E'))]
    photo = [lecture.lined(np.array(a), np.array(b), plane={'id': i, 'n': 100}) for i, (a, b) in enumerate((([0.0, 0.0], [0.0, 4.0]), ([0.7, 0.0], [0.7, 2.0]), ([1.3, 3.0], [1.3, 3.5]),
                                                                                                       ([5.0, 0.0], [5.0, 4.0]), ([0.0, 0.0], [5.0, 0.0]), ([0.0, 4.0], [5.0, 4.0])))]
    pairs = lecture.refine(drawn, photo, 1.0, [0.0, 0.0])[2]
    assert {drawn[i]['name']: j for i, (_, j) in pairs.items()} == {'A': 0, 'B': 1, 'C': 3, 'D': 4, 'E': 5}


def test_outermost_strong_peak_and_hidden_wall():
    column = lambda t, s, n: np.column_stack([np.full(n, t), np.linspace(0.2, 2.2, n), np.full(n, s)])
    cloud = np.vstack([column(t, 0.3, 120) for t in (0.5, 1.5)] + [column(t, -0.1, 60) for t in (0.5, 1.5)] + [column(1.0, -0.4, 6)])
    assert lecture.wall_points(cloud, np.zeros(2), np.array([2.0, 0.0]), np.array([0.0, 1.0]), -1.0, 2.4, [], [])[0] == pytest.approx(-0.1, abs=0.01)
    wall = lecture.lined(np.array([0.0, 0.0]), np.array([0.0, 3.0]), name='A', out=-1.0)
    planes = [{'id': 'p1', 'meuble': True, 'a': np.array([0.05, 0.5]), 'b': np.array([0.05, 2.5])}, {'id': 'p2', 'meuble': True, 'a': np.array([0.6, 0.5]), 'b': np.array([0.6, 2.0])}]
    assert lecture.hidden_walls([wall], {}, planes, 0.0, 1.0, [0.0, 0.0]) == {'A': planes[0]}
    planes.append({'id': 'p3', 'meuble': False, 'a': np.array([-0.4, 1.0]), 'b': np.array([-0.4, 2.0])})
    assert lecture.hidden_walls([wall], {}, planes, 0.0, 1.0, [0.0, 0.0]) == {}


def test_reread_keeps_names_and_clicks(tmp_path):
    quarter_turn(tmp_path / 'plan.png'); new = lecture.read(str(tmp_path / 'plan.png'), ai=False)
    old = {**new, 'walls': [{**w, 'name': n} for w, n in zip(new['walls'][:6], 'PQRSTU')] + [{'name': 'V', 'a': new['walls'][6]['a'], 'b': new['walls'][7]['b']}], 'extra': [],
           'points': [{'corner': 6, 'clicks': [{'image': 'a.jpg', 'u': 1.0, 'v': 2.0}] * 2}, {'corner': 2, 'clicks': []}], 'refused': {'6': ['b.jpg']}}
    kept = lecture.renamed(new, old)
    assert [w['name'] for w in kept['walls']][:6] == list('PQRSTU') and {kept['walls'][6]['name'], kept['walls'][7]['name']} & {'V'}
    assert [c['corner'] for c in kept['points']] == [6, 2] and kept['refused'] == {'6': ['b.jpg']} and kept['rules'][0]['walls'] == ['S', 'T', 'U']
    assert len({w['name'] for w in kept['walls']} | {e['name'] for e in kept['extra'] if e.get('name')}) == 9
