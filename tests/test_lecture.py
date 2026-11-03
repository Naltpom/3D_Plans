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
    assert names == list('ABCDEF') and read['walls'][0]['a'][1] < 110 and read['walls'][0]['b'][0] > 590
    for w in read['walls']:
        d = np.subtract(w['b'], w['a']) @ np.array([[np.cos(read['angle']), -np.sin(read['angle'])], [np.sin(read['angle']), np.cos(read['angle'])]])
        assert min(abs(d)) < 1e-3 * max(abs(d))
    assert {(o['type'], o['wall']) for o in read['openings']} == {('porte', 'A'), ('porte_fenetre', 'D')}
    assert len(read['stairs']) == 1 and {e['inside'] for e in read['extra']} == {True, False}
    assert {item['color']: item['role'] for item in read['legend']} == {'noir': 'mur', 'gris': 'escalier', 'rouge': 'porte', 'bleu': 'porte_fenetre'}
    ignored = lecture.read(str(tmp_path / 'plan.png'), [{**item, 'role': 'ignorer' if item['color'] == 'rouge' else item['role']} for item in read['legend']], ai=False)
    assert {o['type'] for o in ignored['openings']} == {'porte_fenetre'}


def test_apply_replaces_and_undo(tmp_path, monkeypatch):
    monkeypatch.setattr(lecture, 'vision_roles', lambda png: {})
    drawing(tmp_path / 'plan.png'); ctx = Ctx(); rpc.call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'})
    rpc.call(ctx, 'plan.sketch', {'path': str(tmp_path / 'plan.png')}); read = rpc.call(ctx, 'plan.read_drawing', {})
    read['extra'] = [{**e, 'name': 'G'} if e['inside'] else e for e in read['extra']]
    for _ in range(2):
        done = rpc.call(ctx, 'plan.apply', {'plan': read}); rpc.call(ctx, 'update', {})
        loaded = components.load(ctx.project)
        assert sorted(c['label'] for c in loaded.values() if c['type'] == 'mur') == [f'Mur {n}' for n in 'ABCDEFG'] and len(done['openings']) == 2 and done['stairs']
        assert len(loaded) == 10
    width = np.ptp([p for w in plan.walls(ctx) for p in (w['a'], w['b'])], 0)
    assert width == pytest.approx([6.0, 5.0 / 6.0 * 6.0], rel=0.05)
    assert rpc.call(ctx, 'history.undo', {})['done'] and len(components.load(ctx.project)) == 10
    assert rpc.call(ctx, 'history.undo', {})['done'] and not components.load(ctx.project)
    assert rpc.call(ctx, 'history.redo', {})['done'] and len(components.load(ctx.project)) == 10
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
    for corner in range(4):
        x, z = lecture.frame(read, fit['k'])[0](read['walls'][corner]['a']); seen = rpc.call(ctx, 'photos.visible', {'source': 'photos', 'points_xyz_m': [[x, 1.0, z]], 'pose': fit['pose']})
        if len(seen) >= 2: break
    clicks = [{'image': name, 'u': uv[0], 'v': uv[1]} for name in seen[:2] for uv in rpc.call(ctx, 'photos.project_points', {'source': 'photos', 'image': name, 'points_xyz_m': [[x, 1.0, z]], 'pose': fit['pose']})['uv']]
    assert len(clicks) == 2 and rpc.call(ctx, 'plan.check_corner', {'clicks': clicks, 'pose': fit['pose'], 'point': [x, z]})['ok']
    wrong = [clicks[0], {**clicks[1], 'u': clicks[1]['u'] + 300}]
    assert not rpc.call(ctx, 'plan.check_corner', {'clicks': wrong, 'pose': fit['pose'], 'point': [x, z]})['ok']
    read['points'] = [{'corner': corner, 'clicks': clicks}]; rpc.call(ctx, 'plan.apply', {'plan': read}); result = project.update()
    lengths = sorted(np.hypot(*np.subtract(w['b'], w['a'])) for w in plan.walls(ctx))
    assert result.converged and not result.conflicts() and lengths[-1] / lengths[0] == pytest.approx(5 / 4, rel=0.03)


def test_order_constrained_match():
    drawn = [lecture.lined(np.array([x, 0.0]), np.array([x, 2.0]), name=n) for x, n in ((0.0, 'A'), (1.0, 'B'), (3.0, 'C'), (4.0, 'D'))]
    photo = [lecture.lined(np.array([x, 0.0]), np.array([x, 2.0])) for x in (0.0, 1.45, 3.0, 4.75)]
    pairs = lecture.match(drawn, photo, 1.0, [0.0, 0.0])
    assert {drawn[i]['name']: j for i, (_, j) in pairs.items()} == {'A': 0, 'B': 1, 'C': 2}
    crossed = [lecture.lined(np.array([x, 0.0]), np.array([x, 2.0])) for x in (0.0, 0.5, 0.6)]
    assert sorted(j for _, j in lecture.match(drawn[:2], crossed, 1.0, [0.0, 0.0]).values()) == [0, 2]
