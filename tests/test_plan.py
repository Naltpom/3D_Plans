import json, os
import numpy as np
import pytest
from plans3d import Project, cadastre, components, plan, terrain
from plans3d.rpc import Ctx, read_cotes
from test_cadastre import make_geojson


@pytest.fixture
def ctx(tmp_path):
    return Ctx(project=Project(str(tmp_path / 'projet')))


def shapes(project, result): return {s['id']: s for s in components.geometry(project, result)}


def test_trace_room_shares_corners(ctx):
    created = plan.rpc_trace(ctx, [[0, 0], [4, 0], [4, 5], [0, 5]], True, [4.0, None, None, None])['created']
    loaded = components.load(ctx.project); assert created == ['mur1', 'mur2', 'mur3', 'mur4']
    assert sum(len(c['params']) for c in loaded.values()) == 4 + 4 + 1 and [c['options']['axe'] for c in loaded.values()] == ['x', 'z', 'x', 'z']
    assert str(loaded['mur1']['expressions']['start_x']) == 'mur4_x' and str(loaded['mur1']['expressions']['end_x']) == 'mur2_x'
    assert [row['id'] for row in read_cotes(ctx.project)] == ['trace_mur1']
    result = ctx.project.update(); found = shapes(ctx.project, result)
    assert not result.contradictions and np.isfinite(result.sigma).all()
    assert sorted(round(found[c]['size'][0], 2) for c in created) == pytest.approx([4.0, 4.0, 5.0, 5.0], abs=0.01)
    assert found['mur1']['center'][2] == pytest.approx(-0.1) and found['mur2']['center'][0] == pytest.approx(4.1)


def test_trace_open_slanted(ctx):
    plan.rpc_trace(ctx, [[0, 0], [3, 0], [5, 2]], False)
    loaded = components.load(ctx.project)
    assert loaded['mur1']['options']['axe'] == 'x' and 'axe' not in loaded['mur2']['options']
    assert str(loaded['mur1']['expressions']['end_x']) == str(loaded['mur2']['expressions']['start_x']) == 'mur2_x0'
    assert str(loaded['mur2']['expressions']['start_z']) == 'mur1_z'
    walls = {w['id']: w for w in plan.walls(ctx)}
    assert walls['mur2']['a'] == pytest.approx([3, 0]) and walls['mur2']['b'] == pytest.approx([5, 2]) and walls['mur2']['height'] == 2.5
    with pytest.raises(ValueError): plan.rpc_trace(ctx, [[0, 0], [1, 1]], True)


def test_place_openings(ctx):
    plan.rpc_trace(ctx, [[0, 0], [4, 0], [4, 5], [0, 5]], True)
    door = plan.place(ctx, 'porte', [2.0, 0.1]); window = plan.place(ctx, 'fenetre', [3.9, 2.5], 1.0)
    assert door['options']['mur'] == 'mur1' and window['options']['mur'] == 'mur2'
    with pytest.raises(ValueError, match='mur'): plan.place(ctx, 'porte', [2.0, 2.5])
    result = ctx.project.update(); found = shapes(ctx.project, result)
    assert np.isfinite(result.sigma).all() and found[door['id']]['center'][0] == pytest.approx(2.0) and found[window['id']]['center'][2] == pytest.approx(2.5)


def test_sketch_saved_and_calibrated(ctx, tmp_path):
    from PIL import Image
    Image.new('RGB', (40, 30), 'white').save(tmp_path / 'croquis.png')
    stored = plan.sketch(ctx, str(tmp_path / 'croquis.png'))
    assert os.path.exists(stored['path']) and stored['file'] == 'croquis.png'
    with pytest.raises(ValueError): plan.sketch(ctx, points=[[1, 1], [1, 1]], length=2)
    calibrated = plan.sketch(ctx, points=[[5, 5], [35, 5]], length=3.0, size=[40, 30])
    assert calibrated['length'] == 3.0 and json.load(open(os.path.join(ctx.project.folder, 'project.json')))['croquis']['size'] == [40, 30]
    assert plan.sketch(ctx, opacity=0.3)['opacity'] == 0.3 and plan.sketch(ctx)['length'] == 3.0
    with pytest.raises(ValueError): plan.sketch(ctx, opacity=2)


def test_cadastre_walls_and_parcel(ctx, tmp_path):
    cadastre.add_source(ctx.project, make_geojson(str(tmp_path / 'parcelle.geojson')))
    found = plan.from_cadastre(ctx)
    assert found['house']['walls'] == 4 and found['house']['perimeter'] == pytest.approx(28, abs=0.01) and found['parcel']['area'] == pytest.approx(200, abs=0.1)
    created = plan.from_cadastre(ctx, create=True)['created']
    assert len(created) == 5 and plan.from_cadastre(ctx, create=True)['created'] == created
    loaded = components.load(ctx.project); house = [loaded[c] for c in created[:4]]
    assert all(c['params'] == [f"{c['id']}_ep"] and str(c['expressions']['h']).endswith('_b0_h') for c in house)
    result = ctx.project.update(); found = shapes(ctx.project, result)
    assert not result.contradictions and found['mur1']['size'][1] == pytest.approx(6.0)
    middle = np.mean(plan.main_building(ctx.project)[1]['pts'], 0)
    for wall in plan.walls(ctx):
        line, box = np.mean([wall['a'], wall['b']], 0), np.array(found[wall['id']]['center'])[[0, 2]]
        assert np.linalg.norm(box - middle) < np.linalg.norm(line - middle)


def test_read_sketch_with_fake_server(ctx, tmp_path, monkeypatch):
    import threading
    from http.server import ThreadingHTTPServer
    from PIL import Image
    from plans3d import ai
    from test_ai import Fake
    replies = {'corner': [{'point_2d': [100, 100], 'label': 'corner'}, {'point_2d': [602, 98], 'label': 'corner'}, {'point_2d': [600, 500], 'label': 'corner'},
                          {'point_2d': [100, 503], 'label': 'corner'}, {'point_2d': [101, 100], 'label': 'corner'}],
               'dimension': [{'bbox_2d': [300, 60, 380, 90], 'text_content': '5,00 m'}, {'bbox_2d': [620, 280, 680, 320], 'text_content': '3 m'}],
               'doors': [{'bbox_2d': [300, 480, 400, 520], 'label': 'door'}, {'bbox_2d': [1, 1, 5, 5], 'label': 'fireplace'}]}
    reply = lambda body: {'choices': [{'message': {'content': '```json\n' + json.dumps(next(v for k, v in replies.items() if k in body['messages'][0]['content'][1]['text'])) + '\n```'}}]}
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Fake); httpd.seen = []
    httpd.routes = {'/v1/chat/completions': reply}
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(ai, 'SETTINGS', str(tmp_path / 'settings.json')); monkeypatch.setattr(ai, 'URL', f'http://127.0.0.1:{httpd.server_port}/v1')
    Image.new('RGB', (1000, 800), 'white').save(tmp_path / 'croquis.png')
    plan.sketch(ctx, str(tmp_path / 'croquis.png')); plan.sketch(ctx, points=[[100, 100], [600, 100]], length=5.0, size=[1000, 800])
    found = plan.read_sketch(ctx); httpd.shutdown(); httpd.server_close()
    body = httpd.seen[0][2]
    assert len(httpd.seen) == 3 and body['model'] == ai.VISION and body['messages'][0]['content'][0]['image_url']['url'].startswith('data:image/png;base64,')
    wall = found['walls'][0]
    assert wall['closed'] and np.allclose(wall['points'], [[0, 0], [5, 0], [5, 4], [0, 4]], atol=0.03) and wall['length'] == pytest.approx(18, abs=0.1)
    good, bad = found['cotes']
    assert good['ok'] and (good['wall'], good['edge']) == (0, 0) and not bad['ok'] and 'écart' in bad['message'] and bad['value_cm'] == 300
    assert found['openings'] == [{'type': 'porte', 'point': pytest.approx([2.5, 4.0], abs=0.03)}]
    plan.rpc_trace(ctx, wall['points'], True, [5.0, None, None, None], 'lu', 2.0)
    assert [(row['status'], row['sigma']) for row in read_cotes(ctx.project)] == [('lu', 2.0)]


def test_house_roof_and_terrain(ctx, tmp_path):
    cadastre.add_source(ctx.project, make_geojson(str(tmp_path / 'parcelle.geojson'))); plan.from_cadastre(ctx, create=True)
    components.add(ctx.project, 'terrain', 'terrain', {'x0': -10, 'z0': -10, 'nx': 2, 'nz': 2, 'pas_x': 40, 'pas_z': 40, 'altitudes': [[1.0, 1.0], [1.0, 1.0]]})
    moved = plan.settle(ctx)['moved']; made = plan.roof(ctx)
    assert len(moved) == 4 and made['created'] and made['measured'] and not plan.roof(ctx)['created']
    result = ctx.project.update(); found = shapes(ctx.project, result)
    assert not result.contradictions and found['mur1']['center'][1] == pytest.approx(1.0 + 3.0, abs=0.01)
    vertices = [v for s in components.geometry(ctx.project, result) if s['type'] == 'toit' for v in s['vertices']]
    heights = sorted({round(v[1], 2) for v in vertices})
    assert heights[0] == pytest.approx(7.0, abs=0.01) and heights[-1] > 7.0
    xs, zs = zip(*[(v[0], v[2]) for v in vertices])
    assert min(xs) < 4.0 and max(xs) > 12.0 and min(zs) < 3.0 and max(zs) > 9.0


def test_vehicle_body(ctx):
    created = plan.vehicle(ctx, 330, 170, 180, 90, 20, 30, 80, 'droite', 100, 130, 60, 150, 160)['created']
    loaded = components.load(ctx.project)
    assert [loaded[c]['type'] for c in created] == ['mur'] * 4 + ['sol', 'boite', 'boite', 'porte', 'porte']
    assert loaded[created[7]]['options']['mur'] == 'mur3' and loaded[created[8]]['options']['mur'] == 'mur4'
    result = ctx.project.update(); found = shapes(ctx.project, result)
    assert not result.contradictions and sorted(round(found[c]['size'][0], 2) for c in created[:4]) == [1.7, 1.7, 3.3, 3.3]
    assert found['mur1']['size'][1] == pytest.approx(1.8) and found[created[7]]['size'][1] == pytest.approx(1.3)
    with pytest.raises(ValueError, match='déjà'): plan.vehicle(ctx, 300, 160, 170)
