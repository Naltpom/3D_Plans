import json, os
import numpy as np
import pytest
from plans3d import Project, cadastre, components, plan
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
