import json, os
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from plans3d import Project, components, photos, plan, rpc, walls
from plans3d.rpc import Ctx, cotes_add, read_cotes

SCALE, ROOM, CAMERA_HEIGHT = 0.4, (4.0, 2.5, 5.0), 1.6
ROTATION = Rotation.from_euler('xyz', [10, 30, -20], degrees=True).as_matrix()
SHIFT = np.array([1.0, 2.0, -3.0])
SPOTS = [(1.0, 1.0), (3.0, 1.0), (3.0, 4.0), (1.0, 4.0), (2.0, 2.5), (2.0, 0.8)]


def to_photo(points_m): return np.asarray(points_m, float) @ ROTATION.T / SCALE + SHIFT


def face(rng, count, low, high, noise=0.003):
    points = rng.uniform(low, high, (count, 3)); flat = np.flatnonzero(np.array(low) == np.array(high))
    points[:, flat] += rng.normal(0, noise, (count, len(flat))); return points


def room(rng):
    w, h, d = ROOM
    parts = [face(rng, 2500, [0, 0, 0], [w, 0, d]), face(rng, 2500, [0, h, 0], [w, h, d]), face(rng, 2500, [0, 0, 0], [0, h, d]),
             face(rng, 2500, [w, 0, 0], [w, h, d]), face(rng, 2500, [0, 0, 0], [w, h, 0]), face(rng, 180, [0, 0, d], [w, h, d]),
             face(rng, 900, [0.6, 0, 1.0], [0.6, 2.0, 3.0]), face(rng, 400, [1.0, 0.4, 4.2], [3.0, 0.9, 4.2]), rng.uniform(0, 1, (300, 3)) * ROOM]
    return to_photo(np.vstack(parts))


def model(rng):
    up, poses = ROTATION @ [0, 1, 0], []
    for x, z in SPOTS:
        forward = photos.unit(to_photo([2.0, 1.2, 2.5]) - to_photo([x, CAMERA_HEIGHT, z])); image_up = photos.unit(up - (up @ forward) * forward)
        rotation = np.array([np.cross(-image_up, forward), -image_up, forward]); poses.append((rotation, -rotation @ to_photo([x, CAMERA_HEIGHT, z])))
    count = len(poses)
    return {'names': np.array([f'img{i}.jpg' for i in range(count)]), 'q': np.array([np.roll(Rotation.from_matrix(r).as_quat(), 1) for r, _ in poses]),
            't': np.array([t for _, t in poses]), 'K': np.tile([1000.0, 1000.0, 800.0, 600.0, 0, 0, 0, 0], (count, 1)),
            'wh': np.tile([1600, 1200], (count, 1)), 'orientation': np.ones(count, int), 'P': room(rng)}


def write_tracks(cache, m):
    folder = os.path.join(cache, 'sparse', '0'); os.makedirs(folder, exist_ok=True); local = np.einsum('nij,pj->npi', photos.rotation_matrices(m['q']), m['P']) + m['t'][:, None]
    fx, fy, cx, cy = m['K'][0, :4]; u, v = fx * local[..., 0] / local[..., 2] + cx, fy * local[..., 1] / local[..., 2] + cy
    seen = (local[..., 2] > 0) & (u > 0) & (u < 1600) & (v > 0) & (v < 1200)
    with open(os.path.join(folder, 'images.txt'), 'w') as f: f.write('# images\n' + ''.join(f'{i + 7} 1 0 0 0 0 0 0 1 {name}\n\n' for i, name in enumerate(m['names'])))
    with open(os.path.join(folder, 'points3D.txt'), 'w') as f:
        f.write('# points\n' + ''.join(f"{n + 1} {x} {y} {z} 0 0 0 0.5 {' '.join(f'{i + 7} 0' for i in np.flatnonzero(seen[:, n]))}\n" for n, (x, y, z) in enumerate(m['P'])))


@pytest.fixture
def setup(tmp_path):
    folder = str(tmp_path); project = Project(folder); ctx = Ctx(project=project)
    photos.add(ctx, [os.path.join(folder, 'a.jpg')]); cache = photos.source_cache(project, 'photos')[1]; os.makedirs(cache)
    np.savez(os.path.join(cache, 'model.npz'), **model(np.random.default_rng(3)))
    with open(os.path.join(cache, 'meta.json'), 'w') as f: json.dump({'root': folder, 'registered': 6, 'model': '0'}, f)
    mapping = ''
    for row in np.array(photos.planes(ctx)['R']):
        direction = ROTATION.T @ row; axis = int(np.abs(direction).argmax()); mapping += ('+' if direction[axis] > 0 else '-') + 'xyz'[axis]
    photos.axes(ctx, 'photos', mapping); photos.extract(ctx)
    return ctx, project


def by_position(found):
    return {(w['axis'], round(w['position'] - min(v['position'] for v in found['walls'] if v['axis'] == w['axis']), 1)): w for w in found['walls']}


def test_propose_walls(setup):
    ctx, _ = setup; found = walls.propose_walls(ctx); listed = by_position(found)
    assert [w['id'] for w in found['walls']] == list('ABCDE') and found['floor'] and found['ceiling'] and not found['dense']
    scale = photos.CAMERA_HEIGHT_M / CAMERA_HEIGHT
    assert sorted(listed) == sorted([('x', 0.0), ('x', round(0.6 * scale, 1)), ('x', round(4 * scale, 1)), ('z', 0.0), ('z', round(5 * scale, 1))])
    assert [w['sure'] for w in found['walls']] == [True, True, True, False, False]
    assert listed[('z', round(5 * scale, 1))]['reason'] == 'peu de points'
    assert listed[('x', round(0.6 * scale, 1))]['reason'] == f"devant le mur {listed[('x', 0.0)]['id']} : meuble ?"
    assert listed[('x', 0.0)]['length'] == pytest.approx(5 * scale, abs=0.05) and listed[('x', 0.0)]['height'] == pytest.approx(2.5 * scale, abs=0.05)
    assert all(w['photos'] > 0 and w['component'] is None for w in found['walls'])


def test_furniture_seen_through(setup):
    ctx, project = setup; cache = photos.source_cache(project, 'photos')[1]; write_tracks(cache, photos.load_model(cache))
    listed = by_position(walls.propose_walls(ctx)); scale = photos.CAMERA_HEIGHT_M / CAMERA_HEIGHT
    assert listed[('x', round(0.6 * scale, 1))]['reason'] == walls.FURNITURE and listed[('x', round(0.6 * scale, 1))]['meuble'] and not listed[('x', round(0.6 * scale, 1))]['sure']
    assert all(listed[key]['reason'] != walls.FURNITURE for key in (('x', 0.0), ('x', round(4 * scale, 1)), ('z', 0.0)))
    assert os.path.exists(os.path.join(cache, 'tracks.npz')) and walls.point_tracks(cache, '0')[1].min() == 0


def test_create_walls_scale_and_replay(setup):
    ctx, project = setup; listed = by_position(walls.propose_walls(ctx)); scale = photos.CAMERA_HEIGHT_M / CAMERA_HEIGHT
    room_walls = [listed[key]['id'] for key in (('x', 0.0), ('x', round(4 * scale, 1)), ('z', 0.0), ('z', round(5 * scale, 1)))]
    with pytest.raises(ValueError): walls.create_walls(ctx, 'photos', room_walls[:2])
    created = walls.create_walls(ctx, 'photos', room_walls)['created']; found = walls.propose_walls(ctx)
    assert sorted(created) == sorted(f'mur{letter}' for letter in room_walls)
    assert all(all(w['angles'].values()) for w in found['walls'] if w['component']) and found['measure']
    loaded = components.load(project); measured = loaded[found['measure']]
    for comp in loaded.values():
        axis = 'z' if comp['options']['axe'] == 'x' else 'x'
        assert str(comp['expressions'][f'start_{axis}']) == '0' or f"{comp['id']}_{axis}" in comp['params']
    expr = components.measure_expr((measured, 'a'), (measured, 'b'))
    cotes_add(ctx, walls.SCALE_COTE, expr, 500 if measured['options']['axe'] == 'z' else 400)
    result = project.update(); values = dict(zip(result.names, result.x))
    assert not result.contradictions and values['photos_s'] == pytest.approx(SCALE, rel=0.005)
    shapes = {s['id']: s for s in components.geometry(project, result)}
    lengths = sorted(round(shapes[c]['size'][0], 2) for c in created)
    assert lengths == pytest.approx([4.0, 4.0, 5.0, 5.0], abs=0.02)
    assert shapes[created[0]]['size'][1] == pytest.approx(2.5, abs=0.02)
    walls.create_walls(ctx, 'photos', room_walls[:3] + [listed[('x', round(0.6 * scale, 1))]['id']])
    assert walls.SCALE_COTE not in [row['id'] for row in read_cotes(project)] and len(components.load(project)) == 4
    cotes_add(ctx, 'm1', next(iter(components.load(project).values()))['params'][0], 100); project.update()
    walls.create_walls(ctx, 'photos', room_walls)
    assert len(components.load(project)) == 4 and 'm1' not in [row['id'] for row in read_cotes(project)] and project.update().converged
    rpc.call(ctx, 'plan.trace', {'points': [[0, 0], [4, 0], [4, 5], [0, 5]]}); walls.create_walls(ctx, 'photos', room_walls)
    assert sorted(components.load(project)) == sorted(created) and project.db.execute('select count(*) from photos_assoc').fetchone()[0] > 4


def test_cloud_and_visible(setup):
    ctx, _ = setup; points = np.array(photos.cloud(ctx, limit=500)['points']).reshape(-1, 3)
    assert len(points) == 500 and np.ptp(points[:, 1]) == pytest.approx(2.5 * photos.CAMERA_HEIGHT_M / CAMERA_HEIGHT, rel=0.1)
    middle = points.mean(0); seen = photos.visible(ctx, 'photos', [middle.tolist()])
    assert seen and len(set(seen)) == len(seen) and all(name.startswith('img') for name in seen)


def test_guide_saved_in_project(tmp_path):
    ctx = Ctx(); rpc.call(ctx, 'project.create', {'path': str(tmp_path / 'salon'), 'name': 'Salon'})
    rpc.call(ctx, 'project.guide', {'guide': {'type': 'piece', 'step': 4, 'done': [1, 2, 3]}})
    info = rpc.call(ctx, 'project.info', {})
    assert info['name'] == 'Salon' and info['created'] and info['guide'] == {'type': 'piece', 'step': 4, 'done': [1, 2, 3]}
    assert {'photos.propose_walls', 'photos.create_walls', 'photos.cloud', 'photos.visible', 'project.guide'} <= set(rpc.METHODS)


def test_dense_cloud_with_colors(setup):
    ctx, project = setup; cache = photos.source_cache(project, 'photos')[1]; points = photos.load_model(cache)['P'][:3000]
    vertices = np.zeros(len(points), [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertices['x'], vertices['y'], vertices['z'], vertices['nz'], vertices['red'] = *points.T, 1, 200
    os.makedirs(os.path.join(cache, 'dense'))
    with open(os.path.join(cache, 'dense', 'fused.ply'), 'wb') as f:
        f.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n'.encode())
        f.write(''.join(f'property {kind} {name}\n' for name, kind in [*((n, 'float') for n in ('x', 'y', 'z', 'nx', 'ny', 'nz')), *((n, 'uchar') for n in ('red', 'green', 'blue'))]).encode() + b'end_header\n')
        f.write(vertices.tobytes())
    found = photos.cloud(ctx, limit=1000)
    assert found['dense'] and len(found['points']) == len(found['colors']) <= 3000 and set(found['colors'][::3]) == {200}
    assert photos.cloud(ctx, limit=1000)['points'] == found['points'] and os.path.exists(os.path.join(cache, 'cloud.npz'))


def test_fit_photos_on_traced_room(setup):
    from plans3d import plan
    ctx, project = setup; photos.axes(ctx, 'photos', '+x+y+z'); plan.rpc_trace(ctx, [[0, 0], [4, 0], [4, 5], [0, 5]], True, [4.0, 5.0, None, None])
    fitted = walls.fit_walls(ctx)
    assert fitted['scale_known'] and len(fitted['pairs']) == 3 and len({wall for _, wall in fitted['pairs']}) == 3
    result = project.update(); values = dict(zip(result.names, result.x))
    assert not result.contradictions and values['photos_s'] == pytest.approx(SCALE, rel=0.01) and np.isfinite(result.sigma[result.names.index('photos_s')])
    assert [values[name] for name in ('mur2_x', 'mur3_z', 'mur4_x', 'mur1_z')] == pytest.approx([4, 5, 0, 0], abs=0.02)
    first = photos.cameras(ctx)['cameras'][0]['center'][0::2]; mirrored = [4 - SPOTS[0][0], 5 - SPOTS[0][1]]
    assert first == pytest.approx(list(SPOTS[0]), abs=0.05) or first == pytest.approx(mirrored, abs=0.05)
    walls.fit_walls(ctx, turn=2); project.update(); turned = photos.cameras(ctx)['cameras'][0]['center'][0::2]
    assert turned == pytest.approx(mirrored if first == pytest.approx(list(SPOTS[0]), abs=0.05) else list(SPOTS[0]), abs=0.05)


def test_fit_photos_on_slanted_walls(setup):
    from plans3d import plan
    ctx, project = setup; photos.axes(ctx, 'photos', '+x+y+z'); angle = np.radians(30)
    turn = lambda x, z: [np.cos(angle) * x - np.sin(angle) * z + 10, np.sin(angle) * x + np.cos(angle) * z - 3]
    plan.rpc_trace(ctx, [turn(0, 0), turn(4, 0), turn(4, 5), turn(0, 5)], True, [4.0, 5.0, None, None])
    fitted = walls.fit_walls(ctx); result = project.update(); values = dict(zip(result.names, result.x))
    assert fitted['scale_known'] and len(fitted['pairs']) == 3 and not result.contradictions and values['photos_s'] == pytest.approx(SCALE, rel=0.01)
    assert np.isfinite(result.sigma[[result.names.index(n) for n in ('photos_s', 'photos_yaw', 'photos_tx', 'photos_tz')]]).all()
    center = photos.cameras(ctx)['cameras'][0]['center'][0::2]
    assert any(center == pytest.approx(turn(*spot), abs=0.05) for spot in (SPOTS[0], (4 - SPOTS[0][0], 5 - SPOTS[0][1])))


def test_fit_keeps_cameras_inside(setup):
    from plans3d import plan
    ctx, project = setup; plan.rpc_trace(ctx, [[0, 0], [4, 0], [4, 5], [0, 5]], True, [4.0, 5.0, None, None])
    traced = walls.traced_walls(project, 'photos'); spots = np.array(SPOTS, float)
    assert walls.inside_share(spots, traced) == 1.0 and walls.inside_share(spots + [10, 0], traced) == 0.0
    walls.fit_walls(ctx); project.update()
    centers = np.array([c['center'] for c in photos.cameras(ctx)['cameras']])[:, [0, 2]]
    assert walls.inside_share(centers, traced) == 1.0 and centers[:, 0].min() > 0 and centers[:, 1].max() < 5


def test_holes_classified():
    rng = np.random.default_rng(5); points = rng.uniform(0, 1, (12000, 2)) * [5.0, 2.5]
    door = (points[:, 0] > 1.0) & (points[:, 0] < 1.9) & (points[:, 1] < 2.05)
    window = (points[:, 0] > 2.6) & (points[:, 0] < 3.6) & (points[:, 1] > 1.0) & (points[:, 1] < 2.2)
    found = sorted(walls.holes(points[~door & ~window], 5.0, 2.5), key=lambda h: h['along'])
    assert [h['type'] for h in found] == ['porte', 'fenetre']
    assert found[0]['width'] == pytest.approx(0.9, abs=0.15) and found[0]['height'] == pytest.approx(2.05, abs=0.15)
    assert found[1]['sill'] == pytest.approx(1.0, abs=0.15) and found[1]['width'] == pytest.approx(1.0, abs=0.15)


def test_snap_rough_walls(setup):
    from plans3d import plan
    ctx, project = setup; listed = walls.propose_walls(ctx)['walls']
    xs = sorted({round(w['position'], 3) for w in listed if w['axis'] == 'x' and w['sure']}); zs = sorted({round(w['position'], 3) for w in listed if w['axis'] == 'z' and w['sure']})
    created = plan.rpc_trace(ctx, [[xs[0] + 0.2, zs[0] - 0.15], [xs[-1] - 0.1, zs[0] - 0.15], [xs[-1] - 0.1, zs[0] + 3.0], [xs[0] + 0.2, zs[0] + 3.0]], True, prior=0.3)['created']
    paired = walls.snap_walls(ctx, 'photos', created)['pairs']
    result = project.update(); values = dict(zip(result.names, result.x))
    assert len(paired) == 3 and not result.contradictions and sorted(round(p['moved_cm']) for p in paired) == [10, 15, 20]
    assert [values['mur4_x'], values['mur2_x'], values['mur1_z']] == pytest.approx([xs[0], xs[-1], zs[0]], abs=0.03)

def test_inconsistent_clicks_keep_scale(setup):
    ctx, project = setup; listed = by_position(walls.propose_walls(ctx)); scale = photos.CAMERA_HEIGHT_M / CAMERA_HEIGHT
    room_walls = [listed[key]['id'] for key in (('x', 0.0), ('x', round(4 * scale, 1)), ('z', 0.0), ('z', round(5 * scale, 1)))]
    created = walls.create_walls(ctx, 'photos', room_walls)['created']; before = project.update().value('photos_s')[0]
    for i, (u, v) in enumerate([(100, 200), (700, 900), (300, 650), (1100, 100)]):
        photos.click(ctx, 'photos', f'img{i % 2}.jpg', u, v, f'{created[0]}:a_sol'); photos.click(ctx, 'photos', f'img{i % 2}.jpg', v, u, f'{created[1]}:b')
    assert project.update().value('photos_s')[0] > 0.5 * before > 0
