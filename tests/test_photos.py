import json, os, struct, sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from plans3d import Project
from plans3d import photos
from plans3d.rpc import Ctx

SCALE, ROOM, CAMERA_HEIGHT = 0.5, (4.0, 2.5, 5.0), 1.6
WORLD_ROTATION = Rotation.from_euler('xyz', [25, -40, 70], degrees=True).as_matrix()
WORLD_SHIFT = np.array([3.0, -2.0, 7.0])
CAMERA_SPOTS = [((1, 1), 1), ((3, 1), 1), ((3, 4), 1), ((1, 4), 1), ((2, 0.5), 6), ((2, 4.5), 1)]
ANCHORS = {'boite:coin': {'x': 'px', 'y': 'py', 'z': 'pz'}, 'mur:a': {'x': 'ex', 'z': 'ez'}, 'mur:face': {'x': 'fx'}}


def to_photo_units(points_m): return np.asarray(points_m, float) @ WORLD_ROTATION.T / SCALE + WORLD_SHIFT


def room_points(rng, per_face=2500, noise=0.003):
    faces = []
    for axis in range(3):
        for level in (0.0, ROOM[axis]):
            face = rng.uniform(0, 1, (per_face, 3)) * ROOM; face[:, axis] = level + rng.normal(0, noise, per_face); faces.append(face)
    platform = rng.uniform(0, 1, (4000, 3)) * [2.0, 0, 2.5] + [0.5, 1.0, 0.5]; platform[:, 1] += rng.normal(0, noise, 4000)
    layer = rng.uniform(0, 1, (1500, 3)) * [0, 2.5, 2.5]; layer[:, 0] = ROOM[0] - 0.02 + rng.normal(0, noise, 1500)
    faces += [platform, layer, rng.uniform(0, 1, (1500, 3)) * ROOM]
    return to_photo_units(np.vstack(faces))


def camera_pose(center, target, orientation):
    up = WORLD_ROTATION @ [0, 1, 0]; forward = photos.unit(to_photo_units(target) - to_photo_units(center)); image_up = photos.unit(up - (up @ forward) * forward)
    right, down = (np.cross(-image_up, forward), -image_up) if orientation == 1 else (-image_up, np.cross(forward, -image_up))
    rotation = np.array([right, down, forward]); return rotation, -rotation @ to_photo_units(center)


def synthetic_model(rng):
    poses = [camera_pose([x, CAMERA_HEIGHT, z], [2.0, 1.2, 2.5], orientation) for (x, z), orientation in CAMERA_SPOTS]; count = len(poses)
    return {'names': np.array([f'img{i}.jpg' for i in range(count)]), 'q': np.array([np.roll(Rotation.from_matrix(r).as_quat(), 1) for r, _ in poses]),
            't': np.array([t for _, t in poses]), 'K': np.tile([1000.0, 1000.0, 800.0, 600.0, 0, 0, 0, 0], (count, 1)),
            'wh': np.tile([1600, 1200], (count, 1)), 'orientation': np.array([o for _, o in CAMERA_SPOTS]), 'P': room_points(rng)}


def display_pixel(model, i, point_m):
    camera = photos.rotation_matrices(model['q'][i:i + 1])[0] @ to_photo_units(point_m) + model['t'][i]
    u, v = 1000 * camera[0] / camera[2] + 800, 1000 * camera[1] / camera[2] + 600
    return photos.raw_to_display(int(model['orientation'][i]), 1600, 1200, u, v)


@pytest.fixture
def setup(tmp_path):
    folder = str(tmp_path); project = Project(folder); ctx = Ctx(project=project); os.makedirs(os.path.join(folder, 'sources'))
    with open(os.path.join(folder, 'sources', 'cotes.csv'), 'w', encoding='utf-8') as f: f.write('id;expr;value;sigma;unit;status;label\nw;w;500;1;cm;mesuré;fond\n')
    project.add_param('w', 4.8, 'z', 'mur du fond'); project.add_source('cotes', os.path.join(folder, 'sources', 'cotes.csv'), 'cotes_csv')
    photos.add(ctx, [os.path.join(folder, name) for name in ('a.jpg', 'b.jpg')]); cache = photos.source_cache(project, 'photos')[1]; os.makedirs(cache)
    model = synthetic_model(np.random.default_rng(2)); np.savez(os.path.join(cache, 'model.npz'), **model)
    with open(os.path.join(cache, 'meta.json'), 'w') as f: json.dump({'root': folder, 'registered': 6, 'model': '0'}, f)
    frame = np.array(photos.planes(ctx)['R']); mapping = ''
    for row in frame:
        direction = WORLD_ROTATION.T @ row; axis = int(np.abs(direction).argmax()); mapping += ('+' if direction[axis] > 0 else '-') + 'xyz'[axis]
    photos.axes(ctx, 'photos', mapping)
    walls = {axis: sorted((p for p in photos.planes(ctx)['planes'] if p.get('proj_axis') == axis and p['n'] > 1500), key=lambda p: p['proj_offset']) for axis in 'xyz'}
    return ctx, project, model, walls


def test_ransac_two_planes():
    rng = np.random.default_rng(0); first = rng.uniform(-1, 1, (3000, 3)); first[:, 2] = rng.normal(0, 0.002, 3000)
    second = rng.uniform(-1, 1, (2000, 3)); second[:, 0] = 0.7 + rng.normal(0, 0.002, 2000); points = np.vstack([first, second])
    planes = photos.ransac_planes(points, photos.point_normals(points), 0.01, 100)
    assert len(planes) == 2 and abs(abs(planes[0][0][2]) - 1) < 1e-3 and abs(abs(planes[1][0][0]) - 1) < 1e-3
    assert len(planes[0][2]) > 2900 and abs(abs(planes[1][1]) - 0.7) < 1e-3


def test_frame_and_planes():
    model = synthetic_model(np.random.default_rng(1))
    result = photos.find_frame_and_planes(model['P'], photos.camera_centers(model), photos.image_ups(model))
    assert np.degrees(np.arccos(np.array(result['R'])[1] @ WORLD_ROTATION @ [0, 1, 0])) < 0.3
    assert result['s_est'] == pytest.approx(SCALE * photos.CAMERA_HEIGHT_M / CAMERA_HEIGHT, rel=0.01)
    offsets = {axis: sorted(p['offset'] for p in result['planes'] if p['axis'] == axis and p['n'] > 1500) for axis in 'xyz'}
    assert offsets['y'] == pytest.approx([0, 1.0 / SCALE, ROOM[1] / SCALE], abs=0.01)
    assert len(offsets['x']) == len(offsets['z']) == 2 and max(p['n'] for p in result['planes'] if p['axis'] in 'xz' and p['axis']) > 3900
    assert sorted(v[1] - v[0] for v in (offsets['x'], offsets['z'])) == pytest.approx([ROOM[0] / SCALE, ROOM[2] / SCALE], abs=0.02)


def test_expand_folder(tmp_path):
    for name in ('b.PNG', 'a.jpg', 'notes.txt'): (tmp_path / name).write_bytes(b'')
    assert photos.expand_paths([str(tmp_path)]) == [str(tmp_path / 'a.jpg'), str(tmp_path / 'b.PNG')]


def test_associate_scale(setup):
    ctx, project, _, walls = setup; near, far = walls['z']
    with pytest.raises(ValueError): photos.associate(ctx, 'photos', near['id'], 'inconnu + 1')
    with pytest.raises(ValueError): photos.axes(ctx, 'photos', '+x+x+z')
    photos.associate(ctx, 'photos', near['id'], '0'); photos.associate(ctx, 'photos', far['id'], 'w')
    result = project.update(); assert 'photos' in result.timing['reextracted']
    assert result.value('photos_s')[0] == pytest.approx(SCALE, rel=1e-3) and result.value('w')[0] == pytest.approx(5.0, abs=1e-4)
    assert len([o for o in result.obs if o.source == 'photos' and o.status == 'photo']) == 2 and not result.contradictions
    photos.invalidate(project, 'photos'); sigmas = {o.sigma for o in project.update().obs if o.source == 'photos' and o.status == 'photo'}
    assert len(sigmas) == 1 and sigmas.pop() == pytest.approx(photos.PLANE_SIGMA_M / result.value('photos_s')[0])
    with pytest.raises(ValueError): photos.associate(ctx, 'photos', far['id'], 'w', sigma=0)
    photos.associate(ctx, 'photos', far['id'], 'w', sigma=0.05); scale = result.value('photos_s')[0]
    assert sorted(o.sigma * scale for o in project.update().obs if o.source == 'photos' and o.status == 'photo') == pytest.approx([photos.PLANE_SIGMA_M, 0.05], rel=1e-3)
    assert photos.cameras(ctx)['cameras'][0]['center'][1:] == pytest.approx([CAMERA_HEIGHT, 1.0], abs=0.01)
    photos.unassociate(ctx, 'photos', far['id']); result = project.update()
    assert [o.id for o in result.obs if o.source == 'photos' and o.status == 'photo'] == [f"photos:{near['id']}"]


def test_clicks_point_edge_face(setup, monkeypatch):
    ctx, project, model, walls = setup
    monkeypatch.setattr(photos, 'anchor_coordinates', lambda project, anchor: ANCHORS[anchor])
    for plane, expr in ((walls['x'][0], '0'), (walls['y'][0], '0'), (walls['z'][0], '0'), (walls['z'][1], 'w')): photos.associate(ctx, 'photos', plane['id'], expr)
    for name, init in (('px', 2.0), ('py', 1.2), ('pz', 2.5), ('ex', 2.5), ('ez', 2.5), ('fx', 3.0)): project.add_param(name, init, name[1], name)
    truth = {'px': 2.5, 'py': 1.0, 'pz': 3.0, 'ex': 3.0, 'ez': 2.0, 'fx': 3.5}
    for i, point, anchor in ((0, [2.5, 1.0, 3.0], 'boite:coin'), (4, [2.5, 1.0, 3.0], 'boite:coin'), (0, [3.0, 0.5, 2.0], 'mur:a'), (3, [3.0, 1.8, 2.0], 'mur:a'),
                             (0, [3.5, 1.1, 2.6], 'mur:face'), (3, [3.5, 1.1, 2.6], 'mur:face')):
        photos.click(ctx, 'photos', f'img{i}.jpg', *display_pixel(model, i, point), anchor)
    with pytest.raises(ValueError): photos.click(ctx, 'photos', 'img0.jpg', 5000, 10, 'boite:coin')
    listed = photos.clicks(ctx)
    assert [c['kind'] for c in listed] == ['point', 'point', 'arête', 'arête', 'face', 'face'] and listed[-1]['photos_for_anchor'] == 2
    result = project.update(); assert not result.contradictions
    assert len([o for o in result.obs if ':clic' in o.id or ':face:' in o.id]) == 7
    for name, value in truth.items(): assert result.value(name)[0] == pytest.approx(value, abs=0.005), name
    photos.unclick(ctx, listed[-1]['id']); result = project.update()
    assert not [o for o in result.obs if ':face:' in o.id]


def test_project_points(setup):
    ctx, project, model, _ = setup; frame = photos.photo_state(project, 'photos', compute=False)[1]
    linear_map, shift, _ = photos.project_transform(project, 'photos', frame)
    points = model['P'][:50]; camera = points @ photos.rotation_matrices(model['q'][4:5])[0].T + model['t'][4]
    u, v = 1000 * camera[:, 0] / camera[:, 2] + 800, 1000 * camera[:, 1] / camera[:, 2] + 600
    result = photos.project_points(ctx, 'photos', 'img4.jpg', (points @ linear_map.T + shift).tolist())
    visible = camera[:, 2] > 0; uv = np.array([p for p in result['uv'] if p is not None])
    assert result['size'] == [1200, 1600] and [p is None for p in result['uv']] == list(~visible)
    assert np.allclose(uv, np.stack([1200 - v, u], 1)[visible], atol=1e-6)


def test_rpc_dispatch(setup):
    rpc = pytest.importorskip('plans3d.rpc'); _, project, _, walls = setup; ctx = rpc.Ctx(); rpc.open_project(ctx, project.folder)
    assert {'photos.add', 'photos.extract', 'photos.densify', 'photos.click', 'photos.clicks', 'photos.unclick'} <= set(rpc.METHODS)
    assert 'photos.extract' in rpc.LONG_METHODS
    plane_id = walls['z'][1]['id']
    assert rpc.call(ctx, 'photos.associate', {'source': 'photos', 'plane_id': plane_id, 'expr': 'w'})['expr'] == 'w'
    assert next(p for p in rpc.call(ctx, 'photos.planes', {})['planes'] if p['id'] == plane_id)['expr'] == 'w'


def test_component_anchors(tmp_path):
    components = pytest.importorskip('plans3d.components'); project = Project(str(tmp_path))
    components.add(project, 'boite', 'b1'); components.add(project, 'mur', 'm1', {'axe': 'x'})
    assert set(photos.anchor_coordinates(project, 'b1:centre')) == set('xyz') and set(photos.anchor_coordinates(project, 'm1:a')) == {'x', 'z'}
    assert len(photos.anchor_coordinates(project, 'm1:face')) == 1
    with pytest.raises(ValueError): photos.anchor_coordinates(project, 'inconnu:a')


def test_plane_ids_survive_merge_and_new_frame(tmp_path):
    model = synthetic_model(np.random.default_rng(3)); np.savez(tmp_path / 'model.npz', **model); first = photos.frame_and_planes(str(tmp_path))
    walls = [p for p in first['planes'] if p['axis'] == 'x' and p['n'] > 1000]
    split = [{**walls[0], 'id': 'p90', 'n': walls[0]['n'] // 2}, {**walls[0], 'id': 'p91', 'offset': walls[0]['offset'] + 0.03, 'n': walls[0]['n'] // 2}]
    shifted = {**first, 'cloud': 'ancien', 't': (np.array(first['t']) + [0.0, 0.5, 0.0]).tolist(),
               'planes': split + [{**p, 'bbox': (np.array(p['bbox']) + [0.0, 0.5, 0.0]).tolist(), 'offset': p['offset'] + (0.5 if p['axis'] == 'y' else 0)}
                                  for p in first['planes'] if p['id'] != walls[0]['id']]}
    for plane in split: plane['bbox'] = (np.array(plane['bbox']) + [0.0, 0.5, 0.0]).tolist()
    with open(tmp_path / 'frame.json', 'w', encoding='utf-8') as f: json.dump(shifted, f)
    second = photos.frame_and_planes(str(tmp_path)); merged = next(p for p in second['planes'] if p['id'] in ('p90', 'p91'))
    assert {merged['id'], *merged['aliases']} == {'p90', 'p91'}
    assert {p['id'] for p in second['planes'] if p['n'] > 1500} == {p['id'] for p in first['planes'] if p['n'] > 1500} - {walls[0]['id']} | {merged['id']}


def test_axes_kept_on_recompute(tmp_path):
    model = synthetic_model(np.random.default_rng(4)); np.savez(tmp_path / 'model.npz', **model); first = photos.frame_and_planes(str(tmp_path))
    turned = np.array(first['R'])[[2, 1, 0]] * [[-1], [1], [1]]
    with open(tmp_path / 'frame.json', 'w', encoding='utf-8') as f: json.dump({**first, 'cloud': 'ancien', 'R': turned.tolist()}, f)
    assert np.allclose(photos.frame_and_planes(str(tmp_path))['R'], turned, atol=1e-6)


def test_axis_offset_and_old_caches(tmp_path):
    rng = np.random.default_rng(5); wall = rng.uniform(0, 1, (3000, 3)) * [0, 2.5, 3.0]; wall[:, 0] = 1.0 + 0.05 * wall[:, 1] + rng.normal(0, 0.002, 3000)
    plane = photos.plane_summary(0, np.array([1.0, 0, 0]), np.arange(3000), wall)
    assert plane['offset'] == pytest.approx(wall[:, 0].mean(), abs=1e-9) and plane['tilt_deg'] == pytest.approx(np.degrees(np.arctan(0.05)), abs=0.1)
    model = synthetic_model(rng); model['ori'] = model.pop('orientation'); np.savez(tmp_path / 'model.npz', **model)
    assert list(photos.load_model(str(tmp_path))['orientation']) == [o for _, o in CAMERA_SPOTS]
    project = Project(str(tmp_path)); project.db.execute('create table photos_assoc(source text, plane_id text, expr text, primary key(source, plane_id))')
    photos.ensure_schema(project); assert 'sigma' in [row[1] for row in project.db.execute('pragma table_info(photos_assoc)')]


def test_link_images_without_spaces(tmp_path):
    folder = tmp_path / 'a b' / '28-08-26 part 2'; folder.mkdir(parents=True)
    for name in ('photo 1.JPG', 'photo 2.jpg'): (folder / name).write_bytes(b'x')
    image_dir, names = photos.link_images([str(folder / 'photo 1.JPG'), str(folder / 'photo 2.jpg')], str(tmp_path / 'cache dir' / 'images'))
    assert names == ['00000.jpg', '00001.jpg'] and all(os.path.exists(os.path.join(image_dir, name)) for name in names)


def test_yaw_from_clicks(setup, monkeypatch):
    ctx, project, model, _ = setup; yaw = np.radians(2.6)
    turn = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    known = {'k1:coin': [2.5, 1.0, 3.0], 'k2:coin': [3.0, 0.5, 2.0], 'k3:coin': [1.5, 1.8, 3.5]}
    anchors = {name: {axis: number for axis, number in zip('xyz', map(photos.number, turn @ point))} for name, point in known.items()}
    anchors['q:coin'] = {'x': 'qx', 'y': 'qy', 'z': 'qz'}; monkeypatch.setattr(photos, 'anchor_coordinates', lambda project, anchor: anchors[anchor])
    for name in ('qx', 'qy', 'qz'): project.add_param(name, 2.0, name[1], name)
    monkeypatch.setattr(photos, 'CLICK_SIGMA_M', 0.001); project.db.execute("update params set init=0.45 where name='photos_s'"); project.db.commit()
    room_q = [3.5, 1.1, 2.6]; known['q:coin'] = room_q
    for anchor, point in known.items():
        for i in (0, 3): photos.click(ctx, 'photos', f'img{i}.jpg', *display_pixel(model, i, point), anchor)
    result = project.update()
    assert np.degrees(result.value('photos_yaw')[0]) == pytest.approx(2.6, abs=0.1) and result.value('photos_s')[0] == pytest.approx(SCALE, rel=1e-3)
    assert [result.value(name)[0] for name in ('qx', 'qy', 'qz')] == pytest.approx(turn @ room_q, abs=0.003)
    project.db.execute("delete from params where name='photos_yaw'"); project.db.commit(); photos.invalidate(project, 'photos')
    assert not any('yaw' in o.expr for o in project.update().obs)


def test_extract_twice_reuses_image_links(tmp_path, monkeypatch):
    from PIL import Image
    folder = tmp_path / 'a b'; folder.mkdir()
    for i in range(3): Image.new('RGB', (40, 30), (i * 60, 0, 0)).save(folder / f'photo {i}.jpg')
    project = Project(str(tmp_path / 'projet')); ctx = Ctx(project=project); photos.add(ctx, [str(folder)])
    def failing_colmap(arguments, *rest, **options): raise ValueError('COLMAP simulé')
    monkeypatch.setattr(photos, 'run_colmap', failing_colmap)
    for _ in range(2):
        with pytest.raises(ValueError, match='simulé'): photos.extract(ctx)
        with pytest.raises(ValueError, match='simulé'): project.update()
    image_dir = os.path.join(photos.source_cache(project, 'photos')[1], 'images'); (folder / 'photo 1.jpg').unlink()
    photos.link_images([str(folder / 'photo 2.jpg'), str(folder / 'photo 0.jpg')], image_dir)
    assert sorted(os.listdir(image_dir)) == ['00000.jpg', '00001.jpg'] and os.path.samefile(os.path.join(image_dir, '00000.jpg'), folder / 'photo 2.jpg')


def test_source_status_in_sources_list(tmp_path, monkeypatch):
    from PIL import Image
    rpc = pytest.importorskip('plans3d.rpc'); folder = tmp_path / 'photos'; folder.mkdir()
    for i in range(2): Image.new('RGB', (40, 30)).save(folder / f'{i}.jpg')
    project = Project(str(tmp_path / 'projet')); ctx = rpc.Ctx(); rpc.open_project(ctx, project.folder); photos.add(ctx, [str(folder)])
    status = lambda: next(s for s in rpc.call(ctx, 'sources.list', {}) if s['id'] == 'photos')
    assert {k: status()[k] for k in ('photos', 'registered', 'state')} == {'photos': 2, 'registered': None, 'state': 'à calculer'}
    seen = []
    def failing_colmap(arguments, *rest, **options): seen.append(status()['state']); raise ValueError('COLMAP simulé')
    monkeypatch.setattr(photos, 'run_colmap', failing_colmap)
    with pytest.raises(ValueError): photos.extract(ctx)
    assert seen == ['en cours'] and status()['state'] == 'échec' and status()['error'] == 'COLMAP simulé'
    cache = photos.source_cache(ctx.project, 'photos')[1]
    with open(os.path.join(cache, 'meta.json'), 'w') as f: json.dump({'registered': 2}, f)
    assert {k: status()[k] for k in ('registered', 'state')} == {'registered': 2, 'state': 'calculé'} and 'error' not in status()


def write_snapshot(folder, model, names):
    os.makedirs(folder)
    with open(os.path.join(folder, 'images.bin'), 'wb') as f:
        f.write(struct.pack('<Q', len(names)))
        for i, name in enumerate(names): f.write(struct.pack('<I7dI', i + 1, *model['q'][i], *model['t'][i], 1) + name.encode() + b'\0' + struct.pack('<Q', 2) + bytes(48))
    with open(os.path.join(folder, 'points3D.bin'), 'wb') as f:
        f.write(struct.pack('<Q', len(model['P'])))
        for i, p in enumerate(model['P']): f.write(struct.pack('<Q3d3BdQ', i, *p, 200, 100, 50, 0.5, 2) + bytes(16))


def test_live_snapshot(tmp_path):
    model = synthetic_model(np.random.default_rng(0)); names = [f'{i:05d}.jpg' for i in range(len(model['q']))]; write_snapshot(str(tmp_path / 's'), model, names)
    read = photos.read_snapshot(str(tmp_path / 's'), limit=2000)
    assert read[0] == names and len(read[3]) <= 2000 and np.allclose(read[1], model['q']) and set(read[4].ravel()) == {200, 100, 50}
    view = photos.live_view(*read, {name: i for i, name in enumerate(names)}, list(model['orientation']))
    centers = np.array([c['center'] for c in view['cameras']]); points = np.array(view['points']).reshape(-1, 3)
    assert [c['index'] for c in view['cameras']] == list(range(len(names))) and len(view['colors']) == points.size
    assert np.ptp(centers[:, 1]) < 0.05 and np.median(centers[:, 1]) == pytest.approx(photos.CAMERA_HEIGHT_M)
    assert np.percentile(points[:, 1], 2) == pytest.approx(0, abs=0.05) and np.ptp(points[:, 1]) < 3


def test_stop_and_thumbnail(tmp_path, monkeypatch):
    from PIL import Image
    folder = tmp_path / 'photos'; folder.mkdir(); Image.new('RGB', (400, 300), (10, 200, 30)).save(folder / 'a.jpg'); (folder / 'b.jpg').write_bytes(b'rien')
    assert photos.thumbnail(str(folder / 'a.jpg')).startswith('data:image/jpeg;base64,') and photos.thumbnail(str(folder / 'b.jpg')) == ''
    project = Project(str(tmp_path / 'projet')); ctx = Ctx(project=project); photos.add(ctx, [str(folder)]); cache = photos.source_cache(project, 'photos')[1]; os.makedirs(cache)
    monkeypatch.setattr(photos, 'colmap_executable', lambda: sys.executable); seen = []
    def watch(line): seen.append(line); photos.stop(ctx)
    with pytest.raises(photos.Stopped): photos.run_colmap(['-c', 'import time; print("go", flush=True); time.sleep(60)'], os.path.join(cache, 'x.log'), photos.no_progress, 0, 1, '', watch=watch)
    assert seen == ['go\n'] and not photos.PROCESSES
    def stopped(arguments, *rest, **options): raise photos.Stopped('calcul arrêté')
    monkeypatch.setattr(photos, 'run_colmap', stopped)
    assert photos.extract(ctx) == {'stopped': True} and photos.source_status(project, 'photos', photos.source_cache(project, 'photos')[0])['state'] == 'à calculer'


def test_mapper_watch_counts_registered_frames(tmp_path):
    import sqlite3
    database = str(tmp_path / 'db.db'); connection = sqlite3.connect(database); connection.execute('create table images(image_id integer, name text)')
    connection.executemany('insert into images values (?,?)', [(i + 1, f'{i:05d}.jpg') for i in range(6)]); connection.commit(); connection.close()
    (tmp_path / 'snap').mkdir(); events = []
    watch = photos.mapper_watch(database, str(tmp_path / 'snap'), {f'{i:05d}.jpg': i for i in range(6)}, [1] * 6, lambda **event: events.append(event.get('placed')))
    for line in ['Registering initial image pair #3 and #1', 'Registering image #2 (num_reg_frames=2)', 'Registering image #5 (num_reg_frames=3)',
                 'Registering image #6 (num_reg_frames=3)', 'Registering initial image pair #4 and #5']: watch(line)
    assert events == [[2, 0], [2, 0, 1], [2, 0, 1, 4], [2, 0, 1, 5]]
