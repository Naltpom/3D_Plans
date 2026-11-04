import datetime, json, os
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from plans3d import Project, groupes, photos
from plans3d.rpc import Ctx

ROOM = np.array([4.0, 2.5, 5.0])
BOXES = [np.array([0.7, 0.4, 2.5]), np.array([3.3, 0.4, 1.2])]
HALF = 0.35


def surface(rng, low, high, count):
    points = rng.uniform(low, high, (count, 3)); axis = rng.integers(0, 3, count)
    points[np.arange(count), axis] = np.where(rng.random(count) < 0.5, low[axis], high[axis])
    return points


def pose(center, yaw, pitch=-0.35):
    forward = np.array([np.cos(pitch) * np.cos(yaw), np.sin(pitch), np.cos(pitch) * np.sin(yaw)]); right = photos.unit(np.cross(forward, [0, 1, 0]))
    rotation = np.array([right, np.cross(forward, right), forward]); return rotation, -rotation @ center


def hidden_by(box, centers, points):
    vectors = points - centers; low, high = box - HALF + 0.02, box + HALF - 0.02
    with np.errstate(divide='ignore', invalid='ignore'):
        a, b = (low - centers) / vectors, (high - centers) / vectors
    near, far = np.nanmax(np.minimum(a, b), 1), np.nanmin(np.maximum(a, b), 1)
    return (near <= far) & (far > 0) & (near < 0.98)


def scene(folder):
    rng = np.random.default_rng(4); room = surface(rng, np.zeros(3), ROOM, 30000)
    boxes = [surface(rng, box - HALF, box + HALF, 1500) for box in BOXES]; points = np.vstack([room, *boxes])
    labels = np.repeat([0, 1, 2], [len(room), 1500, 1500]); centers, rotations, names = [], [], []
    for g, start in enumerate([datetime.datetime(2026, 10, 30, 11, 0), datetime.datetime(2026, 10, 30, 19, 0)]):
        for k in range(12):
            yaw = 2 * np.pi * k / 12 + 0.2 * g; center = np.array([2.0, 1.4, 2.5]) + 0.6 * np.array([np.cos(yaw), 0, np.sin(yaw)])
            rotation, _ = pose(center, yaw); centers.append(center); rotations.append(rotation)
            names.append((start + datetime.timedelta(seconds=20 * k)).strftime('%Y%m%d_%H%M%S.jpg'))
    track = [[] for _ in points]
    for i, (center, rotation) in enumerate(zip(centers, rotations)):
        g = i // 12; local = (points - center) @ rotation.T; inside = (local[:, 2] > 0.1) & (np.abs(local[:, :2] / local[:, 2:]).max(1) < 0.7)
        seen = inside & ((labels == 0) | (labels == g + 1)) & ~hidden_by(BOXES[g], np.tile(center, (len(points), 1)), points)
        for p in np.flatnonzero(seen): track[p].append(i)
    kept = [p for p, t in enumerate(track) if len(t) >= 2]; centers, rotations = np.array(centers), np.array(rotations)
    model = {'names': np.array(names), 'q': np.array([np.roll(Rotation.from_matrix(r).as_quat(), 1) for r in rotations]),
             't': -np.einsum('nij,nj->ni', rotations, centers), 'K': np.tile([800.0, 800.0, 640, 480, 0, 0, 0, 0], (len(names), 1)),
             'wh': np.tile([1280, 960], (len(names), 1)), 'orientation': np.ones(len(names), int), 'P': points[kept], 'C': np.zeros((len(kept), 3), np.uint8),
             'Tptr': np.concatenate([[0], np.cumsum([len(track[p]) for p in kept])]), 'Timg': np.array([i for p in kept for i in track[p]])}
    for name in names: open(os.path.join(folder, name), 'wb').close()
    return model, labels[kept]


def setup(tmp_path):
    folder = str(tmp_path); project = Project(folder); ctx = Ctx(project=project); model, labels = scene(folder)
    photos.add(ctx, [os.path.join(folder, str(name)) for name in model['names']]); cache = photos.source_cache(project, 'photos')[1]; os.makedirs(cache)
    np.savez(os.path.join(cache, 'model.npz'), **model)
    with open(os.path.join(cache, 'meta.json'), 'w') as f: json.dump({'root': folder, 'registered': len(model['names']), 'model': '0'}, f)
    return ctx, project, model, labels


def test_capture_time_sources(tmp_path):
    named = os.path.join(tmp_path, '20261030_222345.jpg'); Image.new('RGB', (8, 8)).save(named)
    tagged = os.path.join(tmp_path, '20261030_090000.jpg'); exif = Image.Exif(); exif.get_ifd(0x8769)[0x9003] = '2026:10:31 07:15:00'
    Image.new('RGB', (8, 8)).save(tagged, exif=exif); plain = os.path.join(tmp_path, 'img.jpg'); Image.new('RGB', (8, 8)).save(plain); os.utime(plain, (1e9, 1e9))
    assert groupes.read_time(tagged) == datetime.datetime(2026, 10, 31, 7, 15)
    assert groupes.read_time(named) == datetime.datetime(2026, 10, 30, 22, 23, 45)
    assert groupes.read_time(plain) == datetime.datetime.fromtimestamp(1e9)
    minute = lambda m: datetime.datetime(2026, 1, 1) + datetime.timedelta(minutes=m)
    assert groupes.split([minute(m) for m in (0, 4, 30, 8, 34.5, 200)], 300) == [[0, 1, 3], [2, 4], [5]]


def test_sessions_rename_exclude(tmp_path):
    ctx, project, model, _ = setup(tmp_path); listed = groupes.groups(ctx)['groups']
    assert [(g['count'], g['start'][11:16], g['end'][11:16]) for g in listed] == [(12, '11:00', '11:03'), (12, '19:00', '19:03')]
    groupes.set_group(ctx, 'photos', listed[1]['id'], name='Soir'); assert groupes.groups(ctx)['groups'][1]['name'] == 'Soir'
    groupes.set_gap(ctx, 'photos', 600); assert len(groupes.groups(ctx)['groups']) == 1
    groupes.set_gap(ctx, 'photos', 5); cache = photos.source_cache(project, 'photos')[1]
    groupes.set_group(ctx, 'photos', listed[0]['id'], excluded=True)
    assert groupes.groups(ctx)['groups'][0]['excluded'] and photos.source_cache(project, 'photos')[1] != cache
    assert photos.active_photos(project, 'photos', photos.source_cache(project, 'photos')[0]) == groupes.sessions(project, 'photos')[1]['photos']
    groupes.set_group(ctx, 'photos', listed[0]['id'], excluded=False); assert photos.source_cache(project, 'photos')[1] == cache


def test_moved_box_is_mobile(tmp_path):
    ctx, project, model, labels = setup(tmp_path); result = groupes.mobility(project, 'photos'); mobile = result['point_mobile']
    seen = result['point_groups']; assert set(seen[labels == 1]) == {1} and set(seen[labels == 2]) == {2} and (seen[labels == 0] == 3).mean() > 0.5
    assert mobile[labels > 0].mean() > 0.8 and mobile[labels == 0].mean() < 0.03
    assert (groupes.fixed(project, 'photos', model['P']) == ~mobile).mean() > 0.97
    stats = groupes.mobility_rpc(ctx); assert [g['registered'] for g in stats['groups']] == [12, 12] and 0 < stats['mobile_share'] < 0.2
    colours = photos.cloud(ctx, color='mobilite')['colors']; assert tuple(colours[:3]) in (groupes.FIXED, groupes.MOBILE)
