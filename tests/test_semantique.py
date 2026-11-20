import os
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from plans3d import semantique as S

ADE_OF = {'mur': 0, 'sol': 3, 'plafond': 5, 'lit': 7, 'porte': 14, 'fenetre': 8}
L_ROOM = [(0, 0), (4, 0), (4, 2), (2.5, 2), (2.5, 3.2), (0, 3.2)]
HEIGHT, W, H = 2.5, 400, 300


def camera(center, yaw, pitch=-0.15):
    forward = np.array([np.cos(yaw) * np.cos(pitch), np.sin(pitch), np.sin(yaw) * np.cos(pitch)]); down = np.array([0, -1, 0.0])
    down = down - forward * (down @ forward); down /= np.linalg.norm(down); rotation = np.array([np.cross(down, forward), down, forward])
    return rotation, -rotation @ center


def hit(origin, direction, polygon, box, door):
    best, label = np.inf, 'inconnu'
    for level, name in ((0.0, 'sol'), (HEIGHT, 'plafond')):
        t = (level - origin[1]) / direction[1] if abs(direction[1]) > 1e-9 else -1
        if 0 < t < best: best, label = t, name
    for k in range(len(polygon)):
        a, b = np.array(polygon[k], float), np.array(polygon[(k + 1) % len(polygon)], float); axis = 0 if a[0] == b[0] else 2
        if abs(direction[axis]) < 1e-9: continue
        t = (a[axis // 2] - origin[axis]) / direction[axis]; p = origin + t * direction; run = 2 - axis; low, high = sorted((a[1 - axis // 2], b[1 - axis // 2]))
        if 0 < t < best and low <= p[run] <= high and 0 <= p[1] <= HEIGHT:
            best = t; label = 'porte' if k == door[0] and door[1] <= p[run] <= door[2] and p[1] <= 2.0 else 'mur'
    low, high = box
    with np.errstate(divide='ignore', invalid='ignore'):
        t0, t1 = (np.array(low) - origin) / direction, (np.array(high) - origin) / direction
    near, far = np.minimum(t0, t1).max(), np.maximum(t0, t1).min()
    if near < far and 0 < near < best: label = 'lit'
    return label


def scene(tmp_path):
    rng, names, q, t, maps = np.random.default_rng(1), [], [], [], []
    spots = [(1.0, 1.0), (3.0, 1.0), (1.2, 2.5), (2.0, 1.5), (0.8, 2.0)]; box, door = ((0.1, 0, 0.1), (1.5, 0.6, 0.9)), (1, 0.6, 1.4)
    fx = 260.0; cache = str(tmp_path); os.makedirs(S.folder(cache, 'cartes'))
    for k, (x, z) in enumerate(spots):
        for yaw in np.linspace(0, 2 * np.pi, 8, endpoint=False) + k * 0.3:
            rotation, translation = camera(np.array([x, 1.4, z]), yaw); name = f'p{len(names):03d}.jpg'; names.append(name)
            q.append(Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]); t.append(translation)
            rows, cols = 60, 80; grid = np.zeros((rows, cols), np.uint8)
            for r in range(rows):
                for c in range(cols):
                    d = rotation.T @ np.array([((c + 0.5) * W / cols - W / 2) / fx, ((r + 0.5) * H / rows - H / 2) / fx, 1.0])
                    grid[r, c] = ADE_OF.get(hit(np.array([x, 1.4, z]), d / np.linalg.norm(d), L_ROOM, box, door), 41)
            Image.fromarray(grid).save(S.map_path(cache, name))
    wall = np.array([[rng.uniform(0, 4), rng.uniform(0.3, 2.2), 0.0] for _ in range(200)])
    model = {'names': np.array(names), 'q': np.array(q), 't': np.array(t), 'K': np.tile([fx, fx, W / 2, H / 2, 0, 0, 0, 0], (len(names), 1)),
             'wh': np.tile([W, H], (len(names), 1)), 'orientation': np.ones(len(names), int), 'P': wall}
    frame = {'R': np.eye(3).tolist(), 't': [0.0, 0.0, 0.0], 's_est': 1.0, 'planes': []}
    return {'P': wall, 'L': np.full(len(wall), S.INDEX['mur'], np.uint8), 'frame': frame, 'model': model, 'cache': cache}


def test_closed_room_from_labels(tmp_path):
    data = scene(tmp_path); room = S.room(data)
    corners = sorted((round(e['a'][0], 1), round(e['a'][1], 1)) for e in room['edges'])
    assert corners == sorted((float(x), float(z)) for x, z in L_ROOM)
    for e in room['edges']:
        exact = [p for p in (0, 4, 2, 2.5, 3.2) if abs(e['position'] - p) < 0.08]
        assert exact and abs(e['position'] - exact[0]) < 0.06
    grids, seen = S.wall_hits(data, room['edges'], room['height'] or HEIGHT)
    doors = [(e, o) for e, g in zip(room['edges'], grids) for o in S.openings(g)]
    assert len(doors) == 1 and doors[0][1]['type'] == 'porte' and abs(doors[0][1]['width'] - 0.8) < 0.15 and doors[0][0]['axis'] == 'x' and abs(doors[0][0]['position'] - 4) < 0.1


def test_vote_keeps_majority():
    model = {'q': np.array([[1.0, 0, 0, 0]] * 3), 't': np.zeros((3, 3)), 'K': np.tile([100.0, 100, 50, 50, 0, 0, 0, 0], (3, 1)), 'wh': np.tile([100, 100], (3, 1))}
    maps = [np.full((10, 10), S.INDEX['mur'], np.uint8), np.full((10, 10), S.INDEX['mur'], np.uint8), np.full((10, 10), S.INDEX['lit'], np.uint8)]
    labels, share = S.vote(model, maps, np.array([[0, 0, 2.0]]), np.array([0, 0, 0]), np.array([0, 1, 2]))
    assert S.KEYS[labels[0]] == 'mur' and abs(share[0] - 2 / 3) < 1e-6


def test_visibility_reader(tmp_path):
    file = str(tmp_path / 'fused.ply.vis'); np.array([2, 0, 2, 5, 7, 1, 3], np.uint32).tofile(file)
    owner, image = S.read_visibility(file)
    assert owner.tolist() == [0, 0, 1] and image.tolist() == [5, 7, 3]


def test_every_ade_label_has_a_class():
    assert S.ADE.shape == (150,) and S.ADE[0] == S.INDEX['mur'] and S.ADE[7] == S.INDEX['lit'] and S.ADE[2] == S.INDEX['ciel'] and len(S.PALETTE) == len(S.KEYS)

