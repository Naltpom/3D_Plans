import datetime, io, json, os, shutil, subprocess, urllib.error, urllib.request
import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation
from plans3d import groupes, photos, scan, telephone
from plans3d.rpc import Ctx, call

FILTRE = os.path.join(os.path.dirname(scan.__file__), 'scan_filtre.js')
NODE = shutil.which('node')
SCENARIO = r'''
const F = require(process.argv[1]);
let seed = 1;
const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
const W = 320, H = 240, texture = Array.from({ length: 4 * W * H }, () => 255 * rand());
const image = (dx, k = 0, dark = 1) => {
  const g = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = 0;
    for (let i = -k; i <= k; i++) s += texture[y * 2 * W + Math.min(2 * W - 1, Math.max(0, x + dx + i))];
    g[y * W + x] = dark * s / (2 * k + 1);
  }
  return g;
};
const run = (frames) => { const f = new F.Filtre(2), out = []; frames.forEach((g, i) => { const r = f.offre(i * 125, g, W, H); if (r.decision) out.push(r.decision); }); return { out, f }; };
const sharp = image(0), blur = image(0, 6);
const moving = run(Array.from({ length: 33 }, (_, i) => image(4 * i)));
const still = run(Array.from({ length: 33 }, () => sharp));
const shaky = run([...Array.from({ length: 17 }, (_, i) => image(6 * i)), ...Array.from({ length: 16 }, (_, i) => image(100 + 6 * i, 6))]);
const dark = run(Array.from({ length: 9 }, (_, i) => image(8 * i, 0, 0.12)));
console.log(JSON.stringify({ sharp: F.nettete(sharp, W, H), blur: F.nettete(blur, W, H), same: F.ecart(F.signature(sharp, W, H), F.signature(sharp, W, H)),
  moving: moving.out.map((d) => d.raison), still: still.out.map((d) => d.raison), shaky: shaky.out.map((d) => [d.raison, d.conseil]),
  dark: dark.out.map((d) => d.conseil), counts: [still.f.gardees, still.f.doublons, shaky.f.floues] }));
'''


@pytest.mark.skipif(not NODE, reason='node absent')
def test_phone_filter_keeps_sharp_distinct_frames_only():
    r = json.loads(subprocess.run([NODE, '-e', SCENARIO, FILTRE], capture_output=True, text=True, check=True).stdout)
    assert r['sharp'] > 5 * r['blur'] and r['same'] < 1e-6
    assert len(r['moving']) == 8 and all(reason is None for reason in r['moving'])
    assert r['still'][0] is None and set(r['still'][1:]) == {'doublon'} and r['counts'][:2] == [1, 7]
    assert [reason for reason, _ in r['shaky'][:4]] == [None] * 4 and [reason for reason, _ in r['shaky'][4:]] == ['floue'] * 4 and r['counts'][2] == 4
    assert r['shaky'][-1][1] == 'ralentis' and r['dark'] == ['lumiere', 'lumiere']


def test_passages_are_separate_sessions(tmp_path):
    times = [datetime.datetime(2026, 11, 19, 21, 0, s) for s in (0, 1, 2, 3)]
    assert groupes.split(times, 300) == [[0, 1, 2, 3]]
    assert groupes.split(times, 300, ['a', 'a', 'b', 'b']) == [[0, 1], [2, 3]]
    assert groupes.passage(str(tmp_path / 'scan' / 'passage-2' / '00001.jpg'))[1] == '2' and not groupes.passage(str(tmp_path / 'photos' / 'a.jpg'))


def test_saved_jpeg_has_capture_date_and_no_gps(tmp_path):
    exif = Image.Exif(); exif[0x8825] = {1: 'N', 2: (48.0, 51.0, 0.0)}; exif[0x010F] = 'Phone'
    buffer = io.BytesIO(); Image.new('RGB', (64, 48), (90, 120, 150)).save(buffer, 'JPEG', exif=exif.tobytes())
    path = tmp_path / 'a.jpg'; path.write_bytes(scan.dated(buffer.getvalue(), datetime.datetime(2026, 11, 19, 21, 2, 3, 45000).timestamp() * 1000))
    with Image.open(path) as image:
        tags = image.getexif(); image.load()
        assert not tags.get_ifd(0x8825) and 0x010F not in tags and image.size == (64, 48)
    assert photos.read_exif(str(path))['date'] == '2026:11:19 21:02:03.045' and groupes.read_time(str(path)) == datetime.datetime(2026, 11, 19, 21, 2, 3)
    with pytest.raises(ValueError): scan.dated(b'nope', 0)


def model_of(centers, targets):
    up, rows = np.array([0.0, 0.0, 1.0]), []
    for c, t in zip(centers, targets):
        z = (t - c) / np.linalg.norm(t - c); y = -up + (up @ z) * z; y /= np.linalg.norm(y); rows.append(np.array([np.cross(y, z), y, z]))
    R = np.array(rows); q = Rotation.from_matrix(R).as_quat()[:, [3, 0, 1, 2]]
    return {'q': q, 't': -np.einsum('nij,nj->ni', R, np.array(centers, float)), 'names': np.array([f'{i:05d}.jpg' for i in range(len(centers))])}


def test_room_coverage_tells_what_is_missing_from_the_last_view():
    angles = np.radians([*range(-90, 0, 15), *range(90, 0, -15), 0])
    model = model_of([np.zeros(3)] * len(angles), [np.array([np.cos(a), np.sin(a), 0.0]) for a in angles])
    found = scan.coverage(model, 'piece')
    assert len(found['sectors']) == 8 and found['sectors'][4] == 0 and 'derrière toi' in found['gaps'] and 'devant toi' not in found['gaps']
    assert 'le plafond' in found['gaps'] and 'le sol' in found['gaps']


def test_object_turn_and_direction():
    angles = np.radians(np.arange(0, 271, 30))
    model = model_of([2 * np.array([np.cos(a), np.sin(a), 0.3]) for a in angles], [np.zeros(3)] * len(angles))
    found = scan.coverage(model, 'objet')
    assert found['turn'] == 310 and 'continue vers la' in found['gaps'] and 'tour plus haut' in found['gaps'] and np.linalg.norm(found['center']) < 1e-6


@pytest.fixture
def phone(tmp_path, monkeypatch):
    monkeypatch.setattr(telephone, 'tailscale', lambda *args: (None, ''))
    ctx = Ctx(lambda event: None)
    call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'})
    ctx.phone = telephone.Phone(ctx, '127.0.0.1')
    base = f'http://127.0.0.1:{ctx.phone.port}/3d_plans/'
    with urllib.request.urlopen(urllib.request.Request(base + 'session', json.dumps({'code': ctx.phone.token}).encode(), {'Content-Type': 'application/json'})) as r:
        cookie = r.headers['Set-Cookie'].split(';')[0]
    yield ctx, base, cookie
    call(ctx, 'phone.stop', {})


def send(url, body=b'', cookie='', **headers):
    request = urllib.request.Request(url, body if isinstance(body, bytes) else json.dumps(body).encode(), {'Cookie': cookie, **{k.replace('_', '-'): v for k, v in headers.items()}})
    try:
        with urllib.request.urlopen(request, timeout=5) as r: return r.status, json.loads(r.read() or b'null')
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b'null')


def jpeg():
    buffer = io.BytesIO(); Image.new('RGB', (48, 64), (200, 100, 50)).save(buffer, 'JPEG'); return buffer.getvalue()


def test_scan_page_and_uploads_into_photo_source(phone):
    ctx, base, cookie = phone
    with urllib.request.urlopen(base + 'scan') as r:
        page = r.read().decode(); assert "FILTRE.Filtre" in page and '/*FILTRE*/' not in page and 'connect-src' in r.headers['Content-Security-Policy']
    assert send(base + 'scan/state', {})[0] == 403 and send(base + 'scan/state', {}, cookie) == (200, {'active': False})
    assert send(base + 'scan/image', jpeg(), cookie, X_Scan='x', X_Passage='1', X_Index='1', X_Time='0')[0] == 409
    live = ctx.scan = scan.Scan(ctx, 'photos', 'piece')
    head = lambda passage, index, sid=live.id: dict(X_Scan=sid, X_Passage=str(passage), X_Index=str(index), X_Time=str(1.7e12 + 1000 * index))
    for passage, index in ((1, 1), (1, 2), (1, 2), (2, 3)): assert send(base + 'scan/image', jpeg(), cookie, **head(passage, index))[0] == 200
    assert send(base + 'scan/image', jpeg(), cookie, **head(1, 9, 'ancien')) == (409, {'error': "Image d'un ancien scan.", 'old': True})
    assert send(base + 'scan/image', b'pas une image', cookie, **head(1, 10))[0] == 400
    status, state = send(base + 'scan/state', {'passage': 2, 'gardees': 3, 'pirate': 1}, cookie)
    assert status == 200 and state['received'] == 3 and state['passages'] == {'1': 2, '2': 1} and state['phone'] == {'passage': 2, 'gardees': 3}
    manifest = photos.read_manifest(os.path.join(ctx.project.folder, 'sources', 'photos.photos.txt'))
    assert len(manifest) == 3 and all(os.path.exists(p) for p in manifest) and [s['name'] for s in groupes.sessions(ctx.project, 'photos')] == ['Passage 1', 'Passage 2']
    assert send(base + 'scan/cmd', {'op': 'mode', 'mode': 'objet'}, cookie)[1]['mode'] == 'piece'
    assert send(base + 'scan/cmd', {'op': 'finish'}, cookie)[1]['active'] is False and live.done
    assert send(base + 'scan/image', jpeg(), cookie, **head(2, 4))[0] == 409


def test_object_isolated_from_floor_with_dimensions_and_gallery_entry(tmp_path, monkeypatch):
    rng = np.random.default_rng(1); size = np.array([1.6, 0.9, 0.7]); yaw = np.radians(25)
    faces = rng.uniform(-0.5, 0.5, (4000, 3)); axis = rng.integers(0, 3, 4000); faces[np.arange(4000), axis] = np.sign(faces[np.arange(4000), axis]) * 0.5
    box = faces * size + [0, 0, 0.35]; box = box[box[:, 2] > 0.01]
    turn = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]); floor = np.c_[rng.uniform(-3, 3, (3000, 2)), np.zeros(3000)]
    points = np.r_[box @ turn.T, floor] + [5.0, -2.0, 0.0]
    angles = np.radians(np.arange(0, 360, 20)); centers = [np.array([5 + 2.5 * np.cos(a), -2 + 2.5 * np.sin(a), 1.2 + 0.4 * (i % 2)]) for i, a in enumerate(angles)]
    model = {**model_of(centers, [np.array([5.0, -2.0, 0.35])] * len(centers)), 'P': points * 0.37, 'Tptr': np.arange(0, 3 * len(points) + 1, 3)}
    model['t'] *= 0.37
    found, shown = scan.isolate(model)
    assert np.allclose(np.array(found['dims']) / found['dims'][2], size / size[2], rtol=0.05) and len(shown) <= scan.PHONE_POINTS and not found['simple']
    from plans3d import furniture
    monkeypatch.setattr(furniture, 'FOLDER', str(tmp_path / 'catalog'))
    ctx = Ctx(lambda event: None); call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'})
    live = scan.Scan(ctx, 'photos', 'objet'); live.object = found
    with pytest.raises(ValueError): live.keep({'axe': 'hauteur', 'cm': 9000})
    state = live.keep({'axe': 'hauteur', 'cm': 70, 'nom': 'Canapé'})
    assert state['final'] == 'ajouté' and np.allclose(state['object']['added']['dims_cm'], [160, 90, 70], rtol=0.05)
    item = next(e for e in furniture.entries() if e['source'] == 'perso')
    assert item['label'] == 'Canapé' and np.allclose(item['size'], [1.6, 0.7, 0.9], rtol=0.05)


def test_simple_shape_becomes_stacked_boxes():
    rng = np.random.default_rng(2)
    def box(n, lo, hi):
        f = rng.uniform(0, 1, (n, 3)); a = rng.integers(0, 3, n); f[np.arange(n), a] = np.round(f[np.arange(n), a]); return np.array(lo) + f * (np.array(hi) - lo)
    points = np.r_[box(3000, [-0.25, -0.25, 0], [0.25, 0.25, 0.45]), box(2000, [-0.25, 0.17, 0.45], [0.25, 0.25, 0.95]), np.c_[rng.uniform(-3, 3, (3000, 2)), np.zeros(3000)]]
    angles = np.radians(np.arange(0, 360, 20)); centers = [np.array([2 * np.cos(a), 2 * np.sin(a), 1.3]) for a in angles]
    found, _ = scan.isolate({**model_of(centers, [np.array([0, 0, 0.45])] * len(centers)), 'P': points, 'Tptr': np.arange(0, 3 * len(points) + 1, 3)})
    assert found['simple'] and len(found['boxes']) == 2 and abs(found['dims'][2] - 0.95) < 0.03 and min(found['boxes'][1]['lx'], found['boxes'][1]['lz']) < 0.12
