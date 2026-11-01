import hashlib, io, json, math, os
import pytest
from plans3d import furniture as F
from plans3d.rpc import Ctx, RpcError, call
from test_components import build_room

ROWS = '''// rows
export const ROWS = `TESTA|chair|Chaise test|45|90|50|seatH=46|bois
ÖRNA|wallcab|Wall cabinet, 60 cm|60|40|30|doors=2
CASSE|sofa|Ligne illisible|a|b|c
MYSTÈRE|unknownkind|Objet inconnu|50|50|50||rare|mystere-slug
GÉANT|wardrobe|Trop grand|600|200|60|doors=2
ÉTROIT|wardrobe|Armoire|60|200|60|doors=40,arms=wide,rows=3,shelves=2.5`;
'''


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setattr(F, 'FOLDER', str(tmp_path / 'catalog'))
    return Ctx(project=build_room(str(tmp_path / 'p')))


def serve(monkeypatch, data, sha256=None):
    monkeypatch.setattr(F.urllib.request, 'urlopen', lambda url, timeout: io.BytesIO(data))
    monkeypatch.setattr(F, 'IKEA_SHA256', sha256 or hashlib.sha256(data).hexdigest())


def add(ctx, kind, x, z, rot=0.0, **extra): return call(ctx, 'furniture.add', {'item': f'gabarit:{kind}', 'x': x, 'z': z, 'rot': rot, **extra})
def codes(result): return sorted(w['code'] for w in result['warnings'])
def messages(result): return [w['message'] for w in result['warnings']]


@pytest.mark.parametrize('kind', list(F.GABARITS))
def test_templates_stay_inside_their_size(ctx, kind):
    obj = add(ctx, kind, 0.0, 0.0)['object']; (w, h, d) = obj['size']
    for _, (x, y, z), (lx, ly, lz), _ in F.parts(F.get(ctx.project, obj['id'])):
        assert abs(x) + lx / 2 <= w / 2 + 1e-9 and abs(z) + lz / 2 <= d / 2 + 1e-9 and -1e-9 <= y - ly / 2 and y + ly / 2 <= h + 1e-9
    shapes = call(ctx, 'furniture.geometry', {}); json.dumps(shapes)
    assert {s['kind'] for s in shapes} <= {'box', 'mesh'} and all(s['id'] == obj['ref'] and s['type'] == 'meuble' for s in shapes)
    assert any(s.get('light') == {'puissance': 60, 'teinte': '#ffd8a8'} for s in shapes) == (kind in F.LIGHTS)


def test_import_search_and_kinds(ctx, monkeypatch):
    serve(monkeypatch, ROWS.encode('utf-8'))
    assert call(ctx, 'catalog.import_ikea', {})['count'] == 4
    chair, cabinet, other, narrow = F.read('ikea.json')
    assert narrow['params'] == {'rows': 3} and cabinet['params'] == {'doors': 2}
    assert chair['size'] == [0.45, 0.9, 0.5] and chair['params'] == {'seatH': 46.0} and chair['kind'] == 'chaise'
    assert cabinet['kind'] == 'armoire' and cabinet['y'] == 1.4 and other['kind'] == 'commode' and other['id'] == 'ikea:mystere-slug'
    assert [e['id'] for e in call(ctx, 'catalog.search', {'q': 'orna cabinet'})['items']] == [cabinet['id']]
    assert [e['label'] for e in call(ctx, 'catalog.search', {'q': 'chaise'})['items']] == ['Chaise', 'TESTA — Chaise']
    assert (cabinet['label'], other['label']) == ('ÖRNA — Élément mural', 'MYSTÈRE — Objet inconnu')
    assert call(ctx, 'catalog.search', {'kind': 'chaise', 'limit': 1})['total'] == 2
    assert {k['kind']: k['count'] for k in call(ctx, 'catalog.kinds', {})}['commode'] == 2
    assert call(ctx, 'furniture.add', {'item': cabinet['id'], 'x': 2.0, 'z': 0.15})['object']['y'] == 1.4


def test_import_is_pinned_and_capped(ctx, monkeypatch):
    assert F.IKEA_URL.startswith('https://raw.githubusercontent.com/Nutlope/bygga/') and len(F.IKEA_SHA256) == 64
    with pytest.raises(RpcError, match='paramètre inconnu'): call(ctx, 'catalog.import_ikea', {'url': 'https://example.com/rows.ts'})
    serve(monkeypatch, ROWS.encode('utf-8'), '0' * 64)
    with pytest.raises(RpcError, match='empreinte'): call(ctx, 'catalog.import_ikea', {})
    serve(monkeypatch, ROWS.encode('utf-8')); monkeypatch.setattr(F, 'MAX_DOWNLOAD', 100)
    with pytest.raises(RpcError, match='Mo'): call(ctx, 'catalog.import_ikea', {})
    for url in ('http://raw.githubusercontent.com/a/b', 'https://example.com/a', 'file:///c:/rows.ts'):
        monkeypatch.setattr(F, 'IKEA_URL', url)
        with pytest.raises(RpcError, match='refusée'): call(ctx, 'catalog.import_ikea', {})
    assert not os.path.exists(os.path.join(F.FOLDER, 'ikea.json'))


def test_add_rejects_out_of_bounds(ctx):
    for options in ({'doors': 0}, {'doors': 21}, {'rows': 2.5}, {'doors': 'deux'}, {'seatH': math.nan}, {'inconnu': 1}, {'teinte': 'rouge'}, [1]):
        with pytest.raises(RpcError): call(ctx, 'furniture.add', {'item': 'gabarit:armoire', 'options': options})
    for params in ({'x': 'a'}, {'z': math.inf}, {'rot': None}, {'y': '1'}):
        with pytest.raises(RpcError): call(ctx, 'furniture.add', {'item': 'gabarit:table', **params})
    big = {'x': 0, 'y': 0, 'z': 0, 'lx': 6.0, 'ly': 1.0, 'lz': 1.0}
    for custom in ({'boxes': [big]}, {'boxes': [{**big, 'lx': 0.01, 'ly': 0.01, 'lz': 0.01}]}, {'boxes': [{**big, 'lx': 1.0}] * 101}, {'boxes': [{'x': 0}]}):
        with pytest.raises(RpcError): call(ctx, 'furniture.add', {'custom': custom})
    assert call(ctx, 'furniture.add', {'item': 'gabarit:etagere', 'options': {'shelves': 4}})['object']['id'] == 1
    assert sum(1 for _, (_, _, _), (_, ly, _), _ in F.parts(F.get(ctx.project, 1)) if ly < 0.05) == 6
    with pytest.raises(RpcError): call(ctx, 'furniture.move', {'id': 1, 'x': math.nan, 'z': 0})
    os.makedirs(F.FOLDER)
    with open(os.path.join(F.FOLDER, 'ikea.json'), 'w', encoding='utf-8') as fh: json.dump([{'id': 'ikea:x', 'label': 'X', 'kind': 'table', 'size': [9, 1, 1], 'params': {}}], fh)
    with open(os.path.join(F.FOLDER, 'perso.json'), 'w', encoding='utf-8') as fh: json.dump([{'boxes': [big]}, 'texte'], fh)
    with pytest.raises(RpcError, match='hors bornes'): call(ctx, 'furniture.add', {'item': 'ikea:x'})
    assert call(ctx, 'catalog.search', {'kind': 'sur_mesure'})['total'] == 0


def test_custom_and_personal_items(ctx):
    boxes = [{'x': 0, 'y': 0, 'z': 0, 'lx': 1.0, 'ly': 0.5, 'lz': 0.4}, {'x': 1.0, 'y': 0, 'z': 0, 'lx': 0.5, 'ly': 1.2, 'lz': 0.4}]
    obj = call(ctx, 'furniture.add', {'custom': {'label': 'Bloc', 'boxes': boxes}, 'x': 2.0, 'z': 1.5})['object']
    assert obj['size'] == [1.5, 1.2, 0.4] and obj['kind'] == 'sur_mesure'
    assert [b['x'] for b in F.get(ctx.project, obj['id'])['boxes']] == [-0.75, 0.25]
    os.makedirs(F.FOLDER)
    with open(os.path.join(F.FOLDER, 'perso.json'), 'w', encoding='utf-8') as fh: json.dump([{'id': 'perso:bloc', 'label': 'Bloc perso', 'boxes': boxes}], fh)
    assert [e['id'] for e in call(ctx, 'catalog.search', {'q': 'bloc perso', 'kind': 'sur_mesure'})['items']] == ['perso:bloc']
    assert call(ctx, 'furniture.add', {'item': 'perso:bloc', 'x': 2.0, 'z': 2.5})['object']['label'] == 'Bloc perso'
    with pytest.raises(RpcError): call(ctx, 'furniture.add', {'custom': {'boxes': [{**boxes[0], 'lx': 0}]}})


def test_collisions_with_objects_walls_and_stairs(ctx):
    first = add(ctx, 'table', 2.4, 1.4); assert codes(first) == []
    second = add(ctx, 'table', 2.6, 1.5); assert 'Collision : Table chevauche Table' in messages(second)
    for result in (first, second): call(ctx, 'furniture.remove', {'id': result['object']['id']})
    wall = add(ctx, 'commode', 2.4, 0.1); assert messages(wall) == ['Collision : Commode / tiroirs chevauche Mur m1']
    call(ctx, 'furniture.remove', {'id': wall['object']['id']})
    stairs = add(ctx, 'commode', 1.5, 3.2); assert 'Collision : Commode / tiroirs chevauche Escalier droit e1' in messages(stairs)
    assert stairs['object']['id'] == 4
    call(ctx, 'furniture.remove', {'id': 4})
    assert codes(add(ctx, 'lampe', 3.9, 3.2)) == []


def test_passage_door_snap_and_rotation(ctx):
    assert messages(add(ctx, 'commode', 2.4, 0.74)) == ['Passage de 50 cm entre Commode / tiroirs et Mur m1 (minimum 80 cm)']
    moved = call(ctx, 'furniture.move', {'id': 1, 'x': 2.4, 'z': 0.29})
    assert moved['object']['z'] == pytest.approx(0.24) and moved['warnings'] == []
    assert call(ctx, 'furniture.move', {'id': 1, 'x': 0.45, 'z': 0.31})['object'] | {'x': 0.4, 'z': 0.24} == call(ctx, 'furniture.list', {})[0]
    assert 'porte' in codes(call(ctx, 'furniture.move', {'id': 1, 'x': 0.9, 'z': 0.6, 'snap_walls': False}))
    assert call(ctx, 'furniture.rotate', {'id': 1, 'rot': -90})['object']['rot'] == 270
    assert F.bounds(F.outline(F.get(ctx.project, 1))[0]) == pytest.approx((0.9 - 0.24, 0.9 + 0.24, 0.6 - 0.4, 0.6 + 0.4))
    with pytest.raises(RpcError): call(ctx, 'furniture.move', {'id': 9, 'x': 0, 'z': 0})


def test_door_swing_window_access_and_height(ctx):
    lamp = add(ctx, 'lampe', 1.3, 0.85)['object']['id']
    assert codes(call(ctx, 'furniture.check', {})) == ['porte']
    for sens, expected in [('interieur_gauche', []), ('interieur_droite', ['porte']), ('exterieur_droite', [])]:
        call(ctx, 'components.remove', {'id': 'p1'}); call(ctx, 'components.add', {'type': 'porte', 'id': 'p1', 'options': {'mur': 'm1', 'pos': 0.45, 'l': 0.95, 'sens': sens}})
        assert codes(call(ctx, 'furniture.check', {})) == expected, sens
    with pytest.raises(ValueError): call(ctx, 'components.add', {'type': 'porte', 'id': 'p2', 'options': {'mur': 'm3', 'sens': 'dedans'}})
    with pytest.raises(ValueError): call(ctx, 'components.add', {'type': 'fenetre', 'id': 'f2', 'options': {'mur': 'm3', 'sens': 'interieur_gauche'}})
    call(ctx, 'furniture.remove', {'id': lamp})
    shelf = add(ctx, 'etagere', 4.70, 1.5, 90)
    assert messages(shelf) == ['Étagère / bibliothèque gêne l’accès à Fenêtre f1']
    call(ctx, 'furniture.remove', {'id': shelf['object']['id']})
    stool = add(ctx, 'tabouret', 4.65, 1.5)['object']; assert codes(call(ctx, 'furniture.check', {})) == []
    raised = call(ctx, 'furniture.move', {'id': stool['id'], 'x': 4.65, 'z': 1.5, 'y': 0.6, 'snap_walls': False})
    assert raised['object']['y'] == 0.6 and codes(raised) == ['fenetre']
    assert all(s['objet']['x'] == 4.65 for s in call(ctx, 'furniture.geometry', {}))


def test_update_reports_furniture(ctx):
    add(ctx, 'table', 2.4, 1.4); add(ctx, 'chaise', 2.4, 1.4)
    assert any(w['code'] == 'collision' and w['meuble'] == 1 for w in call(ctx, 'update', {})['warnings'])
    assert [w['code'] for w in call(ctx, 'furniture.check', {})['warnings']] == ['collision']
