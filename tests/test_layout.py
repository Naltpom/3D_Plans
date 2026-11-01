import pytest
from plans3d import Project, components as C, layout as L
from plans3d.rpc import Ctx, RpcError, call

ITEMS = ['canape', 'meuble_tv', 'table', 'chaise', 'chaise', 'chaise', 'chaise', 'etagere', 'lampadaire', 'fauteuil']


@pytest.fixture
def ctx(tmp_path):
    project = Project(str(tmp_path))
    def add(type, comp_id, **options): C.add(project, type, comp_id, options)
    add('mur', 'm1', axe='x', z='0', x0='0', x1=5.2, sens=-1); add('mur', 'm2', axe='z', x='m1_x1', z0='0', z1=4.0)
    add('mur', 'm3', axe='x', z='m2_z1', x0='0', x1='m1_x1'); add('mur', 'm4', axe='z', x='0', z0='0', z1='m2_z1', sens=-1)
    add('sol', 's0'); add('porte', 'p1', mur='m1', pos=0.4, l=0.9, sens='interieur_gauche'); add('fenetre', 'f1', mur='m3', pos=2.0, l=1.4, al=0.9)
    ctx = Ctx(project=project)
    for k, item in enumerate(ITEMS): call(ctx, 'furniture.add', {'item': f'gabarit:{item}', 'x': 0.5 + 0.4 * k, 'z': 2.0})
    return ctx


def place(ctx, poses):
    for id, (x, z, rot) in poses.items(): call(ctx, 'furniture.move', {'id': id, 'x': x, 'z': z, 'snap_walls': False}); call(ctx, 'furniture.rotate', {'id': id, 'rot': rot})


def test_rules_are_measured(ctx):
    place(ctx, {1: (2.6, 0.45, 0), 2: (2.6, 3.75, 180), 3: (2.6, 2.0, 0), 4: (1.0, 2.0, 0), 5: (1.6, 2.0, 0), 9: (3.9, 0.3, 0)})
    rules = [{'type': 'contre_mur', 'objet': 1}, {'type': 'face_a', 'objet': 2, 'cible': 1, 'distance': [2, 3.5]}, {'type': 'face_a', 'objet': 2, 'cible': 1, 'distance': [2, 3]},
             {'type': 'orientation', 'objet': 2, 'vers': 'meuble-1'}, {'type': 'orientation', 'objet': 1, 'vers': 90}, {'type': 'aligne', 'objets': [4, 5]},
             {'type': 'centre', 'objet': 3}, {'type': 'loin_de', 'objet': 3, 'cible': 'porte', 'min': 1.5}, {'type': 'pres_de', 'objet': 9, 'cible': 1, 'max': 0.3},
             {'type': 'contre_mur', 'objet': 2, 'mur': 'm1'}]
    scene = L.Scene(ctx.project, None).compile(rules)
    details = scene.report(scene.objects, scene.movable)['details']
    assert [d['ok'] for d in details[:len(rules)]] == [True, True, False, True, False, True, True, True, True, False]
    assert [d['text'] for d in details[:3]] == ['Canapé contre un mur : OK', 'Meuble TV face à Canapé à 3,3 m : OK', 'Meuble TV face à Canapé à 3,3 m (attendu 2,0 m à 3,0 m) : pénalité']
    assert details[4]['text'] == 'Canapé orienté à 90° (écart 90°) : pénalité' and details[7]['text'].startswith('Table loin d’une porte (')
    assert details[len(rules) - 1]['text'] == 'Meuble TV contre Mur m1 (écart 395 cm) : pénalité'


def test_propose_and_apply(ctx):
    rules = [{'type': 'face_a', 'objet': 2, 'cible': 1, 'distance': [2, 3]}, {'type': 'contre_mur', 'objet': 8},
             {'type': 'pres_de', 'objet': 9, 'cible': 1, 'max': 0.3}, {'type': 'loin_de', 'objet': 3, 'cible': 'porte', 'min': 1.5}]
    progress = []
    out = L.propose(ctx, rules=rules, n=3, progress=lambda fraction, message='': progress.append(fraction))
    proposals = out['proposals']
    assert out['seconds'] < 3 and len(proposals) == 3 and progress[-1] == 1.0 and call(ctx, 'layout.rules', {}) == rules
    assert proposals[0]['ok'] and proposals[0]['score'] >= 90 and [p['score'] for p in proposals] == sorted((p['score'] for p in proposals), reverse=True)
    assert len({tuple((o['x'], o['z'], o['rot']) for o in p['objects']) for p in proposals}) == 3
    assert {s['id'] for s in proposals[0]['shapes']} == {f'meuble-{k}' for k in range(1, 11)} and proposals[0]['details'][-1]['text'].endswith(': OK')
    applied = call(ctx, 'layout.apply', {'index': 0})
    assert applied['warnings'] == [] and [{k: o[k] for k in ('id', 'x', 'z', 'rot')} for o in call(ctx, 'furniture.list', {})] == proposals[0]['objects']
    fixed = call(ctx, 'layout.propose', {'objects': [9], 'n': 1})['proposals'][0]['objects']
    assert [o['id'] for o in fixed] == [9]


def test_imposed_angle(ctx):
    rules = [{'type': 'orientation', 'objet': 1, 'vers': 30}, {'type': 'contre_mur', 'objet': 8}]
    for p in L.propose(ctx, rules=rules, n=2)['proposals']:
        assert next(o['rot'] for o in p['objects'] if o['id'] == 1) == 30 and p['details'][0]['text'] == 'Canapé orienté à 30° : OK'


def test_errors(ctx):
    for rules in ([{'type': 'devant'}], [{'type': 'contre_mur', 'objet': 42}], [{'type': 'face_a', 'objet': 2, 'cible': 1, 'distance': 2}],
                  [{'type': 'pres_de', 'objet': 2, 'cible': 'cheminee'}], [{'type': 'aligne', 'objets': [1]}], [{'type': 'contre_mur', 'objet': 1, 'mur': 'p1'}], {}):
        with pytest.raises(RpcError): call(ctx, 'layout.rules', {'rules': rules})
    with pytest.raises(RpcError): call(ctx, 'layout.apply', {'index': 0})
    with pytest.raises(RpcError): call(ctx, 'layout.propose', {'objects': [99]})
