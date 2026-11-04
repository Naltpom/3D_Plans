import pytest
from plans3d import Project, plan
from plans3d.rpc import Ctx, call
import plans3d.mesures  # noqa: F401


@pytest.fixture
def ctx(tmp_path):
    ctx = Ctx(project=Project(str(tmp_path / 'projet')))
    plan.trace(ctx, [[0, 0], [4, 0], [4, 3], [0, 3]], True, sigma=0.3, source='plan', names='ABCD')
    door = plan.place(ctx, 'porte', [1, 0], width=0.8, sigma=0.3)['id']
    plan.place(ctx, 'porte', [1, 3], width=0.8, sigma=0.3, share=door)
    call(ctx, 'update', {})
    return ctx


def measure(ctx, item, cm):
    call(ctx, 'cotes.add', {'id': item['key'], 'expr': item['expr'], 'value': cm, 'sigma': 1, 'unit': 'cm', 'label': item['label']})
    call(ctx, 'update', {})
    return call(ctx, 'mesures.list', {})


def test_one_measure_rescales_the_drawing(ctx):
    r = call(ctx, 'mesures.list', {}); first, side = r['candidates'][0], next(c for c in r['candidates'] if c['key'] == 'longueur.murB')
    assert r['supposed'] and not r['precise'] and first['ends'] and first['value'] == pytest.approx(4)
    assert [c['twin'] for c in r['candidates'] if c['value'] == pytest.approx(4)] == [None, first['key'], first['key']]
    r = measure(ctx, first, 480); found = {c['key']: c for c in r['candidates']}
    assert not r['supposed'] and not r['conflicts'] and r['measures'][0]['resid'] == pytest.approx(0, abs=1e-3)
    assert next(s['value'] for s in r['scales'] if s['name'] == 'plan_s') == pytest.approx(1.2, abs=0.02)
    assert found['longueur.murB']['value'] == pytest.approx(3.6, abs=0.04) and found['longueur.murB']['sigma'] < side['sigma']
    assert found['largeur.porte1']['covers'] == ['la porte 2'] and 'largeur.porte2' not in found
    groups = {g['group']: g for g in r['groups']}
    assert groups['murs']['ok'] < groups['murs']['n'] and groups['ouvertures']['worst'] > 0.02


def test_measures_between_elements_and_contradiction(ctx):
    before = call(ctx, 'mesures.list', {})['candidates']; twin = next(c for c in before if c['twin'] == before[0]['key'])
    r = measure(ctx, before[0], 480)
    pair = call(ctx, 'mesures.between', {'a': 'porte1', 'b': 'murC'})
    assert pair['label'] == 'Entre la porte 1 et le mur C' and pair['value'] == pytest.approx(3.6, abs=0.04) and pair['key'] == 'entre.murC.porte1'
    with pytest.raises(Exception, match='différents'): call(ctx, 'mesures.between', {'a': 'murA', 'b': 'murA'})
    r = measure(ctx, twin, 400)
    assert r['conflicts'] and 'ne colle pas' in r['conflicts'][0]['text'] and all(m['conflict'] for m in r['measures'])


def test_precise_once_every_margin_is_small(tmp_path):
    ctx = Ctx(project=Project(str(tmp_path / 'piece')))
    plan.trace(ctx, [[0, 0], [4, 0], [4, 3], [0, 3]], True, sigma=0.3, source='plan', names='ABCD')
    call(ctx, 'update', {}); found = {c['key']: c for c in call(ctx, 'mesures.list', {})['candidates']}
    for key, cm in (('longueur.murA', 480), ('longueur.murB', 350)): r = measure(ctx, found[key], cm)
    assert not r['precise'] and {g['group'] for g in r['groups'] if g['worst'] > r['threshold']} == {'plafond'}
    r = measure(ctx, next(c for c in r['candidates'] if c['group'] == 'plafond'), 245)
    assert r['precise'] and not r['conflicts'] and len(r['measures']) == 3
