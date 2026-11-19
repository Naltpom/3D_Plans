import numpy as np
import pytest
from plans3d import components, lecture, rpc
from plans3d.rpc import Ctx, read_cotes
from test_lecture import drawing

ROOM = [((200, 0), (400, 0)), ((400, 0), (400, 500)), ((400, 500), (0, 500)), ((0, 500), (0, 200)), ((0, 200), (200, 200)), ((200, 200), (200, 0))]
SEEN = {'A': ((0.4, 0.0), (2.0, 0.0)), 'B': ((2.0, 0.0), (2.0, 5.0)), 'C': ((2.0, 5.0), (-2.0, 5.0)), 'D': ((-2.0, 5.0), (-2.0, 2.0)), 'E': ((-2.0, 2.0), (0.4, 2.0))}


def solve(cotes):
    drawn = {'angle': 0.0, 'walls': [{'name': n, 'a': list(a), 'b': list(b)} for n, (a, b) in zip('ABCDEF', ROOM)],
             'extra': [{'name': 'M', 'a': [200, 100], 'b': [300, 100], 'inside': True}], 'openings': [{'type': 'porte', 'wall': 'C', 't0': 0.25, 't1': 0.45}]}
    ring = lecture.ring_model(drawn, lecture.frame(drawn, 0.01)[0]); rows = [{'id': n, 'value': v, 'unit': 'cm', 'trace': t} for n, (t, v) in cotes.items()]
    d, _, solved = lecture.adjust(ring, {n: tuple(map(np.array, s)) for n, s in SEEN.items()}, np.zeros((0, 3)), 2.5, [], [], lecture.cote_ends(drawn, ring, rows))
    size = {n: float(np.hypot(*(b - a))) * solved['unit'] for n, (a, b) in zip(ring['names'], lecture.ring_ends(ring, d))}
    return size, solved


def length(wall): return [{'wall': wall, 's': 0.0}, {'wall': wall, 's': 1.0}]


def test_cotes_reshape_unseen_wall_partition_and_door():
    size, solved = solve({})
    assert size['E'] == pytest.approx(2.0, abs=0.02) and size['A'] == pytest.approx(2.0, abs=0.02) and size['M'] == pytest.approx(1.0, abs=0.01) and solved['unit'] == 1.0
    door = [{'wall': 'C', 's': 0.0}, {'wall': 'C', 's': 0.25, 'edge': 1}]
    size, solved = solve({'E': (length('E'), 240), 'A': (length('A'), 160), 'M': (length('M'), 60), 'porte': (door, 130)})
    assert size['E'] == pytest.approx(2.4, abs=0.01) and size['A'] == pytest.approx(1.6, abs=0.01) and size['M'] == pytest.approx(0.6, abs=0.01)
    assert solved['edges'][(0, 0)] == pytest.approx(1.3 / 4.0, abs=0.005) and solved['unit'] == pytest.approx(1.0, abs=0.01)
    assert not any(c['conflict'] for c in solved['cotes']) and {c['id']: round(c['plan'], 2) for c in solved['cotes']}['E'] == 2.4


def test_cote_against_photos_is_a_conflict():
    _, solved = solve({'E': (length('E'), 240), 'A': (length('A'), 160), 'C': (length('C'), 400), 'D': (length('D'), 345)})
    assert [c['id'] for c in solved['cotes'] if c['conflict']] == ['D']


def test_cotes_survive_validation_and_reach_step_five(tmp_path, monkeypatch):
    import plans3d.mesures  # noqa: F401
    monkeypatch.setattr(lecture, 'vision_roles', lambda png: {})
    drawing(tmp_path / 'plan.png'); ctx = Ctx(); rpc.call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'})
    rpc.call(ctx, 'plan.sketch', {'path': str(tmp_path / 'plan.png')}); read = rpc.call(ctx, 'plan.read_drawing', {})
    rpc.call(ctx, 'cotes.place', {'trace': length('C'), 'value': 410, 'label': 'Longueur du mur C'})
    assert [(r['id'], r['expr']) for r in read_cotes(ctx.project)] == [('cote.1', '')]
    assert rpc.call(ctx, 'cotes.place', {'trace': length('C')[::-1], 'value': 412})['id'] == 'cote.1' and len(read_cotes(ctx.project)) == 1
    assert rpc.call(ctx, 'plan.apply', {'plan': read})['kept'] == ['longueur.murC']; rpc.call(ctx, 'update', {})
    free = [{'wall': 'B', 's': 0.5, 'r': 0.2}, {'wall': 'E', 's': 0.5}]; value = rpc.call(ctx, 'cotes.preview', {'trace': free})['value']
    assert rpc.call(ctx, 'cotes.place', {'trace': free, 'value': round(value * 100, 1), 'label': 'Près du mur B → mur E'})['id'] == 'cote.1'
    for previous in (None, read):
        plan = rpc.call(ctx, 'plan.read_drawing', {'previous': previous}) if previous else read
        done = rpc.call(ctx, 'plan.apply', {'plan': plan}); rows = {r['id']: r for r in read_cotes(ctx.project)}
        assert sorted(done['kept']) == ['cote.1', 'longueur.murC'] and not done['dropped'] and sorted(rows) == ['cote.1', 'longueur.murC']
        assert rows['longueur.murC']['value'] == 412 and rows['longueur.murC']['trace'] and rows['cote.1']['trace'][0]['r'] == 0.2 and 'hypot' in rows['cote.1']['expr']
    rpc.call(ctx, 'update', {}); found = rpc.call(ctx, 'mesures.list', {}); mine = {m['id']: m for m in found['measures']}
    assert mine['longueur.murC']['key'] == 'longueur.murC' and mine['cote.1']['ends'] and not found['conflicts']
    near = [{'wall': 'C', 's': 0.05}, {'wall': 'C', 's': 0.95}]
    assert rpc.call(ctx, 'cotes.preview', {'trace': near})['value'] == pytest.approx(0.9 * 4.12, abs=0.03)
    rpc.call(ctx, 'cotes.place', {'trace': near, 'value': 400, 'label': 'Mur C (ligne)'}); rpc.call(ctx, 'update', {}); found = rpc.call(ctx, 'mesures.list', {})
    assert found['conflicts'] and {'cote.2', 'longueur.murC'} & {c for g in found['conflicts'] for c in g['cotes']}
    assert ctx.project.update().converged


def test_cote_fixing_the_scale_alone_is_disputed():
    from plans3d.core import Obs, solve
    obs = [Obs('p1', 'a / s', 1.0, 0.02, 'photos', 'photo'), Obs('p2', 'b / s', 2.0, 0.02, 'photos', 'photo'), Obs('s', 's', 1.0, 0.1, 'photos', 'supposé'),
           Obs('cotes:a', 'a', 1.0, 0.01, 'cotes'), Obs('cotes:b', 'b', 2.4, 0.01, 'cotes')]
    res = solve({'a': 1.0, 'b': 2.0, 's': 1.0}, obs)
    flagged = lambda groups: {o.id for g in groups for o, _ in g}
    assert 'cotes:b' not in flagged(res.conflicts()) and 'cotes:b' in flagged(res.disputed())
