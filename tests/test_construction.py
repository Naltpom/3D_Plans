import json
import pytest
from plans3d import construction, mesures, plan  # noqa: F401
from plans3d.rpc import Ctx, RpcError, call, read_cotes


@pytest.fixture
def ctx(tmp_path):
    ctx = Ctx(); call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'})
    return ctx


def op(ctx, **o): return call(ctx, 'construction.op', {'o': o})


def room(ctx):
    op(ctx, op='start', point=[0, 0])
    for p, l in (([4.2, 0], 420), ([4.2, 3], 300), ([0, 3], None)): op(ctx, op='add', point=p, length=l)
    return op(ctx, op='close', length=300)


def test_room_point_by_point_with_lengths(ctx):
    doc = room(ctx)
    t = doc['traces'][0]
    assert t['closed'] and t['names'] == ['A', 'B', 'C', 'D'] and t['typed'] == [420, 300, None, 300] and doc['open'] is None
    with pytest.raises(RpcError): op(ctx, op='add', point=[1, 1])


def test_retouch_length_moves_the_following_walls_and_angle_rotates_them(ctx):
    room(ctx)
    t = op(ctx, op='length', wall='A', value=450)['traces'][0]
    assert t['points'] == [[0, 0], [4.5, 0], [4.5, 3], [0, 3]] and t['typed'][0] == 450
    t = op(ctx, op='angle', wall='B', value=90)['traces'][0]
    assert t['points'][2] == [4.5, 3]
    t = op(ctx, op='angle', wall='B', value=135)['traces'][0]
    assert t['points'][2][0] == pytest.approx(4.5 + 3 * 2 ** -0.5, abs=1e-3) and t['points'][2][1] == pytest.approx(3 * 2 ** -0.5, abs=1e-3)
    with pytest.raises(RpcError): op(ctx, op='angle', wall='A', value=90)


def test_partition_back_finish_and_remove(ctx):
    room(ctx)
    op(ctx, op='start', point=[2, 0], cloison=True); op(ctx, op='add', point=[2, 1.5], length=150); op(ctx, op='add', point=[3, 1.5])
    doc = op(ctx, op='back')
    assert doc['traces'][1]['names'] == ['E'] and doc['open'] == 't2'
    with pytest.raises(RpcError): op(ctx, op='close')
    doc = op(ctx, op='finish')
    assert doc['open'] is None and doc['traces'][1]['cloison']
    t = op(ctx, op='remove', wall='B')['traces'][0]
    assert not t['closed'] and t['names'] == ['C', 'D', 'A'] and t['points'][0] == [4.2, 3]
    op(ctx, op='start', point=[9, 9]); assert len(op(ctx, op='finish')['traces']) == 2


def test_openings_and_height(ctx):
    room(ctx)
    doc = op(ctx, op='opening', type='fenetre', wall='B', pos=80, width=140, height=115, sill=95)
    assert doc['openings'] == [{'id': 'o1', 'type': 'fenetre', 'wall': 'B', 'width': 140, 'height': 115, 'sill': 95, 'pos': 80}]
    doc = op(ctx, op='opening', id='o1', type='fenetre', wall='B', pos=500, width=140)
    assert doc['openings'][0]['pos'] == 160 and doc['openings'][0]['height'] == 115
    with pytest.raises(RpcError): op(ctx, op='opening', type='porte', wall='B', width=400)
    assert op(ctx, op='height', value=245)['height'] == 245
    assert op(ctx, op='remove', opening='o1')['openings'] == []


def test_apply_builds_the_plan_with_lengths_as_scale_and_keeps_cotes(ctx):
    room(ctx)
    op(ctx, op='opening', type='porte', wall='A', pos=60, width=83, height=204)
    op(ctx, op='opening', type='fenetre', wall='C', pos=100, width=120, height=115, sill=95)
    op(ctx, op='start', point=[2, 0], cloison=True); op(ctx, op='add', point=[2, 1.5], length=150); op(ctx, op='finish')
    op(ctx, op='height', value=245)
    made = call(ctx, 'construction.apply', {})
    assert made['walls'] == ['murA', 'murB', 'murC', 'murD', 'murE'] and len(made['openings']) == 2 and not made['skipped']
    res = call(ctx, 'update', {})
    values = {p['name']: p['value'] for p in res['params']}
    assert values['murA_h'] == pytest.approx(2.45, abs=1e-3) and not res['conflicts']
    assert {r['id'] for r in read_cotes(ctx.project)} >= {'trace_murA', 'trace_murB', 'trace_murD', 'trace_murE', 'construction_hauteur'}
    call(ctx, 'cotes.place', {'trace': [{'wall': 'C', 's': 0}, {'wall': 'C', 's': 1}], 'value': 421, 'label': 'Longueur du mur C'})
    op(ctx, op='length', wall='A', value=421)
    again = call(ctx, 'construction.apply', {})
    assert again['walls'][:4] == ['murA', 'murB', 'murC', 'murD'] and any(r['label'] == 'Longueur du mur C' for r in read_cotes(ctx.project))
    assert call(ctx, 'history.undo', {})['done'] and json.load(open(construction.path(ctx.project), encoding='utf-8'))['traces'][0]['typed'][0] == 421
