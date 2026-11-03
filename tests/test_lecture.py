import numpy as np
import pytest
from PIL import Image, ImageDraw
from plans3d import components, lecture, plan, rpc
from plans3d.rpc import Ctx


def drawing(path):
    image = Image.new('RGB', (800, 700), 'white'); pen = ImageDraw.Draw(image)
    room = [(100, 100), (600, 103), (602, 300), (700, 300), (700, 600), (98, 602)]
    pen.line(room + [room[0]], fill=(0, 0, 0), width=5)
    pen.line([(100, 100), (60, 60)], fill=(0, 0, 0), width=4)
    pen.line([(300, 602), (300, 450)], fill=(0, 0, 0), width=4)
    pen.line([(250, 96), (350, 96)], fill=(230, 40, 30), width=7)
    pen.line([(700, 380), (700, 520)], fill=(30, 140, 230), width=7)
    for k in range(8): pen.line([(450 + 20 * k, 450), (450 + 20 * k, 580)], fill=(128, 128, 128), width=2)
    pen.rectangle([440, 450, 610, 580], outline=(128, 128, 128), width=2)
    image.save(path)


def test_read_drawing(tmp_path):
    drawing(tmp_path / 'plan.png'); read = lecture.read(str(tmp_path / 'plan.png'), ai=False)
    names = [w['name'] for w in read['walls']]
    assert names == list('ABCDEF') and read['walls'][0]['a'][1] < 110 and read['walls'][0]['b'][0] > 590
    for w in read['walls']:
        d = np.subtract(w['b'], w['a']) @ np.array([[np.cos(read['angle']), -np.sin(read['angle'])], [np.sin(read['angle']), np.cos(read['angle'])]])
        assert min(abs(d)) < 1e-3 * max(abs(d))
    assert {(o['type'], o['wall']) for o in read['openings']} == {('porte', 'A'), ('porte_fenetre', 'D')}
    assert len(read['stairs']) == 1 and {e['inside'] for e in read['extra']} == {True, False}
    assert {item['color']: item['role'] for item in read['legend']} == {'noir': 'mur', 'gris': 'escalier', 'rouge': 'porte', 'bleu': 'porte_fenetre'}
    ignored = lecture.read(str(tmp_path / 'plan.png'), [{**item, 'role': 'ignorer' if item['color'] == 'rouge' else item['role']} for item in read['legend']], ai=False)
    assert {o['type'] for o in ignored['openings']} == {'porte_fenetre'}


def test_apply_replaces_and_undo(tmp_path):
    drawing(tmp_path / 'plan.png'); ctx = Ctx(); rpc.call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'})
    rpc.call(ctx, 'plan.sketch', {'path': str(tmp_path / 'plan.png')}); read = rpc.call(ctx, 'plan.read_drawing', {})
    read['extra'] = [{**e, 'name': 'G'} if e['inside'] else e for e in read['extra']]
    for _ in range(2):
        done = rpc.call(ctx, 'plan.apply', {'plan': read}); rpc.call(ctx, 'update', {})
        loaded = components.load(ctx.project)
        assert sorted(c['label'] for c in loaded.values() if c['type'] == 'mur') == [f'Mur {n}' for n in 'ABCDEFG'] and len(done['openings']) == 2 and done['stairs']
        assert len(loaded) == 10
    width = np.ptp([p for w in plan.walls(ctx) for p in (w['a'], w['b'])], 0)
    assert width == pytest.approx([6.0, 5.0 / 6.0 * 6.0], rel=0.05)
    assert rpc.call(ctx, 'history.undo', {})['done'] and len(components.load(ctx.project)) == 10
    assert rpc.call(ctx, 'history.undo', {})['done'] and not components.load(ctx.project)
    assert rpc.call(ctx, 'history.redo', {})['done'] and len(components.load(ctx.project)) == 10
    rpc.call(ctx, 'components.remove', {'id': 'murD'})
    assert 'porte_fenetre1' not in components.load(ctx.project) and ctx.project.update().converged
    rpc.call(ctx, 'project.guide', {'guide': {'step': 4}, 'undo': True}); assert rpc.call(ctx, 'history.undo', {})['done']
