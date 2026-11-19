import base64, io, os
import numpy as np
import pytest
from PIL import Image, ImageDraw
from plans3d import components, dessin, lecture, parcours, plan, rpc
from plans3d.rpc import Ctx

line = lambda *pts, **extra: {'kind': 'trait', 'points': [list(p) for p in pts], **extra}
text = lambda x, y, value: {'kind': 'texte', 'points': [[x, y]], 'text': value, 'size': 24}


def sheet(*items, cell=40, cm=50):
    return {'w': 1600, 'h': 1000, 'cell': cell, 'cm': cm, 'grid': True, 'items': [{'id': f'i{n}', 'color': 'noir', 'width': 6, **item} for n, item in enumerate(items)]}


def room():
    return sheet(line((100, 100), (603, 100)), line((600, 97), (600, 300)), line((600, 300), (700, 300)), line((700, 300), (700, 380)), line((700, 520), (700, 600)),
                 line((700, 600), (100, 600)), line((100, 604), (100, 100)), line((300, 600), (300, 450)), line((100, 100), (60, 60)),
                 line((250, 100), (350, 100), color='rouge'), line((700, 380), (700, 520), color='vert'), text(350, 75, '5 m'), text(730, 450, '300'), text(400, 400, 'lit'))


def png(d, texts=True):
    image = Image.new('RGB', (d['w'], d['h']), 'white'); pen = ImageDraw.Draw(image)
    for item in d['items']:
        if item['kind'] != 'texte': pen.line([tuple(p) for p in dessin.outline(item)], fill=dessin.COLORS[item['color']], width=item['width'])
        elif texts: pen.text(tuple(item['points'][0]), item['text'], fill='black')
    buffer = io.BytesIO(); image.save(buffer, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def test_shapes_read_as_geometry():
    read = dessin.read(None, doc=room())
    assert read['formes'] and [w['name'] for w in read['walls']] == list('ABCDEFG')
    assert read['walls'][0]['a'] == [100.0, 100.0] and read['walls'][0]['b'] == [600.0, 100.0] and read['walls'][3]['b'] == [700.0, 600.0]
    assert {(e.get('name'), e['inside']) for e in read['extra']} == {('H', True), (None, False)}
    assert [(o['type'], o['wall'], o['t0'], o['t1']) for o in read['openings']] == [('porte', 'A', 0.3, 0.5), ('fenetre', 'D', pytest.approx(0.2667, abs=1e-3), pytest.approx(0.7333, abs=1e-3))]
    assert [(c['wall'], c['value_cm']) for c in read['cotes']] == [('A', 500.0), ('D', 300.0)] and read['k'] == pytest.approx(0.01)
    assert {item['color']: item['role'] for item in read['legend']} == {'noir': 'mur', 'rouge': 'porte', 'vert': 'fenetre'}


def test_curves_and_open_outline():
    circle = dessin.read(None, doc=sheet({'kind': 'cercle', 'points': [[100, 100], [500, 500]]}))
    assert len(circle['walls']) == dessin.CURVE and all(np.hypot(*np.subtract(w['b'], w['a'])) == pytest.approx(2 * np.pi * 200 / 16, rel=0.02) for w in circle['walls'])
    bay = dessin.read(None, doc=sheet(line((100, 500), (100, 100)), line((100, 100), (500, 100)), line((500, 100), (500, 500)), {'kind': 'arc', 'points': [[500, 500], [300, 600], [100, 500]]}))
    assert len(bay['walls']) > 5 and max(w['a'][1] for w in bay['walls']) == pytest.approx(600, abs=6)
    rect = dessin.read(None, doc=sheet({'kind': 'rect', 'points': [[200, 200], [800, 600]]}, text(500, 180, '2m40')))
    assert len(rect['walls']) == 4 and rect['cotes'][0]['value_cm'] == 240 and rect['k'] == pytest.approx(0.004)
    with pytest.raises(ValueError, match='pas fermé'): dessin.read(None, doc=sheet(line((100, 100), (500, 100)), line((500, 100), (500, 400)), line((100, 400), (100, 100))))
    assert dessin.length_cm('4', 4.1) == 400 and dessin.length_cm('25', 0.3) == 25 and dessin.length_cm('3,5 m', 1) == 350 and dessin.length_cm('lit', 1) is None


def test_freehand_read_as_image():
    d = sheet({'kind': 'pinceau', 'points': [[100, 100], [350, 101], [600, 100], [600, 300], [599, 500], [100, 500], [100, 100]]}, text(350, 75, '5 m'))
    read = dessin.read(None, doc=d, traits=png(d, False))
    assert not read['formes'] and len(read['walls']) == 4 and read['cotes'][0]['wall'] == 'A' and read['k'] == pytest.approx(0.01, rel=0.03)
    live = rpc.call(Ctx(), 'dessin.read', {'doc': d})
    assert live['image'] and 'error' in live


def test_save_read_apply_undo(tmp_path):
    ctx, d = Ctx(), room(); rpc.call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'})
    saved = rpc.call(ctx, 'dessin.save', {'doc': d, 'png': png(d)})
    assert saved['dessin'] and saved['length'] == pytest.approx(1.0) and os.path.exists(saved['path']) and rpc.call(ctx, 'dessin.get', {}) == d
    assert rpc.call(ctx, 'plan.sketch', {})['file'] == 'dessin.png' and rpc.call(ctx, 'guide.route', {'type': 'piece'})['sources'] == ['croquis']
    read = rpc.call(ctx, 'plan.read_drawing', {})
    cotes = {r['trace'][0]['wall']: r for r in rpc.call(ctx, 'cotes.list', {})}
    assert set(cotes) == {'A', 'D'} and cotes['A']['label'] == 'Mur A (écrit sur le dessin)' and cotes['A']['value'] == 500
    rpc.call(ctx, 'plan.apply', {'plan': read}); rpc.call(ctx, 'update', {})
    lengths = {w['label']: np.hypot(*np.subtract(w['b'], w['a'])) for w in plan.walls(ctx)}
    assert lengths['Mur A'] == pytest.approx(5.0, abs=0.02) and lengths['Mur D'] == pytest.approx(3.0, abs=0.02) and len(lengths) == 8
    moved = {**d, 'items': [i for i in d['items'] if i['id'] != 'i12']}; rpc.call(ctx, 'dessin.save', {'doc': moved, 'png': png(moved)})
    rpc.call(ctx, 'plan.read_drawing', {'previous': read})
    assert [(r['trace'][0]['wall'], r['value']) for r in rpc.call(ctx, 'cotes.list', {})] == [('A', 500)]
    rpc.call(ctx, 'cotes.place', {'value': 512, 'trace': [{'wall': 'A', 's': 0}, {'wall': 'A', 's': 1}]}); rpc.call(ctx, 'plan.read_drawing', {'previous': read})
    assert [r['value'] for r in rpc.call(ctx, 'cotes.list', {})] == [512]
    assert rpc.call(ctx, 'history.undo', {})['done'] and rpc.call(ctx, 'history.undo', {})['done'] and rpc.call(ctx, 'dessin.get', {}) == d


def test_sketch_reading_without_ai(tmp_path, monkeypatch):
    from plans3d import ai
    monkeypatch.setattr(ai, 'ask_image', lambda *a: pytest.fail('pas d’IA pour un dessin'))
    ctx, d = Ctx(), room(); rpc.call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'})
    rpc.call(ctx, 'dessin.save', {'doc': d, 'png': png(d), 'crop': [50, 50]}); found = rpc.call(ctx, 'plan.read_sketch', {})
    assert found['walls'][0]['closed'] and found['walls'][0]['points'][0] == [0.5, 0.5] and len(found['walls'][0]['points']) == 7 and not found['walls'][1]['closed']
    assert [(c['wall'], c['edge'], c['value_cm']) for c in found['cotes']] == [(0, 0, 500.0), (0, 3, 300.0)] and [o['type'] for o in found['openings']] == ['porte', 'fenetre']
    rpc.call(ctx, 'plan.trace', {'points': found['walls'][0]['points'], 'lengths': [5.0, None, None, 3.0, None, None, None], 'cote_status': 'lu'})
    assert len([c for c in components.load(ctx.project).values() if c['type'] == 'mur']) == 7
