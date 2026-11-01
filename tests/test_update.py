import json, os
import pytest
from plans3d.rpc import Ctx, RpcError, call


def open_ctx(tmp_path):
    events = []; ctx = Ctx(events.append)
    call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'essai'})
    for name, axis in (('w1', 'x'), ('w2', 'x'), ('h', 'y'), ('ph_s', 'échelle'), ('ph_t', 'photo')):
        call(ctx, 'params.add', {'name': name, 'init': 1.0, 'axis': axis})
    for cid, expr, value, unit, status in (('a', 'w1', 120, 'cm', 'mesuré'), ('p1', '(w1 - ph_t) / ph_s', 1.0, 'u', 'photo'),
                                           ('p2', '(w2 - ph_t) / ph_s', 5.0, 'u', 'photo'), ('porte', 'w2 - w1', 204, 'cm', 'supposé'),
                                           ('plafond', 'h', 250, 'cm', 'supposé')):
        call(ctx, 'cotes.add', {'id': cid, 'expr': expr, 'value': value, 'unit': unit, 'status': status, 'sigma': 0.01 if unit == 'u' else 1})
    return ctx, events


def test_progress_events(tmp_path):
    ctx, events = open_ctx(tmp_path)
    call(ctx, 'update', {})
    progress = [(e['fraction'], e['message']) for e in events if e.get('event') == 'progress' and e['method'] == 'update']
    messages = [m for _, m in progress]
    assert messages == ['Extraction des sources', 'Extraction des sources : cotes', 'Résolution', 'Export']
    assert [f for f, _ in progress] == sorted(f for f, _ in progress)
    ctx.project.db.close()


def test_scale_from_assumption_only(tmp_path):
    ctx, _ = open_ctx(tmp_path)
    result = call(ctx, 'update', {})
    sigma = {p['name']: p['sigma'] for p in result['params']}
    warnings = {w['code']: w for w in result['warnings']}
    assert set(warnings) == {'echelle_supposee', 'non_determines'} and warnings['echelle_supposee']['param'] == 'ph_s'
    assert warnings['non_determines']['params'] == ['w2', 'ph_s', 'ph_t'] and warnings['non_determines']['inutilises'] == 0
    assert call(ctx, 'predict', {'expr': 'w2 - w1'})['sigma'] is None and 0 < call(ctx, 'predict', {'expr': 'h'})['sigma'] < 0.02
    assert call(ctx, 'predict', {'expr': '(w2 - ph_t) / ph_s'})['value'] == pytest.approx(5.0)
    assert call(ctx, 'predict', {'expr': '(w2 - ph_t) / ph_s'})['sigma'] < 0.02
    assert all(sigma[n] == float('inf') for n in ('ph_s', 'ph_t', 'w2')) and sigma['w1'] < 0.02 and sigma['h'] < 0.02
    with open(os.path.join(ctx.project.folder, 'model.json'), encoding='utf-8') as fh: model = json.load(fh)
    assert model['warnings'] == result['warnings'] and model['params']['ph_s']['sigma'] is None

    call(ctx, 'cotes.add', {'id': 'b', 'expr': 'w2', 'value': 324, 'unit': 'cm', 'status': 'mesuré'})
    result = call(ctx, 'update', {})
    assert result['warnings'] == [] and all(0 < p['sigma'] < 0.05 for p in result['params'])
    assert 0 < call(ctx, 'predict', {'expr': 'w2 - w1'})['sigma'] < 0.02
    with open(os.path.join(ctx.project.folder, 'model.json'), encoding='utf-8') as fh: assert json.load(fh)['warnings'] == []
    ctx.project.db.close()


def test_gauge_and_predict(tmp_path):
    ctx = Ctx(); call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'essai'})
    for name in ('w1', 'w2', 'w3', 'libre'): call(ctx, 'params.add', {'name': name, 'init': 1.0, 'axis': 'x'})
    for cid, expr, value in (('a', 'w2 - w1', 130), ('b', 'w3 - w2', 60)): call(ctx, 'cotes.add', {'id': cid, 'expr': expr, 'value': value})
    with pytest.raises(RpcError, match='aucun résultat'): call(ctx, 'predict', {'expr': 'w1'})
    (warning,) = call(ctx, 'update', {})['warnings']
    assert warning == {'code': 'non_determines', 'params': ['w1', 'w2', 'w3'], 'inutilises': 1,
                       'message': '3 paramètres non déterminés (ex. w1, w2, w3) : ajoute une cote ou fixe une origine.'}
    distance = call(ctx, 'predict', {'expr': 'w3 - w1'})
    assert distance['value'] == pytest.approx(1.9) and 0 < distance['sigma'] < 0.02 and call(ctx, 'predict', {'expr': 'w1'})['sigma'] is None
    with pytest.raises(RpcError, match='paramètre inconnu'): call(ctx, 'predict', {'expr': 'w9'})
    with pytest.raises(RpcError, match='expression invalide'): call(ctx, 'predict', {'expr': 'w1 +'})
    ctx.project.db.close()


def make_ctx(tmp_path, params, cotes):
    ctx = Ctx(); call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'essai'})
    for name, axis in params: call(ctx, 'params.add', {'name': name, 'init': 1.0, 'axis': axis})
    for cid, expr, value, unit, sigma, status in cotes:
        call(ctx, 'cotes.add', {'id': cid, 'expr': expr, 'value': value, 'unit': unit, 'sigma': sigma, 'status': status})
    return ctx


def test_fragile_scale(tmp_path):
    photo = [('a', 'w1', 120, 'cm', 1, 'mesuré'), ('p1', '(w1 - ph_t) / ph_s', 1.0, 'u', 0.01, 'photo'), ('p2', '(w2 - ph_t) / ph_s', 5.0, 'u', 0.01, 'photo'),
             ('b', 'w2 - w1', 324, 'cm', 1, 'mesuré'), ('porte', 'w2 - w1', 204, 'cm', 1, 'supposé')]
    ctx = make_ctx(tmp_path, [('w1', 'x'), ('w2', 'x'), ('ph_s', 'échelle'), ('ph_t', 'photo')], photo)
    result = call(ctx, 'update', {})
    assert [(w['code'], w['param'], w['obs']) for w in result['warnings']] == [('echelle_fragile', 'ph_s', 'cotes:b')]
    call(ctx, 'cotes.add', {'id': 'c', 'expr': 'w2', 'value': 444, 'unit': 'cm', 'status': 'mesuré'})
    assert call(ctx, 'update', {})['warnings'] == []
    ctx.project.db.close()


def test_next_best_targets_conflicts(tmp_path):
    cotes = [('d1', 'w1', 120, 'cm', 1, 'mesuré'), ('d2', 'w2 - w1', 130, 'cm', 1, 'mesuré'), ('d3', 'w3 - w2', 60, 'cm', 1, 'mesuré'),
             ('d4', 'w3', 310, 'cm', 1, 'mesuré'), ('d5', 'w3 - w1', 230, 'cm', 1, 'mesuré'),
             ('d6', 'w4 - w3', 100, 'cm', 50, 'lu'), ('d7', 'w5 - w4', 100, 'cm', 50, 'lu')]
    ctx = make_ctx(tmp_path, [(f'w{k}', 'x') for k in range(1, 6)], cotes)
    call(ctx, 'update', {})
    best = call(ctx, 'next_best', {'k': 3})
    assert not any(n in best[0]['expr'] for n in ('w4', 'w5')) and {'w1', 'w3'} & set(best[0]['expr'].split(' - '))
    assert 'w5' in call(ctx, 'next_best', {'k': 1, 'targets': ['w5']})[0]['expr']
    call(ctx, 'cotes.add', {'id': 'd5', 'expr': 'w3 - w1', 'value': 190})
    call(ctx, 'update', {})
    assert any(n in call(ctx, 'next_best', {'k': 1})[0]['expr'] for n in ('w4', 'w5'))
    ctx.project.db.close()


def test_add_auto(tmp_path):
    from plans3d.server import load_plugins; load_plugins()
    ctx = Ctx(); call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'auto'})
    square = [[3, 46.5], [3.0001, 46.5], [3.0001, 46.5001], [3, 46.5]]
    (tmp_path / 'mesures.csv').write_text('id;expr;value;sigma;unit;status;label\n', encoding='utf-8')
    (tmp_path / 'grille.json').write_text(json.dumps({'nx': 2, 'nz': 2, 'pas': 1, 'origine': {'E': 7e5, 'N': 6.6e6}, 'valeurs': [0, 0, 0, 0]}))
    (tmp_path / 'parcelle.json').write_text(json.dumps({'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': {'numero': '1'}, 'geometry': {'type': 'Polygon', 'coordinates': [square]}}]}))
    (tmp_path / 'photos').mkdir(); (tmp_path / 'photos' / 'a.jpg').write_bytes(b'\xff\xd8\xff\xe0')
    (tmp_path / 'notes.txt').write_text('rien')
    added = [call(ctx, 'sources.add_auto', {'path': str(tmp_path / name)}) for name in ('mesures.csv', 'grille.json', 'parcelle.json', 'photos')]
    assert [a['method'] for a in added] == ['sources.add', 'terrain.add', 'cadastre.add', 'photos.add'] and added[0]['kind'] == 'cotes_csv'
    assert {s['kind'] for s in call(ctx, 'sources.list', {})} == {'cotes_csv', 'grille_alti', 'cadastre_geojson', 'photos_colmap'}
    with pytest.raises(RpcError, match='notes.txt : type de fichier non reconnu'): call(ctx, 'sources.add_auto', {'path': str(tmp_path / 'notes.txt')})
    ctx.project.db.close()
