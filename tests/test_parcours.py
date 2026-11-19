import itertools, json, os
import pytest
from plans3d.parcours import COPIED, FLOWS, OFFERED, PHOTOS, QUESTIONS, SOURCES, parcours
from plans3d.rpc import Ctx, call

COMBOS = [(t, set(c)) for t in FLOWS for n in range(len(OFFERED[t]) + 1) for c in itertools.combinations(OFFERED[t], n)]


def useful(route): return [s['id'] for s in route['steps'] if s['useful']]


@pytest.mark.parametrize('type,chosen', COMBOS, ids=[f"{t}-{'+'.join(sorted(c)) or 'rien'}" for t, c in COMBOS])
def test_every_combination(type, chosen):
    r = parcours(type, chosen | {'inconnue'})
    steps, on = [s['id'] for s in r['steps']], useful(r)
    assert steps == list(FLOWS[type]) and set(r['sources']) == chosen
    assert {'projet', 'sources', 'verifier', 'meubler'} <= set(on)
    for q in QUESTIONS: assert set(r['answers'][q]) == {s for s in chosen if q in SOURCES[s][1]}
    assert r['missing'] == [q for q in QUESTIONS if not r['answers'][q]] and r['ready'] == bool(r['answers']['forme'])
    assert ('positions' in on or 'facades' in on) == bool(chosen & set(PHOTOS))
    if type == 'maison': assert ('scan' in next(s['hint'] for s in r['steps'] if s['id'] == 'facades').lower()) == ('scan' in chosen)
    for step in ('cadastre', 'gabarit', 'relief'):
        if step in steps: assert (step in on) == (step in chosen)
    if r['ready']: assert {'murs', 'cadastre', 'gabarit', 'facades'} & set(on)
    else: assert not {'murs', 'echelle'} & set(on)
    sized = r['ready'] and not {'longueurs', 'echelle'} & set(r['missing'])
    if 'echelle' in steps: assert ('echelle' in on) == (r['ready'] and (not sized or set(r['answers']['echelle']) <= COPIED))
    assert len(r['todo']) == len(set(r['todo']))
    if r['ready'] and not r['missing'] and 'echelle' not in on: assert r['todo'] == []
    assert [o['id'] for o in r['offered']] == OFFERED[type]


def test_manual_only_has_no_photo_step_and_nothing_to_measure():
    for t in FLOWS:
        r = parcours(t, {'manuel'})
        assert not {'positions', 'facades', 'echelle', 'cadastre', 'gabarit'} & set(useful(r)) and 'murs' in useful(r) and r['todo'] == []
    assert parcours('piece', {'manuel'})['steps'][3]['title'] == 'Construire'


def test_sketch_only_lists_every_length():
    r = parcours('piece', {'croquis'}, walls=['A', 'B', 'C', 'D'])
    assert useful(r) == ['projet', 'sources', 'murs', 'echelle', 'verifier', 'meubler'] and set(r['missing']) == {'longueurs', 'echelle', 'hauteur'}
    assert r['todo'][0] == 'La longueur de chaque mur : A, B, C et D' and r['steps'][4]['title'] == 'Les longueurs'
    assert parcours('piece', {'croquis'})['todo'][0] == 'La longueur de chaque mur, au mètre'


def test_photos_only_walls_from_points_then_one_measure():
    r = parcours('piece', {'photos'})
    assert useful(r) == list(FLOWS['piece']) and r['todo'] == ["Une seule mesure au mètre pour l'échelle"]
    assert r['steps'][3]['hint'] == 'Valider ce qui est trouvé' and r['steps'][4]['title'] == "L'échelle"


def test_photos_and_sketch_keep_the_current_flow():
    r = parcours('piece', {'photos', 'croquis'})
    assert [s['id'] for s in r['steps']] == ['projet', 'sources', 'positions', 'murs', 'echelle', 'verifier', 'meubler'] and all(s['useful'] for s in r['steps'])
    assert r['todo'] == ["Une seule mesure au mètre pour l'échelle"]


def test_cadastre_gives_the_outer_outline():
    r = parcours('maison', {'cadastre'})
    assert useful(r) == ['projet', 'sources', 'cadastre', 'toit', 'verifier', 'meubler'] and r['answers']['forme'] == ['cadastre']
    assert r['todo'] == ['La hauteur des murs', 'Les portes et fenêtres : place, largeur et hauteur']
    assert 'murs' in useful(parcours('piece', {'cadastre'}))


def test_house_scan_goes_to_the_facades():
    r = parcours('maison', {'cadastre', 'scan'})
    assert useful(r) == ['projet', 'sources', 'cadastre', 'toit', 'facades', 'verifier', 'meubler'] and r['todo'] == []
    assert r['steps'][6]['hint'] == 'Scan au téléphone, recalé' and parcours('maison', {'scan', 'photos'})['steps'][6]['hint'] == 'Photos et scan recalés'
    alone = parcours('maison', {'scan'})
    assert alone['ready'] and 'facades' in useful(alone) and 'echelle' in useful(alone) and alone['todo'] == ["Une seule mesure au mètre pour l'échelle"]
    assert 'murs' in useful(parcours('maison', {'scan', 'croquis'})) and parcours('maison', {'cadastre'})['steps'][6]['hint'] == 'Photos recalées'


def test_written_plan_is_copied_not_measured():
    r = parcours('piece', {'plan'})
    assert 'echelle' in useful(r) and r['steps'][4]['title'] == 'Les cotes du plan' and r['todo'][0].startswith('Rien au mètre')
    assert 'echelle' not in useful(parcours('piece', {'plan', 'manuel'}))


def test_vehicle_template_and_photos_keep_the_recalage():
    r = parcours('vehicule', {'gabarit', 'photos'})
    assert useful(r) == ['projet', 'sources', 'gabarit', 'positions', 'murs', 'verifier', 'meubler'] and r['steps'][4]['title'] == 'Le recalage'
    assert 'murs' not in useful(parcours('vehicule', {'gabarit'}))


def test_route_detects_sources_of_old_projects(tmp_path):
    ctx = Ctx()
    assert call(ctx, 'guide.route', {'type': 'vehicule'})['sources'] == ['gabarit']
    call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'})
    assert call(ctx, 'guide.route', {'type': 'piece'})['sources'] == [] and call(ctx, 'guide.route', {'type': 'maison'})['sources'] == ['cadastre']
    meta = tmp_path / 'p' / 'project.json'
    json.dump({**json.load(open(meta, encoding='utf-8')), 'croquis': {'file': 'croquis.png'}, 'guide': {'manual': True}}, open(meta, 'w', encoding='utf-8'))
    ctx.project.db.execute("insert into sources values ('photos', ?, 'photos_colmap', '')", (os.fspath(tmp_path),)); ctx.project.db.commit()
    assert call(ctx, 'guide.route', {'type': 'piece'})['sources'] == ['photos', 'croquis', 'manuel']
    assert call(ctx, 'guide.route', {'type': 'piece', 'sources': ['plan'], 'walls': ['A']})['todo'][0].startswith('Rien au mètre')
