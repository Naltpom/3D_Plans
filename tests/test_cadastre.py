import json, math
import numpy as np
import pytest
from plans3d import Project, cadastre
from plans3d.rpc import METHODS, Ctx

pymupdf = pytest.importorskip('pymupdf')
POINTS_PER_MM = 72 / 25.4
EAST, NORTH = 1700112.5, 8200030.0
PARCEL = [(EAST, NORTH + 10), (EAST + 20, NORTH + 10), (EAST + 20, NORTH), (EAST, NORTH)]
HOUSE = [(EAST + 4, NORTH + 7), (EAST + 12, NORTH + 7), (EAST + 12, NORTH + 1), (EAST + 4, NORTH + 1)]
SHED = [(EAST + 15, NORTH + 9), (EAST + 18, NORTH + 9), (EAST + 18, NORTH + 7), (EAST + 15, NORTH + 7)]
NEIGHBOUR = [(EAST - 100, NORTH + 15), (EAST - 1, NORTH + 15), (EAST - 1, NORTH), (EAST - 100, NORTH)]


def make_pdf(path, scale_x, scale_y, cross=None):
    document = pymupdf.open(); page = document.new_page(width=595, height=842)
    page.insert_text((40, 60), "Échelle d'édition : 1/1000", fontsize=8)
    page.insert_text((40, 72), 'Coordonnées en projection : RGF93CC49', fontsize=8)
    page.draw_rect(pymupdf.Rect(60, 200, 535, 800), color=(0, 0, 0), width=0.8)
    to_page = lambda e, n: pymupdf.Point(297.5 + (e - EAST - 10) * 1000 / scale_x * POINTS_PER_MM, 500 - (n - NORTH - 5) * 1000 / scale_y * POINTS_PER_MM)
    for ring, fill in ((HOUSE, (1, .8, .2)), (SHED, (1, .9, .6)), (PARCEL, None), (NEIGHBOUR, None)):
        points = [to_page(*p) for p in ring]
        page.draw_polyline(points + points[:1], color=None if fill else (0, 0, 0), fill=fill, width=0.5)
    if cross:
        page.draw_line(to_page(cross[0] - 5, cross[1]), to_page(cross[0] + 5, cross[1]), width=0.8)
        page.draw_line(to_page(cross[0], cross[1] - 5), to_page(cross[0], cross[1] + 5), width=0.8)
    document.save(path); return path


def building(ring, height): return {'type': 'Feature', 'properties': {'hauteur': height, 'construction_legere': False, 'precision_planimetrique': 1.5},
                                    'geometry': {'type': 'MultiPolygon', 'coordinates': [[ring + ring[:1]]]}}


def make_geojson(path, parcel=True, buildings=(HOUSE,)):
    features = [building(ring, 6.0) for ring in buildings]
    if parcel: features.insert(0, {'type': 'Feature', 'properties': {'numero': '0001', 'contenance': 200},
                                   'geometry': {'type': 'Polygon', 'coordinates': [PARCEL + PARCEL[:1]]}})
    crs = {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:EPSG::3949'}}
    json.dump({'type': 'FeatureCollection', 'crs': crs, 'features': features}, open(path, 'w', encoding='utf-8'))
    return path


def find(data, kind, principal=True): return next(e for e in data['entites'] if e['type'] == kind and e['principal'] == principal)


def test_projections():
    assert np.allclose(cadastre.lambert_conic(2154)[0](3, 46.5), (7e5, 6.6e6))
    assert np.allclose(cadastre.lambert_conic(3949)[0](3, 49), (1.7e6, 8.2e6))
    cc49 = cadastre.reproject([(1.9, 48.7)], 4326, 3949)
    assert np.allclose(cadastre.reproject(cc49, 3949, 4326)[0], (1.9, 48.7), atol=1e-10)
    assert math.dist(cadastre.reproject([(1.9, 48.7)], 4326, 2154)[0], cadastre.reproject(cc49, 3949, 2154)[0]) < 1e-4


def test_pdf_stated_scale(tmp_path):
    observations, params, data = cadastre.extract(make_pdf(str(tmp_path / 'plan.pdf'), 1000, 1000), 'plan')
    parcel = find(data, 'parcelle')
    assert data['echelle_source'] == 'texte' and data['epsg'] == 3949 and not data['georeference'] and not data['avertissements']
    assert tuple(parcel['pts'][0]) == (0, 0) and max(z for _, z in parcel['pts']) == pytest.approx(10, abs=0.02)
    assert abs(cadastre.area(parcel['pts']) - 200) < 0.3 and abs(np.hypot(*np.diff(parcel['pts'] + parcel['pts'][:1], axis=0).T).sum() - 60) < 0.05
    house, shed = find(data, 'bati'), find(data, 'bati', False)
    assert (house['nature'], shed['nature']) == ('dur', 'léger')
    assert abs(cadastre.area(house['pts']) - 48) < 0.2 and abs(cadastre.area(shed['pts']) - 6) < 0.1
    neighbour = find(data, 'parcelle', False)
    assert neighbour['partiel'] and sum(o.id.startswith(f"plan:{neighbour['id']}_") for o in observations) == 4
    assert all(o.status == 'lu' and o.sigma == cadastre.SIGMA_COORDINATE and o.expr in params for o in observations)
    assert 'plan_p0_0_x' in params and len(observations) == 2 * (4 + 4 + 4 + 2)


def test_stretched_pdf_georeferenced_on_geojson(tmp_path):
    project = Project(str(tmp_path / 'projet'))
    added_geojson = cadastre.add_source(project, make_geojson(str(tmp_path / 'parcelle.geojson')))
    added_pdf = cadastre.add_source(project, make_pdf(str(tmp_path / 'plan.pdf'), 250, 200, cross=(1700100, 8200000)))
    assert added_geojson['georeference'] and added_pdf['georeference'] and 'contredite' in added_pdf['avertissements'][0]
    pdf = cadastre.entities(project.folder, added_pdf['id'])
    assert np.allclose(pdf['echelle'], (250, 200), rtol=2e-3) and pdf['origine']['epsg'] == 3949 and pdf['origine']['E'] == EAST
    reference, extracted = find(cadastre.entities(project.folder, 'parcelle'), 'parcelle'), find(pdf, 'parcelle')
    assert max(min(math.dist(a, b) for b in reference['pts']) for a in extracted['pts']) < 0.02
    result = project.update(); shapes = cadastre.geometry(project.folder, result)
    house = next(e for e in shapes['entites'] if e['source'] == 'parcelle' and e['type'] == 'bati')
    assert house['principal'] and house['h'] == pytest.approx(6.0)
    assert result.value('parcelle_b0_h')[1] == pytest.approx(0.5, rel=1e-3) and result.value('parcelle_b0_0_x')[1] == pytest.approx(1.5, rel=1e-3)
    assert {e['source'] for e in shapes['entites']} == {'parcelle', 'plan'} and shapes['origine']['source'] == 'parcelle'


def test_neighbour_buildings_display_only(tmp_path):
    far = [(x + 40, y) for x, y in HOUSE]; adjoining = [(EAST + 20, NORTH + 6), (EAST + 26, NORTH + 6), (EAST + 26, NORTH + 2), (EAST + 20, NORTH + 2)]
    project = Project(str(tmp_path / 'projet'))
    added = cadastre.add_source(project, make_geojson(str(tmp_path / 'batiments.geojson'), parcel=False, buildings=(HOUSE, far, adjoining)))
    assert added['parametres'] == 0
    cadastre.add_source(project, make_geojson(str(tmp_path / 'parcelle.geojson'), buildings=()))
    names = set(project.params())
    assert {'batiments_b0_h', 'batiments_b2_h'} <= names and not any(n.startswith('batiments_b1_') for n in names)
    observed = {e['id']: e['observe'] for e in cadastre.geometry(project.folder)['entites'] if e['source'] == 'batiments'}
    assert observed == {'b0': True, 'b1': False, 'b2': True}
    result = project.update()
    assert result.value('batiments_b0_0_x')[1] == pytest.approx(1.5, rel=1e-3) and len(result.obs) == 8 + 1 + 2 * 9


def test_geojson_wgs84_to_lambert93(tmp_path):
    square = [(3.0, 46.5), (3.0001, 46.5), (3.0001, 46.5001), (3.0, 46.5001)]
    path = tmp_path / 'p.geojson'
    json.dump({'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'properties': {'idu': '00000000ZZ0001'},
                                                          'geometry': {'type': 'Polygon', 'coordinates': [square + square[:1]]}}]}, open(path, 'w'))
    project = Project(str(tmp_path / 'pj')); cadastre.add_source(project, str(path))
    data = cadastre.entities(project.folder, 'p'); origin = data['origine']
    assert origin['epsg'] == 2154 and np.allclose((origin['E'], origin['N']), (7e5, 6.6e6)) and origin['k'] == pytest.approx(0.99905, abs=1e-5)
    assert data['entites'][0]['label'] == 'ZZ0001'
    x, z = np.array(data['entites'][0]['pts']).T
    phi, e2 = math.radians(46.5), 0.00669438
    w = math.sqrt(1 - e2 * math.sin(phi) ** 2)
    assert x[1] == pytest.approx(math.radians(1e-4) * 6378137 * math.cos(phi) / w, rel=1e-5)
    assert z[2] == pytest.approx(-math.radians(1e-4) * 6378137 * (1 - e2) / w ** 3, rel=1e-5)


def test_etalab_buildings_and_contenance(tmp_path):
    rect = lambda lon, lat, w, h: [(lon, lat), (lon + w, lat), (lon + w, lat + h), (lon, lat + h)]
    feature = lambda props, ring: {'type': 'Feature', 'properties': props, 'geometry': {'type': 'Polygon', 'coordinates': [ring + ring[:1]]}}
    features = [feature({'section': 'AB', 'numero': f'000{i}', 'contenance': c}, rect(0.8 + i * 4e-4, 44.7, 4e-4, 3e-4)) for i, c in enumerate((1057, 1200))]
    features += [feature({'type': t}, rect(lon, lat, 5e-5, 5e-5)) for t, lon, lat in (('01', 0.8, 44.7), ('02', 0.8002, 44.7001))]
    path = tmp_path / 'etalab.geojson'; json.dump({'type': 'FeatureCollection', 'features': features}, open(path, 'w'))
    project = Project(str(tmp_path / 'pj')); added = cadastre.add_source(project, str(path))
    data = cadastre.entities(project.folder, added['id'])
    assert added['entites'] == 4 and [e['nature'] for e in data['entites'] if e['type'] == 'bati'] == ['dur', 'léger']
    assert [e['label'] for e in data['entites'][:2]] == ['AB0000', 'AB0001'] and added['parametres'] == 2 * (4 + 2 + 3 + 4)
    assert data['entites'][1]['refs'] == ['p0_1', 'p1_1', 'p1_2', 'p0_2'] and data['entites'][2]['refs'][0] == 'p0_0'
    result = project.update(); areas = [o for o in result.obs if o.id.endswith('_aire')]
    assert [(o.value, o.sigma, o.status) for o in areas] == [(1057, 10.57, 'lu'), (1200, 12, 'lu')]
    assert result.predict(areas[0].expr)[0] == pytest.approx(cadastre.area(data['entites'][0]['pts']), abs=1)
    assert [o.id for group in result.conflicts() for o, _ in group] == ['etalab:p1_aire']


def test_rpc(tmp_path):
    ctx = Ctx(project=Project(str(tmp_path / 'pj')))
    added = METHODS['cadastre.add'](ctx, path=make_geojson(str(tmp_path / 'g.geojson')))
    assert added['entites'] == 2 and len(METHODS['cadastre.geometry'](ctx)['entites']) == 2
    assert METHODS['cadastre.entities'](ctx, source=added['id'])['kind'] == 'cadastre_geojson'
    with pytest.raises(ValueError): METHODS['cadastre.geometry'](Ctx())
