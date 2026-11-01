import json, os
import numpy as np
import pytest
from plans3d import Project, cadastre, terrain
from plans3d import components as C
from plans3d.rpc import Ctx, call

EAST, NORTH = 700000.0, 6600000.0
FERME = os.environ.get('PLANS3D_GRILLE_TEST', '')  # grille réelle facultative (chemin local)

def write_json(path, values, step, origin, **extra):
    values = np.asarray(values, float)
    json.dump({'nx': values.shape[1], 'nz': values.shape[0], 'pas': step, 'origine': origin, 'valeurs': values.ravel().tolist(), **extra}, open(path, 'w'))
    return str(path)


def write_asc(path, values, step, west, south, nodata=-9999):
    values = np.where(np.isnan(values), nodata, values)
    head = f'ncols {values.shape[1]}\nnrows {values.shape[0]}\nxllcorner {west}\nyllcorner {south}\ncellsize {step}\nNODATA_value {nodata}\n'
    open(path, 'w').write(head + '\n'.join(' '.join(f'{v:.3f}' for v in row) for row in values))
    return str(path)


def solved(project):
    result = project.update(); values = dict(zip(result.names, result.x))
    return result, values, lambda expr: C.evaluate(expr, values)


def normals_up(mesh):
    v, f = np.array(mesh['vertices']), np.array(mesh['faces'])
    return np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])[:, 1].min() > 0


def test_plan_incline(tmp_path):
    j, i = np.mgrid[0:5, 0:6]; plane = 100 + 0.1 * i + 0.2 * j
    project = Project(str(tmp_path / 'p'))
    info = terrain.add(project, write_json(tmp_path / 'plan.json', plane, 2.0, {'E': EAST, 'N': NORTH, 'epsg': 2154}))
    origin = cadastre.load_cache(project.folder, 'origine')
    assert (info['nx'], info['nz'], info['alt0'], origin['E'], origin['N']) == (6, 5, 100.0, EAST, NORTH)
    assert np.allclose(info['pas'], 2.0 / origin['k'])
    result, values, value = solved(project)
    assert len(result.obs) == 30 and {o.status for o in result.obs} == {'lu'} and np.allclose(result.sigma, 0.2)
    assert np.allclose([values[f'terrain_y_{a}_{b}'] for a, b in zip(i.ravel(), j.ravel())], (plane - 100).ravel())
    comp = C.load(project)['terrain']; px, pz = info['pas']
    assert value(C.anchor(comp, '3.3, 4.1')['y']) == pytest.approx(0.1 * 3.3 / px + 0.2 * 4.1 / pz)
    assert value(C.anchor(comp, 'coin_11')['y']) == pytest.approx(0.1 * 5 + 0.2 * 4)
    C.add(project, 'boite', 'b1', {'x0': 3.0, 'z0': 4.0})
    assert value(C.measure_expr((C.load(project)['b1'], 'bas'), (comp, '3,4'), 'y')) == pytest.approx(0.1 * 3 / px + 0.2 * 4 / pz)
    mesh = next(s for s in C.geometry(project) if s['type'] == 'terrain')
    assert mesh['kind'] == 'mesh' and len(mesh['vertices']) == 30 and len(mesh['faces']) == 40 and normals_up(mesh)


def test_colline_sous_echantillonnee_asc_tiff(tmp_path):
    j, i = np.mgrid[0:61, 0:81]; hill = 50 + 8 * np.exp(-((i - 40) ** 2 + (j - 32) ** 2) / 200); hill[0, 0] = np.nan
    asc = terrain.read_grid(write_asc(tmp_path / 'colline.asc', hill, 1.0, EAST - 0.5, NORTH - 60.5))
    assert (asc['E'], asc['N'], asc['dx'], asc['epsg']) == (EAST, NORTH, 1.0, 2154) and np.isnan(asc['y'][0, 0])
    Image, ifd = pytest.importorskip('PIL.Image'), pytest.importorskip('PIL.TiffImagePlugin').ImageFileDirectory_v2()
    for tag, value, kind in ((33550, (1.0, 1.0, 0.0), 12), (33922, (0.0, 0.0, 0.0, EAST - 0.5, NORTH + 0.5, 0.0), 12),
                             (34735, (1, 1, 0, 1, 3072, 0, 1, 2154), 3), (42113, '-9999', 2)):
        ifd[tag] = value; ifd.tagtype[tag] = kind
    Image.fromarray(np.where(np.isnan(hill), -9999, hill).astype(np.float32)).save(tmp_path / 'colline.tif', tiffinfo=ifd, compression='tiff_lzw')
    tif = terrain.read_grid(str(tmp_path / 'colline.tif'))
    assert all(tif[k] == asc[k] for k in ('E', 'N', 'dx', 'dy', 'epsg')) and np.allclose(tif['y'], asc['y'], atol=1e-3, equal_nan=True)
    project = Project(str(tmp_path / 'p'))
    info = terrain.add(project, str(tmp_path / 'colline.asc'), max_nodes=500, alt0=50)
    assert info['noeuds'] <= 500 and info['nx'] > 15 and info['pas'][0] > 1 and info['altitudes'] == [50.0, pytest.approx(58)]
    result, values, value = solved(project)
    assert len(result.obs) == info['noeuds'] - 1 and result.value('terrain_y_0_0')[1] == np.inf
    top = value(C.anchor(C.load(project)['terrain'], '40,32')['y'])
    assert top == pytest.approx(8, abs=0.1)


def test_rpc_deux_sources(tmp_path):
    ctx = Ctx(); call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 't'})
    j, i = np.mgrid[0:4, 0:5]; base = 200 + i - j
    first = write_json(tmp_path / 'a.json', base, [2.0, 3.0], {'lon': 0.79, 'lat': 44.71})
    info = call(ctx, 'terrain.add', {'path': first})
    assert (info['nx'], info['nz'], info['alt0']) == (5, 4, 200.0) and np.allclose(info['pas'], [2.0, 3.0], rtol=1e-9)
    call(ctx, 'terrain.add', {'path': write_json(tmp_path / 'b.json', base + 0.2, [2.0, 3.0], {'lon': 0.79, 'lat': 44.71}, sigma=0.2)})
    result = call(ctx, 'update', {})
    assert len(result['obs']) == 40 and {o['source'] for o in result['obs']} == {'a', 'b'}
    mesh = call(ctx, 'terrain.geometry', {})[0]
    assert np.allclose(np.array(mesh['vertices'])[:, 1], (base - 200 + 0.1).ravel(), atol=1e-6) and mesh['alt0'] == 200.0


@pytest.mark.skipif(not os.path.exists(FERME), reason='grille de la ferme absente')
def test_grille_ferme(tmp_path):
    text = open(FERME, encoding='utf-8').read(); cad = json.JSONDecoder().raw_decode(text[text.index('{'):])[0]
    values = np.reshape(cad['z'], (cad['ny'], cad['nx']))
    path = write_json(tmp_path / 'ferme.json', values, [cad['w'] / (cad['nx'] - 1), cad['h'] / (cad['ny'] - 1)], cad['origine'])
    project = Project(str(tmp_path / 'p')); info = terrain.add(project, path)
    assert info['noeuds'] == 2464 and info['altitudes'] == [cad['zmin'], cad['zmax']]
    result, values, value = solved(project)
    mesh = next(s for s in C.geometry(project, result) if s['type'] == 'terrain'); y = np.array(mesh['vertices'])[:, 1] + mesh['alt0']
    assert len(result.obs) == 2464 and (y.min(), y.max()) == (pytest.approx(cad['zmin']), pytest.approx(cad['zmax'])) and normals_up(mesh)
