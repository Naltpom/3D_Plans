import json, os, stat
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from plans3d import photos
from test_groupes import setup


def write_text(folder, model, linked):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'cameras.txt'), 'w') as f:
        f.writelines(f'{i + 1} OPENCV {w} {h} {" ".join(map(str, k))}\n' for i, ((w, h), k) in enumerate(zip(model['wh'], model['K'])))
    with open(os.path.join(folder, 'images.txt'), 'w') as f:
        f.writelines(f'{i + 1} {" ".join(map(str, q))} {" ".join(map(str, t))} {i + 1} {name}\n\n' for i, (q, t, name) in enumerate(zip(model['q'], model['t'], linked)))
    with open(os.path.join(folder, 'points3D.txt'), 'w') as f:
        for k, p in enumerate(model['P']):
            track = ' '.join(f'{i + 1} 0' for i in model['Timg'][model['Tptr'][k]:model['Tptr'][k + 1]])
            f.write(f'{k + 1} {p[0]} {p[1]} {p[2]} 0 0 0 0.5 {track}\n')


def moved(model, scale=2.0, angle=0.7, shift=(1.0, -2.0, 3.0)):
    R = Rotation.from_euler('y', angle).as_matrix(); rotations = photos.rotation_matrices(model['q']) @ R.T
    return {**model, 'P': scale * model['P'] @ R.T + shift, 'q': np.roll(Rotation.from_matrix(rotations).as_quat(), 1, axis=1), 't': scale * model['t'] - rotations @ shift}


@pytest.fixture
def old(tmp_path, monkeypatch):
    ctx, project, model, _ = setup(tmp_path); cache = photos.source_cache(project, 'photos')[1]; linked = [f'{i:05d}.jpg' for i in range(len(model['names']))]
    write_text(os.path.join(cache, 'sparse', '0'), model, linked); photos.planes(ctx); os.makedirs(os.path.join(cache, 'images'))
    open(os.path.join(cache, 'images', linked[0]), 'w').close(); os.chmod(os.path.join(cache, 'images', linked[0]), stat.S_IREAD)
    project.db.execute("insert into photos_assoc values ('photos', 'p01', 'mur_x', null)"); project.db.commit()

    def reconstruct(paths, fresh, progress, emit):
        os.makedirs(fresh, exist_ok=True); new = moved(model); write_text(os.path.join(fresh, 'sparse', '0'), new, linked); np.savez(os.path.join(fresh, 'model.npz'), **new)
        json.dump({'root': str(tmp_path), 'image_dir': os.path.join(fresh, 'images'), 'registered': len(linked), 'model': '0'}, open(os.path.join(fresh, 'meta.json'), 'w'))
    monkeypatch.setattr(photos, 'reconstruct', reconstruct)
    return ctx, project, model, cache, linked


def test_restart_realigned_keeps_frame_and_links(old, monkeypatch):
    ctx, project, model, cache, linked = old; ids = [p['id'] for p in photos.planes(ctx)['planes']]

    def colmap(arguments, *rest, **options):
        if arguments[0] == 'model_aligner': write_text(arguments[4], model, linked)
    monkeypatch.setattr(photos, 'run_colmap', colmap)
    result = photos.extract(ctx, fresh=True)
    assert result['fresh'] and result['realigned_m'] < 1e-6 and not os.path.exists(cache + photos.FRESH)
    assert np.allclose(photos.load_model(cache)['P'], model['P']) and photos.load_json(os.path.join(cache, 'meta.json'))['image_dir'] == os.path.join(cache, 'images')
    assert [p['id'] for p in photos.planes(ctx)['planes']] == ids and project.db.execute('select count(*) from photos_assoc').fetchone()[0] == 1
    assert os.listdir(os.path.join(cache, 'images')) == [linked[0]] and os.listdir(os.path.dirname(cache)) == [os.path.basename(cache)]


def test_restart_new_frame_drops_links(old, monkeypatch):
    ctx, project, model, cache, _ = old

    def colmap(arguments, *rest, **options):
        if arguments[0] == 'model_aligner': raise ValueError('COLMAP model_aligner a échoué')
    monkeypatch.setattr(photos, 'run_colmap', colmap)
    assert photos.extract(ctx, fresh=True)['realigned_m'] is None
    assert np.allclose(photos.load_model(cache)['P'], moved(model)['P']) and project.db.execute('select count(*) from photos_assoc').fetchone()[0] == 0


def test_stopped_restart_keeps_old(old, monkeypatch):
    ctx, project, model, cache, _ = old

    def stopped(paths, fresh, progress, emit): raise photos.Stopped('calcul arrêté')
    monkeypatch.setattr(photos, 'reconstruct', stopped)
    assert photos.extract(ctx, fresh=True) == {'stopped': True}
    assert not os.path.exists(cache + photos.FRESH) and np.allclose(photos.load_model(cache)['P'], model['P']) and os.path.exists(os.path.join(cache, 'images'))
    assert project.db.execute('select count(*) from photos_assoc').fetchone()[0] == 1
