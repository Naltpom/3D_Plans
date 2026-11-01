import base64, json, math
from datetime import datetime, timezone
import pytest
from plans3d import cadastre as K, furniture as F, lumiere as Lu
from plans3d.rpc import Ctx, RpcError, call
from test_components import build_room

REFERENCE = [(48.8566, 2.3522, '2025-06-21T16:00', 35.730, 265.485), (48.8566, 2.3522, '2025-12-21T11:00', 16.887, 168.326),
             (-33.8688, 151.2093, '2025-01-15T02:00', 77.192, 4.708), (40.7128, -74.006, '2025-03-20T17:00', 49.410, 178.735),
             (35.6762, 139.6503, '2026-09-01T06:30', 31.326, 256.690), (48.794, 1.964, '2026-11-01T14:00', 19.048, 217.085)]


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setattr(F, 'FOLDER', str(tmp_path / 'catalog'))
    return Ctx(project=build_room(str(tmp_path / 'p')))


def test_lumens_and_kelvins():
    assert F.kelvin_rgb(6600) == (255, 255, 255) and F.kelvin_rgb(2700) == (255, 167, 87) and F.kelvin_rgb(1000)[2] == 0
    assert F.kelvin_rgb(10000)[2] == 255 and F.kelvin_rgb(10000)[0] < 210 and F.kelvin_hex(4000) == '#ffcea6'
    assert [F.kelvin_rgb(k)[2] for k in (2000, 3000, 4000, 5000)] == sorted(F.kelvin_rgb(k)[2] for k in (2000, 3000, 4000, 5000))
    assert F.candela(806) == pytest.approx(806 / (4 * math.pi)) and F.candela(806, True) == pytest.approx(806 / (2 * math.pi * (1 - math.cos(math.radians(30)))))


def test_lamp_lighting_and_switch(ctx):
    lamp = call(ctx, 'furniture.add', {'item': 'gabarit:lampadaire', 'x': 1.0, 'z': 1.0, 'options': {'lumens': 1500, 'kelvin': 4000, 'spot': 1}})['object']
    light = next(s['light'] for s in call(ctx, 'furniture.geometry', {}) if 'light' in s)
    assert light | {'luminance': 0} == {'lumens': 1500.0, 'kelvin': 4000.0, 'spot': True, 'allume': True, 'faisceau': 60, 'couleur': '#ffcea6',
                                        'candela': round(F.candela(1500, True), 3), 'luminance': 0} and light['luminance'] > 0
    call(ctx, 'furniture.options', {'id': lamp['id'], 'options': {'allume': 0}})
    assert next(s['light'] for s in call(ctx, 'furniture.geometry', {}) if 'light' in s)['allume'] is False
    assert F.get(ctx.project, lamp['id'])['params'] == {'lumens': 1500.0, 'kelvin': 4000.0, 'spot': 1, 'allume': 0}
    for options in ({'lumens': -1}, {'kelvin': 20000}, {'allume': 2}, {'teinte': '#ffffff'}):
        with pytest.raises(RpcError): call(ctx, 'furniture.options', {'id': lamp['id'], 'options': options})
    table = call(ctx, 'furniture.add', {'item': 'gabarit:table', 'size': [1.2, 0.75, 1.2], 'label': 'Table ronde', 'options': {'round': 1, 'couleur': '#B08A5A'}})['object']
    assert table['size'] == [1.2, 0.75, 1.2] and table['label'] == 'Table ronde'
    assert {s['color'] for s in call(ctx, 'furniture.geometry', {}) if s['id'] == table['ref']} == {'#b08a5a'}
    with pytest.raises(RpcError, match='hors bornes'): call(ctx, 'furniture.add', {'item': 'gabarit:table', 'size': [6, 0.75, 1]})


@pytest.mark.parametrize('lat, lon, utc, altitude, azimuth', REFERENCE)
def test_sun_matches_reference(lat, lon, utc, altitude, azimuth):
    alt, az = Lu.sun(lat, lon, datetime.fromisoformat(utc).replace(tzinfo=timezone.utc))
    assert abs(alt - altitude) < 1 and abs((az - azimuth + 180) % 360 - 180) < 1


def test_place_and_sun(ctx):
    assert call(ctx, 'lumiere.lieu', {}) == Lu.DEFAULT
    assert call(ctx, 'lumiere.lieu', {'lat': 45.75, 'lon': 4.85}) == {'lat': 45.75, 'lon': 4.85, 'source': 'projet'}
    assert json.load(open(Lu.meta_path(ctx.project), encoding='utf-8'))['lieu'] == {'lat': 45.75, 'lon': 4.85}
    for bad in ({'lat': 91, 'lon': 0}, {'lat': 'nord', 'lon': 0}, {'lat': 10}):
        with pytest.raises(RpcError): call(ctx, 'lumiere.lieu', bad)
    east, north = K.lambert_conic(2154)[0](1.964, 48.794)
    K.save_cache(ctx.project.folder, 'origine', {'E': east, 'N': north, 'epsg': 2154})
    assert call(ctx, 'lumiere.lieu', {}) == {'lat': 48.794, 'lon': 1.964, 'source': 'cadastre'}
    day, night = (call(ctx, 'lumiere.soleil', {'quand': t}) for t in ('2026-06-21T16:00:00Z', '2026-06-21T23:00:00+02:00'))
    assert day['source'] == 'cadastre' and 30 < day['hauteur'] < 40 and 260 < day['azimut'] < 275 and day['direct'] > 50000 and day['diffus'] > 5000
    assert math.isclose(math.hypot(*day['direction']), 1) and day['direction'][0] < 0 and day['direction'][1] > 0
    assert night['hauteur'] < 0 and night['direct'] == 0
    for bad in ('2026-06-21T16:00', 'demain'):
        with pytest.raises(RpcError): call(ctx, 'lumiere.soleil', {'quand': bad})


def test_export_png(ctx, tmp_path):
    png = b'\x89PNG\r\n\x1a\n' + b'0' * 10
    out = call(ctx, 'lumiere.export', {'path': str(tmp_path / 'rendu'), 'data': 'data:image/png;base64,' + base64.b64encode(png).decode()})
    assert out == {'path': str(tmp_path / 'rendu.png'), 'size': len(png)} and open(out['path'], 'rb').read() == png
    for data in ('data:image/png;base64,' + base64.b64encode(b'GIF89a').decode(), 'pas du base64 !'):
        with pytest.raises(RpcError, match='PNG'): call(ctx, 'lumiere.export', {'path': str(tmp_path / 'x.png'), 'data': data})
