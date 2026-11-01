import base64, binascii, json, math, os
from datetime import datetime, timezone
from . import cadastre as K, furniture as F
from .rpc import RpcError, rpc

DEFAULT = {'lat': 48.8566, 'lon': 2.3522, 'source': 'défaut'}
J2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
PNG = b'\x89PNG\r\n\x1a\n'


def sun(lat, lon, when):
    d = (when - J2000).total_seconds() / 86400
    g = math.radians(357.529 + 0.98560028 * d); q = 280.459 + 0.98564736 * d
    ecliptic, tilt = math.radians(q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)), math.radians(23.439 - 3.6e-7 * d)
    ra = math.atan2(math.cos(tilt) * math.sin(ecliptic), math.cos(ecliptic)); dec = math.asin(math.sin(tilt) * math.sin(ecliptic))
    hour = math.radians((18.697374558 + 24.06570982441908 * d) % 24 * 15 + lon) - ra; la = math.radians(lat)
    alt = math.asin(math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(hour))
    az = math.atan2(-math.sin(hour), math.tan(dec) * math.cos(la) - math.sin(la) * math.cos(hour))
    return math.degrees(alt), math.degrees(az) % 360


def direction(alt, az):
    a, z = math.radians(alt), math.radians(az)
    return [math.cos(a) * math.sin(z), math.sin(a), -math.cos(a) * math.cos(z)]


def daylight(alt):
    s = math.sin(math.radians(alt))
    if alt <= 0: return 0.0, max(0.0, 1000 * (1 + alt / 6))
    return 120000 * math.exp(-0.18 / max(s, 0.04)), 1000 + 14000 * s ** 0.6


def meta_path(project): return os.path.join(project.folder, 'project.json')


def meta(project):
    try:
        with open(meta_path(project), encoding='utf-8') as fh: data = json.load(fh)
    except (OSError, ValueError): data = {}
    return data if isinstance(data, dict) else {}


def place(project):
    origin = K.load_cache(project.folder, 'origine')
    if origin:
        lon, lat = K.lambert_conic(origin['epsg'])[1](origin['E'], origin['N'])
        return {'lat': round(lat, 5), 'lon': round(lon, 5), 'source': 'cadastre'}
    saved = meta(project).get('lieu')
    try: return {'lat': F.bounded(saved['lat'], -90, 90, 'latitude'), 'lon': F.bounded(saved['lon'], -180, 180, 'longitude'), 'source': 'projet'}
    except (TypeError, KeyError, RpcError): return dict(DEFAULT)


@rpc('lumiere.lieu')
def lieu(ctx, lat=None, lon=None):
    project = ctx.require_project()
    if lat is not None or lon is not None:
        data = {**meta(project), 'lieu': {'lat': F.bounded(lat, -90, 90, 'latitude'), 'lon': F.bounded(lon, -180, 180, 'longitude')}}
        with open(meta_path(project), 'w', encoding='utf-8') as fh: json.dump(data, fh, ensure_ascii=False, indent=1)
    return place(project)


@rpc('lumiere.soleil')
def soleil(ctx, quand):
    try: when = datetime.fromisoformat(str(quand).replace('Z', '+00:00'))
    except ValueError: raise RpcError(f'date invalide : {quand}') from None
    if when.tzinfo is None: raise RpcError('date : fuseau horaire attendu (ISO 8601, ex. 2025-06-21T16:00:00Z)')
    where = place(ctx.require_project()); alt, az = sun(where['lat'], where['lon'], when); direct, diffus = daylight(alt)
    return {**where, 'hauteur': round(alt, 3), 'azimut': round(az, 3), 'direction': direction(alt, az), 'direct': round(direct, 1), 'diffus': round(diffus, 1)}


@rpc('lumiere.export')
def export(ctx, path, data):
    try: raw = base64.b64decode(str(data).split(',')[-1], validate=True)
    except (binascii.Error, ValueError): raw = b''
    if not raw.startswith(PNG): raise RpcError('export : image PNG attendue')
    path = os.path.abspath(str(path)); path += '' if path.lower().endswith('.png') else '.png'
    with open(path, 'wb') as fh: fh.write(raw)
    return {'path': path, 'size': len(raw)}
