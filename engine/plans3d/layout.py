import json, math, time
import numpy as np
from . import cadastre, components as C, furniture as F
from .rpc import RpcError, rpc

SCHEMA = 'create table if not exists layout_rules(id integer primary key, rules text)'
RULES = {'contre_mur': 'contre un mur', 'face_a': 'face à', 'pres_de': 'près de', 'loin_de': 'loin de', 'aligne': 'alignés', 'centre': 'au centre', 'orientation': 'orienté vers'}
TYPES = {'fenetre': 'une fenêtre', 'porte': 'une porte', 'porte_fenetre': 'une porte-fenêtre', 'mur': 'un mur', 'escalier': 'l’escalier', 'boite': 'une boîte'}
HARD, CONTACT, EPS, COS10 = 50.0, 0.05, 0.005, math.cos(math.radians(10))
CHAINS, STEPS, DISTINCT = 40, 2000, 0.5
I, J = np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])


def m(v): return f'{v:.1f}'.replace('.', ',') + ' m'
def cm(v): return f'{round(v * 100)} cm'
def de(label): return f'd’{label}' if label[:1].lower() in 'aeiouyéèêh' else f'de {label}'
def deg(cos): return math.degrees(math.acos(max(-1.0, min(1.0, float(cos)))))


def obb(points):
    (x0, z0), (x1, z1) = points[0], points[1]; a = math.atan2(z0 - z1, x1 - x0); c, s = math.cos(a), math.sin(a)
    us, vs = [x * c - z * s for x, z in points], [x * s + z * c for x, z in points]
    cu, cv = (max(us) + min(us)) / 2, (max(vs) + min(vs)) / 2
    return cu * c + cv * s, cv * c - cu * s, a, (max(us) - min(us)) / 2, (max(vs) - min(vs)) / 2


def separation(a, b):
    ax, az, ac, as_, ahu, ahv = a; bx, bz, bc, bs, bhu, bhv = b
    kx, kz = np.stack(np.broadcast_arrays(ac, as_, bc, bs)), np.stack(np.broadcast_arrays(-as_, ac, -bs, bc))
    def reach(c, s, hu, hv): return hu * np.abs(c * kx - s * kz) + hv * np.abs(s * kx + c * kz)
    along = (bx - ax) * kx + (bz - az) * kz
    return np.abs(along) - reach(ac, as_, ahu, ahv) - reach(bc, bs, bhu, bhv), along, kx, kz


def pair(a, b):
    sep = separation(a, b)[0]; out = np.maximum(sep, 0)
    return np.maximum(-sep.max(0), 0), np.maximum(np.hypot(out[0], out[1]), np.hypot(out[2], out[3]))


def to_rect(px, pz, b):
    bx, bz, bc, bs, hu, hv = b; dx, dz = px - bx, pz - bz
    return np.hypot(np.maximum(np.abs(dx * bc - dz * bs) - hu, 0), np.maximum(np.abs(dx * bs + dz * bc) - hv, 0))


def facing(a, b):
    dx, dz = b[0] - a[0], b[1] - a[1]; d = np.hypot(dx, dz) + 1e-9
    return (a[3] * dx + a[2] * dz) / d, d


def best(results):
    if len(results) == 1: return results[0]
    k = np.argmin(np.stack(np.broadcast_arrays(*[r[0] for r in results])), 0)
    return tuple(np.stack(np.broadcast_arrays(*column))[k, np.arange(k.size)] for column in zip(*results))


def angle(rule):
    v = rule.get('vers') if isinstance(rule, dict) and rule.get('type') == 'orientation' else None
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def centroid(points):
    x, z = np.asarray(points, float).T; cross = x * np.roll(z, -1) - np.roll(x, -1) * z; a = cross.sum() / 2
    if abs(a) < 1e-9: return float(x.mean()), float(z.mean())
    return float(((x + np.roll(x, -1)) * cross).sum() / (6 * a)), float(((z + np.roll(z, -1)) * cross).sum() / (6 * a))


class Scene:
    def __init__(self, project, result, ids=None):
        self.shapes, self.objects = C.geometry(project, result), F.load(project)
        known = {o['id'] for o in self.objects}; ids = known if ids is None else {int(i) for i in ids}
        if ids - known: raise RpcError(f"objet inconnu : {', '.join(map(str, sorted(ids - known)))}")
        self.movable = [o for o in self.objects if o['id'] in ids]
        if not self.movable: raise RpcError('aucun objet à agencer')
        labels = [o['label'] for o in self.objects]
        rows, self.refs, self.pieces, self.names, self.walls, self.polygons = [], {}, {}, [], [], {}
        def add(x, z, a, hu, hv, y0, y1, role, walking=False, keys=(), name='', front=False):
            for key in keys: self.refs.setdefault(str(key), []).append(len(rows))
            rows.append((x, z, a, hu, hv, y0, y1, role, walking, front)); self.names.append(name)
        for o in self.movable + [o for o in self.objects if o['id'] not in ids]:
            _, y0, y1 = F.outline(o); key = str(o['id'])
            self.pieces[key] = self.pieces[f"meuble-{key}"] = len(rows)
            add(o['x'], o['z'], math.radians(o['rot']), o['size'][0] / 2, o['size'][2] / 2, y0, y1, 'meuble', y0 < F.WALKING and o['kind'] not in F.LIGHTS,
                (key, f'meuble-{key}'), o['label'] if labels.count(o['label']) == 1 else f"{o['label']} {key}", True)
        for s in self.shapes:
            if s['kind'] == 'box' and s['type'] in F.FIXED:
                if s['type'] == 'mur': self.walls.append(len(rows))
                poly, y0, y1 = F.fixed_part(s); add(*obb(poly), y0, y1, 'fixe', True, (s['id'], s['type']), s['label'])
            elif s['kind'] == 'box' and s['type'] in F.DOORS + ('fenetre',): add(*obb(C.footprint(s)), 0, 0, 'cible', keys=(s['id'], s['type']), name=s['label'])
            elif s['kind'] == 'polygon' and len(s['vertices']) >= 3: self.polygons[s['id']] = (s['type'], s['label'], s['vertices'])
        for _, _, zones in F.openings(self.shapes):
            for poly, y0, y1 in zones: add(*obb(poly), y0, y1, 'zone')
        x, z, a, self.hu, self.hv, y0, y1, role, walking, self.front = (np.array(v) for v in zip(*rows))
        self.x0, self.z0, self.a0, self.n, n = x, z, a, len(self.movable), len(rows)
        over = (y0[None, :] < y1[:self.n, None] - 1e-3) & (y0[:self.n, None] < y1[None, :] - 1e-3)
        other = np.arange(n)[None, :] != np.arange(self.n)[:, None]
        self.solid = over & other & np.isin(role, ('meuble', 'fixe'))[None, :]
        self.zone = over & (role == 'zone')[None, :]
        self.near = walking[:self.n, None] & other & ((role == 'fixe') | (role == 'meuble') & walking)[None, :]
        self.block = self.solid | self.zone; self.cols = [np.flatnonzero(b | c) for b, c in zip(self.block, self.near)]
        floors = [v for t, _, v in self.polygons.values() if t == 'sol']
        if floors: self.room = max(floors, key=cadastre.area)
        elif self.walls:
            xs, zs = zip(*[p for s in self.shapes if s['type'] == 'mur' and s['kind'] == 'box' for p in C.footprint(s)])
            self.room = [[min(xs), min(zs)], [max(xs), min(zs)], [max(xs), max(zs)], [min(xs), max(zs)]]
        else: raise RpcError('aucune pièce : ajoutez des murs ou un sol')
        start, end = np.array(self.room, float), np.roll(np.array(self.room, float), -1, 0)
        ex, ez = (end - start).T; flat = np.abs(ez) < 1e-12
        self.edges = start[:, 0], start[:, 1], ex, ez, 1 / np.maximum(ex * ex + ez * ez, 1e-12), np.where(flat, 0, ex / np.where(flat, 1, ez))
        self.low, self.high = start.min(0), start.max(0)
        bases = {round(math.degrees(a[w]) % 90, 3) for w in self.walls} | {0.0}
        self.rots = np.radians(sorted({(b + 90 * k) % 360 for b in bases for k in range(4)}))
        self.rules, self.fixed = [], {}

    def state(self, chains):
        return [np.tile(v, (chains, 1)) for v in (self.x0, self.z0, np.cos(self.a0), np.sin(self.a0))]

    def getter(self, X, Z, Cs, Sn, j=-1, new=()):
        return lambda t: (*(new if t == j else (X[:, t], Z[:, t], Cs[:, t], Sn[:, t])), self.hu[t], self.hv[t])

    def piece(self, ref):
        if str(ref) not in self.pieces: raise RpcError(f'objet inconnu : {ref}')
        return self.pieces[str(ref)]

    def target(self, ref, what='cible'):
        if str(ref) in self.pieces: return [self.pieces[str(ref)]], self.names[self.pieces[str(ref)]]
        if not self.refs.get(str(ref)): raise RpcError(f'{what} inconnue : {ref}')
        cols = self.refs[str(ref)]
        return cols, TYPES.get(str(ref)) or self.names[cols[0]]

    def number(self, rule, key, default):
        value = rule.get(key, default)
        try: return [float(v) for v in value] if isinstance(value, (list, tuple)) else float(value)
        except (TypeError, ValueError): raise RpcError(f"{rule['type']} : valeur « {key} » invalide") from None

    def compile(self, rules):
        if not isinstance(rules, list): raise RpcError('règles : liste attendue')
        self.rules = []
        for r in rules:
            weight, fn, text = self.rule(r); used, get = set(), self.getter(*self.state(1))
            fn(lambda t: used.add(int(t)) or get(t)); self.rules.append((weight, fn, text, used))
        self.involves = [[k for k, rule in enumerate(self.rules) if j in rule[3]] for j in range(self.n)]
        self.fixed = {j: math.radians(a) for r in rules if (a := angle(r)) is not None and (j := self.piece(r.get('objet'))) < self.n}
        return self

    def rule(self, rule):
        kind = rule.get('type') if isinstance(rule, dict) else None
        if kind not in RULES: raise RpcError(f"règle inconnue : {kind} (disponibles : {', '.join(RULES)})")
        weight = self.number(rule, 'poids', 1.0)
        if kind == 'aligne':
            cols = [self.piece(r) for r in rule.get('objets') or []]
            if len(cols) < 2: raise RpcError('aligne : au moins 2 objets')
            names = ', '.join(self.names[c] for c in cols)
            def fn(get):
                bodies = [get(c) for c in cols]; c0, s0 = bodies[0][2], bodies[0][3]
                us, vs = np.stack([b[0] * c0 - b[1] * s0 for b in bodies]), np.stack([b[0] * s0 + b[1] * c0 for b in bodies])
                spread, turn = np.minimum(np.ptp(us, 0), np.ptp(vs, 0)), sum(1 - (b[2] * c0 + b[3] * s0) for b in bodies)
                return spread + 2 * turn, (spread < 0.05) & (turn < 1e-3), spread, turn
            return weight, fn, lambda ok, spread, turn: f'{names} alignés' + ('' if ok else f' (écart {cm(spread)}' + (', orientations différentes)' if turn >= 1e-3 else ')'))
        o = self.piece(rule.get('objet')); name = self.names[o]
        if kind == 'contre_mur':
            walls, label = (self.walls, 'un mur') if rule.get('mur') in (None, '') else self.target(rule['mur'], 'mur')
            if not set(walls) <= set(self.walls) or not walls: raise RpcError(f"contre_mur : mur inconnu : {rule.get('mur')}")
            def fn(get):
                x, z, c, s, hu, hv = b = get(o); bx, bz = x - hv * s, z - hv * c
                d = best([(sum(to_rect(bx + k * hu * c, bz - k * hu * s, get(w)) for k in (-1, 1)) / 2,) for w in walls])[0]
                return d, d < 0.02, d
            return weight, fn, lambda ok, d: f'{name} contre {label}' + ('' if ok else f' (écart {cm(d)})')
        if kind == 'centre':
            zone = rule.get('zone')
            if zone in (None, ''): point, label = centroid(self.room), 'la pièce'
            elif isinstance(zone, list): point, label = centroid(zone), 'la zone'
            elif str(zone) in self.polygons: point, label = centroid(self.polygons[str(zone)][2]), self.polygons[str(zone)][1]
            else: cols, label = self.target(zone, 'zone'); point = (float(self.x0[cols].mean()), float(self.z0[cols].mean()))
            def fn(get): d = np.hypot(get(o)[0] - point[0], get(o)[1] - point[1]); return d, d < 0.1, d
            return weight, fn, lambda ok, d: f'{name} au centre {de(label)}' + ('' if ok else f' (écart {cm(d)})')
        if angle(rule) is not None:
            a = math.radians(angle(rule))
            def fn(get): cos = get(o)[2] * math.cos(a) + get(o)[3] * math.sin(a); return 2 * (1 - cos), cos > COS10, cos
            return weight, fn, lambda ok, cos: f"{name} orienté à {round(rule['vers'])}°" + ('' if ok else f' (écart {round(deg(cos))}°)')
        cols, label = self.target(rule.get('vers' if kind == 'orientation' else 'cible'))
        if kind == 'orientation':
            def fn(get): cos = best([(facing(get(o), get(t))[0],) for t in cols])[0]; return 2 * (1 - cos), cos > COS10, cos
            return weight, fn, lambda ok, cos: f'{name} orienté vers {label}' + ('' if ok else f' (écart {round(deg(cos))}°)')
        if kind == 'face_a':
            bounds = self.number(rule, 'distance', [1.5, 3.5])
            if not isinstance(bounds, list) or len(bounds) != 2 or bounds[0] > bounds[1]: raise RpcError('face_a : distance = [min, max] en m')
            lo, hi = bounds
            def one(get, t):
                c1, d = facing(get(o), get(t)); c2 = facing(get(t), get(o))[0] if self.front[t] else np.ones_like(d); worst = np.minimum(c1, c2)
                return 2 * (2 - c1 - c2) + np.maximum(lo - d, 0) + np.maximum(d - hi, 0), (worst > COS10) & (d > lo - 0.02) & (d < hi + 0.02), d, worst
            def fn(get): return best([one(get, t) for t in cols])
            def text(ok, d, worst):
                problem = '' if ok else f" (pas en vis-à-vis, écart {round(deg(worst))}°)" if worst <= COS10 else f' (attendu {m(lo)} à {m(hi)})'
                return f'{name} face à {label} à {m(d)}{problem}'
            return weight, fn, text
        limit = self.number(rule, 'max' if kind == 'pres_de' else 'min', 0.5 if kind == 'pres_de' else 1.5)
        sign = 1 if kind == 'pres_de' else -1
        def fn(get):
            g = best([(pair(get(o), get(t))[1],) for t in cols])[0]
            return np.maximum(sign * (g - limit), 0), sign * (g - limit) <= 0.01, g
        return weight, fn, lambda ok, g: f"{name} {'près' if sign > 0 else 'loin'} {de(label)} ({cm(g) if g < 1 else m(g)}, {'max' if sign > 0 else 'min'} {cm(limit) if limit < 1 else m(limit)})"

    def outside(self, x, z, c, s, hu, hv):
        px = (x[..., None] + I * (hu * c)[..., None] + J * (hv * s)[..., None])[..., None]
        pz = (z[..., None] - I * (hu * s)[..., None] + J * (hv * c)[..., None])[..., None]
        x1, z1, ex, ez, inv, slope = self.edges
        inside = (((z1 > pz) != (z1 + ez > pz)) & (px < x1 + slope * (pz - z1))).sum(-1) % 2 == 1
        t = np.clip(((px - x1) * ex + (pz - z1) * ez) * inv, 0, 1); d = np.hypot(x1 + t * ex - px, z1 + t * ez - pz).min(-1)
        return np.where(inside | (d < EPS), 0, d + CONTACT).sum(-1)

    def local(self, j, x, z, c, s, X, Z, Cs, Sn):
        cols = self.cols[j]; mine = cols < self.n
        pen, gap = pair((x[..., None], z[..., None], c[..., None], s[..., None], self.hu[j], self.hv[j]), (X[:, cols], Z[:, cols], Cs[:, cols], Sn[:, cols], self.hu[cols], self.hv[cols]))
        hit = pen > EPS; band = self.near[j, cols] & ~hit & (gap > F.NARROW) & (gap < F.PASSAGE)
        cost = (self.block[j, cols] & hit) * (pen + CONTACT) + band * (np.minimum(gap - F.NARROW, F.PASSAGE - gap) + CONTACT)
        return HARD * (cost[..., ~mine].sum(-1) + self.outside(x, z, c, s, self.hu[j], self.hv[j])), HARD * cost[..., mine].sum(-1)

    def search(self, chains, steps, seed, progress):
        rng, n, walls = np.random.default_rng(seed), self.n, np.array(self.walls)
        X, Z, Cs, Sn = self.state(chains)
        X[1:, :n], Z[1:, :n] = (rng.uniform(lo, hi, (chains - 1, n)) for lo, hi in zip(self.low, self.high))
        A = np.tile(self.a0[:n], (chains, 1)); A[1:] = rng.choice(self.rots, (chains - 1, n))
        for j, a in self.fixed.items(): A[:, j] = a
        Cs[:, :n], Sn[:, :n] = np.cos(A), np.sin(A)
        costs = [w * fn(self.getter(X, Z, Cs, Sn))[0] * np.ones(chains) for w, fn, *_ in self.rules]; E = sum(costs, np.zeros(chains))
        for j in range(n):
            st, mv = self.local(j, X[:, j], Z[:, j], Cs[:, j], Sn[:, j], X, Z, Cs, Sn); E += st + mv / 2
        best_E, best = E.copy(), [v[:, :n].copy() for v in (X, Z, A)]
        for k in range(steps):
            f = k / steps; temperature, sigma = 2.0 * 0.004 ** f, 0.6 * 0.03 ** f
            j = int(rng.integers(n)); h, d, kind = self.hu[j], self.hv[j], rng.random(chains)
            jump, stick = (kind >= 0.15) & (kind < 0.2), (kind >= 0.2) & (kind < 0.35) & bool(len(walls))
            nx = np.where(jump, rng.uniform(self.low[0], self.high[0], chains), X[:, j] + rng.normal(0, sigma, chains))
            nz = np.where(jump, rng.uniform(self.low[1], self.high[1], chains), Z[:, j] + rng.normal(0, sigma, chains))
            na = np.where((kind < 0.15) | jump, rng.choice(self.rots, chains), A[:, j])
            if len(walls):
                w, side = rng.choice(walls, chains), rng.choice([-1.0, 1.0], chains); wc, ws = np.cos(self.a0[w]), np.sin(self.a0[w])
                along, out = rng.uniform(-1, 1, chains) * np.maximum(self.hu[w] - h, 0), side * (self.hv[w] + d + 0.002)
                nx, nz = np.where(stick, self.x0[w] + along * wc + out * ws, nx), np.where(stick, self.z0[w] - along * ws + out * wc, nz)
                na = np.where(stick, self.a0[w] + (side < 0) * math.pi, na)
            if j in self.fixed: na = np.full(chains, self.fixed[j])
            nc, ns = np.cos(na), np.sin(na)
            st, mv = self.local(j, *(np.stack([v[:, j], new]) for v, new in zip((X, Z, Cs, Sn), (nx, nz, nc, ns))), X, Z, Cs, Sn)
            delta, fresh, get = st[1] + mv[1] - st[0] - mv[0], {}, self.getter(X, Z, Cs, Sn, j, (nx, nz, nc, ns))
            for r in self.involves[j]: fresh[r] = self.rules[r][0] * self.rules[r][1](get)[0]; delta = delta + fresh[r] - costs[r]
            ok = (delta < 0) | (rng.random(chains) < np.exp(-np.maximum(delta, 0) / temperature))
            for v, new in zip((X, Z, Cs, Sn, A), (nx, nz, nc, ns, na)): v[ok, j] = new[ok]
            for r, value in fresh.items(): costs[r] = np.where(ok, value, costs[r])
            E = E + np.where(ok, delta, 0); better = E < best_E - 1e-9; best_E = np.where(better, E, best_E)
            for v, src in zip(best, (X, Z, A)): v[better] = src[better, :n]
            if k % 100 == 0: progress(0.9 * f, 'Recherche des agencements')
        return best_E, best

    def polish(self, x, z, a):
        objects = {o['id']: dict(o) for o in self.objects}
        moved = [objects[o['id']] for o in self.movable]
        for o, *pose in zip(moved, x, z, a): o['x'], o['z'], o['rot'] = float(pose[0]), float(pose[1]), round(math.degrees(pose[2]) % 360, 6)
        snap = lambda: [o.update(zip(('x', 'z'), F.snap(self.shapes, o))) for o in moved]
        snap()
        for _ in range(3):
            X, Z, Cs, Sn = (v[None, :].copy() for v in (self.x0, self.z0, np.cos(self.a0), np.sin(self.a0)))
            for i, o in enumerate(moved): X[0, i], Z[0, i], Cs[0, i], Sn[0, i] = o['x'], o['z'], math.cos(math.radians(o['rot'])), math.sin(math.radians(o['rot']))
            for i, o in enumerate(moved):
                for t in np.flatnonzero(self.solid[i] | self.zone[i]):
                    a = (o['x'], o['z'], math.cos(math.radians(o['rot'])), math.sin(math.radians(o['rot'])), self.hu[i], self.hv[i])
                    sep, along, kx, kz = separation(a, (X[0, t], Z[0, t], Cs[0, t], Sn[0, t], self.hu[t], self.hv[t])); k = int(np.argmax(sep))
                    if -0.05 < sep[k] < 0:
                        shift = (0.002 - sep[k]) * -math.copysign(1.0, along[k]); o['x'] += shift * kx[k]; o['z'] += shift * kz[k]
                        X[0, i], Z[0, i] = o['x'], o['z']
        snap()
        for o in moved: o['x'], o['z'] = round(o['x'], 4), round(o['z'], 4)
        return list(objects.values()), moved

    def report(self, objects, moved):
        X, Z, Cs, Sn = (v[None, :].copy() for v in (self.x0, self.z0, np.cos(self.a0), np.sin(self.a0)))
        for i, o in enumerate(moved): X[0, i], Z[0, i], Cs[0, i], Sn[0, i] = o['x'], o['z'], math.cos(math.radians(o['rot'])), math.sin(math.radians(o['rot']))
        get, details, satisfied = self.getter(X, Z, Cs, Sn), [], []
        for weight, fn, text, _ in self.rules:
            cost, ok, *values = (np.asarray(v).reshape(-1)[0] for v in fn(get)); ok = bool(ok)
            details.append({'ok': ok, 'text': f"{text(ok, *map(float, values))} : {'OK' if ok else 'pénalité'}"}); satisfied.append(1.0 if ok else math.exp(-weight * float(cost)))
        hard = F.conflicts(objects, self.shapes)
        details += [{'ok': False, 'text': f"{w['message']} : pénalité"} for w in hard] or [{'ok': True, 'text': 'Collisions, passages, portes et fenêtres : OK'}]
        score = max(0, round(100 * (sum(satisfied) / len(satisfied) if satisfied else 1.0)) - 20 * len(hard))
        return {'score': score, 'ok': not hard, 'details': details, 'objects': [{k: o[k] for k in ('id', 'x', 'z', 'rot')} for o in moved], 'shapes': F.geometry(objects)}


def distance(first, second):
    def key(o): return o['item'], tuple(o['size'])
    return max(min(math.hypot(p['x'] - q['x'], p['z'] - q['z']) + (DISTINCT if abs((p['rot'] - q['rot'] + 180) % 360 - 180) > 1 else 0)
                   for q in second if key(q) == key(p)) for p in first)


def saved(project):
    project.db.execute(SCHEMA); row = project.db.execute('select rules from layout_rules where id=1').fetchone()
    return json.loads(row[0]) if row else []


@rpc('layout.rules')
def rules_(ctx, rules=None):
    project = ctx.require_project()
    if rules is None: return saved(project)
    Scene(project, ctx.result).compile(rules)
    project.db.execute(SCHEMA); project.db.execute('insert or replace into layout_rules values (1, ?)', (json.dumps(rules, ensure_ascii=False),)); project.db.commit()
    return rules


@rpc('layout.propose', long=True)
def propose(ctx, objects=None, rules=None, n=3, seed=0, progress=lambda fraction, message='': None):
    project, start = ctx.require_project(), time.perf_counter()
    if rules is not None: rules_(ctx, rules)
    scene = Scene(project, ctx.result, objects).compile(saved(project))
    energy, (x, z, a) = scene.search(CHAINS, STEPS, int(seed), progress)
    progress(0.9, 'Vérification des propositions')
    candidates = []
    for chain in np.argsort(energy)[:max(4 * int(n), 8)]:
        everything, moved = scene.polish(x[chain], z[chain], a[chain])
        candidates.append((scene.report(everything, moved), moved, float(energy[chain])))
    candidates.sort(key=lambda c: (not c[0]['ok'], -c[0]['score'], c[2]))
    chosen = []
    for report, moved, _ in candidates:
        if len(chosen) < int(n) and all(distance(moved, other) > DISTINCT for _, other in chosen): chosen.append((report, moved))
    ctx.layout = (project.folder, [moved for _, moved in chosen])
    progress(1.0, 'Terminé')
    return {'seconds': time.perf_counter() - start, 'proposals': [{'index': k, **report} for k, (report, _) in enumerate(chosen)]}


@rpc('layout.apply')
def apply(ctx, index):
    project = ctx.require_project(); folder, proposals = getattr(ctx, 'layout', None) or (None, [])
    if folder != project.folder or not 0 <= int(index) < len(proposals): raise RpcError('proposition inconnue : relancez « Proposer »')
    for o in proposals[int(index)]: F.get(project, o['id'])
    project.db.executemany('update furniture set x=?, z=?, rot=? where id=?', [(o['x'], o['z'], o['rot'], o['id']) for o in proposals[int(index)]]); project.db.commit()
    return F.report(ctx)
