import csv, hashlib, json, os, sqlite3, threading, time
import numpy as np
from scipy.sparse import diags_array
from scipy.sparse.linalg import lsqr
from .core import Obs, solve

SCHEMA = '''
create table if not exists params(name text primary key, init real, axis text, label text);
create table if not exists sources(id text primary key, path text, kind text, hash text);
create table if not exists obs(id text primary key, source text, expr text, value real, sigma real, status text, label text);
'''
UNITS = {'m': 1.0, 'cm': 0.01, 'mm': 0.001, 'u': 1.0}
EXTRACTORS = {}
SCALE_WARNING = 'Échelle fixée uniquement par des hypothèses : ajoute une cote mesurée.'
FRAGILE_WARNING = 'Échelle reposant sur une seule cote, en conflit ({}) : ajoute une deuxième cote mesurée.'
NOT_CONVERGED = 'Résolution interrompue sans convergence : vérifie les unités et les valeurs aberrantes.'


def extractor(kind):
    def reg(fn): EXTRACTORS[kind] = fn; return fn
    return reg


@extractor('cotes_csv')
def cotes_csv(path, sid):
    out = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter=';'):
            k = UNITS[r['unit']]
            out.append(Obs(f"{sid}:{r['id']}", r['expr'], float(r['value']) * k, float(r['sigma']) * k, sid, r['status'] or 'mesuré', r['label']))
    return out


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


class Project:
    def __init__(self, folder):
        os.makedirs(folder, exist_ok=True); self.folder, self.local = folder, threading.local()
        self.db.executescript(SCHEMA)

    @property
    def db(self):
        if not hasattr(self.local, 'db'): self.local.db = sqlite3.connect(os.path.join(self.folder, 'project.db'))
        return self.local.db

    def add_param(self, name, init, axis='', label=''):
        self.db.execute('insert or replace into params values (?,?,?,?)', (name, init, axis, label)); self.db.commit()

    def add_source(self, sid, path, kind):
        self.db.execute('insert or replace into sources values (?,?,?,coalesce((select hash from sources where id=?), ""))',
                        (sid, os.path.abspath(path), kind, sid)); self.db.commit()

    def free_source_id(self, base):
        existing = {row[0] for row in self.db.execute('select id from sources')}; sid, n = base, 2
        while sid in existing: sid, n = f'{base}_{n}', n + 1
        return sid

    def params(self):
        return {n: (i, a, l) for n, i, a, l in self.db.execute('select * from params')}

    def observations(self, exclude=()):
        return [Obs(r[0], r[2], r[3], r[4], r[1], r[5], r[6]) for r in self.db.execute('select * from obs') if r[0] not in exclude]

    def extract(self, progress=lambda fraction, message: None):
        done = []; rows = self.db.execute('select * from sources').fetchall()
        for i, (sid, path, kind, old) in enumerate(rows):
            h = file_hash(path)
            if h == old: continue
            progress(0.5 * i / len(rows), f'Extraction des sources : {sid}')
            obs = EXTRACTORS[kind](path, sid)
            self.db.execute('delete from obs where source=?', (sid,))
            self.db.executemany('insert into obs values (?,?,?,?,?,?,?)', [(o.id, sid, o.expr, o.value, o.sigma, o.status, o.label) for o in obs])
            self.db.execute('update sources set hash=? where id=?', (h, sid)); done.append(sid)
        self.db.commit(); return done

    def update(self, exclude=(), progress=lambda fraction, message: None):
        progress(0.0, 'Extraction des sources')
        t0 = time.perf_counter(); done = self.extract(progress); t1 = time.perf_counter()
        progress(0.5, 'Résolution')
        P, last = self.params(), self.last_solution()
        res = solve({n: last.get(n, v[0]) for n, v in P.items()}, self.observations(exclude)); t2 = time.perf_counter()
        from .components import inclusion_warnings, load
        from .furniture import warnings as furniture_warnings
        owned = {n for c in load(self).values() for n in c['params']}
        res.warnings = scale_guard(res, P) + determination_warnings(res, P, owned) + inclusion_warnings(self, res) + furniture_warnings(self, res) + ([] if res.converged else [{'code': 'non_converge', 'message': NOT_CONVERGED}])
        progress(0.9, 'Export')
        self.export(res, P)
        res.timing = {'extract_s': t1 - t0, 'solve_s': t2 - t1, 'reextracted': done}
        return res

    def model(self):
        path = os.path.join(self.folder, 'model.json')
        if not os.path.exists(path): return None
        with open(path, encoding='utf-8') as f: return json.load(f)

    def last_solution(self):
        try:
            m = self.model() or {}
            if any(p.get('axis') == 'échelle' and p['value'] <= 1e-6 for p in m['params'].values()): return {}
            return {n: float(p['value']) for n, p in m['params'].items() if p['sigma'] is not None and np.isfinite(p['value'])} if m.get('converged', True) else {}
        except (OSError, ValueError, KeyError, TypeError): return {}

    def export(self, res, P):
        uses = {n: [] for n in res.names}
        for o, d in zip(res.obs, res.deps):
            for j in d: uses[res.names[j]].append(o.id)
        m = {'units': 'm', 'chi2': res.chi2, 'dof': res.dof, 'converged': res.converged,
             'params': {n: {'value': float(x), 'sigma': None if s == float('inf') else float(s), 'axis': P[n][1], 'label': P[n][2], 'obs': uses[n]}
                        for n, x, s in zip(res.names, res.x, res.sigma)},
             'obs': [{'id': o.id, 'expr': o.expr, 'value': o.value, 'sigma': o.sigma, 'status': o.status, 'label': o.label,
                      'resid': float(r), 'w': float(w), 'redundancy': float(q)} for o, r, w, q in zip(res.obs, res.resid, res.w, res.redundancy)],
             'conflicts': [[{'id': o.id, 'w': w} for o, w in g] for g in res.conflicts()], 'warnings': res.warnings}
        json.dump(m, open(os.path.join(self.folder, 'model.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)


def scaled(J):
    d = np.sqrt(np.asarray(J.multiply(J).sum(0)).ravel()); d[d == 0] = 1
    return J @ diags_array(1 / d), d


def free_part(J, d, g):
    g = g / d; v = g - lsqr(J, J @ g, atol=1e-14, btol=1e-14, iter_lim=20 * len(g))[0]
    return v, v @ v > 1e-6 * (g @ g)


def undetermined(res, g):
    if np.isfinite(res.sigma[g != 0]).all(): return False
    Q, dq = res.gauge; u = (g / dq) @ Q
    return free_part(*scaled(res.J), g)[1] or u @ u > 1e-6 * np.sum((g / dq) ** 2)


def scale_guard(res, P):
    n = len(res.x); e = lambda i: np.eye(1, n, i)[0]; res.gauge = np.zeros((n, 0)), np.ones(n)
    scales = [i for i, name in enumerate(res.names) if P[name][1] == 'échelle' and np.isfinite(res.sigma[i])]
    if not scales: return []
    rows = [k for k, o in enumerate(res.obs) if o.status != 'supposé']; known = [res.obs[k] for k in rows]; J, d = scaled(res.J[rows])
    V = {i: free_part(J, d, e(i)) for i in scales}; bad = [i for i in scales if V[i][1]]
    if bad:
        res.gauge = np.linalg.qr(np.array([V[i][0] for i in bad]).T)[0], d
        res.sigma[(res.gauge[0] ** 2).sum(1) > 1e-6] = np.inf
    conflict = {o.id for g in res.conflicts() for o, _ in g}
    fragile = [(i, known[k]) for k in range(len(known)) if known[k].id in conflict for i in scales
               if i not in bad and free_part(J[np.delete(np.arange(len(known)), k)], d, e(i))[1]]
    return ([{'code': 'echelle_supposee', 'param': res.names[i], 'message': SCALE_WARNING} for i in bad] +
            [{'code': 'echelle_fragile', 'param': res.names[i], 'obs': o.id, 'message': FRAGILE_WARNING.format(o.label or o.id)} for i, o in fragile])


def determination_warnings(res, P, owned):
    used = {j for d in res.deps for j in d}
    free = [res.names[j] for j in sorted(used) if not np.isfinite(res.sigma[j])]
    if not free: return []
    ex = ', '.join(P[n][2] or n for n in free[:3]) + (', …' if len(free) > 3 else ''); s = 's' if len(free) > 1 else ''
    return [{'code': 'non_determines', 'params': free, 'inutilises': len(res.x) - len(used),
             'sans_observation': [n for j, n in enumerate(res.names) if j not in used and n in owned],
             'message': f'{len(free)} paramètre{s} non déterminé{s} (ex. {ex}) : ajoute une cote ou fixe une origine.'}]
