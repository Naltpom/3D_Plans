"""Projet = dossier : project.db (paramètres, sources, observations en cache), model.json (export)."""
import csv, hashlib, json, os, re, sqlite3, time
from .core import Obs, solve

SCHEMA = '''
create table if not exists params(name text primary key, init real, axis text, label text);
create table if not exists sources(id text primary key, path text, kind text, hash text);
create table if not exists obs(id text primary key, source text, expr text, value real, sigma real, status text, label text);
'''
UNITS = {'m': 1.0, 'cm': 0.01, 'mm': 0.001, 'u': 1.0}   # u : unité d'une reconstruction photo (échelle = paramètre)
EXTRACTORS = {}


def extractor(kind):
    def reg(fn): EXTRACTORS[kind] = fn; return fn
    return reg


@extractor('cotes_csv')
def cotes_csv(path, sid):
    """CSV ; : id;expr;value;sigma;unit;status;label"""
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
        os.makedirs(folder, exist_ok=True); self.folder = folder
        self.db = sqlite3.connect(os.path.join(folder, 'project.db')); self.db.executescript(SCHEMA)

    def add_param(self, name, init, axis='', label=''):
        self.db.execute('insert or replace into params values (?,?,?,?)', (name, init, axis, label)); self.db.commit()

    def add_source(self, sid, path, kind):
        self.db.execute('insert or replace into sources values (?,?,?,coalesce((select hash from sources where id=?), ""))',
                        (sid, os.path.abspath(path), kind, sid)); self.db.commit()

    def params(self):
        return {n: (i, a, l) for n, i, a, l in self.db.execute('select * from params')}

    def observations(self, exclude=()):
        return [Obs(r[0], r[2], r[3], r[4], r[1], r[5], r[6]) for r in self.db.execute('select * from obs') if r[0] not in exclude]

    def extract(self):
        """Ré-extrait seulement les sources dont l'empreinte a changé. Retourne les ids ré-extraits."""
        done = []
        for sid, path, kind, old in self.db.execute('select * from sources').fetchall():
            h = file_hash(path)
            if h == old: continue
            obs = EXTRACTORS[kind](path, sid)
            self.db.execute('delete from obs where source=?', (sid,))
            self.db.executemany('insert into obs values (?,?,?,?,?,?,?)', [(o.id, sid, o.expr, o.value, o.sigma, o.status, o.label) for o in obs])
            self.db.execute('update sources set hash=? where id=?', (h, sid)); done.append(sid)
        self.db.commit(); return done

    def update(self, exclude=()):
        """Bouton « Mettre à jour » : extraction incrémentale, résolution complète, export model.json."""
        t0 = time.perf_counter(); done = self.extract(); t1 = time.perf_counter()
        P = self.params(); res = solve({n: v[0] for n, v in P.items()}, self.observations(exclude)); t2 = time.perf_counter()
        self.export(res, P)
        res.timing = {'extract_s': t1 - t0, 'solve_s': t2 - t1, 'reextracted': done}
        return res

    def export(self, res, P):
        uses = {n: [o.id for o in res.obs if n in re.findall(r'[A-Za-z_]\w*', o.expr)] for n in res.names}
        m = {'units': 'm', 'chi2': res.chi2, 'dof': res.dof,
             'params': {n: {'value': float(x), 'sigma': None if s == float('inf') else float(s), 'axis': P[n][1], 'label': P[n][2], 'obs': uses[n]}
                        for n, x, s in zip(res.names, res.x, res.sigma)},
             'obs': [{'id': o.id, 'expr': o.expr, 'value': o.value, 'sigma': o.sigma, 'status': o.status, 'label': o.label,
                      'resid': float(r), 'w': float(w), 'redundancy': float(q)} for o, r, w, q in zip(res.obs, res.resid, res.w, res.redundancy)],
             'conflicts': [[{'id': o.id, 'w': w} for o, w in g] for g in res.conflicts()]}
        json.dump(m, open(os.path.join(self.folder, 'model.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
