import csv, inspect, json, os, re, threading
import numpy as np
from datetime import datetime
from .core import FUNCS, STATUS, compile_expr, next_best as core_next_best, variance_factors as core_variance_factors
from .project import EXTRACTORS, UNITS, Project, undetermined

METHODS, LONG_METHODS = {}, set()
ID_PATTERN = re.compile(r'[\w\-.]+')
COTES_COLUMNS = ['id', 'expr', 'value', 'sigma', 'unit', 'status', 'label']
GROUP_FIELDS = {'source', 'status', 'label'}
MAX_CANDIDATES = 20000
IMAGE_MAGIC, TIFF_MAGIC = (b'\xff\xd8\xff', b'\x89PNG'), (b'II*\x00', b'MM\x00*')
HEIC_BRANDS = (b'ftypheic', b'ftypheix', b'ftyphevc', b'ftypmif1', b'ftypmsf1')


class RpcError(ValueError): pass


def rpc(name, long=False):
    def register(fn):
        METHODS[name] = fn
        (LONG_METHODS.add if long else LONG_METHODS.discard)(name)
        return fn
    return register


class Ctx:
    def __init__(self, send=None, project=None):
        self.project, self.result = project, None
        self.send = send or (lambda message: None)
        self.update_lock = threading.Lock()
        self.current = threading.local()

    def emit(self, event):
        request_id = getattr(self.current, 'request_id', None)
        self.send(event if request_id is None else {'request': request_id, **event})

    def require_project(self):
        if self.project is None: raise RpcError('aucun projet ouvert')
        return self.project


def call(ctx, name, params):
    fn = METHODS.get(name)
    if fn is None: raise RpcError(f'méthode inconnue : {name}')
    declared = list(inspect.signature(fn).parameters.values())[1:]
    names = {p.name for p in declared}
    if not any(p.kind == p.VAR_KEYWORD for p in declared):
        unknown = sorted(set(params) - names)
        if unknown: raise RpcError(f'{name} : paramètre inconnu {", ".join(unknown)}')
    missing = [p.name for p in declared if p.default is p.empty and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL) and p.name not in params]
    if missing: raise RpcError(f'{name} : paramètre manquant {", ".join(missing)}')
    if 'progress' in names and 'progress' not in params:
        params = {**params, 'progress': lambda fraction, message='': ctx.emit(
            {'event': 'progress', 'method': name, 'fraction': fraction, 'message': message})}
    return fn(ctx, **params)


def open_project(ctx, folder):
    if ctx.update_lock.locked(): raise RpcError('mise à jour en cours : réessayez une fois terminée')
    if ctx.project: ctx.project.db.close()
    for sub in ('sources', 'cache'): os.makedirs(os.path.join(folder, sub), exist_ok=True)
    ctx.project, ctx.result = Project(folder), None
    return project_info(ctx)


@rpc('project.create')
def project_create(ctx, path, name):
    folder = os.path.abspath(path)
    if any(os.path.exists(os.path.join(folder, f)) for f in ('project.json', 'project.db')): raise RpcError(f'un projet existe déjà dans {folder}')
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'project.json'), 'w', encoding='utf-8') as fh:
        json.dump({'name': name, 'created': datetime.now().astimezone().isoformat(timespec='seconds')}, fh, ensure_ascii=False, indent=1)
    return open_project(ctx, folder)


@rpc('project.open')
def project_open(ctx, path):
    folder = os.path.abspath(path)
    if not os.path.exists(os.path.join(folder, 'project.db')): raise RpcError(f'aucun projet dans {folder}')
    return open_project(ctx, folder)


@rpc('project.info')
def project_info(ctx):
    project = ctx.project
    if project is None: return None
    meta = {'name': os.path.basename(project.folder), 'created': None}
    meta_path = os.path.join(project.folder, 'project.json')
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as fh: meta.update(json.load(fh))
    count = lambda table: project.db.execute(f'select count(*) from {table}').fetchone()[0]
    return {**meta, 'path': project.folder, 'params': count('params'), 'sources': count('sources'), 'obs': count('obs'),
            'model': os.path.exists(os.path.join(project.folder, 'model.json'))}


def project_meta(project, **changes):
    path = os.path.join(project.folder, 'project.json')
    meta = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}
    if changes:
        meta.update(changes)
        with open(path, 'w', encoding='utf-8') as fh: json.dump(meta, fh, ensure_ascii=False, indent=1)
    return meta


@rpc('project.guide')
def project_guide(ctx, guide):
    return project_meta(ctx.require_project(), guide=guide)['guide']


@rpc('params.list')
def params_list(ctx):
    return [{'name': n, 'init': init, 'axis': axis, 'label': label} for n, (init, axis, label) in ctx.require_project().params().items()]


@rpc('params.add')
def params_add(ctx, name, init=0.0, axis='', label=''):
    if not re.fullmatch(r'[A-Za-z_]\w*', name) or name in FUNCS: raise RpcError(f'nom de paramètre invalide : {name!r}')
    ctx.require_project().add_param(name, float(init), axis, label)
    return {'name': name, 'init': float(init), 'axis': axis, 'label': label}


@rpc('params.remove')
def params_remove(ctx, name):
    project = ctx.require_project()
    if not project.db.execute('delete from params where name=?', (name,)).rowcount: raise RpcError(f'paramètre inconnu : {name}')
    project.db.commit()


@rpc('sources.list')
def sources_list(ctx):
    project = ctx.require_project()
    from .photos import source_status
    obs_count = dict(project.db.execute('select source, count(*) from obs group by source').fetchall())
    return [{'id': sid, 'path': path, 'kind': kind, 'extracted': bool(hash_), 'obs': obs_count.get(sid, 0),
             **(source_status(project, sid, path) if kind == 'photos_colmap' else {})}
            for sid, path, kind, hash_ in project.db.execute('select * from sources').fetchall()]


@rpc('sources.add')
def sources_add(ctx, path, kind, id=None):
    project = ctx.require_project(); path = os.path.abspath(path)
    if kind not in EXTRACTORS: raise RpcError(f'type de source inconnu : {kind} (disponibles : {", ".join(sorted(EXTRACTORS))})')
    if not os.path.exists(path): raise RpcError(f'fichier introuvable : {path}')
    if id is None: id = project.free_source_id(re.sub(r'[^\w\-.]+', '_', os.path.splitext(os.path.basename(path))[0]) or 'source')
    elif not ID_PATTERN.fullmatch(id): raise RpcError(f'identifiant de source invalide : {id!r}')
    elif project.free_source_id(id) != id: raise RpcError(f'source déjà existante : {id}')
    project.add_source(id, path, kind)
    return {'id': id, 'path': path, 'kind': kind}


def source_method(path):
    if os.path.isdir(path): return 'photos.add', {'paths': [path]}
    with open(path, 'rb') as fh: head = fh.read(4096)
    if head.startswith(b'%PDF'): return 'cadastre.add', {'path': path}
    if head.startswith(IMAGE_MAGIC) or head[4:12] in HEIC_BRANDS: return 'photos.add', {'paths': [path]}
    if head.startswith(TIFF_MAGIC) or head.lstrip().lower().startswith(b'ncols'): return 'terrain.add', {'path': path}
    text = head.decode('utf-8-sig', 'replace')
    if set(text.partition('\n')[0].strip().split(';')) >= set(COTES_COLUMNS): return 'sources.add', {'path': path, 'kind': 'cotes_csv'}
    if text.lstrip().startswith('{'):
        with open(path, encoding='utf-8-sig') as fh: data = json.load(fh)
        if 'features' in data: return 'cadastre.add', {'path': path}
        if 'valeurs' in data: return 'terrain.add', {'path': path}
    raise RpcError(f'{os.path.basename(path)} : type de fichier non reconnu')


@rpc('sources.add_auto')
def sources_add_auto(ctx, path):
    ctx.require_project(); path = os.path.abspath(path)
    if not os.path.exists(path): raise RpcError(f'fichier introuvable : {path}')
    method, params = source_method(path)
    return {'method': method, **call(ctx, method, params)}


@rpc('sources.remove')
def sources_remove(ctx, id):
    project = ctx.require_project()
    if not project.db.execute('delete from sources where id=?', (id,)).rowcount: raise RpcError(f'source inconnue : {id}')
    project.db.execute('delete from obs where source=?', (id,)); project.db.commit()


def cotes_path(project): return os.path.join(project.folder, 'sources', 'cotes.csv')


def read_cotes(project):
    if not os.path.exists(cotes_path(project)): return []
    with open(cotes_path(project), encoding='utf-8', newline='') as fh:
        return [{**row, 'value': float(row['value']), 'sigma': float(row['sigma'])} for row in csv.DictReader(fh, delimiter=';')]


def write_cotes(project, rows):
    with open(cotes_path(project), 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, COTES_COLUMNS, delimiter=';', extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    if not project.db.execute("select 1 from sources where id='cotes'").fetchone(): project.add_source('cotes', cotes_path(project), 'cotes_csv')


@rpc('cotes.list')
def cotes_list(ctx): return read_cotes(ctx.require_project())


@rpc('cotes.add')
def cotes_add(ctx, id, expr, value, sigma=1.0, unit='cm', label='', status='mesuré'):
    project = ctx.require_project(); id = str(id)
    if not ID_PATTERN.fullmatch(id): raise RpcError(f'identifiant de cote invalide : {id!r}')
    if unit not in UNITS: raise RpcError(f'unité inconnue : {unit} (disponibles : {", ".join(UNITS)})')
    if status not in STATUS: raise RpcError(f'statut inconnu : {status} (disponibles : {", ".join(STATUS)})')
    if not float(sigma) > 0: raise RpcError('σ doit être > 0')
    try: compile_expr(expr, {n: i for i, n in enumerate(project.params())})
    except KeyError as e: raise RpcError(e.args[0]) from None
    except (ValueError, SyntaxError) as e: raise RpcError(f'expression invalide {expr!r} : {e}') from None
    row = {'id': id, 'expr': expr, 'value': float(value), 'sigma': float(sigma), 'unit': unit, 'status': status, 'label': label}
    rows = read_cotes(project)
    position = next((i for i, r in enumerate(rows) if r['id'] == id), len(rows))
    rows[position:position + 1] = [row]
    write_cotes(project, rows)
    return row


@rpc('cotes.remove')
def cotes_remove(ctx, id):
    project = ctx.require_project(); rows = read_cotes(project)
    kept = [r for r in rows if r['id'] != str(id)]
    if len(kept) == len(rows): raise RpcError(f'cote inconnue : {id}')
    write_cotes(project, kept)


@rpc('update', long=True)
def update(ctx, exclude=(), progress=lambda fraction, message: None):
    from .components import obs_unit
    project = ctx.require_project()
    with ctx.update_lock: result = ctx.result = project.update(set(exclude), progress)
    params = project.params()
    return {'timing': result.timing, 'chi2': result.chi2, 'dof': result.dof, 'warnings': result.warnings,
            'params': [{'name': n, 'value': x, 'sigma': s, 'axis': params[n][1], 'label': params[n][2]}
                       for n, x, s in zip(result.names, result.x, result.sigma)],
            'conflicts': [[{'id': o.id, 'w': w, 'label': o.label} for o, w in group] for group in result.conflicts()],
            'obs': [{'id': o.id, 'source': o.source, 'expr': o.expr, 'value': o.value, 'sigma': o.sigma, 'status': o.status, 'label': o.label, 'unit': obs_unit(o),
                     'resid': resid, 'w': w, 'redundancy': red} for o, resid, w, red in zip(result.obs, result.resid, result.w, result.redundancy)]}


@rpc('model')
def model(ctx):
    found = ctx.require_project().model()
    if found is None: raise RpcError('Aucun résultat : cliquez sur « Mettre à jour »')
    return found


def require_result(ctx):
    ctx.require_project()
    if ctx.result is None: raise RpcError('Aucun résultat : cliquez sur « Mettre à jour »')
    return ctx.result


@rpc('predict')
def predict(ctx, expr):
    result = require_result(ctx)
    try: (value, sigma), g = result.predict(expr), result.grad(expr)[1]
    except KeyError as e: raise RpcError(e.args[0]) from None
    except (ValueError, SyntaxError) as e: raise RpcError(f'expression invalide {expr!r} : {e}') from None
    return {'value': value, 'sigma': None if undetermined(result, g) else sigma}



def default_targets(project, result, components):
    from .cadastre import cached_sources, param_prefix
    sources = cached_sources(project.folder); sources = [s for s in sources if s['georeference']] or sources
    main = next(((param_prefix(s['source']), e) for s in sources for e in s['entites'] if e['type'] == 'parcelle' and e['principal']), None)
    refs = (main[1].get('refs') or [f"{main[1]['id']}_{i}" for i in range(len(main[1]['pts']))]) if main else []
    names = {n for c in components if c['type'] != 'terrain' for n in c['params']} | {f'{main[0]}_{r}_{a}' for r in refs for a in 'xz'}
    return names & set(result.names)


@rpc('next_best')
def next_best(ctx, k=5, targets=None, sigma=0.01):
    from .components import load
    result = require_result(ctx); params = ctx.project.params(); components = load(ctx.project).values()
    unknown = sorted(set(targets or ()) - set(result.names))
    if unknown: raise RpcError(f'paramètre inconnu : {", ".join(unknown)}')
    terrain = {n for c in components if c['type'] == 'terrain' for n in c['params']}
    if targets is None:
        scope = default_targets(ctx.project, result, components) or set(result.names) - terrain
        targets = scope & {n for g in result.conflicts() for o, _ in g for n in re.findall(r'[A-Za-z_]\w*', o.expr)} or scope
    idx, label = result._idx, lambda n: params[n][2] or n
    by_axis = {}
    for n in sorted(result.names, key=lambda n: -np.nan_to_num(result.sigma[idx[n]], posinf=0)):
        if n in params and params[n][1] in ('x', 'y', 'z') and n not in terrain: by_axis.setdefault(params[n][1], []).append(n)
    m = int(np.sqrt(2 * MAX_CANDIDATES / max(len(by_axis), 1)))
    candidates = [(label(n), n, sigma) for v in by_axis.values() for n in v]
    candidates += [(f'{label(a)} → {label(b)}', f'{a} - {b}', sigma) for v in by_axis.values() for p, x in enumerate(v[:m]) for y in v[p + 1:m]
                   for a, b in [sorted((x, y), key=idx.get)]]
    return [{'gain': gain, 'label': lab, 'expr': expr, 'sigma': current}
            for gain, lab, expr, current in core_next_best(result, candidates[:MAX_CANDIDATES], sorted(targets), int(k))]


@rpc('variance_factors')
def variance_factors(ctx, by=('source',)):
    result = require_result(ctx)
    unknown = sorted(set(by) - GROUP_FIELDS)
    if unknown: raise RpcError(f'champ de regroupement inconnu : {", ".join(unknown)}')
    groups = core_variance_factors(result, lambda o: tuple(getattr(o, field) for field in by))
    return [{'group': dict(zip(by, key)), 'factor': factor, 'redundancy': red, 'n': n} for key, (factor, red, n) in groups.items()]


@rpc('methods')
def methods(ctx):
    return [{'name': n, 'long': n in LONG_METHODS, 'params': [p for p in list(inspect.signature(fn).parameters)[1:] if p != 'progress']} for n, fn in sorted(METHODS.items())]
