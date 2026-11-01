import json, math, os, re, shutil, subprocess, threading, time, urllib.error, urllib.parse, urllib.request
from . import furniture as F, layout as L
from .rpc import RpcError, rpc

URL, MODEL = 'http://127.0.0.1:11434/v1', 'qwen2.5:7b-instruct'
SETTINGS = os.path.join(os.path.dirname(F.FOLDER), 'settings.json')
OLLAMA = os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Programs', 'Ollama', 'ollama.exe')
WINGET = ['winget', 'install', '--id', 'Ollama.Ollama', '-e', '--silent', '--accept-package-agreements', '--accept-source-agreements', '--disable-interactivity']
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
LOCAL = ('127.0.0.1', 'localhost', '::1')
KEYS = {'contre_mur': ('objet', 'mur'), 'face_a': ('objet', 'cible', 'distance'), 'pres_de': ('objet', 'cible', 'max'), 'loin_de': ('objet', 'cible', 'min'),
        'aligne': ('objets',), 'centre': ('objet',), 'orientation': ('objet', 'vers')}
REQUIRED = ('objet', 'objets', 'cible', 'vers')
GENERIC = ('mur', 'porte', 'porte_fenetre', 'fenetre', 'escalier')
MAX_TEXT, MARGIN = 2000, 0.2
LOCK = threading.Lock()
RULES_HELP = '''Règles possibles :
- {"type":"contre_mur","objet":ID} ; option "mur": id d'un mur précis
- {"type":"face_a","objet":ID,"cible":CIBLE,"distance":[min,max]} : objet et cible se font face, distance en m ; pour « à d m » mets [d-0.2,d+0.2]
- {"type":"pres_de","objet":ID,"cible":CIBLE,"max":0.5} : écart maximal en m
- {"type":"loin_de","objet":ID,"cible":CIBLE,"min":1.5} : écart minimal en m
- {"type":"aligne","objets":[ID,ID]}
- {"type":"centre","objet":ID} : au centre de la pièce
- {"type":"orientation","objet":ID,"vers":CIBLE} : objet tourné vers la cible ; ou {"type":"orientation","objet":ID,"angle":degrés}
ID = id numérique d'un meuble listé. CIBLE = id d'un meuble, id d'un élément fixe listé, ou un type générique (le plus proche) : '''
ANSWER = 'Traite chaque souhait séparément : dans « X près de / loin de / face à / vers Y », objet = id de X et cible (ou vers) = Y, soit l’id d’un meuble, soit l’id ou le type d’un élément fixe (porte, fenetre, mur). Aucune règle non demandée, aucune règle pour un meuble non cité. Ne donne jamais de coordonnées. Réponds uniquement par un objet JSON {"regles":[...]}.'


def read():
    try:
        with open(SETTINGS, encoding='utf-8') as fh: data = json.load(fh)
    except (OSError, ValueError): data = {}
    return data if isinstance(data, dict) else {}


def settings():
    saved = read().get('ai')
    return {'url': URL, 'model': MODEL, 'key': '', **{k: v for k, v in (saved if isinstance(saved, dict) else {}).items() if k in ('url', 'model', 'key') and isinstance(v, str)}}


def local(s): return urllib.parse.urlsplit(s['url']).hostname in LOCAL
def public(s): return {'url': s['url'], 'model': s['model'], 'key': bool(s['key']), 'local': local(s)}
def installed(): return shutil.which('ollama') or (OLLAMA if os.path.exists(OLLAMA) else None)


def opener(s): return urllib.request.build_opener(*([urllib.request.ProxyHandler({})] if local(s) else []))


def request(s, path, body=None, timeout=5.0):
    headers = {'Content-Type': 'application/json', **({'Authorization': f"Bearer {s['key']}"} if s['key'] else {})}
    req = urllib.request.Request(s['url'].rstrip('/') + path, None if body is None else json.dumps(body).encode(), headers)
    try:
        with opener(s).open(req, timeout=timeout) as response: data = response.read()
    except urllib.error.HTTPError as e: raise RpcError(f"serveur IA : erreur {e.code} {e.read()[:300].decode('utf-8', 'replace')}") from None
    except (OSError, ValueError) as e: raise RpcError(f"serveur IA injoignable ({s['url']}) : {getattr(e, 'reason', e)}") from None
    try: return json.loads(data)
    except ValueError: raise RpcError('réponse du serveur IA illisible') from None


@rpc('ai.settings')
def settings_(ctx, url=None, model=None, key=None):
    s, changes = settings(), {k: str(v).strip() for k, v in (('url', url), ('model', model), ('key', key)) if v is not None}
    if changes.get('url', s['url']).rstrip('/') != s['url'].rstrip('/'): changes.setdefault('key', '')
    if changes:
        s.update(changes)
        if urllib.parse.urlsplit(s['url']).scheme not in ('http', 'https') or not s['model']: raise RpcError('réglages IA : URL http(s) et modèle requis')
        data = {**read(), 'ai': s}; os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
        with open(SETTINGS, 'w', encoding='utf-8') as fh: json.dump(data, fh, ensure_ascii=False, indent=1)
    return public(s)


@rpc('ai.status')
def status(ctx):
    s = settings(); out = {**public(s), 'installed': bool(installed()), 'server': False, 'ready': False}
    try: data = request(s, '/models', timeout=3)
    except RpcError as e: return {**out, 'error': str(e)}
    models = {m.get('id') for m in (data.get('data') if isinstance(data, dict) else None) or [] if isinstance(m, dict)}
    return {**out, 'server': True, 'ready': s['model'] in models or f"{s['model']}:latest" in models}


def winget(progress):
    progress(0.0, 'Installation d’Ollama (winget)…')
    try: proc = subprocess.Popen(WINGET, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
    except OSError: raise RpcError('winget introuvable : installez Ollama depuis https://ollama.com/download') from None
    tail = []
    for raw in proc.stdout:
        if line := raw.decode('utf-8', 'replace').strip(): tail.append(line); progress(min(0.15 * len(tail), 0.85), line[:120])
    if proc.wait() and not installed(): raise RpcError(f"installation d’Ollama échouée (winget {proc.returncode}) : {' / '.join(tail[-3:])}")


def serve(s, progress):
    progress(0.9, 'Démarrage d’Ollama…')
    for k in range(80):
        if k == 10 and installed():
            env = {**os.environ, 'OLLAMA_HOST': urllib.parse.urlsplit(s['url']).netloc}
            subprocess.Popen([installed(), 'serve'], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
        try: return request(s, '/models', timeout=2)
        except RpcError: time.sleep(0.5)
    raise RpcError('Ollama ne répond pas : relancez l’appli ou démarrez Ollama')


@rpc('ai.install', long=True)
def install(ctx, progress=None):
    with LOCK:
        s = settings()
        if not local(s): raise RpcError(f"serveur IA distant ({s['url']}) : rien à installer")
        if not installed(): winget(progress)
        if not status(ctx)['server']: serve(s, progress)
        progress(1.0, 'Ollama prêt')
        return status(ctx)


@rpc('ai.pull', long=True)
def pull(ctx, progress=None):
    with LOCK:
        s = settings(); totals, done = {}, {}
        req = urllib.request.Request(re.sub(r'/v1$', '', s['url'].rstrip('/')) + '/api/pull', json.dumps({'model': s['model']}).encode(), {'Content-Type': 'application/json'})
        progress(0.0, f"Téléchargement de {s['model']}…")
        try:
            with opener(s).open(req, timeout=600) as response:
                for line in filter(bytes.strip, response):
                    event = json.loads(line)
                    if event.get('error'): raise RpcError(f"téléchargement du modèle : {event['error']}")
                    if event.get('total'): totals[event.get('digest')], done[event.get('digest')] = event['total'], event.get('completed', 0)
                    total = sum(totals.values())
                    if total: progress(0.99 * sum(done.values()) / total, f"Téléchargement de {s['model']} : {sum(done.values()) >> 20} / {total >> 20} Mo")
                    else: progress(0.0, str(event.get('status', '')))
        except RpcError: raise
        except (OSError, ValueError) as e: raise RpcError(f"téléchargement du modèle impossible : {getattr(e, 'reason', e)}") from None
        progress(1.0, 'Modèle prêt')
        return status(ctx)


def prompt(scene):
    fixed = {s['id']: s for s in scene.shapes if s['type'] in L.TYPES and s['kind'] == 'box'}
    width, depth = (float(v) for v in scene.high - scene.low)
    return '\n'.join([
        'Tu traduis une description d’agencement de pièce en règles JSON pour un solveur de placement.',
        f'Pièce : {L.m(width)} × {L.m(depth)}.', 'Meubles (id : type « libellé ») :',
        *[f"- {o['id']} : {o['kind']} « {o['label']} »" for o in scene.objects], 'Éléments fixes (id : type « libellé ») :',
        *([f"- {s['id']} : {s['type']} « {s['label']} »" for s in fixed.values()] or ['- aucun']),
        RULES_HELP + ', '.join(t for t in GENERIC if any(s['type'] == t for s in fixed.values())) + '.', ANSWER])


def parse(content):
    try: data = json.loads(content)
    except (TypeError, ValueError):
        found = re.search(r'\{.*\}', str(content), re.S)
        try: data = json.loads(found[0]) if found else None
        except ValueError: data = None
    if isinstance(data, dict): data = data.get('regles', data.get('rules'))
    if not isinstance(data, list): raise RpcError('réponse du modèle illisible : JSON {"regles": [...]} attendu')
    return data


def metres(value, top, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= top: raise RpcError(f'{what} : {value!r} hors bornes (0 à {L.m(top)})')
    return round(float(value), 2)


def piece(ids, value):
    text = str(value).strip().removeprefix('meuble-')
    if isinstance(value, bool) or not text.isdigit() or int(text) not in ids: raise RpcError(f'meuble inconnu : {value}')
    return int(text)


def names(scene):
    labels = [o['label'].lower() for o in scene.objects]
    found = {label: f"meuble-{o['id']}" for o, label in zip(scene.objects, labels) if labels.count(label) == 1}
    for s in scene.shapes:
        if s['type'] in L.TYPES and s['kind'] == 'box': found.update(dict.fromkeys((s['id'].lower(), s['label'].lower(), f"{s['type']} {s['id']}".lower()), s['id']))
    return found


def check(scene, raw):
    kind = raw.get('type') if isinstance(raw, dict) else None
    if kind not in KEYS: raise RpcError(f'type de règle inconnu : {kind}')
    angle = raw.get('angle') if kind == 'orientation' else None
    keys = ('objet',) if angle is not None else KEYS[kind]
    missing = [k for k in keys if k in REQUIRED and raw.get(k) in (None, '', [])]
    if missing: raise RpcError(f"champ manquant : {', '.join(missing)}")
    ids, top, rule, alias = {o['id'] for o in scene.objects}, math.hypot(*(scene.high - scene.low)), {'type': kind}, names(scene)
    def name(v): return alias.get(v.strip().lower(), v) if isinstance(v, str) else v
    for key in keys:
        value = name(raw.get(key))
        if value in (None, ''): continue
        if key == 'objet': rule[key] = piece(ids, value)
        elif key == 'objets':
            rule[key] = [piece(ids, name(v)) for v in (value if isinstance(value, list) else [value])]
            if len(set(rule[key])) < max(len(rule[key]), 2): raise RpcError('aligne : au moins 2 meubles différents')
        elif key == 'distance':
            if isinstance(value, (int, float)) and not isinstance(value, bool): value = [max(metres(value, top, 'distance') - MARGIN, 0), value + MARGIN]
            if not isinstance(value, list) or len(value) != 2: raise RpcError('distance : [min, max] en m attendu')
            rule[key] = [metres(v, top, 'distance') for v in value]
            if rule[key][0] > rule[key][1]: raise RpcError('distance : min > max')
        elif key in ('max', 'min'): rule[key] = metres(value, top, key)
        elif re.fullmatch(r'(meuble-)?\d+', str(value).strip()):
            target = piece(ids, value); rule[key] = target if key == 'cible' else f'meuble-{target}'
            if target == rule.get('objet'): raise RpcError('un meuble ne peut pas se viser lui-même')
        else: rule[key] = str(value).strip()
    if angle is not None:
        if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not abs(angle) <= 360: raise RpcError(f'angle : {angle!r} hors bornes (-360 à 360)')
        rule['vers'] = round(float(angle) % 360, 1)
    scene.rule(rule)
    return rule


@rpc('ai.rules_from_text', long=True)
def rules_from_text(ctx, text, progress=None):
    text, start = str(text).strip(), time.perf_counter()
    if not text: raise RpcError('décrivez l’agencement souhaité')
    if len(text) > MAX_TEXT: raise RpcError(f'description trop longue ({MAX_TEXT} caractères max)')
    scene, s = L.Scene(ctx.require_project(), ctx.result), settings()
    progress(0.1, f"Interrogation de {s['model']}…")
    body = {'model': s['model'], 'temperature': 0, 'stream': False, 'response_format': {'type': 'json_object'},
            'messages': [{'role': 'system', 'content': prompt(scene)}, {'role': 'user', 'content': text}]}
    answer = request(s, '/chat/completions', body, timeout=600)
    try: content = answer['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError): raise RpcError('réponse du serveur IA inattendue') from None
    rules, rejected = [], []
    for raw in parse(content):
        try:
            rule = check(scene, raw)
            if rule in rules: raise RpcError('doublon')
            rules.append(rule)
        except RpcError as e: rejected.append({'rule': raw, 'reason': str(e)})
    progress(1.0, 'Terminé')
    return {'rules': rules, 'rejected': rejected, 'model': s['model'], 'seconds': time.perf_counter() - start}
