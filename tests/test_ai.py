import json, socket, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from plans3d import Project, ai, components as C
from plans3d.rpc import Ctx, RpcError, call

ITEMS = ['canape', 'meuble_tv', 'etagere', 'lampadaire']
REAL = 'la télé face au canapé à 2,5 m, la bibliothèque contre un mur, le lampadaire près du canapé'


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self): self.reply(self.server.routes[self.path](None))

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.seen.append((self.path, dict(self.headers), body)); self.reply(self.server.routes[self.path](body))

    def reply(self, data):
        raw = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(200); self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)


def answer(content): return lambda body: {'choices': [{'message': {'content': content if isinstance(content, str) else json.dumps({'regles': content})}}]}


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setattr(ai, 'SETTINGS', str(tmp_path / 'settings.json'))
    project = Project(str(tmp_path))
    def add(type, comp_id, **options): C.add(project, type, comp_id, options)
    add('mur', 'm1', axe='x', z='0', x0='0', x1=5.2, sens=-1); add('mur', 'm2', axe='z', x='m1_x1', z0='0', z1=4.0)
    add('mur', 'm3', axe='x', z='m2_z1', x0='0', x1='m1_x1'); add('mur', 'm4', axe='z', x='0', z0='0', z1='m2_z1', sens=-1)
    add('sol', 's0'); add('porte', 'p1', mur='m1', pos=0.4, l=0.9, sens='interieur_gauche')
    ctx = Ctx(project=project)
    for k, item in enumerate(ITEMS): call(ctx, 'furniture.add', {'item': f'gabarit:{item}', 'x': 1 + k, 'z': 2.0})
    return ctx


@pytest.fixture
def server(ctx, monkeypatch):
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Fake); httpd.seen = []
    httpd.routes = {'/v1/models': lambda body: {'data': [{'id': ai.MODEL}]}}
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(ai, 'URL', f'http://127.0.0.1:{httpd.server_port}/v1')
    yield httpd
    httpd.shutdown(); httpd.server_close()


def test_settings_and_status(server, ctx):
    assert call(ctx, 'ai.status', {}) | {'url': ''} == {'url': '', 'model': ai.MODEL, 'key': False, 'local': True, 'installed': bool(ai.installed()), 'server': True, 'ready': True}
    assert call(ctx, 'ai.settings', {'model': 'autre', 'key': 'secret'})['key'] is True
    assert json.load(open(ai.SETTINGS, encoding='utf-8'))['ai']['model'] == 'autre' and not call(ctx, 'ai.status', {})['ready']
    with pytest.raises(RpcError, match='URL http'): call(ctx, 'ai.settings', {'url': 'ftp://x'})
    server.routes['/v1/chat/completions'] = answer([])
    call(ctx, 'ai.rules_from_text', {'text': 'rien'})
    assert server.seen[-1][1]['Authorization'] == 'Bearer secret' and server.seen[-1][2]['model'] == 'autre'
    assert call(ctx, 'ai.settings', {'url': ai.URL + '/'})['key'] and not call(ctx, 'ai.settings', {'url': 'http://127.0.0.1:1/v1'})['key']


def test_valid_rules(server, ctx):
    server.routes['/v1/chat/completions'] = answer([
        {'type': 'face_a', 'objet': 2, 'cible': 1, 'distance': 2.5}, {'type': 'contre_mur', 'objet': '3', 'note': 'x'}, {'type': 'pres_de', 'objet': 4, 'cible': 'meuble-1', 'max': 0.5},
        {'type': 'orientation', 'objet': 2, 'vers': 1}, {'type': 'loin_de', 'objet': 1, 'cible': 'porte', 'min': 1}, {'type': 'orientation', 'objet': 1, 'angle': -90},
        {'type': 'loin_de', 'objet': 'lampadaire', 'cible': 'porte p1', 'min': 1}, {'type': 'contre_mur', 'objet': 2, 'mur': 'Mur M3'}, {'type': 'pres_de', 'objet': 3, 'cible': 'Canapé'}])
    out = call(ctx, 'ai.rules_from_text', {'text': REAL, 'progress': lambda *a: None})
    assert out['rejected'] == [] and out['model'] == ai.MODEL and out['rules'] == [
        {'type': 'face_a', 'objet': 2, 'cible': 1, 'distance': [2.3, 2.7]}, {'type': 'contre_mur', 'objet': 3}, {'type': 'pres_de', 'objet': 4, 'cible': 1, 'max': 0.5},
        {'type': 'orientation', 'objet': 2, 'vers': 'meuble-1'}, {'type': 'loin_de', 'objet': 1, 'cible': 'porte', 'min': 1.0}, {'type': 'orientation', 'objet': 1, 'vers': 270.0},
        {'type': 'loin_de', 'objet': 4, 'cible': 'p1', 'min': 1.0}, {'type': 'contre_mur', 'objet': 2, 'mur': 'm3'}, {'type': 'pres_de', 'objet': 3, 'cible': 1}]
    path, headers, body = server.seen[-1]
    system = body['messages'][0]['content']
    assert path == '/v1/chat/completions' and 'Authorization' not in headers and body['temperature'] == 0 and body['response_format'] == {'type': 'json_object'}
    assert '- 1 : canape « Canapé »' in system and '- p1 : porte' in system and body['messages'][1]['content'] == REAL
    assert call(ctx, 'layout.rules', {}) == [] and call(ctx, 'layout.rules', {'rules': out['rules']}) == out['rules']


def test_invalid_json(server, ctx):
    server.routes['/v1/chat/completions'] = answer('Voici : {"regles": [{"type": "centre", "objet": 1}]} voilà')
    assert call(ctx, 'ai.rules_from_text', {'text': 'canapé au centre'})['rules'] == [{'type': 'centre', 'objet': 1}]
    for content in ('pas du json', '{"regles": "non"}', '[1, 2'):
        server.routes['/v1/chat/completions'] = answer(content)
        with pytest.raises(RpcError, match='illisible'): call(ctx, 'ai.rules_from_text', {'text': 'canapé au centre'})
    server.routes['/v1/chat/completions'] = lambda body: {'oups': 1}
    with pytest.raises(RpcError, match='inattendue'): call(ctx, 'ai.rules_from_text', {'text': 'canapé au centre'})
    with pytest.raises(RpcError, match='décrivez'): call(ctx, 'ai.rules_from_text', {'text': '  '})


def test_rejections(server, ctx):
    good = {'type': 'centre', 'objet': 1}
    server.routes['/v1/chat/completions'] = answer([
        good, {'type': 'face_a', 'objet': 9, 'cible': 1, 'distance': [2, 3]}, {'type': 'pres_de', 'objet': 4, 'cible': 'Fauteuil'}, {'type': 'pres_de', 'objet': 4},
        {'type': 'loin_de', 'objet': 1, 'cible': 2, 'min': 50}, {'type': 'face_a', 'objet': 2, 'cible': 1, 'distance': [3, 2]},
        {'type': 'face_a', 'objet': 2, 'cible': 1, 'distance': [-1, 2]}, {'type': 'orientation', 'objet': 2, 'angle': 720}, {'type': 'teleporter', 'objet': 1},
        {'type': 'contre_mur', 'objet': 3, 'mur': 'p1'}, {'type': 'pres_de', 'objet': 1, 'cible': 1}, {'type': 'aligne', 'objets': [1, 1]}, good, 'texte',
        {'type': 'pres_de', 'objet': True, 'cible': 2}])
    out = call(ctx, 'ai.rules_from_text', {'text': 'tout'})
    assert out['rules'] == [good] and [r['reason'] for r in out['rejected']] == [
        'meuble inconnu : 9', 'cible inconnue : Fauteuil', 'champ manquant : cible', 'min : 50 hors bornes (0 à 6,6 m)', 'distance : min > max',
        'distance : -1 hors bornes (0 à 6,6 m)', 'angle : 720 hors bornes (-360 à 360)', 'type de règle inconnu : teleporter', 'contre_mur : mur inconnu : p1',
        'un meuble ne peut pas se viser lui-même', 'aligne : au moins 2 meubles différents', 'doublon', 'type de règle inconnu : None', 'meuble inconnu : True']
    assert out['rejected'][1]['rule'] == {'type': 'pres_de', 'objet': 4, 'cible': 'Fauteuil'}


def test_server_absent(ctx, monkeypatch):
    with socket.socket() as s: s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    monkeypatch.setattr(ai, 'URL', f'http://127.0.0.1:{port}/v1')
    status = call(ctx, 'ai.status', {})
    assert not status['server'] and not status['ready'] and 'injoignable' in status['error']
    with pytest.raises(RpcError, match='injoignable'): call(ctx, 'ai.rules_from_text', {'text': 'canapé au centre'})
    call(ctx, 'ai.settings', {'url': 'https://api.exemple.fr/v1'})
    with pytest.raises(RpcError, match='rien à installer'): call(ctx, 'ai.install', {})


def test_install_and_pull(server, ctx, monkeypatch):
    monkeypatch.setattr(ai, 'installed', lambda: None)
    monkeypatch.setattr(ai, 'WINGET', [sys.executable, '-c', 'print("Found Ollama"); print(""); print("Downloading"); print("Successfully installed")'])
    seen = []
    assert call(ctx, 'ai.install', {'progress': lambda f, m='': seen.append((round(f, 2), m))})['server']
    assert seen[1:4] == [(0.15, 'Found Ollama'), (0.3, 'Downloading'), (0.45, 'Successfully installed')] and seen[-1] == (1.0, 'Ollama prêt')
    monkeypatch.setattr(ai, 'WINGET', [sys.executable, '-c', 'print("échec réseau"); raise SystemExit(3)'])
    with pytest.raises(RpcError, match='winget 3'): call(ctx, 'ai.install', {})
    events = [{'status': 'pulling manifest'}, {'digest': 'a', 'total': 100 << 20, 'completed': 50 << 20}, {'digest': 'a', 'total': 100 << 20, 'completed': 100 << 20}, {'status': 'success'}]
    server.routes['/api/pull'] = lambda body: b''.join(json.dumps(e).encode() + b'\n' for e in events)
    seen.clear()
    assert call(ctx, 'ai.pull', {'progress': lambda f, m='': seen.append((round(f, 3), m))})['ready']
    assert server.seen[-1][2] == {'model': ai.MODEL} and (0.495, f'Téléchargement de {ai.MODEL} : 50 / 100 Mo') in seen and seen[-1] == (1.0, 'Modèle prêt')
    server.routes['/api/pull'] = lambda body: b'{"error": "modele introuvable"}\n'
    with pytest.raises(RpcError, match='téléchargement du modèle : modele introuvable'): call(ctx, 'ai.pull', {})


def test_real_ollama(ctx):
    if not call(ctx, 'ai.status', {})['ready']: pytest.skip('Ollama ou le modèle absent')
    out = call(ctx, 'ai.rules_from_text', {'text': REAL})
    rules = {r['type']: r for r in out['rules']}
    assert out['rejected'] == [] and len(out['rules']) == 3 and set(rules) == {'face_a', 'contre_mur', 'pres_de'}
    assert {rules['face_a']['objet'], rules['face_a']['cible']} == {1, 2} and rules['face_a']['distance'][0] <= 2.5 <= rules['face_a']['distance'][1]
    assert rules['contre_mur']['objet'] == 3 and (rules['pres_de']['objet'], rules['pres_de']['cible']) == (4, 1)
