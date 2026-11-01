import json, os, queue, subprocess, sys, threading
import pytest

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'engine'))
FAKE_PLUGIN = r'''
import os, subprocess, sys, threading
from plans3d.rpc import rpc
released = threading.Event()

@rpc('test.long', long=True)
def long_task(ctx, n=3):
    for i in range(n):
        ctx.emit({'event': 'progress', 'done': i, 'total': n})
        if i == 0 and not released.wait(20): raise RuntimeError('jamais libérée')
    return 'fini'

@rpc('test.release')
def release(ctx):
    released.set()
    return True

@rpc('test.noisy')
def noisy(ctx):
    print('bruit print')
    os.write(1, b'bruit fd1\n')
    subprocess.run([sys.executable, '-c', 'print("bruit enfant")'])
    return 'ok'

@rpc('test.child', long=True)
def child(ctx):
    done = subprocess.run([sys.executable, '-c', 'import sys; print(repr(sys.stdin.read()))'], stdout=subprocess.PIPE, text=True, timeout=15)
    return done.stdout.strip()

@rpc('test.progress')
def with_progress(ctx, progress=None):
    progress(0.5, 'moitié')
    progress(1.0)
    return 2
'''


class Server:
    def __init__(self, folder):
        (folder / 'fake_plugin.py').write_text(FAKE_PLUGIN, encoding='utf-8')
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join([ENGINE, str(folder)]), 'PLANS3D_PLUGINS': 'fake_plugin'}
        self.process = subprocess.Popen([sys.executable, '-m', 'plans3d.server'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, cwd=str(folder))
        self.messages, self.last_id, self.events = queue.Queue(), 0, []
        threading.Thread(target=lambda: [self.messages.put(json.loads(line)) for line in self.process.stdout], daemon=True).start()
        assert self.next_message() == {'event': 'ready'}

    def next_message(self): return self.messages.get(timeout=20)

    def write(self, data):
        self.process.stdin.write(data); self.process.stdin.flush()

    def send(self, method, **params):
        self.last_id += 1
        self.write((json.dumps({'id': self.last_id, 'method': method, 'params': params}, ensure_ascii=False) + '\n').encode('utf-8'))
        return self.last_id

    def wait(self, request_id):
        while 'event' in (message := self.next_message()) or message.get('id') != request_id: self.events.append(message)
        return message

    def __call__(self, method, **params):
        response = self.wait(self.send(method, **params))
        if 'error' in response: raise RuntimeError(response['error'])
        return response['result']

    def close(self):
        self.process.stdin.close()
        assert self.process.wait(20) == 0


@pytest.fixture
def server(tmp_path):
    started = Server(tmp_path)
    yield started
    if started.process.poll() is None: started.close()


def test_full_scenario(server, tmp_path):
    with pytest.raises(RuntimeError, match='aucun projet ouvert'): server('params.list')
    with pytest.raises(RuntimeError, match='méthode inconnue'): server('nope')
    path = str(tmp_path / 'salon é')
    info = server('project.create', path=path, name='Salon d’essai')
    with open(os.path.join(path, 'project.json'), encoding='utf-8') as fh: assert json.load(fh)['created']
    assert info['name'] == 'Salon d’essai' and info['params'] == 0
    assert all(os.path.isdir(os.path.join(path, sub)) for sub in ('sources', 'cache'))
    with pytest.raises(RuntimeError, match='existe déjà'): server('project.create', path=path, name='x')

    for name in ('w1', 'w2', 'w3'): server('params.add', name=name, init=1.0, axis='x', label=f'mur {name}')
    assert [p['name'] for p in server('params.list')] == ['w1', 'w2', 'w3']
    with pytest.raises(RuntimeError, match='paramètre manquant name'): server('params.add', init=1)
    with pytest.raises(RuntimeError, match='paramètre inconnu w9'): server('cotes.add', id='x', expr='w9 - w1', value=1)
    contradictory = 230
    cotes = [('d1', 'w1', 120), ('d2', 'w2 - w1', 130), ('d3', 'w3 - w2', 60), ('d4', 'w3', 310), ('d5', 'w3 - w1', contradictory), ('d6', 'w2', 250.5)]
    for cote_id, expr, value in cotes: server('cotes.add', id=cote_id, expr=expr, value=value, label=f'cote {cote_id}')
    assert server('sources.list') == [{'id': 'cotes', 'path': os.path.join(path, 'sources', 'cotes.csv'), 'kind': 'cotes_csv', 'extracted': False, 'obs': 0}]

    result = server('update')
    assert result['dof'] == 3 and result['timing']['reextracted'] == ['cotes'] and len(result['obs']) == 6
    assert [[c['id'] for c in group] for group in result['conflicts']] == [['cotes:d5']] and result['conflicts'][0][0]['label'] == 'cote d5'
    params = {p['name']: p for p in result['params']}
    assert abs(params['w3']['value'] - 3.10) < 0.02 and 0 < params['w3']['sigma'] < 0.02 and params['w1']['label'] == 'mur w1'
    best = server('next_best', k=3, targets=['w2'])
    assert len(best) == 3 and best[0]['gain'] >= best[1]['gain'] >= best[2]['gain'] > 0 and 'w2' in best[0]['expr']
    assert {c['expr'] for c in server('next_best', k=20)} == {'w1', 'w2', 'w3', 'w1 - w2', 'w1 - w3', 'w2 - w3'}
    factors = server('variance_factors')
    assert factors[0]['group'] == {'source': 'cotes'} and factors[0]['n'] == 6 and factors[0]['factor'] > 3
    assert server('model')['params']['w2']['obs'] == ['cotes:d2', 'cotes:d3', 'cotes:d6']

    server('cotes.add', id='d5', expr='w3 - w1', value=190, label='cote d5 corrigée')
    result = server('update')
    assert result['conflicts'] == [] and result['timing']['reextracted'] == ['cotes'] and len(server('cotes.list')) == 6
    server('cotes.remove', id='d6')
    assert server('update')['dof'] == 2
    with pytest.raises(RuntimeError, match='cote inconnue'): server('cotes.remove', id='d6')
    model = server('model')
    assert set(model['params']) == {'w1', 'w2', 'w3'} and model['conflicts'] == []
    server.close()

    reopened = Server(tmp_path)
    try:
        assert reopened('project.info') is None
        info = reopened('project.open', path=path)
        assert info['name'] == 'Salon d’essai' and info['params'] == 3 and info['obs'] == 5 and info['model']
        with pytest.raises(RuntimeError, match='aucun résultat'): reopened('next_best')
        result = reopened('update')
        assert result['timing']['reextracted'] == [] and result['dof'] == 2
        reopened('sources.remove', id='cotes')
        assert reopened('sources.list') == [] and reopened('project.info')['obs'] == 0
        result = reopened('update')
        assert result['obs'] == [] and [p['sigma'] for p in result['params']] == [None] * 3
        with pytest.raises(RuntimeError, match='expression invalide'): reopened('cotes.add', id='e', expr='w1 +', value=1)
    finally:
        reopened.close()


def test_long_method_does_not_block(server):
    assert {'name': 'update', 'long': True, 'params': ['exclude']} in server('methods')
    long_id = server.send('test.long', n=3)
    assert server.next_message() == {'event': 'progress', 'request': long_id, 'done': 0, 'total': 3}
    assert server('project.info') is None
    assert server('test.release') is True
    assert server.wait(long_id) == {'id': long_id, 'result': 'fini'}
    assert [e['done'] for e in server.events if e.get('request') == long_id] == [1, 2]


def test_protocol_errors_and_stdout_isolation(server):
    server.write(b'pas du json\n{"id": 7, "method": 3}\n')
    assert server.next_message() == {'id': None, 'error': 'requête invalide : {"id", "method", "params"} attendu'}
    assert server.next_message()['id'] == 7
    assert server('test.noisy') == 'ok' and server('project.info') is None


def test_injected_progress(server):
    assert server('test.progress') == 2
    assert [(e['fraction'], e['message']) for e in server.events if e.get('method') == 'test.progress'] == [(0.5, 'moitié'), (1.0, '')]


def test_child_process_does_not_inherit_protocol_stdin(server):
    child_id = server.send('test.child')
    assert server.wait(child_id) == {'id': child_id, 'result': "''"}
    assert server('project.info') is None
