import json, threading, urllib.error, urllib.request
import pytest
from plans3d import mesures, telephone  # noqa: F401
from plans3d.rpc import Ctx, call, read_cotes

TRACE = [{'wall': 'A', 's': 0.0}, {'wall': 'A', 's': 1.0}]


@pytest.fixture
def phone(tmp_path, monkeypatch):
    monkeypatch.setattr(telephone, 'HOST', '127.0.0.1')
    events = []
    ctx = Ctx(events.append)
    call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'})
    links = call(ctx, 'phone.start', {})['links']
    base = f'http://127.0.0.1:{ctx.phone.server.server_address[1]}/{ctx.phone.token}/'
    yield ctx, base, events, links
    call(ctx, 'phone.stop', {})


def get(url, data=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(data).encode() if data else None, headers={'Content-Type': 'application/json'}), timeout=5) as r: return r.status, r.read()
    except urllib.error.HTTPError as e: return e.code, e.read()


def stream(url, n, out):
    with urllib.request.urlopen(url, timeout=10) as r:
        while len(out) < n:
            line = r.readline().decode()
            if line.startswith('data: '): out.append(json.loads(line[6:]))


def test_token_required_and_nothing_else_served(phone):
    ctx, base, _, links = phone
    root = base.rsplit('/', 2)[0]
    assert all(link['url'].endswith(f'/{ctx.phone.token}/') and len(ctx.phone.token) >= 40 for link in links)
    assert get(base)[0] == 200 and b'<title>3D_Plans' in get(base)[1]
    for url in (root + '/', root + '/events', root + '/x' * 43 + '/', base[:-2] + '/', base + '../project.db', base + '..%2Fproject.db', root + '/' + ctx.phone.token):
        assert get(url)[0] == 404, url
    assert get(root + '/api', {'op': 'place'})[0] == 404 and get(base + 'api', {'op': 'lire', 'path': 'project.db'})[0] == 400


def test_two_phones_and_pc_live(phone):
    ctx, base, events, _ = phone
    seen = [[], []]
    readers = [threading.Thread(target=stream, args=(base + 'events', 2, out)) for out in seen]
    for t in readers: t.start()
    for _ in range(50):
        if ctx.phone.clients == 2: break
        threading.Event().wait(0.1)
    assert ctx.phone.clients == 2 and {'event': 'phone', 'clients': 2} in events
    status, body = get(base + 'api', {'op': 'place', 'trace': TRACE, 'value': 132, 'label': 'Longueur du mur A'})
    assert status == 200 and json.loads(body)['row']['value'] == 132
    for t in readers: t.join(10)
    assert all(out[0]['cotes'] == [] and out[1]['cotes'][0]['value'] == 132 for out in seen)
    assert any(e.get('changed') and e['label'] == 'Longueur du mur A' for e in events)
    call(ctx, 'cotes.place', {'value': 130, 'trace': TRACE})
    assert [r['value'] for r in read_cotes(ctx.project)] == [130]
    status, body = get(base + 'api', {'op': 'hide', 'id': 'cote.1', 'shown': False})
    assert status == 200 and {'event': 'phone', 'hide': 'cote.1', 'shown': False} in events
    assert get(base + 'api', {'op': 'remove', 'id': 'cote.1'})[0] == 200 and read_cotes(ctx.project) == []
    assert json.loads(get(base + 'api', {'op': 'remove', 'id': 'cote.1'})[1])['error'] == 'cote inconnue : cote.1'


def test_stop_closes_everything(phone):
    ctx, base, _, _ = phone
    call(ctx, 'phone.scene', {'scene': {'segs': [], 'title': 'Étape 4.2'}})
    out = []
    stream(base + 'events', 1, out)
    assert out[0]['scene']['title'] == 'Étape 4.2' and out[0]['project'] == 'p'
    call(ctx, 'phone.stop', {})
    with pytest.raises(OSError): urllib.request.urlopen(base, timeout=3)
    assert ctx.phone is None
