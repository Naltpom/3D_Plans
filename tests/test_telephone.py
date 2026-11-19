import json, threading, urllib.error, urllib.request
import pytest
from plans3d import mesures, telephone  # noqa: F401
from plans3d.rpc import Ctx, call, read_cotes

TRACE = [{'wall': 'A', 's': 0.0}, {'wall': 'A', 's': 1.0}]
IMMICH = {'TCP': {'443': {'HTTPS': True}}, 'Web': {'pc.example.ts.net:443': {'Handlers': {'/': {'Proxy': 'http://127.0.0.1:2283'}}}}}
ENABLE = 'Serve is not enabled on your tailnet.\nTo enable, visit:\n\n         https://login.tailscale.com/f/serve?node=abc'


class Tailnet:
    def __init__(self): self.conf, self.serve, self.calls = json.loads(json.dumps(IMMICH)), (True, ''), []

    def __call__(self, *args):
        self.calls.append(args)
        if args[0] == 'status': return True, json.dumps({'BackendState': 'Running', 'Self': {'DNSName': 'pc.example.ts.net.'}})
        if args[1:] == ('status', '--json'): return True, json.dumps(self.conf)
        port = next(a[8:] for a in args if a.startswith('--https='))
        if args[-1] == 'off': del self.conf['TCP'][port], self.conf['Web'][f'pc.example.ts.net:{port}']; return True, ''
        if self.serve[0]: self.conf['TCP'][port], self.conf['Web'][f'pc.example.ts.net:{port}'] = {'HTTPS': True}, {'Handlers': {'/': {'Proxy': args[-1]}}}
        return self.serve

    def add(self, proxy):
        self.conf['TCP']['8738'], self.conf['Web']['pc.example.ts.net:8738'] = {'HTTPS': True}, {'Handlers': {'/': {'Proxy': proxy}}}


@pytest.fixture(autouse=True)
def tailnet(monkeypatch):
    fake = Tailnet()
    monkeypatch.setattr(telephone, 'tailscale', fake)
    monkeypatch.setattr(telephone, 'lan', lambda: None)
    return fake


@pytest.fixture
def project(tmp_path):
    events = []
    ctx = Ctx(events.append)
    call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'})
    yield ctx, events
    call(ctx, 'phone.stop', {})


@pytest.fixture
def phone(project):
    ctx, events = project
    link = call(ctx, 'phone.start', {})['links'][0]
    return ctx, f'http://127.0.0.1:{ctx.phone.port}/3d_plans/', events, link


def get(url, data=None, cookie=''):
    request = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None, headers={'Content-Type': 'application/json', 'Cookie': cookie})
    try:
        with urllib.request.urlopen(request, timeout=5) as r: return r.status, r.read(), r.headers.get('Set-Cookie', '')
    except urllib.error.HTTPError as e: return e.code, e.read(), ''


def login(base, code):
    status, _, cookie = get(base + 'session', {'code': code})
    assert status == 204 and 'HttpOnly' in cookie and 'SameSite=Strict' in cookie and ('Secure' in cookie) == base.startswith('http://127.0.0.1:')
    return cookie.split(';')[0]


def stream(url, cookie, n, out):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'Cookie': cookie}), timeout=10) as r:
        while len(out) < n:
            line = r.readline().decode()
            if line.startswith('data: '): out.append(json.loads(line[6:]))


def test_tailscale_serve_publishes_loopback_server_and_keeps_other_entries(phone, tailnet):
    ctx, base, _, link = phone
    assert [s.server_address[0] for s in ctx.phone.servers] == ['127.0.0.1'] and link['url'] == f'https://pc.example.ts.net:8738/#{ctx.phone.token}'
    assert tailnet.conf['Web']['pc.example.ts.net:8738'] == {'Handlers': {'/': {'Proxy': f'http://127.0.0.1:{ctx.phone.port}/3d_plans'}}}
    call(ctx, 'phone.stop', {})
    assert tailnet.conf == IMMICH and not any('reset' in c for c in tailnet.calls)


def test_wifi_fallback_with_activation_link(project, tailnet, monkeypatch):
    ctx, _ = project
    tailnet.serve = (False, ENABLE)
    monkeypatch.setattr(telephone, 'lan', lambda: '127.0.0.2')
    started = call(ctx, 'phone.start', {})
    assert [s.server_address[0] for s in ctx.phone.servers] == ['127.0.0.1', '127.0.0.2'] and started['tailscale']['url'] == 'https://login.tailscale.com/f/serve?node=abc'
    assert [link['url'] for link in started['links']] == [f'http://127.0.0.2:{ctx.phone.port}/#{ctx.phone.token}']
    login(f'http://127.0.0.2:{ctx.phone.port}/', ctx.phone.token)
    monkeypatch.setattr(telephone, 'tailscale', lambda *args: (None, ''))
    assert 'pas installé' in call(ctx, 'phone.start', {})['tailscale']['text']


def test_only_own_leftover_entry_removed(project, tailnet):
    ctx, _ = project
    tailnet.add('http://127.0.0.1:5555/3d_plans')
    call(ctx, 'phone.stop', {})
    assert tailnet.conf == IMMICH and tailnet.calls[-1] == ('serve', '--https=8738', 'off')
    tailnet.calls.clear()
    tailnet.add('http://127.0.0.1:9000')
    call(ctx, 'phone.stop', {})
    started = call(ctx, 'phone.start', {})
    assert started['links'] == [] and '8738' in started['tailscale']['text'] and tailnet.conf['Web']['pc.example.ts.net:8738']['Handlers']['/']['Proxy'] == 'http://127.0.0.1:9000'
    assert not any(c[-1] == 'off' or '--bg' in c for c in tailnet.calls)


def test_code_exchanged_for_cookie_and_nothing_else_served(phone):
    ctx, base, _, _ = phone
    assert get(base)[0] == 200 and b'<title>3D_Plans' in get(base)[1] and ctx.phone.token.encode() not in get(base)[1]
    assert [get(base + p)[0] for p in ('events', 'session', 'project.db', '../project.db', '..%2Fproject.db', ctx.phone.token + '/')] == [403, 403, 404, 404, 404, 404]
    assert get(base + 'session', {'code': 'x' * len(ctx.phone.token)})[0] == 403 and get(base + 'api', {'op': 'remove', 'id': 'a'})[0] == 403
    assert get(base + 'api', {'op': 'remove', 'id': 'a'}, 's=faux')[0] == 403
    cookie = login(base, ctx.phone.token)
    assert get(base + 'session', cookie=cookie)[0] == 204 and get(base + 'api', {'op': 'lire', 'path': 'project.db'}, cookie)[0] == 400


def test_two_phones_and_pc_live(phone):
    ctx, base, events, _ = phone
    cookies, seen = [login(base, ctx.phone.token) for _ in range(2)], [[], []]
    readers = [threading.Thread(target=stream, args=(base + 'events', c, 2, out)) for c, out in zip(cookies, seen)]
    for t in readers: t.start()
    for _ in range(50):
        if ctx.phone.clients == 2: break
        threading.Event().wait(0.1)
    assert ctx.phone.clients == 2 and {'event': 'phone', 'clients': 2} in events
    status, body, _ = get(base + 'api', {'op': 'place', 'trace': TRACE, 'value': 132, 'label': 'Longueur du mur A'}, cookies[0])
    assert status == 200 and json.loads(body)['row']['value'] == 132
    for t in readers: t.join(10)
    assert all(out[0]['cotes'] == [] and out[1]['cotes'][0]['value'] == 132 for out in seen)
    assert any(e.get('changed') and e['label'] == 'Longueur du mur A' for e in events)
    call(ctx, 'cotes.place', {'value': 130, 'trace': TRACE})
    assert [r['value'] for r in read_cotes(ctx.project)] == [130]
    assert get(base + 'api', {'op': 'hide', 'id': 'cote.1', 'shown': False}, cookies[1])[0] == 200 and {'event': 'phone', 'hide': 'cote.1', 'shown': False} in events
    assert get(base + 'api', {'op': 'remove', 'id': 'cote.1'}, cookies[1])[0] == 200 and read_cotes(ctx.project) == []
    assert json.loads(get(base + 'api', {'op': 'remove', 'id': 'cote.1'}, cookies[1])[1])['error'] == 'cote inconnue : cote.1'


def test_stop_closes_everything(phone):
    ctx, base, _, _ = phone
    call(ctx, 'phone.scene', {'scene': {'segs': [], 'title': 'Étape 4.2'}})
    out = []
    stream(base + 'events', login(base, ctx.phone.token), 1, out)
    assert out[0]['scene']['title'] == 'Étape 4.2' and out[0]['project'] == 'p'
    call(ctx, 'phone.stop', {})
    with pytest.raises(OSError): urllib.request.urlopen(base, timeout=3)
    assert ctx.phone is None
