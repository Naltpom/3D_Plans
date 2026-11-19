import json, threading, urllib.error, urllib.request
import pytest
from plans3d import mesures, telephone  # noqa: F401
from plans3d.rpc import Ctx, call, read_cotes

TRACE = [{'wall': 'A', 's': 0.0}, {'wall': 'A', 's': 1.0}]


@pytest.fixture
def phone(tmp_path, monkeypatch):
    monkeypatch.setattr(telephone, 'addresses', lambda: [('tailscale', '127.0.0.1')])
    events = []
    ctx = Ctx(events.append)
    call(ctx, 'project.create', {'path': str(tmp_path / 'p'), 'name': 'p'})
    link = call(ctx, 'phone.start', {})['links'][0]
    yield ctx, link['url'].split('#')[0], events, link
    call(ctx, 'phone.stop', {})


def get(url, data=None, cookie=''):
    request = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None, headers={'Content-Type': 'application/json', 'Cookie': cookie})
    try:
        with urllib.request.urlopen(request, timeout=5) as r: return r.status, r.read(), r.headers.get('Set-Cookie', '')
    except urllib.error.HTTPError as e: return e.code, e.read(), ''


def login(base, code):
    status, _, cookie = get(base + 'session', {'code': code})
    assert status == 204 and 'HttpOnly' in cookie and 'SameSite=Strict' in cookie
    return cookie.split(';')[0]


def stream(url, cookie, n, out):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'Cookie': cookie}), timeout=10) as r:
        while len(out) < n:
            line = r.readline().decode()
            if line.startswith('data: '): out.append(json.loads(line[6:]))


def test_listens_only_on_shown_addresses(phone):
    ctx, base, _, link = phone
    assert [s.server_address[0] for s in ctx.phone.servers] == ['127.0.0.1'] and link['url'] == f'http://127.0.0.1:{ctx.phone.port}/#{ctx.phone.token}'


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
