import hashlib, hmac, ipaddress, json, os, re, secrets, shutil, socket, subprocess, threading
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
from . import construction
from .rpc import RpcError, call, cotes_path, read_cotes, rpc

PAGE = os.path.join(os.path.dirname(__file__), 'telephone.html')
HOST, PORT, HTTPS, MOUNT = os.environ.get('PLANS3D_PHONE_HOST'), 8737, 8738, '/3d_plans'
TAILSCALE = ipaddress.ip_network('100.64.0.0/10')
OPS = {'place': ('cotes.place', {'value', 'trace', 'id', 'label', 'expr'}), 'remove': ('cotes.remove', {'id'})}
CSP = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:"


def lan():
    home = lambda a: ipaddress.ip_address(a).is_private and ipaddress.ip_address(a) not in TAILSCALE and not ipaddress.ip_address(a).is_loopback
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s: s.connect(('192.0.2.1', 9)); ip = s.getsockname()[0]
    except OSError: ip = None
    return ip if ip and home(ip) else next((a for a in sorted({a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}) if home(a)), None)


def tailscale(*args):
    text = lambda *out: ''.join(o.decode('utf-8', 'replace') if isinstance(o, bytes) else o or '' for o in out).strip()
    try:
        r = subprocess.run([shutil.which('tailscale') or r'C:\Program Files\Tailscale\tailscale.exe', *args], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=6, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return r.returncode == 0, text(r.stdout, r.stderr)
    except subprocess.TimeoutExpired as e: return False, text(e.stdout, e.stderr)
    except OSError: return None, ''


def served():
    ok, out = tailscale('serve', 'status', '--json')
    try: conf = json.loads(out) if ok else {}
    except ValueError: conf = {}
    mounts = [h.get('Proxy') or '' for k, v in (conf.get('Web') or {}).items() if k.endswith(f':{HTTPS}') for h in (v.get('Handlers') or {}).values()]
    return str(HTTPS) in (conf.get('TCP') or {}), bool(mounts) and all(re.fullmatch(rf'http://127\.0\.0\.1:\d+{MOUNT}', m) for m in mounts)


def unpublish():
    taken, mine = served()
    if taken and mine: tailscale('serve', f'--https={HTTPS}', 'off')


def publish(port):
    ok, out = tailscale('status', '--json', '--peers=false')
    if ok is None: return None, {'text': "Tailscale n'est pas installé sur ce PC."}
    try: status = json.loads(out)
    except ValueError: status = {}
    if status.get('BackendState') != 'Running': return None, {'text': 'Tailscale est arrêté sur ce PC : ouvre Tailscale, puis rouvre ce panneau.'}
    taken, mine = served()
    if taken and not mine: return None, {'text': f'Le port {HTTPS} de Tailscale Serve sert déjà à une autre appli.'}
    ok, out = tailscale('serve', '--bg', '--yes', f'--https={HTTPS}', f'http://127.0.0.1:{port}{MOUNT}')
    if ok: return f"{status['Self']['DNSName'].rstrip('.')}:{HTTPS}", None
    url = re.search(r'https://\S+/f/\S+', out)
    return None, {'text': "HTTPS (Tailscale Serve) n'est pas activé sur ton réseau Tailscale : ouvre ce lien, active-le, puis rouvre ce panneau.", 'url': url[0]} if url else {'text': out or 'Tailscale Serve a échoué.'}


class Server(ThreadingHTTPServer):
    allow_reuse_address, daemon_threads = False, True

    def server_bind(self): TCPServer.server_bind(self); self.server_name, self.server_port = self.server_address[:2]


def bind(host):
    for port in (PORT, 0):
        try: return Server((host, port), Handler)
        except OSError: pass
    raise RpcError('aucun port libre pour la saisie sur le téléphone')


class Phone:
    def __init__(self, ctx, host):
        self.ctx, self.project, self.token, self.open, self.sessions = ctx, ctx.require_project(), secrets.token_urlsafe(32), True, set()
        self.scene, self.version, self.clients, self.changed = {}, 0, 0, threading.Condition()
        self.key = hashlib.sha1(os.path.abspath(self.project.folder).encode()).hexdigest()[:12]
        with open(PAGE, 'rb') as fh: self.page = fh.read()
        self.servers = []
        self.port = self.listen(host)

    def listen(self, host):
        port = next((s.server_address[1] for s in self.servers if s.server_address[0] == host), None)
        if port: return port
        server = bind(host); server.phone = self; self.servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True, name='telephone').start()
        return server.server_address[1]

    def notify(self, **event):
        with self.changed: self.version += 1; self.changed.notify_all()
        if event: self.ctx.send({'event': 'phone', **event})

    def connect(self, step):
        with self.changed: self.clients += step; n = self.clients
        self.ctx.send({'event': 'phone', 'clients': n})

    def state(self):
        return self.version, *(os.stat(p).st_mtime_ns if os.path.exists(p) else 0 for p in (cotes_path(self.project), construction.path(self.project)))

    def payload(self):
        with self.ctx.lock: cotes, built = read_cotes(self.project), construction.load(self.project) if self.scene.get('mode') == 'construire' else None
        return json.dumps({'project': os.path.basename(os.path.normpath(self.project.folder)), 'key': self.key, 'scene': self.scene, 'cotes': cotes, 'construction': built}).encode()

    def act(self, op):
        if self.ctx.project is not self.project: raise RpcError('ce projet est fermé sur le PC')
        kind = op.get('op') if isinstance(op, dict) else None
        if kind == 'hide':
            self.ctx.send({'event': 'phone', 'hide': str(op['id']), 'shown': bool(op['shown'])}); return {}
        if kind in ('build', 'undo'):
            if self.scene.get('mode') != 'construire': raise RpcError('la construction est fermée sur le PC')
            with self.ctx.lock:
                if kind == 'undo': call(self.ctx, 'history.undo', {})
                else: call(self.ctx, 'construction.op', {'o': {**{k: v for k, v in op.items() if k not in ('op', 'do', 'at')}, 'op': op.get('do')}})
                built = construction.load(self.project)
            self.notify(built=True); return {'construction': built}
        if kind not in OPS: raise RpcError('opération inconnue')
        method, allowed = OPS[kind]
        with self.ctx.lock:
            old = next((r for r in read_cotes(self.project) if r['id'] == str(op.get('id'))), {})
            row = call(self.ctx, method, {k: v for k, v in op.items() if k in allowed}) or old
        self.notify(changed=row.get('id'), label=row.get('label'), value=None if kind == 'remove' else row.get('value'), removed=kind == 'remove')
        return {'row': row}

    def stop(self):
        with self.changed: self.open = False; self.changed.notify_all()
        for server in self.servers: server.shutdown(); server.server_close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def allowed(self):
        try: return SimpleCookie(self.headers.get('Cookie') or '')['s'].value in self.server.phone.sessions
        except Exception: return False

    def reply(self, code, body=b'', kind='application/json; charset=utf-8', extra=()):
        self.send_response(code)
        for name, value in (('Content-Type', kind), ('Content-Length', str(len(body))), ('Cache-Control', 'no-store'), ('Referrer-Policy', 'no-referrer'), ('X-Content-Type-Options', 'nosniff'), *extra):
            self.send_header(name, value)
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path = self.path.split('?')[0].removeprefix(MOUNT) or '/'
        if path.startswith('/scan'): from . import scan; return scan.get(self, path)
        if path == '/': return self.reply(200, self.server.phone.page, 'text/html; charset=utf-8', [('Content-Security-Policy', CSP)])
        if path not in ('/events', '/session'): return self.reply(404)
        if not self.allowed(): return self.reply(403)
        if path == '/events': return self.events()
        self.reply(204)

    def do_POST(self):
        path, phone, size = self.path.split('?')[0].removeprefix(MOUNT), self.server.phone, int(self.headers.get('Content-Length') or 0)
        if path.startswith('/scan'): from . import scan; return scan.post(self, path, size)
        if path not in ('/api', '/session'): return self.reply(404)
        if not 0 < size <= 65536: return self.reply(413)
        body = self.rfile.read(size)
        if path == '/session':
            try: ok = hmac.compare_digest(str(json.loads(body)['code']).encode(), phone.token.encode())
            except Exception: ok = False
            if not ok: return self.reply(403)
            session = secrets.token_urlsafe(32); phone.sessions.add(session)
            return self.reply(204, extra=[('Set-Cookie', f's={session}; HttpOnly; SameSite=Strict; Path=/' + '; Secure' * (self.server.server_address[0] == '127.0.0.1'))])
        if not self.allowed(): return self.reply(403)
        try: result = phone.act(json.loads(body))
        except Exception as e: return self.reply(400, json.dumps({'error': str(e) if isinstance(e, ValueError) else f'requête refusée : {e!r}'}, ensure_ascii=False).encode())
        self.reply(200, json.dumps(result, ensure_ascii=False).encode())

    def events(self):
        phone, sent, idle = self.server.phone, None, 0
        self.send_response(200)
        for name, value in (('Content-Type', 'text/event-stream'), ('Cache-Control', 'no-store'), ('Referrer-Policy', 'no-referrer')): self.send_header(name, value)
        self.end_headers()
        phone.connect(1)
        try:
            self.wfile.write(b'retry: 2000\n\n')
            while phone.open:
                key = phone.state()
                if key != sent: self.wfile.write(b'data: ' + phone.payload() + b'\n\n'); sent, idle = key, 0
                elif idle >= 5: self.wfile.write(b'event: ping\ndata:\n\n'); idle = 0
                with phone.changed: phone.changed.wait(1)
                idle += 1
        except OSError: pass
        finally: phone.connect(-1)


@rpc('phone.start')
def start(ctx, page=''):
    stop(ctx)
    phone = ctx.phone = Phone(ctx, HOST or '127.0.0.1')
    host, note = publish(phone.port)
    return {'links': [{'kind': 'tailscale', 'host': host, 'url': f'https://{host}/{page}#{phone.token}'}] * bool(host), 'wifi': lan(), 'tailscale': note}


@rpc('phone.wifi')
def wifi(ctx, page=''):
    ip = lan()
    if not (ctx.phone and ip): raise RpcError('pas de réseau local pour la saisie sur le téléphone')
    port = ctx.phone.listen(ip)
    return {'kind': 'wifi', 'host': f'{ip}:{port}', 'url': f'http://{ip}:{port}/{page}#{ctx.phone.token}'}


@rpc('phone.stop')
def stop(ctx):
    phone, ctx.phone = ctx.phone, None
    if phone: phone.stop()
    unpublish()


@rpc('phone.scene')
def scene(ctx, scene):
    if ctx.phone and scene != ctx.phone.scene: ctx.phone.scene = scene; ctx.phone.notify()
