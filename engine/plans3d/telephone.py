import hashlib, hmac, ipaddress, json, os, secrets, socket, subprocess, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .rpc import RpcError, call, cotes_path, read_cotes, rpc

PAGE = os.path.join(os.path.dirname(__file__), 'telephone.html')
HOST, PORT = os.environ.get('PLANS3D_PHONE_HOST', '0.0.0.0'), 8737
TAILSCALE = ipaddress.ip_network('100.64.0.0/10')
OPS = {'place': ('cotes.place', {'value', 'trace', 'id', 'label', 'expr'}), 'remove': ('cotes.remove', {'id'})}
CSP = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:"


def addresses():
    found = sorted({a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)})
    tail = next((a for a in found if ipaddress.ip_address(a) in TAILSCALE), None)
    if not tail:
        try: tail = subprocess.run(['tailscale', 'ip', '-4'], capture_output=True, text=True, timeout=3, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout.split()[0]
        except (OSError, IndexError, subprocess.SubprocessError): pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s: s.connect(('192.0.2.1', 9)); lan = s.getsockname()[0]
    except OSError: lan = None
    home = lambda a: ipaddress.ip_address(a).is_private and ipaddress.ip_address(a) not in TAILSCALE and not ipaddress.ip_address(a).is_loopback
    if not (lan and home(lan)): lan = next((a for a in found if home(a)), None)
    return [(kind, ip) for kind, ip in (('tailscale', tail), ('wifi', lan)) if ip]


class Server(ThreadingHTTPServer):
    allow_reuse_address, daemon_threads = False, True


class Phone:
    def __init__(self, ctx):
        self.ctx, self.project, self.token, self.open = ctx, ctx.require_project(), secrets.token_urlsafe(32), True
        self.scene, self.version, self.clients, self.changed = {}, 0, 0, threading.Condition()
        self.key = hashlib.sha1(os.path.abspath(self.project.folder).encode()).hexdigest()[:12]
        with open(PAGE, 'rb') as fh: self.page = fh.read()
        try: self.server = Server((HOST, PORT), Handler)
        except OSError: self.server = Server((HOST, 0), Handler)
        self.server.phone = self
        threading.Thread(target=self.server.serve_forever, daemon=True, name='telephone').start()

    def notify(self, **event):
        with self.changed: self.version += 1; self.changed.notify_all()
        if event: self.ctx.send({'event': 'phone', **event})

    def connect(self, step):
        with self.changed: self.clients += step; n = self.clients
        self.ctx.send({'event': 'phone', 'clients': n})

    def state(self):
        path = cotes_path(self.project)
        return self.version, os.stat(path).st_mtime_ns if os.path.exists(path) else 0

    def payload(self):
        with self.ctx.lock: cotes = read_cotes(self.project)
        return json.dumps({'project': os.path.basename(os.path.normpath(self.project.folder)), 'key': self.key, 'scene': self.scene, 'cotes': cotes}).encode()

    def act(self, op):
        if self.ctx.project is not self.project: raise RpcError('ce projet est fermé sur le PC')
        kind = op.get('op') if isinstance(op, dict) else None
        if kind == 'hide':
            self.ctx.send({'event': 'phone', 'hide': str(op['id']), 'shown': bool(op['shown'])}); return {}
        if kind not in OPS: raise RpcError('opération inconnue')
        method, allowed = OPS[kind]
        with self.ctx.lock:
            old = next((r for r in read_cotes(self.project) if r['id'] == str(op.get('id'))), {})
            row = call(self.ctx, method, {k: v for k, v in op.items() if k in allowed}) or old
        self.notify(changed=row.get('id'), label=row.get('label'), value=None if kind == 'remove' else row.get('value'), removed=kind == 'remove')
        return {'row': row}

    def stop(self):
        with self.changed: self.open = False; self.changed.notify_all()
        self.server.shutdown(); self.server.server_close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def route(self):
        parts = self.path.split('?')[0].split('/')
        ok = len(parts) > 2 and hmac.compare_digest(parts[1].encode(), self.server.phone.token.encode())
        return '/'.join(parts[2:]) if ok else None

    def reply(self, code, body=b'', kind='application/json; charset=utf-8', extra=()):
        self.send_response(code)
        for name, value in (('Content-Type', kind), ('Content-Length', str(len(body))), ('Cache-Control', 'no-store'), ('Referrer-Policy', 'no-referrer'), ('X-Content-Type-Options', 'nosniff'), *extra):
            self.send_header(name, value)
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        where = self.route()
        if where == '': return self.reply(200, self.server.phone.page, 'text/html; charset=utf-8', [('Content-Security-Policy', CSP)])
        if where == 'events': return self.events()
        self.reply(404)

    def do_POST(self):
        if self.route() != 'api': return self.reply(404)
        size = int(self.headers.get('Content-Length') or 0)
        if not 0 < size <= 65536: return self.reply(413)
        try: result = self.server.phone.act(json.loads(self.rfile.read(size)))
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
def start(ctx):
    stop(ctx)
    ctx.phone = Phone(ctx); port = ctx.phone.server.server_address[1]
    return {'links': [{'kind': kind, 'host': f'{ip}:{port}', 'url': f'http://{ip}:{port}/{ctx.phone.token}/'} for kind, ip in addresses()]}


@rpc('phone.stop')
def stop(ctx):
    phone, ctx.phone = ctx.phone, None
    if phone: phone.stop()


@rpc('phone.scene')
def scene(ctx, scene):
    if ctx.phone and scene != ctx.phone.scene: ctx.phone.scene = scene; ctx.phone.notify()
