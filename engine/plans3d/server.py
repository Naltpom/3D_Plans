import importlib, json, math, os, sys, threading, traceback
import numpy as np
from .rpc import LONG_METHODS, Ctx, RpcError, call

PLUGINS = ['plans3d.components', 'plans3d.photos', 'plans3d.cadastre', 'plans3d.terrain']
INVALID_REQUEST = 'requête invalide : {"id", "method", "params"} attendu'


def to_json(value):
    if isinstance(value, dict): return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)): return [to_json(v) for v in value]
    if isinstance(value, np.ndarray): return to_json(value.tolist())
    if isinstance(value, np.generic): value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def error_message(error):
    if isinstance(error, KeyError) and error.args:
        text = str(error.args[0]); return text if ' ' in text else f'clé inconnue : {text}'
    if isinstance(error, FileNotFoundError): return f'fichier introuvable : {error.filename}'
    if isinstance(error, ValueError): return str(error)
    return f'erreur interne ({type(error).__name__}) : {error}'


def isolate_std_streams():
    sys.stdout.flush()
    protocol_in = os.fdopen(os.dup(0), encoding='utf-8', errors='replace')
    protocol_out = os.fdopen(os.dup(1), 'w', encoding='utf-8', newline='\n')
    null_in = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null_in, 0)
    os.close(null_in)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    if os.name == 'nt':
        import ctypes, msvcrt
        for std_handle, fd in ((-10, 0), (-11, 1)): ctypes.windll.kernel32.SetStdHandle(std_handle, msvcrt.get_osfhandle(fd))
    return protocol_in, protocol_out


def load_plugins():
    for module in PLUGINS + [m for m in os.environ.get('PLANS3D_PLUGINS', '').split(',') if m]:
        try: importlib.import_module(module)
        except ModuleNotFoundError as e:
            if e.name != module: print(f'module {module} non chargé : {e}', file=sys.stderr)
        except Exception: print(f'module {module} non chargé :', traceback.format_exc(), file=sys.stderr)


def main():
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    (protocol_in, protocol), write_lock = isolate_std_streams(), threading.Lock()

    def send(message):
        line = json.dumps(to_json(message), ensure_ascii=False, allow_nan=False)
        with write_lock: protocol.write(line + '\n'); protocol.flush()

    ctx = Ctx(send)
    load_plugins()

    def run(request_id, method, params):
        ctx.current.request_id = request_id
        try: send({'id': request_id, 'result': call(ctx, method, params)})
        except Exception as e:
            if not isinstance(e, RpcError): traceback.print_exc()
            send({'id': request_id, 'error': error_message(e)})
        finally: ctx.current.request_id = None

    send({'event': 'ready'})
    for line in protocol_in:
        if not line.strip(): continue
        request = None
        try:
            request = json.loads(line); method, params = request['method'], request.get('params') or {}
            if not isinstance(method, str) or not isinstance(params, dict): raise ValueError
        except Exception:
            send({'id': request.get('id') if isinstance(request, dict) else None, 'error': INVALID_REQUEST}); continue
        if method in LONG_METHODS: threading.Thread(target=run, args=(request.get('id'), method, params), name=method).start()
        else: run(request.get('id'), method, params)


if __name__ == '__main__':
    main()
