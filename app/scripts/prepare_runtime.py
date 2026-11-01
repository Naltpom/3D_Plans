import hashlib, io, shutil, subprocess, sys, tarfile, urllib.request
from pathlib import Path

PYTHON_URL = 'https://github.com/astral-sh/python-build-standalone/releases/download/20261026/cpython-3.12.16%2B20261026-x86_64-pc-windows-msvc-install_only_stripped.tar.gz'
PYTHON_SHA256 = '1f2ab53880fc67ea323398c6388e05af28addc99a8e389209f08a51fdff4b149'
APP = Path(__file__).resolve().parent.parent
ENGINE = APP.parent / 'engine'
RUNTIME = APP / 'src-tauri' / 'runtime'
PYTHON = RUNTIME / 'python' / 'python.exe'
STAMP = RUNTIME / 'python' / 'stamp.txt'
IMPORT_ALL = "import importlib, pkgutil, sys; sys.path.insert(0, sys.argv[1]); [importlib.import_module('plans3d.' + m.name) for m in pkgutil.iter_modules([sys.argv[1] + '/plans3d'])]"


def run(*args):
    subprocess.run([PYTHON, '-I', *args], check=True)


def install_python():
    stamp = hashlib.sha256(Path(__file__).read_bytes() + (ENGINE / 'requirements.txt').read_bytes()).hexdigest()
    if STAMP.exists() and STAMP.read_text() == stamp:
        return
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    archive = urllib.request.urlopen(PYTHON_URL).read()
    if hashlib.sha256(archive).hexdigest() != PYTHON_SHA256:
        sys.exit(f'empreinte invalide : {PYTHON_URL}')
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(RUNTIME, filter='data')
    run('-m', 'pip', 'install', '--isolated', '--no-cache-dir', '--no-compile', '--no-warn-script-location', '-r', ENGINE / 'requirements.txt')
    lib = RUNTIME / 'python' / 'Lib'
    for unused in [lib / 'idlelib', lib / 'tkinter', lib / 'turtledemo', RUNTIME / 'python' / 'tcl']:
        shutil.rmtree(unused, ignore_errors=True)
    run('-m', 'compileall', '-q', '-j', '0', '--invalidation-mode', 'unchecked-hash', lib)
    STAMP.write_text(stamp)


def copy_engine():
    target = RUNTIME / 'engine'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(ENGINE / 'plans3d', target / 'plans3d', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(APP / 'src-tauri' / 'python', target, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    run('-m', 'compileall', '-q', '--invalidation-mode', 'unchecked-hash', target)
    run('-c', IMPORT_ALL, target)


install_python()
copy_engine()
