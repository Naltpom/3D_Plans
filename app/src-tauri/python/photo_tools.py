import hashlib, os, shutil, threading, urllib.request, zipfile
from plans3d.rpc import RpcError, rpc

TOOLS = os.environ['PLANS3D_TOOLS']
COLMAP = os.path.join(TOOLS, 'colmap', 'bin', 'colmap.exe')
COLMAP_URL = 'https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip'
COLMAP_SHA256 = 'e9c5cbd84c2ea986d2e970a2473fc2d2e6b34a2cdcf5d3df2765c319a63af881'
INSTALL_LOCK = threading.Lock()


def install_colmap(progress):
    archive, extracted = os.path.join(TOOLS, 'colmap.zip'), os.path.join(TOOLS, 'colmap.tmp')
    digest = hashlib.sha256()
    with urllib.request.urlopen(COLMAP_URL) as response, open(archive, 'wb') as file:
        total, done = int(response.headers['Content-Length']), 0
        while chunk := response.read(1 << 20):
            file.write(chunk); digest.update(chunk); done += len(chunk)
            progress(0.9 * done / total, f'téléchargement de COLMAP : {done >> 20} / {total >> 20} Mo')
    if digest.hexdigest() != COLMAP_SHA256:
        os.remove(archive)
        raise RpcError('archive COLMAP corrompue, réessayez')
    shutil.rmtree(extracted, ignore_errors=True)
    with zipfile.ZipFile(archive) as content:
        members = content.infolist()
        for i, member in enumerate(members, 1):
            content.extract(member, extracted)
            progress(0.9 + 0.1 * i / len(members), 'extraction de COLMAP')
    shutil.rmtree(os.path.join(TOOLS, 'colmap'), ignore_errors=True)
    os.replace(extracted, os.path.join(TOOLS, 'colmap'))
    os.remove(archive)


@rpc('tools.status')
def tools_status(ctx):
    return {'path': TOOLS, 'colmap': os.path.exists(COLMAP)}


@rpc('tools.install', long=True)
def tools_install(ctx, progress=None):
    with INSTALL_LOCK:
        os.makedirs(TOOLS, exist_ok=True)
        if not os.path.exists(COLMAP):
            install_colmap(progress)
        progress(1.0, 'outils photo installés')
        return tools_status(ctx)
