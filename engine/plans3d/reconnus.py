import hashlib, json, os, re
import numpy as np
from . import furniture, photos, semantique as S
from .core import Obs
from .rpc import RpcError, rpc

R = 'https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/6b7c523703ae26e39be4b03774966f8571c7f523/'
S.FILES.update({'clip_vision': (R + 'onnx/vision_model_quantized.onnx', '583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299', 89117001),
                'clip_text': (R + 'onnx/text_model_quantized.onnx', '73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a', 64504507),
                'clip_vocab': (R + 'vocab.json', '5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363', 862328),
                'clip_merges': (R + 'merges.txt', '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a', 524619)})
CLIP = ('clip_vision', 'clip_text', 'clip_vocab', 'clip_merges')
KINDS = {'lit': ('lit',), 'canape': ('canape',), 'fauteuil': ('fauteuil',), 'chaise': ('chaise', 'tabouret'), 'table': ('table',), 'bureau': ('bureau', 'table'),
         'armoire': ('armoire', 'commode', 'meuble_tv'), 'etagere': ('etagere',), 'banc': ('banc',)}
MIN_POINTS, MIN_SIZE_M, TOP, SCHEMA = 40, 0.3, 5, 'create table if not exists meubles_echelle(source text, piece text, item text, units real, metres real, sigma real, primary key(source, piece))'
WORD = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+", re.I)
MEAN, STD = np.array([0.48145466, 0.4578275, 0.40821073]), np.array([0.26862954, 0.26130258, 0.27577711])


clip_file = S.path
def clip_ready(): return all(os.path.exists(clip_file(n)) for n in CLIP)


def install_clip(progress):
    S.install(lambda f, m='': progress(0.3 * f, m))
    todo = [n for n in CLIP if not os.path.exists(clip_file(n))]
    for k, name in enumerate(todo):
        S.fetch(name, progress, 0.3 + 0.7 * k / len(todo), 0.3 + 0.7 * (k + 1) / len(todo))


def byte_map():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + len(cs) - 188)
    return dict(zip(bs, map(chr, cs)))


def tokenizer():
    if 'tok' not in S.SESSION:
        vocab = json.load(open(clip_file('clip_vocab'), encoding='utf-8'))
        ranks = {tuple(line.split()): i for i, line in enumerate(open(clip_file('clip_merges'), encoding='utf-8').read().split('\n')[1:]) if len(line.split()) == 2}
        S.SESSION['tok'] = (vocab, ranks, byte_map())
    return S.SESSION['tok']


def encode(text):
    vocab, ranks, bytes_ = tokenizer(); ids = [49406]
    for word in WORD.findall(' '.join(text.lower().split())):
        parts = list(''.join(bytes_[b] for b in word.encode())); parts[-1] += '</w>'
        while len(parts) > 1:
            rank, k = min((ranks.get(pair, 1e9), k) for k, pair in enumerate(zip(parts, parts[1:])))
            if rank == 1e9: break
            parts[k:k + 2] = [parts[k] + parts[k + 1]]
        ids += [vocab.get(p, 0) for p in parts]
    ids = ids[:76] + [49407]
    return ids + [49407] * (77 - len(ids))


def clip_session(name):
    if name not in S.SESSION:
        S.session(); import onnxruntime as ort
        S.SESSION[name] = ort.InferenceSession(clip_file(name), providers=['CPUExecutionProvider'])
    return S.SESSION[name]


def unit(rows): rows = np.atleast_2d(rows); return rows / np.linalg.norm(rows, axis=1, keepdims=True).clip(1e-9)


def describe(entry): return 'a photo of a ' + re.sub(r'[\d×x,]+\s*(cm)?', ' ', entry['keywords']).strip()


def text_embeddings(entries):
    file = os.path.join(S.FOLDER, 'clip_catalogue.npz'); key = hashlib.sha256('|'.join(describe(e) for e in entries).encode()).hexdigest()
    if os.path.exists(file):
        with np.load(file) as saved:
            if str(saved['key']) == key: return saved['E']
    session, found = clip_session('clip_text'), []
    for k in range(0, len(entries), 64):
        found.append(session.run(None, {'input_ids': np.array([encode(describe(e)) for e in entries[k:k + 64]], np.int64)})[0])
    embeddings = unit(np.vstack(found)) if found else np.zeros((0, 512)); np.savez(file, E=embeddings, key=key)
    return embeddings


def image_embedding(path, box, orientation):
    from PIL import Image
    image = Image.open(path).convert('RGB').crop(tuple(int(v) for v in box)); turn = {3: 180, 6: 270, 8: 90}.get(int(orientation))
    image = image.rotate(turn, expand=True) if turn else image; w, h = image.size; f = 224 / min(w, h)
    image = image.resize((max(224, round(w * f)), max(224, round(h * f))), Image.BICUBIC); w, h = image.size; left, top = (w - 224) // 2, (h - 224) // 2
    x = ((np.asarray(image.crop((left, top, left + 224, top + 224)), np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    return unit(clip_session('clip_vision').run(None, {'pixel_values': x})[0])[0]


def best_view(data, thing):
    frame, model = data['frame'], data['model']; low, high = np.array(thing['center']) - np.array(thing['size']) / 2, np.array(thing['center']) + np.array(thing['size']) / 2
    corners = np.array([[(low, high)[i][0], (low, high)[j][1], (low, high)[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    world = (corners / frame['s_est'] - frame['t']) @ np.array(frame['R']); best = (0.0, None)
    for i, name in enumerate(model['names']):
        u, v, depth = S.pixels(model, i, world); (width, height) = model['wh'][i]
        if (depth <= 0).any(): continue
        box = np.array([u.min(), v.min(), u.max(), v.max()]); clipped = np.array([max(box[0], 0), max(box[1], 0), min(box[2], width), min(box[3], height)])
        area = max(clipped[2] - clipped[0], 0) * max(clipped[3] - clipped[1], 0); full = (box[2] - box[0]) * (box[3] - box[1])
        if area <= 0 or area < 0.7 * full: continue
        labels = S.read_map(data['cache'], name)
        if labels is None: continue
        rows, cols = labels.shape; r0, r1, c0, c1 = (int(clipped[1] * rows / height), int(np.ceil(clipped[3] * rows / height)), int(clipped[0] * cols / width), int(np.ceil(clipped[2] * cols / width)))
        share = float((labels[r0:r1, c0:c1] == S.INDEX[thing['kind']]).mean()) if r1 > r0 and c1 > c0 else 0.0
        score = area / (width * height) * share
        if share > 0.3 and score > best[0]: best = (score, (i, clipped))
    return best[1]


def footprint(size):
    horizontal = sorted((size[0], size[2]), reverse=True)
    return np.array([horizontal[0], horizontal[1], size[1]])


def rank(thing, entries, sims=None):
    measured = footprint(thing['size']); found = []
    for k, entry in enumerate(entries):
        w, h, d = entry['size']; target = np.array([max(w, d), min(w, d), h]); weights = np.array([1.0, 1.0, 0.5])
        factor = float(np.exp(np.sum(weights * np.log(target / measured.clip(0.05))) / weights.sum()))
        error = float(np.sqrt(np.sum(weights * ((target - factor * measured) / target) ** 2) / weights.sum()))
        found.append((error + 0.5 * abs(np.log(factor)) - (0.03 * sims[k] if sims is not None else 0.0), k, factor))
    return sorted(found)[:TOP]


def pieces(project, source):
    result = S.understand(project, source)
    if not result: raise RpcError('analyse des photos non faite : « Comprendre la pièce » à l\'étape des murs')
    return result, [dict(t, key=f"{t['kind']}-{k}") for k, t in enumerate(result['furniture']) if t['kind'] in KINDS and t['n'] >= MIN_POINTS and max(t['size']) >= MIN_SIZE_M]


@rpc('meubles.reconnus', long=True)
def reconnus(ctx, source='photos', progress=None):
    progress = progress or photos.no_progress; project = ctx.require_project(); result, things = pieces(project, source)
    catalog = [e for e in furniture.read('ikea.json')]
    if not catalog: return {'catalog': False, 'pieces': []}
    if not clip_ready(): install_clip(lambda f, m='': progress(0.8 * f, m))
    progress(0.85, 'comparaison au catalogue'); data = result['data']; to_project, factor = S.projector(project, source, data['frame'])
    confirmed = {row[0]: row[1] for row in confirmed_rows(project, source)}; listed = []
    for thing in things:
        entries = [e for e in catalog if e['kind'] in KINDS[thing['kind']]]
        if not entries: continue
        view = best_view(data, thing); sims = None
        if view:
            i, box = view; embedded = image_embedding(os.path.join(data['meta']['root'], str(data['model']['names'][i])), box, data['model']['orientation'][i])
            raw = text_embeddings(entries) @ embedded; sims = (raw - raw.mean()) / (raw.std() or 1.0)
        size = footprint(thing['size']) * factor
        listed.append({'key': thing['key'], 'kind': thing['kind'], 'label': thing['label'], 'size': size.round(3).tolist(), 'n': thing['n'], 'photo': view and str(data['model']['names'][view[0]]),
                       'center': to_project(np.array(thing['center'])).round(3).tolist(), 'confirmed': confirmed.get(thing['key']),
                       'candidates': [{'id': entries[k]['id'], 'label': entries[k]['label'], 'size': entries[k]['size'], 'factor': round(f / factor, 3)} for _, k, f in rank({**thing, 'size': thing['size']}, entries, sims)]})
    progress(1.0, 'terminé')
    return {'catalog': True, 'pieces': listed}


def confirmed_rows(project, source):
    project.db.execute(SCHEMA)
    return project.db.execute('select piece, item, units, metres, sigma from meubles_echelle where source=?', (source,)).fetchall()


@rpc('meubles.confirmer')
def confirmer(ctx, source, piece, item):
    project = ctx.require_project(); result, things = pieces(project, source); thing = next((t for t in things if t['key'] == piece), None)
    entry = next((e for e in furniture.entries() if e['id'] == item), None)
    if not thing or not entry: raise RpcError('meuble ou article inconnu')
    frame = result['data']['frame']; to_project, _ = S.projector(project, source, frame); size = np.array(thing['size']); center = np.array(thing['center'])
    along = np.array([1.0, 0, 0]) if size[0] >= size[2] else np.array([0, 0, 1.0]); a, b = to_project(center - along / 2), to_project(center + along / 2)
    angle = float(np.degrees(np.arctan2(-(b - a)[2], (b - a)[0]))); w, h, d = entry['size']; rot = angle if w >= d else angle + 90
    at = to_project(center); placed = furniture.add(ctx, item=item, x=float(at[0]), z=float(at[2]), rot=float((rot + 180) % 360 - 180))
    units, metres = float(max(size[0], size[2]) / frame['s_est']), float(max(w, d))
    photos.ensure_params(project, source); project.db.execute(SCHEMA)
    project.db.execute('insert or replace into meubles_echelle values (?,?,?,?,?,?)', (source, piece, item, units, metres, 0.03 + 0.03 * metres)); photos.invalidate(project, source)
    return {'placed': placed, 'scale_m': metres, 'units': units}


@rpc('meubles.oublier')
def oublier(ctx, source, piece):
    project = ctx.require_project(); project.db.execute(SCHEMA); project.db.execute('delete from meubles_echelle where source=? and piece=?', (source, piece)); photos.invalidate(project, source)
    return {'piece': piece}


def observations(project, source):
    try: rows = confirmed_rows(project, source)
    except Exception: return []
    return [Obs(f'{source}:meuble:{piece}', f'{source}_s * {photos.number(units)}', metres, sigma, source, 'déduit', f'{item} confirmé ({photos.number(metres)} m)')
            for piece, item, units, metres, sigma in rows]
