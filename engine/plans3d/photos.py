import json, os, re, shutil, sqlite3, subprocess, time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from .core import Obs, compile_expr
from .project import Project, extractor, file_hash
from .rpc import rpc

YAW_PRIOR_RAD = np.radians(5)
PLANE_SIGMA_M, CLICK_SIGMA_PX, CAMERA_HEIGHT_M, INLIER_M, PLANE_GAP_M, MERGE_M, FRAME_VERSION = 0.015, 3.0, 1.4, 0.01, 0.3, 0.05, 5
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.heic', '.png')
IMAGE_UP = {1: (0, -1, 0), 3: (0, 1, 0), 6: (-1, 0, 0), 8: (1, 0, 0)}
RUNNING = set()
LINE = 'ligne:'
PROGRESS_PATTERN = re.compile(r'\[(\d+)/(\d+)\]|view (\d+) / (\d+)|num_reg_frames=(\d+)')
SCHEMA = '''create table if not exists photos_assoc(source text, plane_id text, expr text, sigma real, primary key(source, plane_id));
create table if not exists photos_axes(source text primary key, map text);
create table if not exists photos_clicks(id integer primary key autoincrement, source text, image text, u real, v real, anchor text);'''


def no_progress(fraction, message=''): pass
def unit(vector): return vector / np.linalg.norm(vector)
def load_json(path): return json.load(open(path, encoding='utf-8'))
def write_lines(path, lines): open(path, 'w', encoding='utf-8').write(''.join(f'{line}\n' for line in lines))
def number(value): return f'{float(value):.12g}'


def colmap_executable():
    tools = os.environ.get('PLANS3D_TOOLS')
    for path in (os.environ.get('PLANS3D_COLMAP'), shutil.which('colmap'), tools and os.path.join(tools, 'colmap', 'bin', 'colmap.exe')):
        if path and os.path.exists(path): return path
    raise ValueError('COLMAP introuvable : installer les outils photo')


def read_exif(path):
    from PIL import Image
    try:
        with Image.open(path) as image:
            tags = image.getexif(); photo = tags.get_ifd(0x8769)
            return {'size': image.size, 'date': f"{photo.get(0x9003) or tags.get(0x0132) or ''}.{photo.get(0x9291) or ''}",
                    'focal35': float(photo.get(0xA405) or 0), 'orientation': int(tags.get(0x0112) or 1)}
    except OSError:
        return {'size': None, 'date': '', 'focal35': 0.0, 'orientation': 1}


def run_colmap(arguments, log_path, progress, start, end, message, image_count=1, line_total=None):
    with open(log_path, 'w', encoding='utf-8') as log, subprocess.Popen(
            [colmap_executable(), *arguments], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            errors='replace', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as process:
        matched = 0
        for line in process.stdout:
            log.write(line); found = PROGRESS_PATTERN.search(line)
            if not found: continue
            matched += 1
            done, total = (matched, line_total) if line_total else (found[1], found[2]) if found[1] else (found[3], found[4]) if found[3] else (found[5], image_count)
            progress(start + (end - start) * min(1.0, int(done) / int(total)), message)
    if process.returncode: raise ValueError(f'COLMAP {arguments[0]} a échoué (journal : {log_path})')


def neighbour_pairs(count, window=10):
    return {(i, j) for i in range(count) for j in range(i + 1, min(count, i + window + 1))}


def reduced_database(database, reduced, keep=1024):
    source = sqlite3.connect(database); target = sqlite3.connect(reduced); source.backup(target); source.close()
    rows = target.execute('select k.image_id, k.rows, k.cols, k.data, d.data from keypoints k join descriptors d using(image_id)').fetchall()
    for image_id, count, columns, keypoint_blob, descriptor_blob in rows:
        keypoints = np.frombuffer(keypoint_blob, np.float32).reshape(count, columns); descriptors = np.frombuffer(descriptor_blob, np.uint8).reshape(count, -1)
        scale = np.abs(keypoints[:, 2] * keypoints[:, 5] - keypoints[:, 3] * keypoints[:, 4]) if columns == 6 else keypoints[:, 2]
        largest = np.sort(np.argsort(-scale)[:keep])
        target.execute('update keypoints set rows=?, data=? where image_id=?', (len(largest), keypoints[largest].tobytes(), image_id))
        target.execute('update descriptors set rows=?, data=? where image_id=?', (len(largest), descriptors[largest].tobytes(), image_id))
    target.commit(); target.close()


def verified_pairs(database, names, min_inliers=6):
    connection = sqlite3.connect(database); index = {image_id: names.index(name) for image_id, name in connection.execute('select image_id, name from images')}
    pairs = {tuple(sorted((index[pair // 2147483647], index[pair % 2147483647])))
             for pair, in connection.execute('select pair_id from two_view_geometries where rows >= ?', (min_inliers,))}
    connection.close(); return pairs


def read_text_model(folder):
    rows = lambda name: [line.split() for line in open(os.path.join(folder, name), encoding='utf-8') if line.strip() and line[0] != '#']
    cameras = {int(row[0]): (int(row[2]), int(row[3]), [float(value) for value in row[4:]]) for row in rows('cameras.txt')}
    lines = [line for line in open(os.path.join(folder, 'images.txt'), encoding='utf-8') if line[0] != '#']
    images = [line.split(maxsplit=9) for line in lines[::2] if line.strip()]
    points = rows('points3D.txt'); errors = np.array([row[7] for row in points], float); tracks = np.array([(len(row) - 8) // 2 for row in points])
    model = {'names': np.array([row[9].strip() for row in images]), 'q': np.array([row[1:5] for row in images], float),
             't': np.array([row[5:8] for row in images], float), 'K': np.array([cameras[int(row[8])][2] for row in images]),
             'wh': np.array([cameras[int(row[8])][:2] for row in images]), 'P': np.array([row[1:4] for row in points], float).reshape(-1, 3)}
    return model, float((errors * tracks).sum() / max(tracks.sum(), 1))


def link_images(paths, image_dir):
    os.makedirs(image_dir, exist_ok=True); names = [f'{i:05d}{os.path.splitext(path)[1].lower()}' for i, path in enumerate(paths)]
    for stale in set(os.listdir(image_dir)) - set(names): os.remove(os.path.join(image_dir, stale))
    for path, name in zip(paths, names):
        target = os.path.join(image_dir, name)
        if os.path.exists(target):
            if os.path.samefile(path, target): continue
            os.remove(target)
        try: os.link(path, target)
        except OSError: shutil.copy2(path, target)
    return image_dir, names


def reconstruct(paths, cache, progress=no_progress):
    in_cache = lambda *parts: os.path.join(cache, *parts)
    if os.path.exists(in_cache('meta.json')): return load_json(in_cache('meta.json'))
    os.makedirs(cache, exist_ok=True); timings = {}; started = time.perf_counter()
    lap = lambda key: timings.__setitem__(key, time.perf_counter() - started - sum(timings.values()))
    exif = {path: read_exif(path) for path in paths}; paths = sorted(paths, key=lambda path: (exif[path]['date'], path)); count = len(paths)
    root = os.path.commonpath([os.path.dirname(path) for path in paths]); originals = [os.path.relpath(path, root).replace('\\', '/') for path in paths]
    image_dir, names = link_images(paths, in_cache('images')); database = in_cache('database.db')
    for stale in (database, database + '-wal', database + '-shm'):
        if os.path.exists(stale): os.remove(stale)
    shutil.rmtree(in_cache('sparse'), ignore_errors=True); os.makedirs(in_cache('sparse'), exist_ok=True); write_lines(in_cache('images.txt'), names)
    sizes = {exif[path]['size'] for path in paths}; camera = []
    if len(sizes) == 1 and None not in sizes:
        (width, height), = sizes; focal = (exif[paths[0]]['focal35'] or 26.0) * np.hypot(width, height) / 43.267
        camera = ['--ImageReader.single_camera', '1', '--ImageReader.camera_params', f'{focal:.1f},{focal:.1f},{width / 2},{height / 2},0,0,0,0']
    run_colmap(['feature_extractor', '--database_path', database, '--image_path', image_dir, '--image_list_path', in_cache('images.txt'),
                '--ImageReader.camera_model', 'OPENCV', *camera, '--FeatureExtraction.use_gpu', '1', '--SiftExtraction.max_num_features', '16384'],
               in_cache('features.log'), progress, 0, .1, 'points clés', count)
    lap('features_s')
    if count * (count - 1) // 2 <= 1000:
        matcher, pair_count = ['exhaustive_matcher'], count * (count - 1) // 2
    else:
        reduced_database(database, in_cache('pre.db'))
        run_colmap(['exhaustive_matcher', '--database_path', in_cache('pre.db'), '--FeatureMatching.use_gpu', '1', '--TwoViewGeometry.min_num_inliers', '6'],
                   in_cache('prematching.log'), progress, .1, .13, 'pré-appariement')
        pairs = sorted(verified_pairs(in_cache('pre.db'), names) | neighbour_pairs(count)); os.remove(in_cache('pre.db')); pair_count = len(pairs)
        write_lines(in_cache('pairs.txt'), [f'{names[i]} {names[j]}' for i, j in pairs]); lap('prematching_s')
        matcher = ['matches_importer', '--match_list_path', in_cache('pairs.txt'), '--match_type', 'pairs']
    run_colmap([*matcher, '--database_path', database, '--FeatureMatching.use_gpu', '1'], in_cache('matching.log'), progress, .13, .3, 'appariement')
    lap('matching_s')
    run_colmap(['mapper', '--database_path', database, '--image_path', image_dir, '--output_path', in_cache('sparse')], in_cache('mapper.log'), progress, .3, .95, 'poses', count)
    lap('mapper_s'); models = []
    for name in os.listdir(in_cache('sparse')):
        run_colmap(['model_converter', '--input_path', in_cache('sparse', name), '--output_path', in_cache('sparse', name), '--output_type', 'TXT'],
                   in_cache('convert.log'), progress, .95, .95, 'lecture')
        models.append((*read_text_model(in_cache('sparse', name)), name))
    if not models: raise ValueError('COLMAP : aucune photo positionnée')
    model, reprojection_px, best = max(models, key=lambda item: len(item[0]['names'])); index = {name: i for i, name in enumerate(names)}
    model['orientation'] = np.array([exif[paths[index[name]]]['orientation'] for name in model['names']])
    model['names'] = np.array([originals[index[name]] for name in model['names']]); np.savez(in_cache('model.npz'), **model)
    meta = {'root': root, 'image_dir': image_dir, 'photos': count, 'registered': len(model['names']), 'models': len(models), 'model': best, 'points': len(model['P']),
            'pairs': pair_count, 'reproj_px': reprojection_px, 'timings': timings}
    json.dump(meta, open(in_cache('meta.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    return meta


def load_model(cache):
    model = dict(np.load(os.path.join(cache, 'model.npz'))); model.setdefault('orientation', model.get('ori', np.ones(len(model['names']), int)))
    return model


def rotation_matrices(quaternions):
    w, x, y, z = quaternions.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                     2 * (y * z - w * x), 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def camera_centers(model):
    return -np.einsum('nji,nj->ni', rotation_matrices(model['q']), model['t'])


def read_vertices(path):
    types = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'uchar': 'u1', 'uint8': 'u1', 'int': 'i4', 'short': 'i2', 'ushort': 'u2'}
    with open(path, 'rb') as file:
        header = []
        while (line := file.readline().decode('ascii', 'replace').strip()) != 'end_header': header.append(line.split())
        count = next(int(words[2]) for words in header if words[:2] == ['element', 'vertex'])
        return np.fromfile(file, np.dtype([(words[2], '<' + types[words[1]]) for words in header if words[0] == 'property']), count)


def read_ply(path):
    vertices = read_vertices(path); points = np.stack([vertices['x'], vertices['y'], vertices['z']], 1).astype(float)
    normals = np.stack([vertices['nx'], vertices['ny'], vertices['nz']], 1).astype(float) if 'nx' in vertices.dtype.names else None
    return points, normals


def point_normals(points, neighbours=12):
    _, nearest = cKDTree(points).query(points, neighbours); local = points[nearest] - points[nearest].mean(1, keepdims=True)
    return np.linalg.eigh(np.einsum('nki,nkj->nij', local, local))[1][:, :, 0]


def ransac_planes(points, normals, threshold, min_points, max_planes=150, min_cos=0.85, hypotheses=256, sample=20000, rng=None):
    rng = rng or np.random.default_rng(0); remaining = np.arange(len(points)); planes = []
    while len(remaining) >= min_points and len(planes) < max_planes:
        seeds = rng.choice(remaining, min(hypotheses, len(remaining)), replace=False); probe = rng.choice(remaining, min(sample, len(remaining)), replace=False)
        seed_normals = normals[seeds]; seed_offsets = np.einsum('ij,ij->i', seed_normals, points[seeds])
        scores = ((np.abs(points[probe] @ seed_normals.T - seed_offsets) < threshold) & (np.abs(normals[probe] @ seed_normals.T) > min_cos)).sum(0)
        best = scores.argmax()
        if scores[best] * len(remaining) / len(probe) < min_points: break
        normal, offset = seed_normals[best], seed_offsets[best]; candidates, candidate_normals = points[remaining], normals[remaining]
        for _ in range(3):
            inliers = (np.abs(candidates @ normal - offset) < threshold) & (np.abs(candidate_normals @ normal) > min_cos)
            if inliers.sum() < 3: break
            center = candidates[inliers].mean(0); fitted = np.linalg.eigh(np.cov((candidates[inliers] - center).T))[1][:, 0]
            normal = fitted if fitted @ normal > 0 else -fitted; offset = normal @ center
        if inliers.sum() < min_points: break
        planes.append((normal, float(offset), remaining[inliers])); remaining = remaining[~inliers]
    return planes


def connected_cells(coordinates, cell):
    cells = np.floor(coordinates / cell).astype(np.int64); unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    key = lambda c: c[:, 0] * 4_000_003 + c[:, 1]; keys = key(unique); order = np.argsort(keys); sorted_keys = keys[order]; rows, columns = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            wanted = key(unique + [dx, dy]); position = np.searchsorted(sorted_keys, wanted).clip(0, len(keys) - 1); hit = sorted_keys[position] == wanted
            rows.append(np.flatnonzero(hit)); columns.append(order[position[hit]])
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    return connected_components(coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(unique),) * 2), directed=False)[1][inverse.ravel()]


def find_frame_and_planes(points, centers, image_ups, normals=None, min_points=None, previous_x=None):
    normals = point_normals(points) if normals is None else normals
    up = unit(image_ups.sum(0)); camera_level = np.median(centers @ up); rough_scale = CAMERA_HEIGHT_M / (camera_level - np.percentile(points @ up, 2))
    min_points = min_points or max(30, len(points) // 6000)
    planes = ransac_planes(points, normals, INLIER_M / rough_scale, min_points)
    floors = [(len(inliers), normal * np.sign(normal @ up), inliers) for normal, _, inliers in planes
              if abs(normal @ up) > np.cos(np.radians(25)) and (points[inliers] @ up).mean() < camera_level]
    floors = [floor for floor in floors if floor[0] >= 0.25 * max(f[0] for f in floors)]
    floor_level = np.percentile(points @ up, 2)
    if floors:
        _, up, inliers = min(floors, key=lambda floor: np.median(points[floor[2]] @ up)); floor_level = np.median(points[inliers] @ up)
        for band_m in (0.15, 0.08, 0.04):
            near = (np.abs(points @ up - floor_level) < band_m / rough_scale) & (np.abs(normals @ up) > 0.85); center = points[near].mean(0)
            fitted = np.linalg.eigh(np.cov((points[near] - center).T))[1][:, 0]; up = fitted * np.sign(fitted @ up); floor_level = center @ up
    walls = [(len(inliers), normal) for normal, _, inliers in planes if abs(normal @ up) < np.sin(np.radians(25))]
    if walls and floors:
        wall_weight = sum(w for w, _ in walls); balance = sum(w * np.outer(n, n) for w, n in walls) + wall_weight * (np.eye(3) - np.outer(up, up))
        fitted = np.linalg.eigh(balance)[1][:, 0]; up = fitted * np.sign(fitted @ up); floor_level = center @ up
    first = unit(np.cross(up, [1, 0, 0] if abs(up[0]) < .9 else [0, 1, 0])); second = np.cross(up, first)
    weights, wall_normals = (np.array([w for w, _ in walls], float), np.array([n for _, n in walls])) if walls else (1.0, normals[np.abs(normals @ up) < 0.3])
    angle = np.angle((weights * np.exp(4j * np.arctan2(wall_normals @ second, wall_normals @ first))).sum()) / 4
    turns = [np.cos(angle + k * np.pi / 2) * first + np.sin(angle + k * np.pi / 2) * second for k in range(4)]
    x_axis = turns[0] if previous_x is None else max(turns, key=lambda axis: axis @ previous_x); rotation = np.array([x_axis, up, np.cross(x_axis, up)])
    translation = -np.array([np.mean(centers @ rotation[0]), floor_level, np.mean(centers @ rotation[2])]); framed = points @ rotation.T + translation
    scale = CAMERA_HEIGHT_M / max(np.median(centers @ up) - floor_level, 1e-9); pieces = []
    for normal, _, inliers in planes:
        framed_normal = rotation @ normal; axis = int(np.abs(framed_normal).argmax()); framed_normal *= np.sign(framed_normal[axis])
        tangent = unit(np.cross(framed_normal, [1, 0, 0] if abs(framed_normal[0]) < .9 else [0, 1, 0]))
        labels = connected_cells(framed[inliers] @ np.array([tangent, np.cross(framed_normal, tangent)]).T, PLANE_GAP_M / scale)
        pieces += [(axis if framed_normal[axis] > np.cos(np.radians(10)) else None, framed_normal, inliers[labels == label])
                   for label in np.unique(labels) if (labels == label).sum() >= min_points]
    groups = merge_coplanar(pieces, framed, MERGE_M / scale, PLANE_GAP_M / scale)
    candidates = sorted((plane_summary(axis, normal, members, framed) for axis, normal, members in groups), key=lambda plane: -plane['n'])
    return {'R': rotation.tolist(), 't': translation.tolist(), 's_est': float(scale), 'thr': float(INLIER_M / scale), 'planes': candidates}


def plane_position(axis, normal, members, framed):
    return framed[members][:, axis].mean() if axis is not None else (framed[members] @ normal).mean()


def merge_coplanar(pieces, framed, tolerance, gap):
    groups = []
    for axis, normal, members in sorted(pieces, key=lambda piece: -len(piece[2])):
        position, low, high = plane_position(axis, normal, members, framed), framed[members].min(0), framed[members].max(0)
        for group in groups:
            if (group[0] == axis and group[1] @ normal > np.cos(np.radians(5)) and abs(plane_position(*group[:3], framed) - position) <= tolerance
                    and np.all(np.minimum(group[4], high) >= np.maximum(group[3], low) - gap)):
                group[1] = unit(group[1] * len(group[2]) + normal * len(members)); group[2] = np.concatenate([group[2], members])
                group[3], group[4] = np.minimum(group[3], low), np.maximum(group[4], high)
                break
        else:
            groups.append([axis, normal, members, low, high])
    return [group[:3] for group in groups]


def plane_summary(axis, normal, members, framed):
    inliers = framed[members]; center = inliers.mean(0); fitted = np.linalg.eigh(np.cov((inliers - center).T))[1][:, 0]; fitted *= np.sign(fitted @ normal)
    distances = (inliers - center) @ fitted; component = int(np.abs(fitted).argmax())
    offset = center @ fitted if axis is None else center[axis]
    return {'axis': 'xyz'[axis] if axis is not None else '', 'normal': fitted.round(5).tolist(), 'offset': float(offset),
            'tilt_deg': round(float(np.degrees(np.arccos(min(1.0, abs(fitted[component]))))), 2), 'n': len(members), 'sigma': float(distances.std()),
            'sigma_offset': float(distances.std() / np.sqrt(len(members))), 'center': center.round(5).tolist(),
            'bbox': [inliers.min(0).round(4).tolist(), inliers.max(0).round(4).tolist()]}


def plane_center(plane):
    center = np.mean(plane['bbox'], 0)
    if plane['axis']: center['xyz'.index(plane['axis'])] = plane['offset']
    return center


def reframe(plane, change, shift):
    corners = np.array([[plane['bbox'][i][0], plane['bbox'][j][1], plane['bbox'][k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ change.T + shift
    return {'id': plane['id'], 'aliases': plane.get('aliases', []), 'normal': change @ plane['normal'], 'center': change @ plane_center(plane) + shift,
            'low': corners.min(0), 'high': corners.max(0)}


def same_plane(new, old, tolerance):
    return (abs(np.dot(new['normal'], old['normal'])) > np.cos(np.radians(5)) and abs((old['center'] - plane_center(new)) @ new['normal']) <= tolerance
            and np.all(np.minimum(np.array(new['bbox'][1]), old['high']) >= np.maximum(np.array(new['bbox'][0]), old['low']) - tolerance))


def image_ups(model):
    return np.einsum('nji,nj->ni', rotation_matrices(model['q']), np.array([IMAGE_UP.get(int(o), IMAGE_UP[1]) for o in model['orientation']], float))


def frame_and_planes(cache, progress=no_progress):
    in_cache = lambda *parts: os.path.join(cache, *parts)
    dense = next((path for path in (in_cache('dense.ply'), in_cache('dense', 'fused.ply')) if os.path.exists(path)), None)
    signature = f'{FRAME_VERSION}:' + (f'{os.path.getsize(dense)}:{os.path.getmtime(dense)}' if dense else 'épars')
    previous = load_json(in_cache('frame.json')) if os.path.exists(in_cache('frame.json')) else {'cloud': None, 'planes': []}
    if previous['cloud'] == signature: return previous
    progress(0.96, 'plans'); model = load_model(cache); centers, ups = camera_centers(model), image_ups(model); points, normals = model['P'], None
    if dense:
        up = unit(ups.sum(0)); voxel = INLIER_M * (np.median(centers @ up) - np.percentile(points @ up, 2)) / CAMERA_HEIGHT_M
        dense_points, dense_normals = read_ply(dense); _, kept = np.unique(np.floor(dense_points / voxel).astype(np.int64), axis=0, return_index=True)
        dense_points = dense_points[kept]
        dense_normals = point_normals(dense_points) if dense_normals is None else dense_normals[kept] / np.linalg.norm(dense_normals[kept], axis=1, keepdims=True).clip(1e-12)
        points, normals = np.vstack([points, dense_points]), np.vstack([point_normals(points), dense_normals])
    result = find_frame_and_planes(points, centers, ups, normals, previous_x=np.array(previous['R'][0]) if 'R' in previous else None); used = set()
    change = np.array(result['R']) @ np.array(previous.get('R', np.eye(3))).T; shift = np.array(result['t']) - change @ np.array(previous.get('t', np.zeros(3)))
    old_planes = [reframe(plane, change, shift) for plane in previous['planes']]; tolerance = MERGE_M / result['s_est']
    next_number = 1 + max([int(plane['id'][1:]) for plane in previous['planes']], default=0)
    for plane in result['planes']:
        matches = [old for old in old_planes if old['id'] not in used and same_plane(plane, old, tolerance)]
        if matches: plane['id'] = matches[0]['id']; plane['aliases'] = sorted({alias for old in matches for alias in [old['id'], *old['aliases']]} - {plane['id']})
        else: plane['id'] = f'p{next_number:02d}'; next_number += 1
        used.update([plane['id'], *plane.get('aliases', [])])
    result['cloud'] = signature; json.dump(result, open(in_cache('frame.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    return result


def ensure_schema(project):
    project.db.executescript(SCHEMA)
    if 'sigma' not in [row[1] for row in project.db.execute('pragma table_info(photos_assoc)')]: project.db.execute('alter table photos_assoc add column sigma real')
def read_manifest(path): return [line.strip() for line in open(path, encoding='utf-8') if line.strip()]
def invalidate(project, source): project.db.execute("update sources set hash='' where id=?", (source,)); project.db.commit()


def expand_paths(paths):
    expanded = []
    for path in map(os.path.abspath, paths):
        if os.path.isdir(path): expanded += [os.path.join(path, name) for name in sorted(os.listdir(path)) if name.lower().endswith(IMAGE_EXTENSIONS)]
        else: expanded.append(path)
    return expanded


def source_cache(project, source):
    ensure_schema(project); row = project.db.execute('select path from sources where id=?', (source,)).fetchone()
    if not row: raise ValueError(f'source photo inconnue : {source}')
    return row[0], os.path.join(project.folder, 'cache', 'photos', file_hash(row[0]))


def photo_state(project, source, progress=no_progress, compute=True):
    manifest, cache = source_cache(project, source)
    if not os.path.exists(os.path.join(cache, 'meta.json')):
        if not compute: raise ValueError(f'source {source} : reconstruction non faite (photos.extract)')
        key, error_path = (os.path.abspath(project.folder), source), os.path.join(cache, 'error.txt'); RUNNING.add(key)
        try:
            if os.path.exists(error_path): os.remove(error_path)
            reconstruct(read_manifest(manifest), cache, progress)
        except Exception as error:
            os.makedirs(cache, exist_ok=True); open(error_path, 'w', encoding='utf-8').write(str(error) or type(error).__name__); raise
        finally:
            RUNNING.discard(key)
    return load_json(os.path.join(cache, 'meta.json')), frame_and_planes(cache, progress), cache


def source_status(project, source, manifest):
    if not os.path.exists(manifest): return {'photos': 0, 'registered': None, 'state': 'échec', 'error': f'manifeste introuvable : {manifest}'}
    cache = os.path.join(project.folder, 'cache', 'photos', file_hash(manifest)); meta, error = os.path.join(cache, 'meta.json'), os.path.join(cache, 'error.txt')
    status = {'photos': len(read_manifest(manifest)), 'registered': load_json(meta)['registered'] if os.path.exists(meta) else None}
    if (os.path.abspath(project.folder), source) in RUNNING: return {**status, 'state': 'en cours'}
    if status['registered'] is not None: return {**status, 'state': 'calculé'}
    if os.path.exists(error): return {**status, 'state': 'échec', 'error': open(error, encoding='utf-8').read()}
    return {**status, 'state': 'à calculer'}


def axis_mapping(project, source):
    row = project.db.execute('select map from photos_axes where source=?', (source,)).fetchone(); text = row[0] if row else '+x+y+z'
    return {axis: (text[2 * i + 1], 1.0 if text[2 * i] == '+' else -1.0) for i, axis in enumerate('xyz')}


def mapping_matrix(mapping):
    matrix = np.zeros((3, 3))
    for i, axis in enumerate('xyz'): matrix['xyz'.index(mapping[axis][0]), i] = mapping[axis][1]
    return matrix


def mapped_frame(project, source, frame):
    matrix = mapping_matrix(axis_mapping(project, source))
    return matrix @ np.array(frame['R']), matrix @ np.array(frame['t'])


def anchor_coordinates(project, anchor):
    from .components import anchor as component_anchor, load, resolve_ref
    component, name = resolve_ref(load(project), anchor)
    return {axis: str(expr) for axis, expr in component_anchor(component, name).items()}


def image_index(model, image):
    names = [str(name) for name in model['names']]
    for i, name in enumerate(names):
        if name == image or os.path.basename(name) == os.path.basename(image): return i
    raise ValueError(f'photo non positionnée : {image}')


def display_to_raw(orientation, width, height, u, v):
    return {6: (v, height - u), 8: (width - v, u), 3: (width - u, height - v)}.get(orientation, (u, v))


def raw_to_display(orientation, width, height, u, v):
    return {6: (height - v, u), 8: (v, width - u), 3: (width - u, height - v)}.get(orientation, (u, v))


def undistort(camera, u, v):
    fx, fy, cx, cy, k1, k2, p1, p2 = camera; x_distorted, y_distorted = (u - cx) / fx, (v - cy) / fy; x, y = x_distorted, y_distorted
    for _ in range(20):
        r2 = x * x + y * y; radial = 1 + k1 * r2 + k2 * r2 * r2
        x = (x_distorted - 2 * p1 * x * y - p2 * (r2 + 2 * x * x)) / radial; y = (y_distorted - p1 * (r2 + 2 * y * y) - 2 * p2 * x * y) / radial
    return x, y


def linear(coefficients, terms): return ' + '.join(f'{number(c)} * {term}' for c, term in zip(coefficients, terms))


def skew(v): return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def scene_depth(model, i):
    camera = model['P'] @ rotation_matrices(model['q'][i:i + 1])[0].T + model['t'][i]; fx, fy, cx, cy = model['K'][i][:4]; (width, height) = model['wh'][i]
    with np.errstate(divide='ignore', invalid='ignore'):
        u, v = fx * camera[:, 0] / camera[:, 2] + cx, fy * camera[:, 1] / camera[:, 2] + cy
    visible = (camera[:, 2] > 0) & (u >= 0) & (u <= width) & (v >= 0) & (v <= height)
    return float(np.median(camera[visible, 2])) if visible.any() else 1.0


def triangulate(model, cameras, group):
    centers = camera_centers(model); directions = [unit(cameras[i].T @ [x, y, 1.0]) for _, i, x, y, _ in group]
    spread = max(np.linalg.norm(np.cross(a, b)) for a in directions for b in directions)
    if spread < 0.03: return None, spread
    projectors = [np.eye(3) - np.outer(d, d) for d in directions]
    return np.linalg.solve(sum(projectors), sum(p @ centers[i] for p, (_, i, *_) in zip(projectors, group))), spread


def yaw_name(project, source):
    return f'{source}_yaw' if project.db.execute('select 1 from params where name=?', (f'{source}_yaw',)).fetchone() else '0'


def line_expressions(source, yaw, expr, axis):
    values = expr[len(LINE):].split(';'); cos, sin = f'cos({yaw})', f'sin({yaw})'
    first, second = ((cos, f'-{sin}'), (sin, cos))['xz'.index(axis)]
    return [f'({first} * (({x}) - {source}_tx) + {second} * (({z}) - {source}_tz)) / {source}_s' for x, z in zip(values[0::2], values[1::2])]


def axis_expression(source, yaw, expr, axis, along, across=0.0):
    position = f'(({expr}) - {source}_t{axis}) / {source}_s'
    if axis == 'y' or yaw == '0': return position, float(along)
    turn = f' - (cos({yaw}) - 1) * {number(along)}'
    if across: turn += f' {"-" if axis == "x" else "+"} sin({yaw}) * ({number(across)})'
    return position + turn, float(along)


def rotated_terms(source, yaw, coordinates):
    delta = {axis: f'(({coordinates[axis]}) - {source}_t{axis})' if axis in coordinates else '0' for axis in 'xyz'}
    if yaw == '0': return [delta['x'], delta['y'], delta['z'], f'{source}_s']
    return [f'(cos({yaw}) * {delta["x"]} - sin({yaw}) * {delta["z"]})', delta['y'], f'(sin({yaw}) * {delta["x"]} + cos({yaw}) * {delta["z"]})', f'{source}_s']


def click_observations(project, source, model, frame, scale_estimate):
    rotation, offset = mapped_frame(project, source, frame); cameras = rotation_matrices(model['q']); clicks = {}; yaw = yaw_name(project, source)
    for click_id, image, u, v, anchor in project.db.execute('select id, image, u, v, anchor from photos_clicks where source=? order by id', (source,)):
        try: coordinates = anchor_coordinates(project, anchor); i = image_index(model, image)
        except ValueError: continue
        (width, height), orientation = model['wh'][i], int(model['orientation'][i])
        clicks.setdefault(anchor, []).append((click_id, i, *undistort(model['K'][i], *display_to_raw(orientation, width, height, u, v)), coordinates))
    observations = []
    for anchor, group in clicks.items():
        coordinates = group[0][4]; known = [axis for axis in 'xyz' if axis in coordinates]
        point, spread = triangulate(model, cameras, group) if len(known) != 2 else (None, 0.0)
        if len(known) == 1:
            if point is not None: observations.append(face_observation(source, yaw, anchor, group, model, rotation, offset, known[0], coordinates[known[0]], point, spread, scale_estimate))
            continue
        terms = rotated_terms(source, yaw, coordinates)
        for click_id, i, x, y, _ in group:
            to_camera = cameras[i] @ rotation.T; columns = np.column_stack([to_camera, model['t'][i] - to_camera @ offset])
            depth_m = scale_estimate * ((cameras[i] @ point + model['t'][i])[2] if point is not None else scene_depth(model, i))
            fx, fy = model['K'][i][:2]; label = f'clic {click_id} sur {anchor}'
            if len(known) == 3:
                observations += [Obs(f'{source}:clic{click_id}:u', linear(fx / depth_m * (columns[0] - x * columns[2]), terms), 0.0, CLICK_SIGMA_PX, source, 'photo', label),
                                 Obs(f'{source}:clic{click_id}:v', linear(fy / depth_m * (columns[1] - y * columns[2]), terms), 0.0, CLICK_SIGMA_PX, source, 'photo', label)]
                continue
            ray = np.array([x, y, 1.0]); missing = next(axis for axis in 'xyz' if axis not in known); by_axis = dict(zip('xyz', to_camera.T))
            parts = {'y': [('', by_axis['y'])], 'x': [(f'cos({yaw})', by_axis['x']), (f'sin({yaw})', by_axis['z'])],
                     'z': [(f'cos({yaw})', by_axis['z']), (f'sin({yaw})', -by_axis['x'])]}[missing]
            scale = depth_m * np.linalg.norm(np.cross(ray, parts[0][1])[:2])
            distance = ' + '.join(f'{factor} * ({linear(fx / scale * (ray @ (-skew(direction) @ columns)), terms)})' if factor else linear(fx / scale * (ray @ (-skew(direction) @ columns)), terms)
                                  for factor, direction in (parts if yaw != '0' else parts[:1]))
            observations.append(Obs(f'{source}:clic{click_id}:ligne', distance, 0.0, CLICK_SIGMA_PX, source, 'photo', label))
    return observations


def face_observation(source, yaw, anchor, group, model, rotation, offset, axis, expr, point, spread, scale_estimate):
    centers = camera_centers(model); depth = np.mean([np.linalg.norm(point - centers[i]) for _, i, *_ in group])
    focal = np.mean([model['K'][i][0] for _, i, *_ in group]); sigma = max(CLICK_SIGMA_PX * depth / focal / spread, 0.005 / scale_estimate)
    mapped = rotation @ point + offset; other = 'z' if axis == 'x' else 'x'
    expression, value = axis_expression(source, yaw, expr, axis, mapped['xyz'.index(axis)], mapped['xyz'.index(other)])
    return Obs(f'{source}:face:{anchor}', expression, value, sigma, source, 'photo', f'face {anchor} ({len(group)} clics)')


@extractor('photos_colmap')
def photos_colmap(path, source):
    project = Project(os.path.dirname(os.path.dirname(os.path.abspath(path))))
    _, frame, cache = photo_state(project, source); mapping = axis_mapping(project, source); scale = resolved_scale(project, source, frame['s_est'])
    yaw = yaw_name(project, source)
    planes = {name: plane for plane in frame['planes'] for name in [plane['id'], *plane.get('aliases', [])]}; observations, seen = [], set()
    for plane_id, expr, sigma_m in project.db.execute('select plane_id, expr, sigma from photos_assoc where source=? order by plane_id', (source,)):
        plane = planes.get(plane_id)
        if not plane or not plane['axis'] or (plane['id'], expr) in seen: continue
        axis, sign = mapping[plane['axis']]; seen.add((plane['id'], expr))
        expressions = line_expressions(source, yaw, expr, axis) if expr.startswith(LINE) else [axis_expression(source, yaw, expr, axis, sign * plane['offset'])[0]]
        observations += [Obs(f'{source}:{plane_id}' + ':b' * k, expression, sign * plane['offset'], (sigma_m or PLANE_SIGMA_M) / scale, source, 'photo', f'plan {plane["id"]} ({axis})')
                         for k, expression in enumerate(expressions)]
    observations += click_observations(project, source, load_model(cache), frame, scale)
    if yaw != '0': observations.append(Obs(f'{source}:lacet', yaw, project.params()[yaw][0], YAW_PRIOR_RAD, source, 'supposé', 'lacet des photos a priori'))
    project.db.close(); return observations


def resolved_scale(project, source, fallback):
    entry = (project.model() or {'params': {}})['params'].get(f'{source}_s')
    return entry['value'] if entry and entry.get('sigma') is not None and entry['value'] > 0 else fallback


def current_values(project, source):
    values = {name: init for name, init in project.db.execute('select name, init from params where name like ?', (f'{source}%',))}
    solved = os.path.exists(model_path := os.path.join(project.folder, 'model.json'))
    if solved: values.update({name: p['value'] for name, p in load_json(model_path)['params'].items() if name in values})
    yaw = values.get(f'{source}_yaw', 0.0); turn = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    return values.get(f'{source}_s', 1.0), np.array([values.get(f'{source}_t{axis}', 0.0) for axis in 'xyz']), turn, solved


def project_transform(project, source, frame):
    scale, translation, turn, solved = current_values(project, source); rotation, offset = mapped_frame(project, source, frame)
    return scale * turn @ rotation, scale * turn @ offset + translation, solved


def project_to_display(model, i, linear_map, shift, points_m):
    local = np.linalg.solve(linear_map, (np.atleast_2d(np.asarray(points_m, float)) - shift).T).T @ rotation_matrices(model['q'][i:i + 1])[0].T + model['t'][i]
    fx, fy, cx, cy, k1, k2, p1, p2 = model['K'][i]; depth = local[:, 2]; x, y = local[:, 0] / depth, local[:, 1] / depth
    r2 = x * x + y * y; radial = 1 + k1 * r2 + k2 * r2 * r2
    u = fx * (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + cx; v = fy * (y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + cy
    (width, height), orientation = model['wh'][i], int(model['orientation'][i]); u, v = raw_to_display(orientation, width, height, u, v)
    size = [int(height), int(width)] if orientation in (6, 8) else [int(width), int(height)]
    return [[float(a), float(b)] if d > 0 else None for a, b, d in zip(u, v, depth)], size, orientation


def ensure_params(project, source):
    existing = project.params()
    for name, axis, label, init in [(f'{source}_s', 'échelle', 'm par unité photo', 1.0), (f'{source}_yaw', 'photo', 'lacet des photos (rad)', 0.0)] + [
            (f'{source}_t{a}', 'photo', f'translation {a} des photos', 0.0) for a in 'xyz']:
        if name not in existing: project.add_param(name, init, axis, label)


@rpc('photos.add')
def add(ctx, paths, source='photos'):
    project = ctx.require_project(); source = re.sub(r'\W', '_', source); os.makedirs(os.path.join(project.folder, 'sources'), exist_ok=True)
    manifest = os.path.join(project.folder, 'sources', f'{source}.photos.txt'); old = read_manifest(manifest) if os.path.exists(manifest) else []
    photos = old + [path for path in dict.fromkeys(expand_paths(paths)) if path not in old]
    if not photos: raise ValueError('aucune photo')
    write_lines(manifest, photos); project.add_source(source, manifest, 'photos_colmap')
    ensure_params(project, source)
    return {'source': source, 'manifest': manifest, 'photos': len(photos)}


@rpc('photos.extract', long=True)
def extract(ctx, source='photos', progress=no_progress):
    project = ctx.require_project(); started = time.perf_counter(); meta, frame, _ = photo_state(project, source, progress)
    project.db.execute('update params set init=? where name=?', (frame['s_est'], f'{source}_s')); project.db.commit(); progress(1.0, 'terminé')
    return {**meta, 'planes': len(frame['planes']), 's_est': frame['s_est'], 'total_s': time.perf_counter() - started}


@rpc('photos.densify', long=True)
def densify(ctx, source='photos', max_size=1000, progress=no_progress):
    project = ctx.require_project(); meta, _, cache = photo_state(project, source, compute=False); started = time.perf_counter()
    dense = lambda *parts: os.path.join(cache, 'dense', *parts)
    run_colmap(['image_undistorter', '--image_path', meta.get('image_dir', meta['root']), '--input_path', os.path.join(cache, 'sparse', meta['model']), '--output_path', dense(),
                '--max_image_size', str(max_size)], os.path.join(cache, 'undistort.log'), progress, 0, .03, 'redressement')
    run_colmap(['patch_match_stereo', '--workspace_path', dense(), '--PatchMatchStereo.geom_consistency', '1'], os.path.join(cache, 'patchmatch.log'),
               progress, .03, .95, 'profondeurs', line_total=2 * meta['registered'])
    run_colmap(['stereo_fusion', '--workspace_path', dense(), '--output_path', dense('fused.ply')], os.path.join(cache, 'fusion.log'), progress, .95, .99, 'fusion')
    frame = frame_and_planes(cache, progress); progress(1.0, 'terminé')
    return {'dense_points': len(read_ply(dense('fused.ply'))[0]), 'planes': len(frame['planes']), 'total_s': time.perf_counter() - started}


@rpc('photos.planes')
def planes(ctx, source='photos'):
    project = ctx.require_project(); _, frame, _ = photo_state(project, source, compute=False); mapping = axis_mapping(project, source)
    associated = {plane_id: (expr, sigma_m) for plane_id, expr, sigma_m in project.db.execute('select plane_id, expr, sigma from photos_assoc where source=?', (source,))}
    linear_map, shift, solved = project_transform(project, source, frame); listed = []
    for plane in frame['planes']:
        expr, sigma_m = next((associated[name] for name in [plane['id'], *plane.get('aliases', [])] if name in associated), (None, None))
        entry = {**plane, 'expr': expr, 'expr_sigma_m': sigma_m}
        if plane['axis']:
            axis, sign = mapping[plane['axis']]
            center = linear_map @ np.linalg.solve(np.array(frame['R']), np.array(plane.get('center', plane_center(plane))) - frame['t']) + shift
            entry.update(proj_axis=axis, proj_offset=sign * plane['offset'], position_m=float(center['xyz'.index(axis)]) if solved else None)
        listed.append(entry)
    return {'R': frame['R'], 't': frame['t'], 's_est': frame['s_est'], 'axes': {a: f"{'+' if s > 0 else '-'}{b}" for a, (b, s) in mapping.items()}, 'planes': listed}


@rpc('photos.associate')
def associate(ctx, source, plane_id, expr, sigma=None):
    project = ctx.require_project(); _, frame, _ = photo_state(project, source, compute=False)
    plane = next((plane for plane in frame['planes'] if plane['id'] == plane_id), None)
    if not plane: raise ValueError(f'plan inconnu : {plane_id}')
    if not plane['axis']: raise ValueError(f'{plane_id} : plan oblique, association impossible')
    try: compile_expr(expr, {name: i for i, name in enumerate(project.params())})
    except KeyError as error: raise ValueError(error.args[0]) from None
    if sigma is not None and not float(sigma) > 0: raise ValueError(f'σ invalide : {sigma}')
    ensure_params(project, source)
    project.db.execute('insert or replace into photos_assoc values (?,?,?,?)', (source, plane_id, expr, None if sigma is None else float(sigma))); invalidate(project, source)
    return {'source': source, 'plane_id': plane_id, 'expr': expr, 'sigma': sigma}


@rpc('photos.unassociate')
def unassociate(ctx, source, plane_id):
    project = ctx.require_project(); ensure_schema(project)
    project.db.execute('delete from photos_assoc where source=? and plane_id=?', (source, plane_id)); invalidate(project, source)
    return {'source': source, 'plane_id': plane_id}


@rpc('photos.axes')
def axes(ctx, source, mapping):
    if not re.fullmatch(r'([+-][xyz]){3}', mapping) or sorted(mapping[1::2]) != list('xyz'): raise ValueError(f'correspondance d’axes invalide : {mapping}')
    project = ctx.require_project(); ensure_schema(project)
    project.db.execute('insert or replace into photos_axes values (?,?)', (source, mapping)); invalidate(project, source)
    return {'source': source, 'axes': mapping}


@rpc('photos.cameras')
def cameras(ctx, source='photos'):
    project = ctx.require_project(); meta, frame, cache = photo_state(project, source, compute=False); model = load_model(cache)
    linear_map, shift, solved = project_transform(project, source, frame); centers = camera_centers(model) @ linear_map.T + shift
    to_project = (linear_map / np.linalg.norm(linear_map[0])) @ rotation_matrices(model['q']).transpose(0, 2, 1)
    return {'solved': solved, 'cameras': [{'name': str(name), 'path': os.path.join(meta['root'], str(name)), 'center': center.tolist(), 'R': rotation.tolist(),
                                            'K': camera.tolist(), 'size': size.tolist(), 'orientation': int(orientation)}
                                           for name, center, rotation, camera, size, orientation in
                                           zip(model['names'], centers, to_project, model['K'], model['wh'], model['orientation'])]}


@rpc('photos.project_points')
def project_points(ctx, source, image, points_xyz_m):
    project = ctx.require_project(); _, frame, cache = photo_state(project, source, compute=False); model = load_model(cache)
    linear_map, shift, _ = project_transform(project, source, frame)
    uv, size, orientation = project_to_display(model, image_index(model, image), linear_map, shift, points_xyz_m)
    return {'uv': uv, 'size': size, 'orientation': orientation}


@rpc('photos.click')
def click(ctx, source, image, u, v, anchor):
    project = ctx.require_project(); _, _, cache = photo_state(project, source, compute=False); model = load_model(cache)
    i = image_index(model, image); coordinates = anchor_coordinates(project, anchor); width, height = model['wh'][i]
    limits = (height, width) if int(model['orientation'][i]) in (6, 8) else (width, height)
    if not (0 <= u <= limits[0] and 0 <= v <= limits[1]): raise ValueError(f'clic hors de la photo ({u:.0f}, {v:.0f})')
    ensure_params(project, source)
    cursor = project.db.execute('insert into photos_clicks(source, image, u, v, anchor) values (?,?,?,?,?)', (source, str(model['names'][i]), float(u), float(v), anchor))
    invalidate(project, source)
    return {'id': cursor.lastrowid, 'kind': {3: 'point', 2: 'arête', 1: 'face'}[len(coordinates)]}


@rpc('photos.clicks')
def clicks(ctx, source='photos'):
    project = ctx.require_project(); ensure_schema(project); listed = []
    rows = project.db.execute('select id, image, u, v, anchor from photos_clicks where source=? order by id', (source,)).fetchall()
    for click_id, image, u, v, anchor in rows:
        try: kind = {3: 'point', 2: 'arête', 1: 'face'}[len(anchor_coordinates(project, anchor))]
        except ValueError: kind = 'ancre inconnue'
        listed.append({'id': click_id, 'image': image, 'u': u, 'v': v, 'anchor': anchor, 'kind': kind,
                       'photos_for_anchor': len({row[1] for row in rows if row[4] == anchor})})
    return listed


@rpc('photos.unclick')
def unclick(ctx, id):
    project = ctx.require_project(); ensure_schema(project)
    row = project.db.execute('select source from photos_clicks where id=?', (id,)).fetchone()
    if not row: raise ValueError(f'clic inconnu : {id}')
    project.db.execute('delete from photos_clicks where id=?', (id,)); invalidate(project, row[0])
    return {'id': id}


def cloud_sample(cache, limit):
    dense, saved = os.path.join(cache, 'dense', 'fused.ply'), os.path.join(cache, 'cloud.npz')
    if not os.path.exists(dense): points = load_model(cache)['P']; return points[np.random.default_rng(0).permutation(len(points))[:limit]], np.zeros((0, 3), np.uint8)
    signature = f'{os.path.getsize(dense)}:{os.path.getmtime(dense)}:{limit}'
    if os.path.exists(saved):
        with np.load(saved) as data:
            if str(data['signature']) == signature: return data['P'], data['C']
    vertices = read_vertices(dense); kept = np.random.default_rng(0).permutation(len(vertices))[:limit]; vertices = vertices[kept]
    points = np.stack([vertices['x'], vertices['y'], vertices['z']], 1).astype(float)
    colors = np.stack([vertices['red'], vertices['green'], vertices['blue']], 1).astype(np.uint8) if 'red' in vertices.dtype.names else np.zeros((0, 3), np.uint8)
    np.savez(saved, P=points, C=colors, signature=signature)
    return points, colors


@rpc('photos.cloud')
def cloud(ctx, source='photos', limit=50000):
    project = ctx.require_project(); _, frame, cache = photo_state(project, source, compute=False)
    points, colors = cloud_sample(cache, int(limit)); linear_map, shift, _ = project_transform(project, source, frame); points = points @ linear_map.T + shift
    low, high = np.percentile(points, 1, 0), np.percentile(points, 99, 0); margin = 0.1 * (high - low)
    kept = np.all((points > low - margin) & (points < high + margin), 1); kept[np.flatnonzero(kept)[int(limit):]] = False
    return {'points': points[kept].round(3).ravel().tolist(), 'colors': colors[kept].ravel().tolist() if len(colors) else [],
            'dense': bool(len(colors)) or os.path.exists(os.path.join(cache, 'dense', 'fused.ply'))}


@rpc('photos.visible')
def visible(ctx, source, points_xyz_m, limit=8):
    project = ctx.require_project(); _, frame, cache = photo_state(project, source, compute=False); model = load_model(cache)
    linear_map, shift, _ = project_transform(project, source, frame); centers = camera_centers(model) @ linear_map.T + shift; scored, chosen = [], []
    for i in range(len(model['names'])):
        uv, (width, height), _ = project_to_display(model, i, linear_map, shift, points_xyz_m)
        if all(p and 0.05 < p[0] / width < 0.95 and 0.05 < p[1] / height < 0.95 for p in uv):
            scored.append((max(abs(p[0] / width - 0.5) + abs(p[1] / height - 0.5) for p in uv), i))
    for _, i in sorted(scored):
        if all(np.linalg.norm(centers[i] - centers[j]) > 0.5 for j in chosen): chosen.append(i)
    return [str(model['names'][i]) for i in chosen[:int(limit)]]
