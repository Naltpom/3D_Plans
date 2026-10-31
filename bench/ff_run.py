"""Feed-forward reconstruction (VGGT / MapAnything) -> fused cloud npz for align.py.
  .venv python ff_run.py --method {vggt,mapanything} --list LIST --images DIR --undist-cam cam.json --out cloud.npz
                         [--intrinsics] [--poses poses.json] [--conf-drop 0.3] [--voxel-frac 0.004]
cam.json / poses.json come from ff_dump.py (global python, pycolmap)."""
import argparse, json, os, time
import numpy as np, cv2, torch

ap = argparse.ArgumentParser()
ap.add_argument('--method', required=True, choices=['vggt', 'mapanything'])
ap.add_argument('--list', required=True); ap.add_argument('--images', required=True)
ap.add_argument('--undist-cam', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--intrinsics', action='store_true'); ap.add_argument('--poses')
ap.add_argument('--width', type=int, default=1036)
ap.add_argument('--conf-drop', type=float, default=0.3); ap.add_argument('--voxel-frac', type=float, default=0.004)
ap.add_argument('--max-views', type=int, default=0)
ap.add_argument('--pin-poses', action='store_true')  # points de chaque vue reposés sur la pose fournie
a = ap.parse_args()
t0 = time.time(); dev = 'cuda'
names = [l.strip() for l in open(a.list, encoding='utf-8') if l.strip()]
cam = json.load(open(a.undist_cam)); fx, fy, cx, cy = cam['params'][:4]; dist = np.array(cam['params'][4:8])
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
s = a.width / cam['width']; W, H = a.width, int(round(cam['height'] * s))
Kn = np.array([[fx * s, 0, (cx + .5) * s - .5], [0, fy * s, (cy + .5) * s - .5], [0, 0, 1.]])  # undistorted K at W x H
Ks = Kn.copy()  # distorted K at W x H (resize with INTER_AREA first, then undistort)
m1, m2 = cv2.initUndistortRectifyMap(Ks, dist, None, Kn, (W, H), cv2.CV_32FC1)

def load_undist(n):
    im = cv2.imread(os.path.join(a.images, n), cv2.IMREAD_COLOR)
    if im.shape[1] != cam['width']: im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)  # safety; expect landscape
    im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(cv2.remap(im, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), cv2.COLOR_BGR2RGB)

def normals_from_pmap(P, C, v):
    """P (H,W,3) world points, C (3,) camera centre -> (H,W,3) unit normals toward camera, valid mask."""
    dx = P[1:-1, 2:] - P[1:-1, :-2]; dy = P[2:, 1:-1] - P[:-2, 1:-1]
    n = np.cross(dx, dy); ln = np.linalg.norm(n, axis=-1, keepdims=True)
    N = np.zeros_like(P); N[1:-1, 1:-1] = n / np.maximum(ln, 1e-12)
    flip = np.einsum('hwc,hwc->hw', N, C - P) < 0; N[flip] *= -1
    ok = np.zeros(P.shape[:2], bool); ok[1:-1, 1:-1] = (ln[..., 0] > 0) & v[1:-1, 2:] & v[1:-1, :-2] & v[2:, 1:-1] & v[:-2, 1:-1]
    return N, ok

meta = {'method': a.method, 'intrinsics_input': bool(a.intrinsics), 'poses_input': a.poses, 'undist_size': [W, H]}
torch.cuda.reset_peak_memory_stats()
tl = time.time()
if a.method == 'vggt':
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    model = VGGT.from_pretrained('facebook/VGGT-1B'); model.track_head = None; model = model.to(dev).eval()
else:
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs
    model = MapAnything.from_pretrained('facebook/map-anything').to(dev).eval()
load_s = time.time() - tl
tp = time.time(); imgs = [load_undist(n) for n in names]; prep_s = time.time() - tp
torch.cuda.synchronize()

def run_vggt(idx):
    Wv = 518; Hv = int(round(H * Wv / W / 14)) * 14  # VGGT 'crop' preprocessing (slight anisotropic resize like load_fn)
    x = torch.from_numpy(np.stack([cv2.resize(imgs[i], (Wv, Hv), interpolation=cv2.INTER_AREA) for i in idx]))
    x = x.permute(0, 3, 1, 2).float().div(255).to(dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        pr = model(x)
    ext, intr = pose_encoding_to_extri_intri(pr['pose_enc'], x.shape[-2:])
    ext, intr = ext[0].float().cpu().numpy(), intr[0].float().cpu().numpy()
    depth = pr['depth'][0].float().cpu().numpy(); conf = pr['depth_conf'][0].float().cpu().numpy()
    P = unproject_depth_map_to_point_map(depth, ext, intr)
    R, t = ext[:, :, :3], ext[:, :, 3]; C = -np.einsum('nji,nj->ni', R, t)
    col = x.permute(0, 2, 3, 1).cpu().numpy(); valid = np.ones(conf.shape, bool)
    del pr, x; return P, conf, valid, col, C

def pin(Pc, C_out, T_in):
    """Échelle globale (médiane des rapports de distances entre caméras), puis chaque vue posée sur sa pose d'entrée."""
    i, j = np.triu_indices(len(C_out), 1); C_in = T_in[:, :3, 3]
    k = np.median(np.linalg.norm(C_in[i] - C_in[j], axis=1) / np.maximum(np.linalg.norm(C_out[i] - C_out[j], axis=1), 1e-9))
    meta['pin_scale'] = float(k)
    return np.einsum('vab,vhwb->vhwa', T_in[:, :3, :3], k * Pc) + C_in[:, None, None], C_in


def run_ma(idx):
    poses = json.load(open(a.poses)) if a.poses else {}
    order = sorted(idx, key=lambda i: names[i] not in poses)  # a posed view first (reference frame view 0)
    views = []
    for i in order:
        v = {'img': torch.from_numpy(imgs[i])}
        if a.intrinsics: v['intrinsics'] = torch.tensor(Kn, dtype=torch.float32)
        if names[i] in poses:
            v['camera_poses'] = torch.tensor(poses[names[i]], dtype=torch.float32); v['is_metric_scale'] = torch.tensor([False])
        views.append(v)
    meta['n_posed_views'] = sum(names[i] in poses for i in idx)
    pv = preprocess_inputs(views)
    with torch.no_grad():
        out = model.infer(pv, memory_efficient_inference=True, use_amp=True, amp_dtype='bf16', apply_mask=True, mask_edges=True)
    g = lambda k: np.stack([o[k][0].float().cpu().numpy() for o in out])
    P, conf, valid, col = g('pts3d'), g('conf'), g('mask')[..., 0] > 0.5, g('img_no_norm')
    C = g('camera_poses')[:, :3, 3]
    if a.pin_poses: P, C = pin(g('pts3d_cam'), C, np.array([poses[names[i]] for i in order]))
    if 'metric_scaling_factor' in out[0]: meta['metric_scaling_factor'] = float(out[0]['metric_scaling_factor'].float().mean())
    pos = {i: k for k, i in enumerate(order)}; r = [pos[i] for i in idx]  # back to list order
    return P[r], conf[r], valid[r], col[r], C[r]

n = len(names) if not a.max_views else min(a.max_views, len(names)); ti = time.time()
while True:
    idx = sorted(set(np.round(np.linspace(0, len(names) - 1, n)).astype(int).tolist()))
    try:
        ti = time.time(); P, conf, valid, col, C = (run_vggt if a.method == 'vggt' else run_ma)(idx); torch.cuda.synchronize(); break
    except torch.cuda.OutOfMemoryError:
        print(f'OOM with {n} views'); torch.cuda.empty_cache(); n = int(n * 0.85)
        if n < 8: raise
infer_s = time.time() - ti
meta.update(n_views_used=len(idx), peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30, pred_size=list(P.shape[1:3]))
if a.method == 'mapanything': meta['units'] = 'metres (metric scale output)' if not a.poses else 'input-pose frame (D, non-metric poses), MapAnything metric-scaled'
# confidence filter: drop lowest conf_drop fraction among valid pixels
thr = np.quantile(conf[valid], a.conf_drop); keep = valid & (conf >= thr) & np.isfinite(P).all(-1)
pts, nrm, cols = [], [], []
for k in range(len(idx)):
    N, ok = normals_from_pmap(P[k], C[k], valid[k] & np.isfinite(P[k]).all(-1)); m = keep[k] & ok
    pts.append(P[k][m]); nrm.append(N[m]); cols.append(col[k][m])
pts, nrm, cols = np.concatenate(pts), np.concatenate(nrm), np.concatenate(cols)
diag = float(np.linalg.norm(C.max(0) - C.min(0))); vox = a.voxel_frac * diag
key = np.floor((pts - pts.min(0)) / vox).astype(np.int64); key = (key[:, 0] << 42) | (key[:, 1] << 21) | key[:, 2]
_, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
avg = lambda X: np.stack([np.bincount(inv, X[:, j]) for j in range(3)], 1) / cnt[:, None]
P2, C2 = avg(pts), np.clip(avg(cols), 0, 1); N2 = avg(nrm); N2 /= np.maximum(np.linalg.norm(N2, axis=1, keepdims=True), 1e-12)
meta.update(n_raw=int(len(pts)), n_points=int(len(P2)), voxel=vox, cam_bbox_diag=diag, conf_thr=float(thr))
meta['timings'] = {'load_s': load_s, 'prep_s': prep_s, 'infer_s': infer_s, 'total_s': time.time() - t0}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.savez(a.out, names=np.array([names[i] for i in idx]), centers=C.astype(np.float64), points=P2.astype(np.float32),
         normals=N2.astype(np.float32), colors=C2.astype(np.float32), meta=json.dumps(meta))
print(json.dumps(meta))
