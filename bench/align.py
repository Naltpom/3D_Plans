"""Exprime la sortie d'une méthode dans le repère monde de A (similitude sur les centres des caméras) et prépare RUN_DIR.

    python align.py --ref A_SPARSE --colmap REC_DIR --out RUN_DIR      # méthodes COLMAP (A', D) : nuage = points épars
    python align.py --ref A_SPARSE --ff CLOUD.npz --out RUN_DIR        # méthodes feed-forward (ff_run.py)
La similitude ne fixe que la jauge (position, orientation, échelle de départ) : l'échelle finale est refaite par evaluate.py.
"""
import argparse, json, os
import numpy as np
import open3d as o3d
import pycolmap


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0); a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt; s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()
    return s, R, md - s * R @ ms


def robust_sim3(src, dst, thr=0.1, it=500):
    """RANSAC sur triplets de caméras (seuil en u de A ≈ 5 cm), puis moindres carrés sur les inliers."""
    rng = np.random.default_rng(0); best = None
    res = lambda s, R, t: np.linalg.norm(src @ (s * R).T + t - dst, axis=1)
    for _ in range(it):
        i = rng.choice(len(src), 3, replace=False)
        k = res(*umeyama(src[i], dst[i])) < thr
        if best is None or k.sum() > best.sum(): best = k
    k = best
    for _ in range(3):
        s, R, t = umeyama(src[k], dst[k]); r = res(s, R, t); k = r < thr
    return s, R, t, r, k


def centers(rec):
    return {im.name: im.projection_center() for im in rec.images.values() if im.has_pose}


def write_ply(path, P, N, C=None):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pc.normals = o3d.utility.Vector3dVector(N)
    if C is not None: pc.colors = o3d.utility.Vector3dVector(C)
    o3d.io.write_point_cloud(path, pc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True); ap.add_argument('--out', required=True)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument('--colmap'); g.add_argument('--ff')
    a = ap.parse_args()
    ref = pycolmap.Reconstruction(a.ref); cref = centers(ref)
    sp, dn = os.path.join(a.out, 'colmap', 'sparse', '0'), os.path.join(a.out, 'colmap', 'dense')
    os.makedirs(sp, exist_ok=True); os.makedirs(dn, exist_ok=True)
    if a.colmap:
        rec = pycolmap.Reconstruction(a.colmap); cm = centers(rec)
        names = sorted(set(cm) & set(cref))
        s, R, t, r, k = robust_sim3(np.array([cm[n] for n in names]), np.array([cref[n] for n in names]))
        rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(R), t))
        rec.write(sp)
        pts = [p for p in rec.points3D.values() if p.track.length() >= 2]
        P = np.array([p.xyz for p in pts]); C = np.array([p.color for p in pts]) / 255
        cen = {im.image_id: im.projection_center() for im in rec.images.values()}
        V = np.array([cen[p.track.elements[0].image_id] for p in pts]) - P   # vers une caméra qui voit le point
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
        N = np.asarray(pc.normals); N *= np.sign((N * V).sum(1))[:, None]
        n_reg = len(cm)
    else:
        d = np.load(a.ff)
        names = [str(n) for n in d['names']]
        src = d['centers']; keep = [i for i, n in enumerate(names) if n in cref]
        s, R, t, r, k = robust_sim3(src[keep], np.array([cref[names[i]] for i in keep]))
        P = d['points'] @ (s * R).T + t; N = d['normals'] @ R.T; C = d['colors']
        for pid in list(ref.points3D): ref.delete_point3D(pid)   # la chaîne n'utilise que les points épars : aucun ici
        ref.write(sp); n_reg = len(names); names = [names[i] for i in keep]
    write_ply(os.path.join(dn, 'fused.ply'), P, N, C)
    info = {'sim3_scale': float(s), 'n_common': len(names), 'n_registered': n_reg, 'n_inliers': int(k.sum()),
            'center_resid_median_u': float(np.median(r)), 'center_resid_max_u': float(r.max()), 'n_points': len(P)}
    json.dump(info, open(os.path.join(a.out, 'align.json'), 'w'), indent=1)
    print(info)


if __name__ == '__main__':
    main()
