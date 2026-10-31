"""Dump COLMAP camera (A) and optional poses (cam2world OpenCV 4x4 per image name) to JSON. Global python (pycolmap).
  python ff_dump.py --cam colmap/sparse/0 --cam-out cam.json [--poses REC --poses-out poses.json]"""
import argparse, json, numpy as np, pycolmap
ap = argparse.ArgumentParser(); ap.add_argument('--cam'); ap.add_argument('--cam-out')
ap.add_argument('--poses'); ap.add_argument('--poses-out'); a = ap.parse_args()
if a.cam:
    r = pycolmap.Reconstruction(a.cam); cams = list(r.cameras.values())
    c = cams[0]; print('cameras:', len(cams), c.model.name, c.width, c.height)
    json.dump({'model': c.model.name, 'width': c.width, 'height': c.height, 'params': list(map(float, c.params))}, open(a.cam_out, 'w'))
if a.poses:
    r = pycolmap.Reconstruction(a.poses); out = {}
    for im in r.images.values():
        if not im.has_pose: continue
        T = np.eye(4); T[:3] = im.cam_from_world().inverse().matrix() if callable(im.cam_from_world) else im.cam_from_world.inverse().matrix()
        out[im.name] = T.tolist()
    print('poses:', len(out)); json.dump(out, open(a.poses_out, 'w'))
