import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export type Shape = {
  id: string; type: string; label: string; color: string | null; sigma_max: number | null;
  kind: "box" | "polygon" | "mesh"; center: Vec; size: Vec; rotY: number; vertices: number[][]; faces: number[][]; y?: number; normal?: number[]; sens?: string; light?: { puissance: number; teinte: string }; objet?: Pose;
};
type Pose = { x: number; z: number; rot: number; size: number[] };
type Vec = [number, number, number];
type Range = [number, number];

const OPACITY: Record<string, number> = { mur: 0.45, fenetre: 0.6, porte_fenetre: 0.6, parcelle: 0.3, toit: 0.7 };
const EDGES = { ok: 0x202020, large: 0xe8953a, unknown: 0xf07a72 };
const VIEWS: Record<string, Vec> = { dessus: [0, 1, 1e-4], face: [0, 0, 1], "3d": [0.6, 0.5, 0.8] };
const SNAP = 0.1;

const canvas = document.querySelector<HTMLCanvasElement>("#scene")!;
const legend = document.querySelector<HTMLElement>("#legend")!;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101318);
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(5, 10, 7);
const model = new THREE.Group();
const photos = new THREE.Group();
const furniture = new THREE.Group();
const swings = new THREE.Group();
let walls: Shape[] = [];
const CONE = new THREE.ConeGeometry(0.14, 0.2, 4, 1, true).rotateY(Math.PI / 4).translate(0, -0.1, 0).rotateX(-Math.PI / 2);
scene.add(new THREE.GridHelper(20, 20, 0x4a5260, 0x2a303a), new THREE.AxesHelper(1), new THREE.HemisphereLight(0xffffff, 0x444444, 2), sun, model, photos, furniture, swings);

const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
camera.position.set(6, 5, 8);
const controls = new OrbitControls(camera, canvas);

export const status = (sigma: number | null | undefined) => sigma === null ? "unknown" : sigma !== undefined && sigma > 0.05 ? "large" : "ok";

function corners(shapes: Shape[]) {
  const near = (a: number, b: number) => Math.abs(a - b) < 1e-3;
  const touch = (a: Range, b: Range) => near(a[1], b[0]) || near(a[0], b[1]);
  const walls = shapes.filter((s) => s.type === "mur" && s.kind === "box").map((shape) => {
    const alongX = Math.abs(Math.sin(shape.rotY)) < 1e-6, alongZ = Math.abs(Math.cos(shape.rotY)) < 1e-6;
    const [hx, hz] = (alongX ? [shape.size[0], shape.size[2]] : [shape.size[2], shape.size[0]]).map((v) => v / 2);
    const [x, y, z] = shape.center, h = shape.size[1] / 2;
    return { shape, alongX, alongZ, x: [x - hx, x + hx] as Range, y: [y - h, y + h] as Range, z: [z - hz, z + hz] as Range };
  });
  return shapes.map((shape): Shape => {
    const a = walls.find((w) => w.shape === shape);
    if (!a?.alongX) return shape;
    const xs = walls.filter((b) => b.alongZ && touch(a.x, b.x) && touch(a.z, b.z) && a.y[0] < b.y[1] && b.y[0] < a.y[1]).flatMap((b) => b.x);
    const lo = Math.min(...a.x, ...xs), hi = Math.max(...a.x, ...xs);
    return { ...shape, center: [(lo + hi) / 2, shape.center[1], shape.center[2]], size: [hi - lo, shape.size[1], shape.size[2]] };
  });
}

function shapeGeometry(shape: Shape) {
  if (shape.kind === "box") return new THREE.BoxGeometry(...shape.size);
  if (shape.kind === "polygon") return new THREE.ShapeGeometry(new THREE.Shape(shape.vertices.map(([x, z]) => new THREE.Vector2(x, -z)))).rotateX(-Math.PI / 2);
  const geometry = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(shape.vertices.flat(), 3)).setIndex(shape.faces.flat());
  geometry.computeVertexNormals();
  return geometry;
}

function mesh(shape: Shape) {
  const geometry = shapeGeometry(shape), terrain = shape.type === "terrain";
  const opacity = OPACITY[shape.type] ?? 1;
  const material = new THREE.MeshLambertMaterial({
    color: shape.color ?? "#bbbbbb", transparent: opacity < 1, opacity, depthWrite: opacity === 1,
    side: shape.type === "plafond" ? THREE.BackSide : shape.kind === "box" || terrain ? THREE.FrontSide : THREE.DoubleSide,
    polygonOffset: shape.kind !== "box", polygonOffsetFactor: 1, polygonOffsetUnits: terrain || shape.type === "parcelle" ? 4 : 1,
  });
  const result = new THREE.Mesh(geometry, material);
  if (shape.kind === "box") {
    result.position.fromArray(shape.center);
    result.rotation.y = shape.rotY;
  } else result.position.y = shape.y ?? 0;
  result.add(new THREE.LineSegments(terrain ? new THREE.WireframeGeometry(geometry) : new THREE.EdgesGeometry(geometry),
    new THREE.LineBasicMaterial({ color: EDGES[status(shape.sigma_max)], transparent: terrain, opacity: terrain ? 0.25 : 1 })));
  result.userData.id = shape.id;
  result.userData.known = shape.sigma_max !== null;
  result.userData.objet = shape.objet;
  return result;
}

function clear(group: THREE.Group) {
  group.traverse((o) => {
    if (o instanceof THREE.Mesh || o instanceof THREE.Line) { o.geometry.dispose(); o.material.dispose(); }
  });
  group.clear();
}

function swing({ center: [x, y, z], size: [l, h, ep], rotY: a, normal, sens = "" }: Shape) {
  const [side, hand] = sens.split("_"), s = side === "interieur" ? -1 : 1, k = hand === "gauche" ? -1 : 1, t = (ep - 0.02) / 2;
  const [nx, nz] = normal ?? [Math.sin(a), Math.cos(a)], ux = Math.cos(a), uz = -Math.sin(a), hx = x + s * t * nx + k * l / 2 * ux, hz = z + s * t * nz + k * l / 2 * uz;
  const points = [new THREE.Vector2(hx, hz), ...Array.from({ length: 13 }, (_, i) => {
    const q = (12 - i) * Math.PI / 24;
    return new THREE.Vector2(hx + l * (s * nx * Math.sin(q) - k * ux * Math.cos(q)), hz + l * (s * nz * Math.sin(q) - k * uz * Math.cos(q)));
  })];
  const fill = new THREE.Mesh(new THREE.ShapeGeometry(new THREE.Shape(points.map((p) => new THREE.Vector2(p.x, -p.y)))).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: 0xf5c451, transparent: true, opacity: 0.15, depthWrite: false, side: THREE.DoubleSide }));
  fill.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points.map((p) => new THREE.Vector3(p.x, 0, p.y))), new THREE.LineBasicMaterial({ color: 0xf5c451 })));
  fill.position.y = y - h / 2 + 0.01;
  return fill;
}

export function show(shapes: Shape[], frame = false) {
  const empty = !model.children.length;
  walls = shapes.filter((s) => s.type === "mur" && s.kind === "box");
  clear(model);
  for (const shape of corners(shapes)) model.add(mesh(shape));
  clear(swings);
  for (const shape of shapes) if (shape.sens && shape.kind === "box") swings.add(swing(shape));
  legend.hidden = shapes.every((s) => status(s.sigma_max) === "ok");
  if (frame || empty) setView("3d");
}

export function showFurniture(shapes: Shape[]) {
  clear(furniture);
  for (const shape of shapes) {
    const object = mesh(shape);
    furniture.add(object);
    if (!shape.light) continue;
    const light = new THREE.PointLight(shape.light.teinte, shape.light.puissance / 20, 0, 2);
    light.position.copy(new THREE.Box3().setFromObject(object).getCenter(new THREE.Vector3()));
    light.userData.id = shape.id;
    furniture.add(light);
  }
}

export const focusPoint = () => ({ x: controls.target.x, z: controls.target.z });

export function highlight(id: string | string[] | null, color = 0x1f6f8a) {
  const ids = [id ?? []].flat();
  for (const o of [...model.children, ...furniture.children])
    if (o instanceof THREE.Mesh) o.material.emissive.set(ids.includes(o.userData.id) ? color : 0x000000);
}

export function showCameras(cameras: { name: string; center: Vec; R: number[][] }[], current: string | null) {
  photos.children.forEach((o) => ((o as THREE.Mesh).material as THREE.Material).dispose());
  photos.clear();
  for (const { name, center, R } of cameras) {
    const cone = new THREE.Mesh(CONE, new THREE.MeshBasicMaterial({ color: name === current ? 0xf5c451 : 0x3fb6d9, wireframe: true, side: THREE.DoubleSide }));
    const column = (j: number) => new THREE.Vector3(R[0][j], R[1][j], R[2][j]);
    cone.matrixAutoUpdate = false;
    cone.matrix.makeBasis(column(0), column(1), column(2)).setPosition(...center);
    cone.userData.photo = name;
    photos.add(cone);
  }
}

document.addEventListener("plans3d:highlight", (event) => highlight((event as CustomEvent<string[]>).detail, 0x8a4a12));

export function setView(name: string) {
  const known = model.children.filter((o) => o.userData.known), box = new THREE.Box3();
  for (const o of known.length ? known : model.children) box.expandByObject(o);
  if (box.isEmpty()) box.set(new THREE.Vector3(-2, 0, -2), new THREE.Vector3(2, 2.5, 2));
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const half = Math.atan(Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * Math.min(1, camera.aspect));
  controls.target.copy(sphere.center);
  camera.position.copy(sphere.center).addScaledVector(new THREE.Vector3(...VIEWS[name]).normalize(), sphere.radius / Math.sin(half));
  controls.update();
  document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach((button) => { button.ariaPressed = String(button.dataset.view === name); });
}

document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach((button) => { button.onclick = () => setView(button.dataset.view!); });

const raycaster = new THREE.Raycaster();
const floor = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
let down = [0, 0];
let drag: { id: string; from: THREE.Vector3; pose: Pose; parts: [THREE.Object3D, THREE.Vector3][] } | null = null;

function snap({ size: [w, , d], rot }: Pose, x: number, z: number) {
  const a = rot * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
  const extent = (px: number, pz: number) => w / 2 * Math.abs(c * px - s * pz) + d / 2 * Math.abs(s * px + c * pz);
  for (const wall of walls) {
    const r = wall.rotY, nx = Math.sin(r), nz = Math.cos(r), ux = Math.cos(r), uz = -Math.sin(r);
    const px = x - wall.center[0], pz = z - wall.center[2], across = px * nx + pz * nz, gap = Math.abs(across) - wall.size[2] / 2 - extent(nx, nz);
    if (Math.abs(gap) >= SNAP || Math.abs(px * ux + pz * uz) >= wall.size[0] / 2 + extent(ux, uz)) continue;
    const side = across < 0 ? -1 : 1;
    x -= side * gap * nx;
    z -= side * gap * nz;
  }
  return [x, z];
}

function aim(event: PointerEvent) {
  const rect = canvas.getBoundingClientRect();
  raycaster.setFromCamera(new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, (rect.top - event.clientY) / rect.height * 2 + 1), camera);
  return raycaster.ray.intersectPlane(floor, new THREE.Vector3());
}

canvas.addEventListener("pointerdown", (event) => {
  down = [event.clientX, event.clientY];
  const from = event.button === 0 ? aim(event) : null, hit = from && raycaster.intersectObjects(furniture.children, false)[0]?.object.userData;
  if (!from || !hit?.id) return;
  if (!hit.objet) {
    event.stopPropagation();
    document.dispatchEvent(new CustomEvent("plans3d:locked"));
    return;
  }
  drag = { id: hit.id, from, pose: hit.objet, parts: furniture.children.filter((o) => o.userData.id === hit.id).map((o) => [o, o.position.clone()]) };
  controls.enabled = false;
  canvas.setPointerCapture(event.pointerId);
}, { capture: true });
canvas.addEventListener("pointermove", (event) => {
  const at = drag && aim(event);
  if (!drag || !at) return;
  const { x, z } = drag.pose, [sx, sz] = snap(drag.pose, x + at.x - drag.from.x, z + at.z - drag.from.z);
  for (const [o, base] of drag.parts) o.position.set(base.x + sx - x, base.y, base.z + sz - z);
});
canvas.addEventListener("pointerup", (event) => {
  const moved = Math.hypot(event.clientX - down[0], event.clientY - down[1]) > 4;
  if (drag) {
    const { id, parts: [[o, base]] } = drag;
    drag = null;
    controls.enabled = true;
    document.dispatchEvent(new CustomEvent(moved ? "plans3d:furniture-move" : "plans3d:furniture", { detail: { id, dx: o.position.x - base.x, dz: o.position.z - base.z } }));
    return;
  }
  if (event.button !== 0 || moved) return;
  aim(event);
  const hit = (raycaster.intersectObjects(photos.children, false)[0] ?? raycaster.intersectObjects(model.children, false)[0])?.object.userData;
  document.dispatchEvent(hit?.photo ? new CustomEvent("plans3d:photo", { detail: hit.photo }) : new CustomEvent("plans3d:pick", { detail: hit?.id ?? null }));
});

new ResizeObserver(() => {
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  camera.aspect = canvas.clientWidth / Math.max(canvas.clientHeight, 1);
  camera.updateProjectionMatrix();
}).observe(canvas);

renderer.setAnimationLoop(() => renderer.render(scene, camera));
