import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export type Shape = {
  id: string; type: string; label: string; color: string | null; sigma_max: number | null;
  kind: "box" | "polygon" | "mesh"; center: Vec; size: Vec; rotY: number; vertices: number[][]; faces: number[][]; y?: number; normal?: number[]; sens?: string; light?: Lamp; objet?: Pose;
};
export type Lamp = { lumens: number; kelvin: number; spot: boolean; allume: boolean; faisceau: number; couleur: string; candela: number; luminance: number };
type Pose = { x: number; z: number; rot: number; size: number[] };
type Vec = [number, number, number];
type Range = [number, number];

const OPACITY: Record<string, number> = { mur: 0.45, fenetre: 0.6, porte_fenetre: 0.6, parcelle: 0.3, toit: 0.7 };
const EDGES = { ok: 0x202020, large: 0xe8953a, unknown: 0xf07a72 };
const VIEWS: Record<string, Vec> = { dessus: [0, 1, 1e-4], face: [0, 0, 1], "3d": [0.6, 0.5, 0.8] };
const SNAP = 0.1;
export const GLASS = ["fenetre", "porte_fenetre"];
const OPENINGS = ["porte", ...GLASS];
const FLAT = ["sol", "terrain", "parcelle"];

const canvas = document.querySelector<HTMLCanvasElement>("#scene")!;
const legend = document.querySelector<HTMLElement>("#legend")!;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.setPixelRatio(window.devicePixelRatio);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101318);
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(5, 10, 7);
const neutral = new THREE.Group().add(new THREE.HemisphereLight(0xffffff, 0x444444, 2), sun);
const helpers = [new THREE.GridHelper(20, 20, 0x4a5260, 0x2a303a), new THREE.AxesHelper(1)];
const model = new THREE.Group();
const photos = new THREE.Group();
const furniture = new THREE.Group();
const swings = new THREE.Group();
const cloud = new THREE.Group();
const overlays = new THREE.Group();
const draft = new THREE.Group();
const sketch = new THREE.Group();
let sketchMap: THREE.Texture | null = null;
let walls: Shape[] = [];
const CONE = new THREE.ConeGeometry(0.14, 0.2, 4, 1, true).rotateY(Math.PI / 4).translate(0, -0.1, 0).rotateX(-Math.PI / 2);
scene.add(...helpers, neutral, model, photos, furniture, swings, cloud, overlays, draft, sketch);

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

function wallGeometry({ center, size: [l, h, t], rotY }: Shape, openings: Shape[]) {
  const c = Math.cos(rotY), s = Math.sin(rotY), V = THREE.Vector2, floor = -h / 2 + 0.005;
  const holes = openings.flatMap((o) => {
    const dx = o.center[0] - center[0], dz = o.center[2] - center[2], u = dx * c - dz * s, y = o.center[1] - center[1];
    if (Math.abs(Math.sin(o.rotY - rotY)) > 1e-3 || Math.abs(dx * s + dz * c) > t / 2 + 0.05) return [];
    const box = [Math.max(u - o.size[0] / 2, 0.01 - l / 2), Math.min(u + o.size[0] / 2, l / 2 - 0.01), Math.max(y - o.size[1] / 2, -h / 2), Math.min(y + o.size[1] / 2, h / 2 - 0.01)];
    return box[1] > box[0] && box[3] > box[2] ? [box] : [];
  });
  if (!holes.length) return new THREE.BoxGeometry(l, h, t);
  const outline = [new V(-l / 2, -h / 2)];
  for (const [u0, u1, , y1] of holes.filter((b) => b[2] <= floor).sort((a, b) => a[0] - b[0])) outline.push(new V(u0, -h / 2), new V(u0, y1), new V(u1, y1), new V(u1, -h / 2));
  const face = new THREE.Shape([...outline, new V(l / 2, -h / 2), new V(l / 2, h / 2), new V(-l / 2, h / 2)]);
  face.holes = holes.filter((b) => b[2] > floor).map(([u0, u1, y0, y1]) => new THREE.Path([new V(u0, y0), new V(u1, y0), new V(u1, y1), new V(u0, y1)]));
  const geometry = new THREE.ExtrudeGeometry(face, { depth: t, bevelEnabled: false }).translate(0, 0, -t / 2);
  geometry.clearGroups();
  return geometry;
}

function shapeGeometry(shape: Shape, openings: Shape[]) {
  if (shape.kind === "box") return shape.type === "mur" ? wallGeometry(shape, openings) : new THREE.BoxGeometry(...shape.size);
  if (shape.kind === "polygon") return new THREE.ShapeGeometry(new THREE.Shape(shape.vertices.map(([x, z]) => new THREE.Vector2(x, -z)))).rotateX(-Math.PI / 2);
  const geometry = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(shape.vertices.flat(), 3)).setIndex(shape.faces.flat());
  geometry.computeVertexNormals();
  return geometry;
}

function mesh(shape: Shape, openings: Shape[] = []) {
  const geometry = shapeGeometry(shape, openings), terrain = shape.type === "terrain";
  const opacity = OPACITY[shape.type] ?? 1;
  const material = new THREE.MeshStandardMaterial({
    color: shape.color ?? "#bbbbbb", roughness: 0.9, metalness: 0, transparent: opacity < 1, opacity, depthWrite: opacity === 1,
    side: shape.type === "plafond" ? THREE.BackSide : shape.kind === "box" || terrain ? THREE.FrontSide : THREE.DoubleSide, shadowSide: shape.kind === "box" ? null : THREE.DoubleSide,
    polygonOffset: shape.kind !== "box", polygonOffsetFactor: 1, polygonOffsetUnits: terrain || shape.type === "parcelle" ? 4 : 1,
  });
  const result = new THREE.Mesh(geometry, material);
  result.castShadow = !FLAT.includes(shape.type) && !GLASS.includes(shape.type) && !shape.light;
  result.receiveShadow = true;
  Object.assign(result.userData, { type: shape.type, opacity, material });
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
    if (o instanceof THREE.Mesh || o instanceof THREE.Line || o instanceof THREE.Points) { o.geometry.dispose(); (o.userData.material ?? o.material).dispose(); }
    if (o instanceof THREE.Sprite) { o.material.map?.dispose(); o.material.dispose(); }
    if (o instanceof THREE.Light) o.dispose();
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
  const openings = shapes.filter((s) => OPENINGS.includes(s.type) && s.kind === "box");
  for (const shape of corners(shapes)) model.add(mesh(shape, openings));
  clear(swings);
  for (const shape of shapes) if (shape.sens && shape.kind === "box") swings.add(swing(shape));
  legend.hidden = shapes.every((s) => status(s.sigma_max) === "ok");
  if (frame || empty) setView("3d");
  document.dispatchEvent(new CustomEvent("plans3d:scene"));
}

export function showFurniture(shapes: Shape[]) {
  clear(furniture);
  for (const shape of shapes) {
    const object = mesh(shape);
    furniture.add(object);
    if (!shape.light) continue;
    const { couleur, candela, spot, faisceau } = shape.light;
    const light = spot ? new THREE.SpotLight(couleur, candela, 0, THREE.MathUtils.degToRad(faisceau / 2), 0.3, 2) : new THREE.PointLight(couleur, candela, 0, 2);
    light.position.copy(new THREE.Box3().setFromObject(object).getCenter(new THREE.Vector3()));
    light.castShadow = true;
    light.shadow.mapSize.set(512, 512);
    light.shadow.bias = -0.002;
    light.shadow.camera.near = 0.05;
    light.userData = { id: shape.id, label: shape.label, lamp: shape.light };
    object.userData.lamp = shape.light;
    furniture.add(light);
    if (light instanceof THREE.SpotLight) {
      light.target.position.copy(light.position).y -= 1;
      light.target.userData.id = shape.id;
      furniture.add(light.target);
    }
  }
  document.dispatchEvent(new CustomEvent("plans3d:scene"));
}

export const focusPoint = () => ({ x: controls.target.x, z: controls.target.z });

export function highlight(id: string | string[] | null, color = 0x1f6f8a) {
  const ids = [id ?? []].flat();
  for (const o of [...model.children, ...furniture.children])
    if (o instanceof THREE.Mesh) o.userData.material.emissive.set(ids.includes(o.userData.id) ? color : o.userData.glow ?? 0x000000);
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

let viewName = "3d";

export function setView(name: string) {
  viewName = name;
  const known = model.children.filter((o) => o.userData.known), box = new THREE.Box3();
  for (const o of known.length ? known : model.children.length ? model.children : cloud.children.length ? cloud.children : sketch.children) box.expandByObject(o);
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

export const pointer: { move: ((x: number, z: number) => void) | null; click: ((x: number, z: number) => void) | null } = { move: null, click: null };

export function setPointer(move: typeof pointer.move, click: typeof pointer.click) {
  Object.assign(pointer, { move, click });
  controls.enableRotate = !click;
  canvas.style.cursor = click ? "crosshair" : "";
}

canvas.addEventListener("pointerdown", (event) => {
  down = [event.clientX, event.clientY];
  if (pointer.click) return;
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
canvas.addEventListener("pointermove", (event) => {
  const at = pointer.move && aim(event);
  if (at) pointer.move!(at.x, at.z);
});
canvas.addEventListener("pointerup", (event) => {
  const moved = Math.hypot(event.clientX - down[0], event.clientY - down[1]) > 4;
  if (pointer.click) {
    const at = event.button === 0 && !moved && aim(event);
    if (at) pointer.click(at.x, at.z);
    return;
  }
  if (drag) {
    const { id, parts: [[o, base]] } = drag;
    drag = null;
    controls.enabled = true;
    document.dispatchEvent(new CustomEvent(moved ? "plans3d:furniture-move" : "plans3d:furniture", { detail: { id, dx: o.position.x - base.x, dz: o.position.z - base.z } }));
    return;
  }
  if (event.button !== 0 || moved) return;
  aim(event);
  const hit = (raycaster.intersectObjects(photos.children, false)[0] ?? raycaster.intersectObjects(model.children, false).find((h) => (h.object as THREE.Mesh).material === h.object.userData.material))?.object.userData;
  document.dispatchEvent(hit?.photo ? new CustomEvent("plans3d:photo", { detail: hit.photo }) : new CustomEvent("plans3d:pick", { detail: hit?.id ?? null }));
});

new ResizeObserver(() => {
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  camera.aspect = canvas.clientWidth / Math.max(canvas.clientHeight, 1);
  camera.updateProjectionMatrix();
}).observe(canvas);

export const view = { scene, camera, renderer, controls, model, furniture, neutral, helpers: [...helpers, photos, swings, cloud, overlays, draft, sketch] };
export const hooks = { frame: () => renderer.render(scene, camera) };
const MOVES: Record<string, [number, number, number]> = { z: [1, 0, 0], w: [1, 0, 0], s: [-1, 0, 0], q: [0, -1, 0], a: [0, -1, 0], d: [0, 1, 0], " ": [0, 0, 1], c: [0, 0, -1] };
const held = new Set<string>();
let last = performance.now();
export const walk = { speed: 2 };
const typing = (target: EventTarget | null) => target instanceof HTMLElement && (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(target.tagName));
window.addEventListener("keydown", (event) => {
  const key = event.key.toLowerCase();
  if (!MOVES[key] || typing(event.target) || event.ctrlKey || event.metaKey || event.altKey) return;
  held.add(key);
  event.preventDefault();
});
window.addEventListener("keyup", (event) => held.delete(event.key.toLowerCase()));
window.addEventListener("blur", () => held.clear());

function stroll(now: number) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  if (!held.size || !canvas.offsetParent) return;
  const ahead = camera.getWorldDirection(new THREE.Vector3()).setY(0).normalize(), side = new THREE.Vector3().crossVectors(ahead, camera.up).normalize();
  const step = new THREE.Vector3();
  for (const key of held) {
    const [f, r, u] = MOVES[key];
    step.addScaledVector(ahead, f).addScaledVector(side, r).y += u;
  }
  if (!step.lengthSq()) return;
  step.normalize().multiplyScalar(walk.speed * (held.has("shift") ? 3 : 1) * dt);
  camera.position.add(step);
  controls.target.add(step);
}
window.addEventListener("keydown", (event) => { if (event.key === "Shift") held.add("shift"); });
renderer.setAnimationLoop((now) => {
  stroll(now);
  hooks.frame();
});

const TONES: Record<string, number> = { sure: 0x3fb6d9, doubt: 0xe8953a, measure: 0xf5c451 };

function tag(text: string, color: number) {
  const canvas = Object.assign(document.createElement("canvas"), { width: 512, height: 128 }), context = canvas.getContext("2d")!;
  context.font = "600 64px Inter, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillStyle = `#${color.toString(16).padStart(6, "0")}`;
  context.fillText(text, 256, 64);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false }));
  sprite.scale.set(1.6, 0.4, 1);
  sprite.renderOrder = 1;
  return sprite;
}

export function showCloud(points: number[], colors: number[] = []) {
  const first = !cloud.children.length;
  clear(cloud);
  if (!points.length) return;
  const geometry = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  if (colors.length) geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors.map((c) => (c / 255) ** 2.2), 3));
  const material = new THREE.PointsMaterial({ color: colors.length ? 0xffffff : 0x8b95a3, vertexColors: colors.length > 0, size: 2, sizeAttenuation: false });
  const dots = new THREE.Points(geometry, material);
  dots.userData.material = material;
  cloud.add(dots);
  if (first && !model.children.length) setView(viewName);
}

export function showWalls(walls: { a: number[]; b: number[]; height: number; tone: string; on: boolean; text: string }[]) {
  clear(overlays);
  for (const { a: [x0, z0], b: [x1, z1], height, tone, on, text } of walls) {
    const thick = tone === "measure", material = new THREE.MeshBasicMaterial({ color: TONES[tone], transparent: true, opacity: on ? thick ? 0.6 : 0.35 : 0.1, depthWrite: false, side: THREE.DoubleSide });
    const panel = new THREE.Mesh(new THREE.BoxGeometry(Math.hypot(x1 - x0, z1 - z0), height, thick ? 0.15 : 0.06), material);
    panel.position.set((x0 + x1) / 2, height / 2, (z0 + z1) / 2);
    panel.rotation.y = Math.atan2(z0 - z1, x1 - x0);
    panel.userData.material = material;
    panel.add(new THREE.LineSegments(new THREE.EdgesGeometry(panel.geometry), new THREE.LineBasicMaterial({ color: TONES[tone], transparent: true, opacity: on ? 1 : 0.4 })));
    overlays.add(panel);
    if (!text) continue;
    const label = tag(text, TONES[tone]);
    label.position.set((x0 + x1) / 2, height + 0.4, (z0 + z1) / 2);
    overlays.add(label);
  }
}

export function showDraft(points: number[][]) {
  clear(draft);
  if (!points.length) return;
  const vertices = points.map(([x, z]) => new THREE.Vector3(x, 0.02, z));
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(vertices), new THREE.LineBasicMaterial({ color: 0xf5c451, depthTest: false }));
  const dots = new THREE.Points(new THREE.BufferGeometry().setFromPoints(vertices), new THREE.PointsMaterial({ color: 0xf5c451, size: 9, sizeAttenuation: false, depthTest: false }));
  line.renderOrder = dots.renderOrder = 2;
  draft.add(line, dots);
}

export function sketchOpacity(value: number) {
  for (const o of sketch.children) if (o instanceof THREE.Mesh) o.material.opacity = value;
}

export function showSketch(url: string | null, place?: { x: number; z: number; width: number; depth: number; angle: number; opacity: number }) {
  clear(sketch);
  sketchMap?.dispose();
  sketchMap = null;
  if (!url || !place) return;
  sketchMap = new THREE.TextureLoader().load(url);
  sketchMap.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.MeshBasicMaterial({ map: sketchMap, transparent: true, opacity: place.opacity, depthWrite: false });
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(place.width, place.depth).rotateX(-Math.PI / 2), material);
  plane.position.set(place.x, 0.003, place.z);
  plane.rotation.y = place.angle;
  plane.userData.material = material;
  sketch.add(plane);
}
