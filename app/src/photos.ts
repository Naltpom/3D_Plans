import { call, host, onEvent } from "./host";
import { $, button, cm, el, number, report, run } from "./ui";
import { refreshSources } from "./projects";
import { showCameras, showCloud, type Shape } from "./view3d";

type UV = [number, number];
type Point = number[];
type Camera = { name: string; path: string; center: [number, number, number]; R: number[][]; size: UV; orientation: number };
type Located = { solved: boolean; cameras: Camera[] };
type Plane = { id: string; axis: string; n: number; offset: number; bbox: Point[]; expr: string | null; proj_axis?: string; position_m?: number | null };
type Planes = { axes: Record<string, string>; planes: Plane[] };
type Click = { id: number; image: string; u: number; v: number; anchor: string; kind: string; photos_for_anchor: number };
type Anchor = { ref: string; label: string; axes: Record<string, string> };
type Model = { params: Record<string, { value: number }> };
type Guided = { anchor: string; text: string; points: Point[]; photos: string[] };

const ROTATIONS: Record<string, string> = { "+x+y+z": "0°", "+z+y-x": "90°", "-x+y-z": "180°", "-z+y+x": "270°" };
const KINDS = ["point", "face", "arête", "point"];
const BOX_EDGES = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
const STEPS = 8;
const TASKS: Record<string, string> = { "tools.install": "Installation des outils photo", "photos.extract": "Calcul des positions", "photos.densify": "Densification" };
const TASK_KEY = "plans3d.photoTask";

const svg = (tag: string, attributes: Record<string, string | number> = {}) => {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
  return node;
};
const fileName = (path: string) => path.split(/[\\/]/).pop()!;
const subdivide = (a: Point, b: Point) => Array.from({ length: STEPS + 1 }, (_, i) => a.map((v, k) => v + (b[k] - v) * i / STEPS));

const section = $("#photos");
const sourceSelect = $<HTMLSelectElement>("#photo-source");
const state = $("#photo-state");
const bars = [$<HTMLProgressElement>("#photo-progress"), $<HTMLProgressElement>("#progress")];
const status = $("#status");
const installButton = $<HTMLButtonElement>("#photo-install");
const extractButton = $<HTMLButtonElement>("#photo-extract");
const densifyButton = $<HTMLButtonElement>("#photo-densify");
const openButton = $<HTMLButtonElement>("#photo-open");
const tools = $("#photo-tools");
const photoBar = $("#photo-bar");
const title = $("#photo-title");
const toolsState = $("#tools-state");
const canvas = $("#scene");
const details = $("#photo-details");
const clickList = $("#click-list");
const axesSelect = $<HTMLSelectElement>("#photo-axes");
const planeList = $("#plane-list");
const viewport = $("#viewport");
const view = $("#photo-view");
const strip = $("#photo-strip");
const stage = $("#photo-stage");
const frame = $("#photo-frame");
const picture = $<HTMLImageElement>("#photo-image");
const overlay = $<SVGSVGElement>("#photo-overlay");
const form = $<HTMLFormElement>("#anchor-form");
const [compSelect, anchorSelect] = ["comp", "anchor"].map((name) => form.elements.namedItem(name) as HTMLSelectElement);
const exprInput = form.elements.namedItem("expr") as HTMLInputElement;
const hint = $("#anchor-hint");
const [wire, outline, aim, marks] = ["wire", "plane", "aim", "marks"].map((name) => overlay.appendChild(svg("g", { class: name })));
const guideBox = $("#photo-guide");
const undoButton = $<HTMLButtonElement>("#photo-guide-undo");

let source: string | null = null;
let installed = true;
let located: Located | null = null;
let planes: Planes | null = null;
let clicks: Click[] = [];
let anchors: Anchor[] = [];
let busy = "";
let projectPath = "";
let step = { fraction: NaN, message: "", started: 0 };
let dirty = false;
let image: string | null = null;
let size: UV = [1, 1];
let zoom = { k: 1, x: 0, y: 0 };
let pending: UV | null = null;
let targetId: string | null = null;
let shown: string | null = null;
let stripKey = "";
let sequence = 0;
let timer = 0;
let guided: Guided | null = null;

const cameras = () => located?.cameras ?? [];
const guidedClicks = () => clicks.filter((c) => c.anchor === guided?.anchor);
const signal = () => document.dispatchEvent(new CustomEvent("plans3d:photos"));

export const photoInfo = () => ({ source, installed, count: cameras().length, busy, ...step });
const target = () => planes?.planes.find((p) => p.id === targetId);

const schedule = () => {
  clearTimeout(timer);
  timer = window.setTimeout(() => run(refresh), 30);
};

async function refresh() {
  const n = ++sequence;
  const sources = (await call<{ id: string; kind: string }[]>("sources.list")).filter((s) => s.kind === "photos_colmap");
  const id = sources.some((s) => s.id === source) ? source : sources[0]?.id ?? null;
  const ask = <T>(method: string, fallback: T) => id ? call<T>(method, { source: id }).catch(() => fallback) : Promise.resolve(fallback);
  const [status, found, detected, listed, cloud] = await Promise.all([
    id ? call<{ colmap: boolean }>("tools.status").catch(() => ({ colmap: true })) : { colmap: true },
    ask<Located | null>("photos.cameras", null), ask<Planes | null>("photos.planes", null), ask<Click[]>("photos.clicks", []), ask("photos.cloud", { points: [] as number[] })]);
  if (found) await host.allowAssets(found.cameras.map((c) => c.path)).catch(report);
  if (n !== sequence) return;
  [source, installed, located, planes, clicks] = [id, status.colmap, found, detected, listed];
  sourceSelect.replaceChildren(...sources.map((s) => new Option(s.id, s.id)));
  sourceSelect.value = source ?? "";
  sourceSelect.hidden = sources.length < 2;
  section.hidden = !source;
  showCloud(cloud.points);
  if (image && !cameras().some((c) => c.name === image)) closePhoto();
  render();
  if (image) await draw();
}

function render() {
  const count = cameras().length, plural = count > 1 ? "s" : "";
  installButton.hidden = installed;
  extractButton.hidden = !installed || Boolean(located);
  densifyButton.hidden = openButton.hidden = !located;
  title.textContent = `${source} · photo ${cameras().findIndex((c) => c.name === image) + 1} / ${count}`;
  toolsState.textContent = source ? `Outils photo : ${installed ? "COLMAP installé" : "non installés"}` : "";
  for (const b of [installButton, extractButton, densifyButton]) b.disabled = Boolean(busy);
  const stopped = interrupted();
  if (stopped) state.replaceChildren(`Calcul interrompu (${TASKS[stopped].toLowerCase()}) — `, button("relancer", "link", "Relancer le calcul", () => run(() => photoTask(stopped))));
  else if (!busy) state.textContent = !installed ? "Outils photo non installés" : !located ? "Positions non calculées" : [`${count} photo${plural} positionnée${plural}`,
    located.solved ? "" : "à recaler : mettre à jour", dirty ? "modifications à intégrer : mettre à jour" : ""].filter(Boolean).join(" · ");
  renderStrip();
  renderClicks();
  renderPlanes();
  renderGuide();
  showCameras(cameras(), image);
  signal();
}

function renderGuide() {
  guideBox.hidden = !guided;
  if (!guided) return;
  const done = new Set(guidedClicks().map((c) => c.image)).size;
  $("#photo-guide-text").textContent = done >= 2 ? "C'est bon : l'angle est vu sur 2 photos. Tu peux revenir au parcours." : guided.text;
  $("#photo-guide-count").textContent = `${Math.min(done, 2)} / 2 photos`;
  $("#photo-guide-done").textContent = done >= 2 ? "Revenir au parcours" : "Annuler";
  $("#photo-guide-done").classList.toggle("primary", done >= 2);
  undoButton.hidden = !guidedClicks().length;
}

function renderStrip() {
  const shown = guided ? guided.photos.flatMap((name) => cameras().filter((c) => c.name === name)) : cameras();
  const key = shown.map((c) => c.path).join("|");
  if (key !== stripKey) {
    stripKey = key;
    strip.replaceChildren(...shown.map((camera) => {
      const thumb = button("", "", camera.name, () => run(() => openPhoto(camera.name)));
      thumb.append(Object.assign(el("img"), { src: host.photoUrl(camera.path), alt: camera.name, loading: "lazy", decoding: "async" }));
      return thumb;
    }));
  }
  [...strip.children].forEach((thumb, i) => thumb.classList.toggle("current", shown[i]?.name === image));
}

function renderClicks() {
  const sorted = [...clicks].sort((a, b) => a.anchor.localeCompare(b.anchor) || a.id - b.id);
  clickList.replaceChildren(...sorted.flatMap((c, i) => {
    const face = c.kind === "face", alone = face && c.photos_for_anchor < 2;
    const item = el("li", `${face ? "member" : ""} ${c.image === image ? "current" : ""}`);
    const open = button("", "item", "Ouvrir la photo", () => run(() => openPhoto(c.image)));
    open.append(el("strong", "", face ? fileName(c.image) : `${c.anchor} · ${c.kind}`), el("small", "", face ? `clic ${c.id}` : fileName(c.image)));
    item.append(open, button("×", "remove", "Supprimer le clic", () => run(async () => {
      await call("photos.unclick", { id: c.id });
      dirty = true;
      await refresh();
    })));
    if (!face || sorted[i - 1]?.anchor === c.anchor) return [item];
    const group = el("li", `group ${alone ? "alone" : ""}`);
    group.append(el("strong", "", `${c.anchor} · face · même point`),
      el("small", alone ? "large" : "", alone ? "à cliquer aussi sur une 2e photo" : `${c.photos_for_anchor} photos`));
    return [group, item];
  }));
  if (!clicks.length) clickList.append(el("li", "empty", "Cliquez un point sur une photo"));
}

function renderPlanes() {
  const axes = planes?.axes, key = axes ? `${axes.x}${axes.y}${axes.z}` : "+x+y+z";
  axesSelect.replaceChildren(...[...new Set([...Object.keys(ROTATIONS), key])].map((k) => new Option(ROTATIONS[k] ? `${ROTATIONS[k]} (${k})` : k, k)));
  axesSelect.value = key;
  const aligned = planes?.planes.filter((p) => p.axis) ?? [];
  planeList.replaceChildren(...aligned.map((p) => {
    const item = el("li", p.id === shown ? "current" : "");
    item.dataset.plane = p.id;
    const open = button("", "item", "Montrer sur la photo", () => run(() => {
      shown = p.id === shown ? null : p.id;
      renderPlanes();
      return drawPlane();
    }));
    open.append(el("strong", "", `${p.id} · axe ${p.proj_axis} · ${number(p.n, 0)} pts`),
      el("small", "", p.expr ? `= ${p.expr}` : p.position_m == null ? "non associé" : `non associé · à ${cm(p.position_m)}`));
    item.append(open, p.expr
      ? button("×", "remove", "Dissocier", () => run(async () => {
        await call("photos.unassociate", { source, plane_id: p.id });
        dirty = true;
        await refresh();
      }))
      : button("Associer", "action", "Associer à une face de composant", () => run(() => openForm(p.id))));
    return item;
  }));
  if (!aligned.length) planeList.append(el("li", "empty", "Aucun plan détecté"));
  if (targetId) planeList.querySelector(`[data-plane="${targetId}"]`)?.append(form);
}

async function openForm(plane: string | null) {
  targetId = plane;
  exprInput.hidden = !plane;
  exprInput.required = Boolean(plane);
  anchorSelect.required = !plane;
  exprInput.value = "";
  (plane && planeList.querySelector(`[data-plane="${plane}"]`) || view).append(form);
  const components = await call<{ id: string; label: string }[]>("components.list");
  const previous = compSelect.value;
  compSelect.replaceChildren(new Option("composant…", ""), ...components.map((c) => new Option(c.label, c.id)));
  form.hidden = false;
  await choose(components.some((c) => c.id === previous) ? previous : "");
}

function closeForm() {
  form.hidden = true;
  targetId = pending = null;
  view.append(form);
  drawMarks();
}

async function choose(id: string) {
  compSelect.value = id;
  const plane = target(), list = id ? await call<Anchor[]>("components.anchors", { id }) : [];
  const score = (a: Anchor) => plane ? 2 * Number(plane.proj_axis! in a.axes) + Number(("y" in a.axes) === (plane.proj_axis === "y")) : 0;
  anchors = list.filter((a) => !plane || Object.keys(a.axes).length === 1).sort((a, b) => score(b) - score(a));
  anchorSelect.replaceChildren(...anchors.map((a) => new Option(`${a.label} (${KINDS[Object.keys(a.axes).length] ?? "point"})`, a.ref)));
  sync();
  place();
}

function sync() {
  const plane = target(), anchor = anchors.find((a) => a.ref === anchorSelect.value), [axis] = Object.keys(anchor?.axes ?? {});
  hint.replaceChildren();
  if (!plane) return void (hint.textContent = anchor && Object.keys(anchor.axes).length === 1
    ? "Face : cliquez le même point (repère net) sur au moins 2 photos." : "Ancre visible au point cliqué : dans la liste, ou clic sur le composant dans la vue 3D à droite.");
  if (!anchor) return void (hint.textContent = "Face qui porte ce plan : dans la liste, ou clic sur le composant dans la vue 3D à droite.");
  if (axis === plane.proj_axis) return void (exprInput.value = anchor.axes[axis]);
  const i = "xyz".indexOf(plane.axis), rotation = Object.keys(ROTATIONS).find((r) => r.slice(2 * i, 2 * i + 2) === `+${axis}`);
  hint.append(`Plan selon ${plane.proj_axis}, face selon ${axis}${rotation ? " : repère photo tourné ? " : " : choisissez une autre face."}`,
    ...rotation ? [button("Tourner les axes photo", "link", rotation, () => run(() => setAxes(rotation)))] : []);
}

async function setAxes(mapping: string) {
  await call("photos.axes", { source, mapping });
  dirty = true;
  await refresh();
  if (targetId) sync();
}

async function openPhoto(name: string) {
  const camera = cameras().find((c) => c.name === name);
  if (!camera) return;
  if (name !== image) {
    pending = null;
    if (!targetId) form.hidden = true;
  }
  image = name;
  size = camera.orientation === 6 || camera.orientation === 8 ? [camera.size[1], camera.size[0]] : [camera.size[0], camera.size[1]];
  Object.assign(frame.style, { width: `${size[0]}px`, height: `${size[1]}px` });
  overlay.setAttribute("viewBox", `0 0 ${size[0]} ${size[1]}`);
  picture.src = host.photoUrl(camera.path);
  view.hidden = false;
  if (!guided) photoBar.append(tools);
  details.prepend(canvas);
  fit();
  render();
  await draw();
}

function closePhoto() {
  const wasGuided = Boolean(guided);
  [image, guided] = [null, null];
  view.classList.remove("guided");
  view.hidden = true;
  section.append(tools);
  viewport.prepend(canvas);
  closeForm();
  render();
  if (wasGuided) document.dispatchEvent(new CustomEvent("plans3d:clicked"));
}

export async function openGuided(task: Guided) {
  guided = task;
  view.classList.add("guided");
  stripKey = "";
  const done = new Set(guidedClicks().map((c) => c.image));
  await openPhoto(task.photos.find((name) => !done.has(name)) ?? task.photos[0]);
}

async function guidedClick(u: number, v: number) {
  await call("photos.click", { source, image, u, v, anchor: guided!.anchor });
  dirty = true;
  await refresh();
  const done = new Set(guidedClicks().map((c) => c.image)), next = guided?.photos.find((name) => !done.has(name));
  if (done.size < 2 && next) await openPhoto(next);
}

function fit() {
  const { width, height } = stage.getBoundingClientRect(), k = Math.min(width / size[0], height / size[1]);
  apply({ k, x: (width - size[0] * k) / 2, y: (height - size[1] * k) / 2 });
}

function apply(next: typeof zoom) {
  zoom = next;
  frame.style.transform = `translate(${zoom.x}px, ${zoom.y}px) scale(${zoom.k})`;
  drawMarks();
}

const scaleAt = (f: number, x: number, y: number) => apply({ k: zoom.k * f, x: x - (x - zoom.x) * f, y: y - (y - zoom.y) * f });

function place() {
  if (!pending || form.parentElement !== view) return;
  const rect = stage.getBoundingClientRect(), x = rect.left + zoom.x + pending[0] * zoom.k, y = rect.top + zoom.y + pending[1] * zoom.k;
  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(v, hi));
  form.style.left = `${clamp(x + 16, rect.left + 8, rect.right - form.offsetWidth - 8)}px`;
  form.style.top = `${clamp(y - 28, rect.top + 8, rect.bottom - form.offsetHeight - 8)}px`;
}

function edges(shapes: Shape[]): Point[][] {
  return shapes.flatMap((shape): Point[][] => {
    if (shape.kind === "mesh") return shape.type === "terrain" ? [] : shape.faces.flatMap((face) => face.map((k, i) => [shape.vertices[k], shape.vertices[face[(i + 1) % face.length]]] as Point[]));
    if (shape.kind === "polygon") return shape.vertices.map(([x, z], i, all) => [[x, shape.y ?? 0, z], [all[(i + 1) % all.length][0], shape.y ?? 0, all[(i + 1) % all.length][1]]]);
    const [cx, cy, cz] = shape.center, [hx, hy, hz] = shape.size.map((v) => v / 2), c = Math.cos(shape.rotY), s = Math.sin(shape.rotY);
    const corner = (i: number) => {
      const x = i & 1 ? hx : -hx, z = i & 4 ? hz : -hz;
      return [cx + c * x + s * z, cy + (i & 2 ? hy : -hy), cz - s * x + c * z];
    };
    return BOX_EDGES.map(([a, b]) => [corner(a), corner(b)]);
  });
}

function runs(uv: (UV | null)[]) {
  const inside = (p: UV | null): p is UV => !!p && p[0] > -size[0] && p[0] < 2 * size[0] && p[1] > -size[1] && p[1] < 2 * size[1];
  const result: UV[][] = [[]];
  for (const p of uv) inside(p) ? result[result.length - 1].push(p) : result.push([]);
  return result.filter((r) => r.length > 1);
}

const project = (name: string, points: Point[]) => call<{ uv: (UV | null)[] }>("photos.project_points", { source, image: name, points_xyz_m: points });

async function drawAim() {
  const name = image;
  if (!guided || !name) return void aim.replaceChildren();
  const { uv } = await project(name, subdivide(guided.points[0], guided.points[1]));
  if (name === image) aim.replaceChildren(...runs(uv).map((run) => svg("polyline", { points: run.map((p) => p.join(",")).join(" ") })));
}

async function drawWire() {
  if (guided) return void wire.replaceChildren();
  const name = image, lines = edges(await call<Shape[]>("components.geometry")).map(([a, b]) => subdivide(a, b));
  const uv = name && lines.length ? (await project(name, lines.flat())).uv : [];
  if (name !== image) return;
  wire.replaceChildren(...lines.flatMap((_, i) => runs(uv.slice(i * (STEPS + 1), (i + 1) * (STEPS + 1))))
    .map((run) => svg("polyline", { points: run.map((p) => p.join(",")).join(" ") })));
}

async function drawPlane() {
  const plane = planes?.planes.find((p) => p.id === shown), name = image, axes = planes?.axes;
  if (!plane || !name || !axes) return void outline.replaceChildren();
  const [params, model] = await Promise.all([call<{ name: string; init: number }[]>("params.list"), call<Model>("model").catch(() => null)]);
  const value = (key: string) => model?.params[key]?.value ?? params.find((p) => p.name === key)?.init ?? 0;
  const scale = value(`${source}_s`), shift = [..."xyz"].map((a) => value(`${source}_t${a}`));
  const k = "xyz".indexOf(plane.axis), [i, j] = [0, 1, 2].filter((n) => n !== k), [lo, hi] = plane.bbox;
  const corner = (a: number, b: number) => {
    const p = [0, 0, 0];
    [p[k], p[i], p[j]] = [plane.offset, a, b];
    return p;
  };
  const loop = [corner(lo[i], lo[j]), corner(hi[i], lo[j]), corner(hi[i], hi[j]), corner(lo[i], hi[j])];
  const points = loop.flatMap((p, n) => subdivide(p, loop[(n + 1) % 4]).slice(0, -1)).map((p) => {
    const q = [...shift];
    p.forEach((x, n) => { q["xyz".indexOf(axes["xyz"[n]][1])] += (axes["xyz"[n]][0] === "-" ? -scale : scale) * x; });
    return q;
  });
  const { uv } = await project(name, points);
  if (name === image && plane.id === shown) outline.replaceChildren(svg("polygon", { points: uv.filter((p) => p).map((p) => p!.join(",")).join(" ") }));
}

function drawMarks() {
  const r = 6 / zoom.k, mark = ([u, v]: UV, text: string, className = "") => {
    const group = svg("g", { class: className });
    group.append(svg("circle", { cx: u, cy: v, r }),
      Object.assign(svg("text", { x: u + 1.6 * r, y: v - 1.6 * r, "font-size": 13 / zoom.k, "stroke-width": 3 / zoom.k }), { textContent: text }));
    return group;
  };
  marks.replaceChildren(...clicks.filter((c) => c.image === image).map((c) => mark([c.u, c.v], c.anchor)), ...pending ? [mark(pending, "?", "pending")] : []);
  place();
}

async function draw() {
  await Promise.all([drawWire(), drawPlane(), drawAim()]);
  drawMarks();
}

function interrupted() {
  const saved = JSON.parse(localStorage.getItem(TASK_KEY) ?? "null") as { method: string; project: string; source: string } | null;
  if (busy || !saved || saved.method !== "tools.install" && (saved.project !== projectPath || saved.source !== source)) return "";
  if (!({ "tools.install": installed, "photos.extract": Boolean(located) } as Record<string, boolean>)[saved.method]) return saved.method;
  localStorage.removeItem(TASK_KEY);
  return "";
}

function showStep() {
  const seconds = Math.round((Date.now() - step.started) / 1000), known = Number.isFinite(step.fraction);
  for (const bar of bars) {
    bar.hidden = false;
    if (known) bar.value = step.fraction;
    else bar.removeAttribute("value");
  }
  state.textContent = status.textContent = `${TASKS[busy]}${step.message ? ` : ${step.message}` : "…"}${known ? ` (${Math.round(step.fraction * 100)} %)` : ""}`
    + ` · ${Math.floor(seconds / 60)} min ${String(seconds % 60).padStart(2, "0")} s`;
  signal();
}

export async function photoTask(method: string) {
  const label = TASKS[method];
  busy = method;
  step = { fraction: NaN, message: "", started: Date.now() };
  localStorage.setItem(TASK_KEY, JSON.stringify({ method, project: projectPath, source }));
  render();
  showStep();
  const ticking = window.setInterval(showStep, 1000);
  try {
    await call(method, method === "tools.install" ? {} : { source });
    status.textContent = `${label} : terminé`;
  } catch (error) {
    status.textContent = `${label} : échec`;
    throw error;
  } finally {
    clearInterval(ticking);
    localStorage.removeItem(TASK_KEY);
    busy = "";
    for (const bar of bars) bar.hidden = true;
    render();
    await Promise.all([refresh(), refreshSources()]).catch(report);
  }
}

let drag: { x: number; y: number; moved: boolean } | null = null;
stage.onpointerdown = (event) => {
  if (event.button !== 0) return;
  drag = { x: event.clientX, y: event.clientY, moved: false };
  stage.setPointerCapture(event.pointerId);
};
stage.onpointermove = (event) => {
  if (!drag || !drag.moved && Math.hypot(event.clientX - drag.x, event.clientY - drag.y) < 4) return;
  apply({ ...zoom, x: zoom.x + event.clientX - drag.x, y: zoom.y + event.clientY - drag.y });
  drag = { x: event.clientX, y: event.clientY, moved: true };
};
stage.onpointerup = (event) => {
  const clicked = drag && !drag.moved;
  drag = null;
  if (!clicked) return;
  const rect = frame.getBoundingClientRect(), u = (event.clientX - rect.left) / rect.width * size[0], v = (event.clientY - rect.top) / rect.height * size[1];
  if (u < 0 || v < 0 || u > size[0] || v > size[1]) return;
  if (guided) return void run(() => guidedClick(u, v));
  pending = [u, v];
  drawMarks();
  if (form.hidden || targetId) run(() => openForm(null));
};
stage.onwheel = (event) => {
  event.preventDefault();
  const rect = stage.getBoundingClientRect();
  scaleAt(Math.exp(-event.deltaY / 500), event.clientX - rect.left, event.clientY - rect.top);
};
document.querySelectorAll<HTMLButtonElement>("[data-zoom]").forEach((zoomButton) => {
  const f = Number(zoomButton.dataset.zoom);
  zoomButton.onclick = () => f ? scaleAt(f, stage.clientWidth / 2, stage.clientHeight / 2) : fit();
});
new ResizeObserver(() => { if (image) fit(); }).observe(stage);

installButton.onclick = () => run(() => photoTask("tools.install"));
extractButton.onclick = () => run(() => photoTask("photos.extract"));
densifyButton.onclick = () => run(() => photoTask("photos.densify"));
openButton.onclick = () => view.hidden ? run(() => openPhoto(cameras()[0]?.name ?? "")) : closePhoto();
$("#photo-close").onclick = closePhoto;
$("#photo-guide-done").onclick = closePhoto;
undoButton.onclick = () => run(async () => {
  const last = guidedClicks().at(-1);
  if (last) await call("photos.unclick", { id: last.id });
  dirty = true;
  await refresh();
});
sourceSelect.onchange = () => {
  source = sourceSelect.value;
  closeForm();
  closePhoto();
  schedule();
};
axesSelect.onchange = () => run(() => setAxes(axesSelect.value));
compSelect.onchange = () => run(() => choose(compSelect.value));
anchorSelect.onchange = sync;
$("#anchor-cancel").onclick = closeForm;
form.onsubmit = (event) => {
  event.preventDefault();
  run(async () => {
    const plane = target();
    if (plane) await call("photos.associate", { source, plane_id: plane.id, expr: exprInput.value.trim() });
    else if (pending) await call("photos.click", { source, image, u: pending[0], v: pending[1], anchor: anchorSelect.value });
    closeForm();
    dirty = true;
    await refresh();
  });
};

onEvent((event) => {
  if (event.event !== "progress" || event.method !== busy) return;
  step = { ...step, fraction: Number(event.fraction), message: String(event.message ?? "") };
  showStep();
});
new MutationObserver(schedule).observe($("#source-list"), { childList: true });
new MutationObserver(() => { if (image) run(drawWire); }).observe($("#component-list"), { childList: true });
document.addEventListener("plans3d:opened", (event) => {
  projectPath = (event as CustomEvent<{ path: string } | null>).detail?.path ?? "";
  [source, located, planes, clicks, dirty, shown] = [null, null, null, [], false, null];
  closeForm();
  closePhoto();
  section.hidden = true;
});
document.addEventListener("plans3d:updated", () => {
  dirty = false;
  schedule();
});
document.addEventListener("plans3d:pick", (event) => {
  const id = (event as CustomEvent<string | null>).detail;
  if (!form.hidden && id) run(() => choose(id));
});
document.addEventListener("plans3d:photo", (event) => {
  if (!document.body.classList.contains("guided")) run(() => openPhoto((event as CustomEvent<string>).detail));
});
