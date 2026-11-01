import { call, host } from "./host";
import { $, button, el, number, run } from "./ui";
import { setPointer, setView, showDraft, showSketch, sketchOpacity } from "./view3d";

type Point = [number, number];
type Wall = { label: string; a: number[]; b: number[] };
export type Sketch = { file: string; path: string; points?: number[][]; length?: number; size?: number[]; opacity?: number } | null;

const GRID = 0.05;
const SQUARE = Math.PI / 30;
const CLOSE = 0.25;
const MAGNET = 0.2;

let points: Point[] = [];
let lengths: (number | null)[] = [];
let cursor: Point | null = null;
let active = "";
let keyboard = false;
let changed = () => {};
let finished: (made?: string[]) => Promise<void> = async () => {};
let rough = false;
let lengthInput: HTMLInputElement | null = null;
let angleInput: HTMLInputElement | null = null;
let hint: HTMLElement | null = null;
let magnet: HTMLElement | null = null;
let existing: () => Wall[] = () => [];

const decimal = (text: string) => Number(text.trim().replace(",", "."));
const round = (v: number) => Math.round(v / GRID) * GRID;
const angleOf = (a: Point, b: Point) => Math.atan2(a[1] - b[1], b[0] - a[0]);
const toward = (a: Point, angle: number, length: number): Point => [a[0] + length * Math.cos(angle), a[1] - length * Math.sin(angle)];
const degrees = (angle: number) => ((angle * 180 / Math.PI) % 360 + 360) % 360;

export const mode = () => active;

function onWall({ a, b }: Wall, x: number, z: number): Point {
  const dx = b[0] - a[0], dz = b[1] - a[1], t = Math.max(0, Math.min(1, ((x - a[0]) * dx + (z - a[1]) * dz) / (dx * dx + dz * dz)));
  return [a[0] + t * dx, a[1] + t * dz];
}

function attract(x: number, z: number): [Point, string] | null {
  for (const wall of existing()) for (const p of [wall.a, wall.b]) if (Math.hypot(x - p[0], z - p[1]) < MAGNET) return [[p[0], p[1]], `angle du ${wall.label[0].toLowerCase()}${wall.label.slice(1)}`];
  const close = existing().map((wall) => [onWall(wall, x, z), wall] as const).filter(([p]) => Math.hypot(x - p[0], z - p[1]) < MAGNET)
    .sort(([p], [q]) => Math.hypot(x - p[0], z - p[1]) - Math.hypot(x - q[0], z - q[1]))[0];
  return close ? [close[0], `${close[1].label[0].toLowerCase()}${close[1].label.slice(1)}`] : null;
}

function snap(x: number, z: number): Point {
  const last = points.at(-1), held = attract(x, z);
  if (magnet) magnet.textContent = points.length > 2 && Math.hypot(x - points[0][0], z - points[0][1]) < CLOSE ? "Aimanté : premier angle (clic pour fermer)" : held ? `Aimanté : ${held[1]}` : "";
  if (points.length > 2 && Math.hypot(x - points[0][0], z - points[0][1]) < CLOSE) return points[0];
  if (held) return held[0];
  if (!last) return [round(x), round(z)];
  const angle = angleOf(last, [x, z]), square = Math.round(angle / (Math.PI / 2)) * Math.PI / 2;
  return Math.abs(angle - square) < SQUARE ? toward(last, square, round(Math.hypot(x - last[0], z - last[1]))) : [round(x), round(z)];
}

function preview() {
  showDraft(cursor ? [...points, cursor] : points);
  const last = points.at(-1);
  if (!lengthInput || !angleInput || !last || !cursor) return;
  lengthInput.placeholder = number(Math.hypot(cursor[0] - last[0], cursor[1] - last[1]) * 100, 0);
  angleInput.placeholder = number(degrees(angleOf(last, cursor)), 0);
}

function add(point: Point, typed: number | null) {
  if (points.length > 2 && point === points[0]) return finish(true);
  points.push(point);
  if (points.length > 1) lengths.push(typed);
  changed();
}

function typedPoint() {
  const last = points.at(-1), length = decimal(lengthInput?.value ?? "") / 100;
  if (!last) throw new Error("Clique d'abord le premier angle dans la vue.");
  if (!(length > 0)) throw new Error("Tape une longueur en centimètres, par exemple 420.");
  const angle = angleInput?.value.trim() ? decimal(angleInput.value) * Math.PI / 180 : cursor ? angleOf(last, cursor) : 0;
  keyboard = true;
  add(toward(last, angle, length), length);
}

async function finish(closed: boolean) {
  if (points.length < (closed ? 3 : 2)) throw new Error(closed ? "Il faut au moins 3 angles pour fermer la pièce." : "Il faut au moins 2 points.");
  const drawn = { points, closed, lengths: closed ? [...lengths, null] : lengths };
  stop();
  const made = await call<{ created: string[] }>("plan.trace", { ...drawn, ...rough ? { prior: 0.3 } : {} });
  await finished(rough ? made.created : []);
}

export function stop() {
  [active, points, lengths, cursor, keyboard] = ["", [], [], null, false];
  setPointer(null, null);
  showDraft([]);
  changed();
}

export function trace(walls: () => Wall[], onChange: () => void, onDone: (made?: string[]) => Promise<void>) {
  [active, points, lengths, cursor, changed, finished, existing] = ["trace", [], [], null, onChange, onDone, walls];
  setView("dessus");
  setPointer((x, z) => { cursor = snap(x, z); preview(); }, (x, z) => run(() => add(snap(x, z), null)));
  changed();
}

function nearest(walls: Wall[], x: number, z: number) {
  const distance = ({ a, b }: Wall) => {
    const dx = b[0] - a[0], dz = b[1] - a[1], t = Math.max(0, Math.min(1, ((x - a[0]) * dx + (z - a[1]) * dz) / (dx * dx + dz * dz)));
    return Math.hypot(x - a[0] - t * dx, z - a[1] - t * dz);
  };
  return walls.filter((w) => distance(w) < 0.6).sort((p, q) => distance(p) - distance(q))[0];
}

export function placeHint() {
  hint = el("p", "status", "Approche le curseur d'un mur.");
  return hint;
}

export function place(type: string, walls: () => Wall[], onChange: () => void, onDone: () => Promise<void>) {
  [active, changed] = [type, onChange];
  setView("dessus");
  setPointer((x, z) => {
    const wall = nearest(walls(), x, z);
    if (hint) hint.textContent = wall ? `Clic : sur le ${wall.label[0].toLowerCase()}${wall.label.slice(1)}.` : "Approche le curseur d'un mur.";
  }, (x, z) => run(async () => {
    await call("plan.place", { type, point: [x, z] });
    await onDone();
  }));
  changed();
}

export function controls() {
  lengthInput = Object.assign(el("input", "measure"), { inputMode: "decimal", autocomplete: "off", placeholder: "cm", ariaLabel: "Longueur en cm" });
  angleInput = Object.assign(el("input", "measure"), { inputMode: "decimal", autocomplete: "off", placeholder: "°", ariaLabel: "Angle en degrés" });
  lengthInput.onkeydown = angleInput.onkeydown = (event) => { if (event.key === "Enter") run(typedPoint); };
  const fields = el("div", "fields");
  for (const [text, input] of [["Longueur (cm)", lengthInput], ["Angle (°)", angleInput]] as const) {
    const label = el("label", "", text);
    label.append(input);
    fields.append(label);
  }
  const actions = el("div", "row");
  actions.append(
    Object.assign(button("Annuler le dernier point", "", "Retirer le dernier angle posé", () => { points.pop(); lengths.pop(); changed(); }), { disabled: !points.length }),
    Object.assign(button("Fermer la pièce", "primary", "Relier le dernier angle au premier", () => run(() => finish(true))), { disabled: points.length < 3 }),
    Object.assign(button("Terminer sans fermer", "", "Créer les murs tracés sans fermer", () => run(() => finish(false))), { disabled: points.length < 2 }),
    button("Abandonner", "", "Quitter le tracé sans rien créer", stop));
  if (keyboard) queueMicrotask(() => lengthInput?.focus());
  keyboard = false;
  preview();
  const status = points.length ? `${points.length} angle${points.length > 1 ? "s" : ""} posé${points.length > 1 ? "s" : ""}. Angle 0° vers la droite, 90° vers le haut du plan.`
    : "Clique le premier angle de la pièce dans la vue.";
  magnet = el("p", "status");
  const unsure = Object.assign(el("input"), { type: "checkbox", checked: rough });
  unsure.onchange = () => { rough = unsure.checked; };
  const option = el("label", "wall", "");
  option.append(unsure, el("span", "", "Je ne suis pas sûr de mes murs : chaque mur sera calé sur la surface photo la plus proche (± 30 cm)"));
  return [el("p", "", status), magnet, fields, option, actions];
}

export function opacity(sketch: NonNullable<Sketch>) {
  const slider = Object.assign(el("input"), { type: "range", min: "0.1", max: "1", step: "0.05", value: String(sketch.opacity ?? 0.55), ariaLabel: "Opacité du croquis" });
  slider.oninput = () => sketchOpacity(Number(slider.value));
  slider.onchange = () => run(() => call("plan.sketch", { opacity: Number(slider.value) }).then((saved) => { sketch.opacity = (saved as NonNullable<Sketch>).opacity; }));
  const label = el("label", "opacity", "Opacité du croquis");
  label.append(slider);
  return label;
}

export function placeSketch(sketch: Sketch) {
  if (!sketch?.points || !sketch.size || !sketch.length) return showSketch(null);
  const [[u0, v0], [u1, v1]] = sketch.points, [width, height] = sketch.size;
  const k = sketch.length / Math.hypot(u1 - u0, v1 - v0), angle = Math.atan2(v1 - v0, u1 - u0), c = Math.cos(angle), s = Math.sin(angle);
  const u = width / 2 - u0, v = height / 2 - v0;
  showSketch(host.photoUrl(sketch.path), { x: k * (c * u + s * v), z: k * (c * v - s * u), width: width * k, depth: height * k, angle, opacity: sketch.opacity ?? 0.55 });
}

export function calibrate(sketch: NonNullable<Sketch>, onDone: () => Promise<void>) {
  const view = $("#sketch-view"), image = $<HTMLImageElement>("#sketch-image"), marks = $<SVGSVGElement>("#sketch-marks"), input = $<HTMLInputElement>("#sketch-length");
  let picked: number[][] = [];
  const draw = () => {
    const svg = (tag: string, attributes: Record<string, number>) => Object.entries(attributes).reduce((node, [k, v]) => (node.setAttribute(k, String(v)), node), document.createElementNS("http://www.w3.org/2000/svg", tag));
    const r = Math.max(image.naturalWidth, image.naturalHeight) / 120;
    marks.setAttribute("viewBox", `0 0 ${image.naturalWidth} ${image.naturalHeight}`);
    marks.replaceChildren(...picked.map(([cx, cy]) => svg("circle", { cx, cy, r })),
      ...picked.length === 2 ? [svg("line", { x1: picked[0][0], y1: picked[0][1], x2: picked[1][0], y2: picked[1][1] })] : []);
    $("#sketch-help").textContent = picked.length < 2 ? `Clique le point ${picked.length + 1} sur 2 : deux points dont tu connais la distance.` : "Indique la distance entre les deux points.";
  };
  image.onload = draw;
  image.onclick = (event) => {
    const box = image.getBoundingClientRect(), point = [(event.clientX - box.left) / box.width * image.naturalWidth, (event.clientY - box.top) / box.height * image.naturalHeight];
    picked = picked.length >= 2 ? [point] : [...picked, point];
    draw();
    if (picked.length === 2) input.focus();
  };
  $("#sketch-ok").onclick = () => run(async () => {
    if (picked.length < 2) throw new Error("Clique deux points sur le croquis.");
    await call("plan.sketch", { points: picked, length: decimal(input.value) / 100, size: [image.naturalWidth, image.naturalHeight] });
    view.hidden = true;
    await onDone();
  });
  $("#sketch-cancel").onclick = () => { view.hidden = true; };
  input.value = sketch.length ? String(sketch.length * 100) : "";
  image.src = host.photoUrl(sketch.path);
  view.hidden = false;
  draw();
}

document.addEventListener("keydown", (event) => {
  if (active !== "trace" || event.target instanceof HTMLInputElement) return;
  if (/^[\d.,]$/.test(event.key) && lengthInput) lengthInput.focus();
  if (event.key === "Escape") stop();
  if (event.key === "Backspace" && points.length) { points.pop(); lengths.pop(); changed(); }
});
