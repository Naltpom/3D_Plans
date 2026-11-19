import { call } from "./host";
import { $, button, el, number, run, typing } from "./ui";
import { node } from "./plan2d";
import { choice } from "./coter";

type Item = { id: string; kind: string; color: string; width: number; points: number[][]; text?: string; size?: number };
type Doc = { w: number; h: number; cell: number; cm: number; grid: boolean; items: Item[] };
type Live = { walls?: { name: string; a: number[]; b: number[] }[]; extra?: { a: number[]; b: number[]; name?: string }[]; openings?: { type: string; wall: string }[];
  stairs?: number; cotes?: { wall: string; value_cm: number }[]; k?: number; image?: boolean; error?: string };
type Gesture = { kind: string; start: number[]; before: string; box?: number[]; m?: DOMMatrix; origin?: number[][]; changed?: boolean };
export type Helpers = {
  intro: (title: string, text: string) => HTMLElement; footer: (...actions: HTMLButtonElement[]) => HTMLElement; primary: (text: string, action: () => unknown) => HTMLButtonElement;
  render: () => void; done: () => Promise<void>; close: () => void;
};

const COLORS: [string, string, string][] = [["noir", "#1a1a1a", "Murs"], ["rouge", "#e0302c", "Portes"], ["vert", "#3aa65a", "Fenêtres"], ["bleu", "#2f8fe0", "Portes-fenêtres"],
  ["gris", "#8a8a8a", "Escalier"], ["violet", "#9a5bd0", "Notes"]];
const HEX: Record<string, string> = { ...Object.fromEntries(COLORS.map(([c, hex]) => [c, hex])), orange: "#e8953a", jaune: "#e3c13b" };
const ROLE: Record<string, string> = { noir: "mur", rouge: "porte", orange: "porte", vert: "fenetre", jaune: "fenetre", bleu: "porte_fenetre", gris: "escalier" };
const TOOLS: [string, string][] = [["choisir", "Sélection"], ["pinceau", "Pinceau"], ["gomme", "Gomme"], ["trait", "Trait"], ["rect", "Rectangle"], ["cercle", "Cercle"], ["arc", "Arc"],
  ["texte", "Texte"], ["porte", "Porte"], ["fenetre", "Fenêtre"], ["porte_fenetre", "Porte-fenêtre"]];
const OPENING: Record<string, string> = { porte: "rouge", fenetre: "vert", porte_fenetre: "bleu" };
const NOUNS: Record<string, [string, string]> = { porte: ["porte", "portes"], fenetre: ["fenêtre", "fenêtres"], porte_fenetre: ["porte-fenêtre", "portes-fenêtres"] };
const HINTS: Record<string, string> = { choisir: "Clic : choisir · glisser : déplacer · double-clic sur un texte : le changer · Suppr : effacer", pinceau: "Glisser : dessiner à main levée",
  gomme: "Glisser : effacer le pinceau, et les formes touchées", trait: "Glisser d'un bout à l'autre · Maj : angles de 15°", rect: "Glisser d'un coin à l'autre · Maj : carré",
  cercle: "Glisser d'un coin à l'autre · Maj : cercle", arc: "Glisser d'un bout à l'autre, puis clic pour courber", texte: "Clic : écrire, par exemple 420 ou 3,10 m sur un mur",
  porte: "Glisser le long d'un mur", fenetre: "Glisser le long d'un mur", porte_fenetre: "Glisser le long d'un mur" };
const overlay = $("#plan2d");

let h: Helpers;
let doc: Doc | null = null;
let saved = "";
let tool = "trait";
let color = "noir";
let width = 6;
let view: number[] | null = null;
let past: string[] = [];
let future: string[] = [];
let picked = "";
let live: Live = {};
let pending: Item | null = null;
let gesture: Gesture | null = null;
let leaving = false;
let timer = 0;
let status = "";
let svg: SVGSVGElement | null = null;
let labels: SVGGElement | null = null;
let layer: SVGGElement | null = null;
let draft: SVGGElement | null = null;
let statusBar: HTMLElement | null = null;

const gap = (p: number[], q: number[]) => Math.hypot(p[0] - q[0], p[1] - q[1]);
const unit = () => 1 / (svg?.getScreenCTM()?.a || 1);
const fresh = () => `i${Date.now().toString(36)}${Math.random().toString(36).slice(2, 5)}`;
const scale = () => live.k ?? doc!.cm / 100 / doc!.cell;
const inked = () => doc!.items.some((i) => i.kind === "pinceau" && ROLE[i.color]);
const shapes = () => doc!.items.filter((i) => !["texte", "pinceau"].includes(i.kind));
export const drawingOpen = () => Boolean(doc);

function arcPoints([p, m, q]: number[][], steps = 64) {
  const b = [m[0] - p[0], m[1] - p[1]], c = [q[0] - p[0], q[1] - p[1]], d = 2 * (b[0] * c[1] - b[1] * c[0]), bb = b[0] ** 2 + b[1] ** 2, cc = c[0] ** 2 + c[1] ** 2;
  if (Math.abs(d) < 1e-6) return [p, q];
  const o = [p[0] + (c[1] * bb - b[1] * cc) / d, p[1] + (b[0] * cc - c[0] * bb) / d], r = gap(p, o), angle = (v: number[]) => Math.atan2(v[1] - o[1], v[0] - o[0]);
  const turn = (x: number) => ((x % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI), a0 = angle(p);
  let sweep = turn(angle(q) - a0);
  if (turn(angle(m) - a0) > sweep) sweep -= 2 * Math.PI;
  const n = Math.max(2, Math.ceil(steps * Math.abs(sweep) / (2 * Math.PI)));
  return Array.from({ length: n + 1 }, (_, i) => [o[0] + r * Math.cos(a0 + sweep * i / n), o[1] + r * Math.sin(a0 + sweep * i / n)]);
}

function outline(item: Item): number[][] {
  const [p, q] = [item.points[0], item.points.at(-1)!];
  if (item.kind === "rect") return [p, [q[0], p[1]], q, [p[0], q[1]], p];
  if (item.kind === "cercle") return Array.from({ length: 65 }, (_, i) => [(p[0] + q[0]) / 2 + Math.abs(q[0] - p[0]) / 2 * Math.cos(i * Math.PI / 32), (p[1] + q[1]) / 2 + Math.abs(q[1] - p[1]) / 2 * Math.sin(i * Math.PI / 32)]);
  return item.kind === "arc" && item.points.length === 3 ? arcPoints(item.points) : item.points;
}

function foot(p: number[], a: number[], b: number[]) {
  const d = [b[0] - a[0], b[1] - a[1]], t = Math.max(0, Math.min(1, ((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / (d[0] ** 2 + d[1] ** 2 || 1))), f = [a[0] + t * d[0], a[1] + t * d[1]];
  return { p: f, d: gap(p, f), a, b };
}

const segments = (items = shapes()) => items.flatMap((i) => { const line = outline(i); return line.slice(1).map((b, k) => [line[k], b]); });
const distance = (p: number[], item: Item) => item.kind === "texte" ? Math.max(0, gap(p, item.points[0]) - (item.size ?? 24) * 0.6) : Math.min(...segments([item]).map(([a, b]) => foot(p, a, b).d), Infinity);
const corners = () => shapes().flatMap((i) => i.kind === "rect" ? outline(i).slice(0, 4) : i.kind === "cercle" ? [] : [i.points[0], i.points.at(-1)!]);

function snap(p: number[], from?: number[], free = false): [number[], string] {
  const tol = 10 * unit(), corner = corners().filter((q) => q !== from).sort((a, b) => gap(a, p) - gap(b, p))[0], step = doc!.cell / 4;
  if (corner && gap(corner, p) < tol) return [corner, "aimanté sur un coin"];
  let q = p, text = "";
  if (from) {
    const angle = Math.atan2(p[1] - from[1], p[0] - from[0]), turn = free ? Math.PI / 12 : Math.PI / 2, s = Math.round(angle / turn) * turn;
    if (free || Math.abs(angle - s) < Math.PI / 36) {
      const length = doc!.grid ? Math.round(gap(p, from) / step) * step : gap(p, from);
      [q, text] = [[from[0] + length * Math.cos(s), from[1] + length * Math.sin(s)], free ? `angle de ${Math.round(((-s * 180 / Math.PI) % 360 + 360) % 360)}°` : "d'équerre"];
    }
  }
  const on = segments().map(([a, b]) => foot(q, a, b)).sort((a, b) => a.d - b.d)[0];
  if (on && on.d < tol && !text) return [on.p, "aimanté sur un trait"];
  return [from || !doc!.grid ? q : q.map((v) => Math.round(v / step) * step), text];
}

function shape(item: Item, editing = true) {
  const p = item.points, q = p.at(-1)!, hex = HEX[item.color] ?? HEX.noir, cls = editing && item.id === picked ? "picked" : "";
  const stroke = { stroke: hex, "stroke-width": item.width, fill: "none", "stroke-linecap": "round", "stroke-linejoin": "round" };
  if (item.kind === "texte") return node("text", { x: p[0][0], y: p[0][1], fill: hex, "font-size": item.size ?? 24, "font-family": "Inter, Arial, sans-serif", "font-weight": 500, "text-anchor": "middle", "dominant-baseline": "central" }, cls, item.text ?? "");
  if (item.kind === "rect") return node("rect", { x: Math.min(p[0][0], q[0]), y: Math.min(p[0][1], q[1]), width: Math.abs(q[0] - p[0][0]), height: Math.abs(q[1] - p[0][1]), ...stroke }, cls);
  if (item.kind === "cercle") return node("ellipse", { cx: (p[0][0] + q[0]) / 2, cy: (p[0][1] + q[1]) / 2, rx: Math.abs(q[0] - p[0][0]) / 2, ry: Math.abs(q[1] - p[0][1]) / 2, ...stroke }, cls);
  return node("polyline", { points: outline(item).map((v) => v.join(",")).join(" "), ...stroke }, cls);
}

function bounds() {
  const d = doc!, pts = d.items.flatMap((i) => i.kind === "texte" ? [-1, 1].flatMap((s) => [[i.points[0][0] + s * (i.size ?? 24) * (i.text ?? "").length * 0.35, i.points[0][1] + s * (i.size ?? 24) * 0.6]]) : outline(i));
  const edge = (k: number, f: (...v: number[]) => number, pad: number, limit: number) => Math.round(Math.min(Math.max(f(...pts.map((p) => p[k])) + pad, 0), limit));
  return pts.length ? [edge(0, Math.min, -2 * d.cell, d.w), edge(1, Math.min, -2 * d.cell, d.h), edge(0, Math.max, 2 * d.cell, d.w), edge(1, Math.max, 2 * d.cell, d.h)] : [0, 0, d.w, d.h];
}

async function picture(texts: boolean, [x0, y0, x1, y1]: number[] = [0, 0, doc!.w, doc!.h]) {
  const page = node("svg", { xmlns: "http://www.w3.org/2000/svg", width: x1 - x0, height: y1 - y0, viewBox: `${x0} ${y0} ${x1 - x0} ${y1 - y0}` });
  page.append(node("rect", { x: x0, y: y0, width: x1 - x0, height: y1 - y0, fill: "#ffffff" }), ...doc!.items.filter((i) => texts || i.kind !== "texte").map((i) => shape(i, false)));
  const image = new Image();
  await new Promise((done, fail) => { image.onload = done; image.onerror = fail; image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(new XMLSerializer().serializeToString(page))}`; });
  const canvas = Object.assign(document.createElement("canvas"), { width: x1 - x0, height: y1 - y0 });
  canvas.getContext("2d")!.drawImage(image, 0, 0);
  return canvas.toDataURL("image/png");
}

function schedule() {
  clearTimeout(timer);
  timer = window.setTimeout(() => run(async () => {
    const sent = JSON.stringify(doc), traits = inked() ? await picture(false) : undefined, found = await call<Live>("dessin.read", { doc, ...traits ? { traits } : {} });
    if (doc && JSON.stringify(doc) === sent) { live = found; paintLive(); h.render(); }
  }), 350);
}

function changed() {
  pending = null;
  draw();
  schedule();
  h.render();
}

function change(edit: () => void) {
  past.push(JSON.stringify(doc));
  future = [];
  edit();
  changed();
}

function travel(from: string[], to: string[]) {
  if (!from.length) return;
  to.push(JSON.stringify(doc));
  doc = JSON.parse(from.pop()!);
  picked = "";
  changed();
}

function commit(item: Item) {
  if (OPENING[tool]) {
    const off = ([a, b]: number[][]) => item.points.reduce((s, p) => s + foot(p, a, b).d, 0), near = segments(shapes().filter((i) => ROLE[i.color] === "mur")).sort((a, b) => off(a) - off(b))[0];
    if (near && off(near) < 48 * unit()) item.points = item.points.map((p) => foot(p, near[0], near[1]).p);
  }
  change(() => doc!.items.push(item));
}

function erase(p: number[]) {
  const r = Math.max(width * 1.5, 8 * unit()), before = doc!.items.length;
  doc!.items = doc!.items.flatMap((i) => {
    if (i.kind !== "pinceau") return distance(p, i) < r + i.width / 2 ? [] : [i];
    const runs: number[][][] = [[]];
    for (const q of i.points) if (gap(q, p) < r + i.width / 2) runs.push([]); else runs.at(-1)!.push(q);
    const kept = runs.filter((s) => s.length > 1);
    if (kept.length === 1 && kept[0].length === i.points.length) return [i];
    gesture!.changed = true;
    return kept.map((points) => ({ ...i, id: fresh(), points }));
  });
  if (doc!.items.length !== before) gesture!.changed = true;
  if (gesture!.changed) paintItems();
}

function writeText(p: number[], item?: Item) {
  const box = overlay.querySelector("section")!, m = svg!.getScreenCTM()!, frame = box.getBoundingClientRect();
  const input = Object.assign(el("input", "dessin-text"), { value: item?.text ?? "", placeholder: "420", ariaLabel: "Texte ou cote, par exemple 420 ou 3,10 m" });
  Object.assign(input.style, { left: `${m.a * p[0] + m.e - frame.left - 60}px`, top: `${m.d * p[1] + m.f - frame.top - 18}px` });
  let done = false;
  const finish = (keep: boolean) => {
    if (done) return;
    done = true;
    input.remove();
    const text = input.value.trim();
    if (!keep || text === (item?.text ?? "")) return;
    if (item) change(() => { const found = doc!.items.find((i) => i.id === item.id)!; if (text) found.text = text; else doc!.items = doc!.items.filter((i) => i !== found); });
    else if (text) change(() => doc!.items.push({ id: fresh(), kind: "texte", color, width, points: [p], text, size: Math.max(20, width * 4) }));
  };
  input.onkeydown = (event) => { if (event.key === "Enter") finish(true); if (event.key === "Escape") finish(false); event.stopPropagation(); };
  input.onblur = () => finish(true);
  box.append(input);
  setTimeout(() => input.focus());
}

function bubble(group: Element, at: number[], text: string) {
  const u = unit(), size = 12 * u, w = text.length * size * 0.62 + 12 * u;
  group.append(node("rect", { x: at[0] + 10 * u, y: at[1] - 30 * u, width: w, height: 20 * u, rx: 10 * u }, "bubble"),
    node("text", { x: at[0] + 10 * u + w / 2, y: at[1] - 20 * u, "font-size": size }, "bubble-text", text));
}

function paintDraft(cursor?: number[], label = "") {
  if (!draft) return;
  draft.replaceChildren();
  if (pending) draft.append(shape(pending));
  const end = pending?.points.at(-1) ?? cursor;
  if (cursor && !pending && label) draft.append(node("circle", { cx: cursor[0], cy: cursor[1], r: 6 * unit() }, "snap"));
  const size = pending && !["pinceau", "texte"].includes(pending.kind) ? gap(pending.points[0], pending.kind === "arc" ? pending.points[2] ?? pending.points[1] : pending.points[1]) : 0;
  const shown = pending?.kind === "rect" || pending?.kind === "cercle" ? `${number(Math.abs(pending.points[1][0] - pending.points[0][0]) * scale(), 2)} × ${number(Math.abs(pending.points[1][1] - pending.points[0][1]) * scale(), 2)} m` : size ? `≈ ${number(size * scale(), 2)} m` : "";
  if (end && shown) bubble(draft, end, shown);
  status = [TOOLS.find(([id]) => id === tool)![1], shown, label].filter(Boolean).join(" · ");
  if (statusBar) statusBar.firstChild!.textContent = status;
}

function paintItems() {
  layer?.replaceChildren(...doc!.items.map((i) => shape(i)));
}

function paintLive() {
  if (!labels || !doc) return;
  labels.replaceChildren();
  const walls = [...live.walls ?? [], ...(live.extra ?? []).filter((e) => e.name) as { name: string; a: number[]; b: number[] }[]], u = unit();
  const centre = [0, 1].map((k) => (live.walls ?? []).reduce((s, w) => s + w.a[k], 0) / Math.max(1, live.walls?.length ?? 0));
  for (const w of walls) {
    const mid = [(w.a[0] + w.b[0]) / 2, (w.a[1] + w.b[1]) / 2], away = gap(mid, centre) || 1, at = [mid[0] + (centre[0] - mid[0]) / away * 22 * u, mid[1] + (centre[1] - mid[1]) / away * 22 * u];
    labels.append(node("rect", { x: at[0] - 11 * u, y: at[1] - 11 * u, width: 22 * u, height: 22 * u, rx: 5 * u }, "tag"), node("text", { x: at[0], y: at[1], "font-size": 13 * u }, "tag-text", w.name));
  }
}

function at(event: PointerEvent | WheelEvent | MouseEvent, m = svg!.getScreenCTM()!.inverse()) {
  const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(m);
  return [p.x, p.y];
}

function setView(box: number[]) {
  view = box;
  svg?.setAttribute("viewBox", box.join(" "));
  paintLive();
}

function start(event: PointerEvent) {
  const p = at(event), before = JSON.stringify(doc), item = pending;
  (document.activeElement as HTMLElement | null)?.blur();
  if (event.button !== 0 || tool === "choisir" && !doc!.items.some((i) => distance(p, i) < 8 * unit() + i.width / 2)) {
    gesture = { kind: "pan", start: [event.clientX, event.clientY], before, box: view!, m: svg!.getScreenCTM()!.inverse() };
    if (event.button === 0) { picked = ""; paintItems(); }
    return;
  }
  if (tool === "choisir") {
    const hit = [...doc!.items].reverse().sort((a, b) => distance(p, a) - distance(p, b))[0];
    picked = hit.id;
    gesture = { kind: "move", start: p, before, origin: hit.points.map((q) => [...q]) };
    return paintItems();
  }
  if (tool === "texte") return writeText(p);
  if (tool === "gomme") { gesture = { kind: "erase", start: p, before }; return erase(p); }
  if (tool === "arc" && item?.points.length === 3) return commit({ ...item, points: [item.points[0], p, item.points[2]] });
  const s = tool === "pinceau" ? p : snap(p)[0];
  pending = { id: fresh(), kind: OPENING[tool] ? "trait" : tool, color: OPENING[tool] ?? color, width: OPENING[tool] ? width + 4 : width, points: [s, s] };
  gesture = { kind: "draw", start: s, before };
  paintDraft();
}

function move(event: PointerEvent) {
  if (!svg || !doc) return;
  const p = at(event);
  if (gesture?.kind === "pan") {
    const [a, b] = [at(new MouseEvent("move", { clientX: gesture.start[0], clientY: gesture.start[1] }), gesture.m), at(event, gesture.m)];
    return setView([gesture.box![0] - b[0] + a[0], gesture.box![1] - b[1] + a[1], gesture.box![2], gesture.box![3]]);
  }
  if (gesture?.kind === "move") {
    const found = doc.items.find((i) => i.id === picked)!;
    found.points = gesture.origin!.map((q) => [q[0] + p[0] - gesture!.start[0], q[1] + p[1] - gesture!.start[1]]);
    gesture.changed = true;
    return paintItems();
  }
  if (gesture?.kind === "erase") return erase(p);
  if (pending && gesture?.kind === "draw") {
    if (pending.kind === "pinceau") { if (gap(pending.points.at(-1)!, p) > 2 * unit()) pending.points.push(p); return paintDraft(); }
    if (pending.kind === "rect" || pending.kind === "cercle") {
      const [q] = snap(p), s = gesture.start, side = Math.max(Math.abs(q[0] - s[0]), Math.abs(q[1] - s[1]));
      pending.points = [s, event.shiftKey ? [s[0] + Math.sign(q[0] - s[0] || 1) * side, s[1] + Math.sign(q[1] - s[1] || 1) * side] : q];
      return paintDraft();
    }
    const [q, label] = snap(p, gesture.start, event.shiftKey);
    pending.points = [gesture.start, q];
    return paintDraft(undefined, label);
  }
  if (pending?.kind === "arc" && pending.points.length === 3) { pending.points = [pending.points[0], p, pending.points[2]]; return paintDraft(); }
  const [q, label] = ["choisir", "pinceau", "gomme", "texte"].includes(tool) ? [p, ""] : snap(p);
  paintDraft(q, label);
}

function end() {
  const g = gesture, item = pending;
  gesture = null;
  if (!g || g.kind === "pan") return;
  if (g.kind !== "draw") {
    if (!g.changed) return;
    past.push(g.before);
    future = [];
    return changed();
  }
  if (!item) return;
  const size = item.kind === "pinceau" ? item.points.length > 1 ? Infinity : 0 : gap(item.points[0], item.points[1]);
  if (size < 3 * unit()) { pending = null; return paintDraft(); }
  if (item.kind === "arc") { pending = { ...item, points: [item.points[0], [(item.points[0][0] + item.points[1][0]) / 2, (item.points[0][1] + item.points[1][1]) / 2], item.points[1]] }; return paintDraft(); }
  commit(item);
}

function toolbar() {
  const bar = el("div", "cote-tools dessin-tools"), more = el("div", "dessin-row");
  const tools = choice("Outil", TOOLS, tool, (id) => { tool = id; pending = null; picked = ""; draw(); h.render(); });
  const palette = choice("Couleur : chaque couleur a un rôle", COLORS.map(([id, , label]) => [id, label]), color, (id) => {
    color = id;
    if (picked) change(() => { doc!.items.find((i) => i.id === picked)!.color = id; });
    else draw();
  });
  palette.classList.add("palette");
  palette.querySelectorAll("label").forEach((label, i) => label.prepend(Object.assign(el("span", "swatch"), { style: `background:${COLORS[i][1]}` })));
  const thick = Object.assign(el("input"), { type: "range", min: "2", max: "24", value: String(width), ariaLabel: "Épaisseur du trait" }), shown = el("span", "m", String(width));
  thick.oninput = () => { width = Number(thick.value); shown.textContent = thick.value; };
  thick.onchange = () => { if (picked) change(() => { doc!.items.find((i) => i.id === picked)!.width = width; }); };
  const thickness = el("label", "check", "Épaisseur");
  thickness.append(thick, shown);
  const grid = Object.assign(el("input"), { type: "checkbox", checked: doc!.grid }), gridLabel = el("label", "check");
  grid.onchange = () => change(() => { doc!.grid = grid.checked; });
  gridLabel.append(grid, "Grille");
  const cell = Object.assign(el("input", "measure"), { inputMode: "decimal", value: String(doc!.cm).replace(".", ","), ariaLabel: "Taille d'un carreau en cm" }), cellLabel = el("label", "check", "1 carreau =");
  cell.onchange = () => { const v = Number(cell.value.replace(",", ".")); if (v > 0) change(() => { doc!.cm = v; }); };
  cellLabel.append(cell, "cm");
  const zoom = (f: number) => { const [x, y, w, hh] = view!; setView([x + w * (1 - f) / 2, y + hh * (1 - f) / 2, w * f, hh * f]); };
  const zooms = el("span", "zoom");
  zooms.append(Object.assign(button("↶", "", "Annuler (Ctrl+Z)", () => travel(past, future)), { disabled: !past.length, ariaLabel: "Annuler" }),
    Object.assign(button("↷", "", "Rétablir (Ctrl+Y)", () => travel(future, past)), { disabled: !future.length, ariaLabel: "Rétablir" }),
    button("−", "", "Dézoomer", () => zoom(1.25)), button("+", "", "Zoomer", () => zoom(0.8)), button("Tout voir", "", "Toute la feuille", () => setView(whole())));
  more.append(palette, thickness, gridLabel, cellLabel, el("span", "spacer"), zooms);
  bar.append(tools, more);
  return bar;
}

const whole = () => [-30, -30, doc!.w + 60, doc!.h + 60];

function draw() {
  if (!doc) return;
  view ??= whole();
  svg = node("svg", { viewBox: view.join(" "), role: "img", "aria-label": "Feuille de dessin du plan" }, `plan dessin ${tool}`) as SVGSVGElement;
  const d = doc, lines: string[] = [];
  if (d.grid) for (let x = d.cell; x < d.w; x += d.cell) lines.push(`M${x} 0V${d.h}`);
  if (d.grid) for (let y = d.cell; y < d.h; y += d.cell) lines.push(`M0 ${y}H${d.w}`);
  [layer, labels, draft] = [node("g", {}) as SVGGElement, node("g", {}) as SVGGElement, node("g", {}) as SVGGElement];
  svg.append(node("rect", { x: 0, y: 0, width: d.w, height: d.h }, "paper"), node("path", { d: lines.join("") || "M0 0" }, "sheet-grid"), layer, labels, draft);
  paintItems();
  svg.addEventListener("wheel", (e) => { e.preventDefault(); const [x, y] = at(e), [bx, by, bw, bh] = view!, f = e.deltaY > 0 ? 1.2 : 1 / 1.2; setView([x - (x - bx) * f, y - (y - by) * f, bw * f, bh * f]); }, { passive: false });
  svg.addEventListener("pointerdown", (e) => { svg!.setPointerCapture(e.pointerId); start(e); });
  svg.addEventListener("pointermove", move);
  svg.addEventListener("pointerup", () => run(end));
  svg.addEventListener("contextmenu", (e) => e.preventDefault());
  svg.addEventListener("dblclick", (e) => { const p = at(e), hit = doc!.items.find((i) => i.kind === "texte" && distance(p, i) < 8 * unit()); if (hit && tool === "choisir") writeText(hit.points[0], hit); });
  statusBar = el("div", "dessin-status");
  const hint = `${HINTS[tool]} · molette : zoom · clic droit : déplacer la feuille`;
  statusBar.append(el("span", "m", status || TOOLS.find(([id]) => id === tool)![1]), Object.assign(el("span", "hint", hint), { title: hint }));
  const box = el("section");
  box.append(toolbar(), svg, statusBar);
  overlay.replaceChildren(box);
  overlay.classList.remove("half");
  overlay.hidden = false;
  paintLive();
  paintDraft();
}

export async function openDrawing(helpers: Helpers) {
  h = helpers;
  doc = await call<Doc | null>("dessin.get") ?? { w: 1600, h: 1000, cell: 40, cm: 50, grid: true, items: [] };
  [saved, past, future, picked, live, pending, view, leaving, status, tool, color, width] = [JSON.stringify(doc), [], [], "", {}, null, null, false, "", "trait", "noir", 6];
  draw();
  schedule();
  h.render();
}

export function closeDrawing() {
  doc = null;
  clearTimeout(timer);
  overlay.hidden = true;
}

async function finish() {
  const traits = inked() ? await picture(false) : undefined, box = bounds();
  await call("dessin.save", { doc, png: await picture(true, box), crop: box.slice(0, 2), ...traits ? { traits } : {} });
  closeDrawing();
  await h.done();
}

function alert(title: string, text: string) {
  const box = el("div", "alert");
  box.append(el("b", "", title), el("span", "", text));
  return box;
}

function line(tag: string, text: string, note = "", cls = "") {
  const box = el("div", "tagged");
  box.append(el("span", `tag ${cls}`, tag), el("span", "fill", text), ...note ? [el("small", "grow", note)] : []);
  return box;
}

export function drawingStep(): Node[] {
  const l = live, walls = l.walls ?? [], named = (l.extra ?? []).filter((e) => e.name), cotes = l.cotes ?? [], drawnItems = doc!.items.filter((i) => i.kind !== "texte").length;
  const kinds = Object.entries(NOUNS).map(([id, [one, many]]) => { const on = (l.openings ?? []).filter((o) => o.type === id); return on.length ? `${on.length} ${on.length > 1 ? many : one} (${[...new Set(on.map((o) => o.wall))].join(", ")})` : ""; }).filter(Boolean);
  const notes = doc!.items.filter((i) => i.kind !== "texte" && !ROLE[i.color]).length, read = l.error ? [] : [
    ...walls.length ? [line(String(walls.length), `Pièce fermée · murs ${walls[0].name} à ${walls.at(-1)!.name}`)] : [],
    ...named.length ? [line(named.map((e) => e.name).join(" "), named.length > 1 ? "Cloisons à bout libre" : "Cloison à bout libre")] : [],
    ...walls.length ? [line(String(cotes.length), cotes.length ? `Cote${cotes.length > 1 ? "s" : ""} écrite${cotes.length > 1 ? "s" : ""} : ${cotes.map((c) => `${c.wall} ${number(c.value_cm, 0)} cm`).join(", ")}` : "Aucune cote écrite : la grille donne l'échelle",
      cotes.length ? "elles donnent l'échelle ; gardées comme cotes ± 1 cm, modifiables à l'étape 4" : `1 carreau = ${number(doc!.cm)} cm`, "warn")] : [],
    ...kinds.length ? [line(String((l.openings ?? []).length), kinds.join(", "), "", "warn")] : [],
    ...l.stairs ? [line("≡", l.stairs > 1 ? `${l.stairs} escaliers` : "Un escalier")] : [],
    ...notes ? [line("·", `${notes} trait${notes > 1 ? "s" : ""} en violet : notes ignorées`)] : []];
  const problem = l.error && drawnItems ? [alert("Pas encore une pièce", `${l.error[0].toUpperCase()}${l.error.slice(1)}.`)] : [];
  const changedDoc = JSON.stringify(doc) !== saved;
  const quit = leaving ? [alert("Fermer sans garder ?", "Tes changements depuis l'ouverture seront perdus.")] : [];
  const close = button(leaving ? "Oui, fermer sans garder" : "Fermer sans garder", "", "Revenir sans enregistrer", () => { if (changedDoc && !leaving) { leaving = true; h.render(); } else { closeDrawing(); h.close(); } });
  return [h.intro("Dessine ton plan", "Les formes sont lues telles quelles, sans IA : un trait noir est un mur, une longueur écrite sur un mur est sa cote. Le pinceau est lu comme une image, d'après ses couleurs."),
    el("span", "kicker", "Ce que je lis"), ...read, ...!drawnItems ? [el("p", "status", "Feuille vide : trace les murs en noir, puis les portes et fenêtres sur les murs.")] : [], ...problem,
    ...l.image ? [el("div", "note", "Le pinceau est aux couleurs d'un rôle : à la fin, tout le dessin est lu comme une image, d'après ses couleurs ; les textes restent des cotes.")] : [],
    el("div", "note", "Les bouts s'aimantent aux coins et aux traits. Avec la grille, un carreau vaut la taille indiquée : le dessin est déjà à l'échelle, une cote écrite la corrige. Enregistré dans le projet : « Retoucher le dessin » le rouvre."),
    ...quit, h.footer(close, Object.assign(h.primary("Terminer le dessin", finish), { disabled: !drawnItems }))];
}

window.addEventListener("keydown", (event) => {
  if (!doc || typing(event.target)) return;
  const key = event.key.toLowerCase(), ctrl = event.ctrlKey || event.metaKey;
  if (ctrl && key === "z" && !event.shiftKey) travel(past, future);
  else if (ctrl && (key === "y" || key === "z" && event.shiftKey)) travel(future, past);
  else if ((event.key === "Delete" || event.key === "Backspace") && picked) change(() => { doc!.items = doc!.items.filter((i) => i.id !== picked); picked = ""; });
  else if (event.key === "Escape") { pending = null; picked = ""; draw(); }
  else return;
  event.preventDefault();
  event.stopImmediatePropagation();
}, true);
