import { call } from "./host";
import { $, button, el, number, run } from "./ui";
import { node } from "./plan2d";
import { choice, drawCote } from "./coter";
import { phoneButton, phonePanel } from "./telephone";

type Trace = { id: string; closed: boolean; cloison: boolean; points: number[][]; names: string[]; typed: (number | null)[] };
type Hole = { id: string; type: string; wall: string; pos: number; width: number; height: number; sill: number };
type Doc = { traces: Trace[]; openings: Hole[]; height: number; open: string | null };
type Wall = { name: string; trace: Trace; k: number; a: number[]; b: number[]; length: number; typed: number | null; angle: number | null };
export type Helpers = {
  intro: (title: string, text: string) => HTMLElement; card: (title: string, text: string, ...actions: HTMLButtonElement[]) => HTMLElement; row: (...nodes: Node[]) => HTMLElement;
  footer: (...actions: HTMLButtonElement[]) => HTMLElement; primary: (text: string, action: () => unknown) => HTMLButtonElement; back: () => HTMLButtonElement; render: () => void;
  act: (text: string, task: () => Promise<unknown>) => Promise<void>; next: () => Promise<void>; finish: () => Promise<void>; extra: () => Node[]; built: () => number;
};

const TOOLS: [string, string][] = [["mur", "Murs"], ["cloison", "Cloison"], ["porte", "Porte"], ["fenetre", "Fenêtre"], ["porte_fenetre", "Porte-fenêtre"], ["retoucher", "Retoucher"]];
const KINDS: Record<string, string> = { porte: "Porte", fenetre: "Fenêtre", porte_fenetre: "Porte-fenêtre" };
const HINTS: Record<string, string> = { mur: "Clique le premier angle de la pièce, puis les suivants. Les chiffres tapés donnent la longueur du mur en cours.", cloison: "Clique le départ de la cloison (sur un mur ou n'importe où), puis son bout.",
  porte: "Clique sur un mur : la porte s'y pose, règle ensuite sa place et sa taille dans la liste.", fenetre: "Clique sur un mur : la fenêtre s'y pose, règle ensuite sa place, sa taille et son allège.",
  porte_fenetre: "Clique sur un mur : la porte-fenêtre s'y pose, règle ensuite sa place et sa taille.", retoucher: "Clique un mur ou une ouverture pour la retrouver dans la liste : longueur, angle, place, taille." };
const overlay = $("#plan2d");
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

let h: Helpers;
let doc: Doc | null = null;
let tool = "mur";
let square = true;
let view: number[] | null = null;
let cursor: number[] | null = null;
let picked = "";
let applied: string | null = null;
let made: { walls: string[]; openings: string[]; skipped: string[] } | null = null;
let lengthInput: HTMLInputElement | null = null;
let angleInput: HTMLInputElement | null = null;

const gap = (p: number[], q: number[]) => Math.hypot(p[0] - q[0], p[1] - q[1]);
const decimal = (text: string) => Number(text.trim().replace(",", "."));
const grid = (v: number) => Math.round(v * 20) / 20;
const open = () => doc?.traces.find((t) => t.id === doc!.open) ?? null;
const fresh = () => { const used = new Set(doc!.traces.flatMap((t) => t.names)); return [...LETTERS].find((n) => !used.has(n)) ?? "?"; };

function walls(): Wall[] {
  return (doc?.traces ?? []).flatMap((t) => t.names.map((name, k) => {
    const n = t.points.length, a = t.points[k], b = t.points[(k + 1) % n], prev = k ? t.points[k - 1] : t.closed ? t.points[n - 1] : null;
    const turn = prev ? Math.atan2((a[0] - prev[0]) * (b[1] - a[1]) - (a[1] - prev[1]) * (b[0] - a[0]), (a[0] - prev[0]) * (b[0] - a[0]) + (a[1] - prev[1]) * (b[1] - a[1])) : null;
    return { name, trace: t, k, a, b, length: gap(a, b) * 100, typed: t.typed[k], angle: turn == null || k === 0 ? null : 180 - Math.abs(turn) * 180 / Math.PI };
  }));
}

async function send(o: Record<string, unknown>) {
  doc = await call<Doc>("construction.op", { o });
  if (applied === null) applied = "";
  h.render();
}

export async function loadBuild() {
  doc = await call<Doc>("construction.get");
}

function snap(p: number[]) {
  const t = open(), last = t?.points.at(-1), points = (doc?.traces ?? []).flatMap((x) => x.points);
  const near = points.filter((q) => !t || q !== last).sort((x, y) => gap(x, p) - gap(y, p))[0];
  if (near && gap(near, p) < 0.2) return near;
  if (!last) {
    const on = walls().map((w) => { const d = [w.b[0] - w.a[0], w.b[1] - w.a[1]], s = Math.max(0, Math.min(1, ((p[0] - w.a[0]) * d[0] + (p[1] - w.a[1]) * d[1]) / (d[0] ** 2 + d[1] ** 2 || 1))); return [w.a[0] + s * d[0], w.a[1] + s * d[1]]; })
      .sort((x, y) => gap(x, p) - gap(y, p))[0];
    return tool === "cloison" && on && gap(on, p) < 0.2 ? on : [grid(p[0]), grid(p[1])];
  }
  const prev = t!.points.at(-2), base = prev ? Math.atan2(last[1] - prev[1], last[0] - prev[0]) : 0, angle = Math.atan2(p[1] - last[1], p[0] - last[0]);
  const quarter = base + Math.round((angle - base) / (Math.PI / 2)) * Math.PI / 2, length = grid(gap(p, last));
  const dir = square && Math.abs(Math.atan2(Math.sin(angle - quarter), Math.cos(angle - quarter))) < Math.PI / 30 ? quarter : angle;
  return [last[0] + length * Math.cos(dir), last[1] + length * Math.sin(dir)];
}

function typed() {
  const t = open(), last = t?.points.at(-1), length = decimal(lengthInput?.value ?? "");
  if (!t || !last) throw new Error("Clique d'abord le premier angle dans le plan.");
  if (!(length > 0)) throw new Error("Tape la longueur du mur en centimètres, par exemple 420.");
  const target = cursor && snap(cursor), prev = t.points.at(-2), aim = target && gap(target, last) > 0.01 ? target : [last[0] + 1, last[1]], free = angleInput?.value.trim() ? decimal(angleInput.value) : null;
  let dir = Math.atan2(aim[1] - last[1], aim[0] - last[0]);
  if (prev && free != null) {
    const base = Math.atan2(last[1] - prev[1], last[0] - prev[0]), side = Math.sign(Math.sin(dir - base)) || 1;
    dir = base + side * (180 - free) * Math.PI / 180;
  }
  return send({ op: "add", point: [last[0] + length / 100 * Math.cos(dir), last[1] + length / 100 * Math.sin(dir)], length });
}

function nearest(p: number[]) {
  const scored = walls().map((w) => {
    const d = [w.b[0] - w.a[0], w.b[1] - w.a[1]], n = d[0] ** 2 + d[1] ** 2 || 1, s = ((p[0] - w.a[0]) * d[0] + (p[1] - w.a[1]) * d[1]) / n;
    return { w, s, d: gap(p, [w.a[0] + Math.max(0, Math.min(1, s)) * d[0], w.a[1] + Math.max(0, Math.min(1, s)) * d[1]]) };
  }).sort((x, y) => x.d - y.d)[0];
  return scored && scored.d < 0.6 ? scored : null;
}

function pick(p: number[]) {
  const t = open();
  if (tool === "mur" || tool === "cloison") {
    const q = snap(p);
    if (!t) return send({ op: "start", point: q, cloison: tool === "cloison" });
    if (t.points.length > 2 && !t.cloison && q === t.points[0]) return send({ op: "close" });
    return send({ op: "add", point: q });
  }
  const hit = nearest(p);
  if (tool === "retoucher") {
    const hole = (doc?.openings ?? []).find((o) => { const w = walls().find((x) => x.name === o.wall); return w && hit?.w.name === w.name && Math.abs(hit.s * w.length - (o.pos + o.width / 2)) < o.width / 2 + 10; });
    picked = hole?.id ?? hit?.w.name ?? "";
    return h.render();
  }
  if (!hit) throw new Error("Clique sur un mur (à moins de 60 cm).");
  const width = { porte: 83, fenetre: 120, porte_fenetre: 140 }[tool] ?? 90;
  return send({ op: "opening", type: tool, wall: hit.w.name, pos: Math.round(Math.max(0, hit.s * hit.w.length - width / 2)) });
}

function plan() {
  const all = (doc?.traces ?? []).flatMap((t) => t.points), xs = all.map((p) => p[0]), zs = all.map((p) => p[1]);
  const ext = all.length > 1 ? [Math.min(...xs), Math.min(...zs), Math.max(...xs), Math.max(...zs)] : [0, 0, 6, 4];
  const m = Math.max(ext[2] - ext[0], ext[3] - ext[1], 2) * 0.15, full = [ext[0] - m, ext[1] - m, ext[2] - ext[0] + 2 * m, ext[3] - ext[1] + 2 * m];
  view ??= full;
  const svg = node("svg", { viewBox: view.join(" "), role: "img", "aria-label": "Plan en construction, vu de dessus" }, "plan") as SVGSVGElement, font = Math.max(view[2], view[3]) * 0.022;
  const lines = node("g", {}, "", "");
  for (let x = Math.floor(view[0]); x < view[0] + view[2]; x++) lines.append(node("path", { d: `M${x} ${view[1]}V${view[1] + view[3]}` }, "grid"));
  for (let z = Math.floor(view[1]); z < view[1] + view[3]; z++) lines.append(node("path", { d: `M${view[0]} ${z}H${view[0] + view[2]}` }, "grid"));
  const centre = [0, 1].map((i) => all.reduce((s, p) => s + p[i], 0) / Math.max(1, all.length)), group = node("g", {});
  for (const w of walls()) {
    group.append(node("line", { x1: w.a[0], y1: w.a[1], x2: w.b[0], y2: w.b[1] }, `wall ${w.trace.cloison ? "cloison" : ""} ${picked === w.name ? "selected" : ""}`));
    const mid = [(w.a[0] + w.b[0]) / 2, (w.a[1] + w.b[1]) / 2], away = gap(mid, centre) || 1;
    group.append(node("text", { x: mid[0] + (mid[0] - centre[0]) / away * font * 1.4, y: mid[1] + (mid[1] - centre[1]) / away * font * 1.4, "font-size": font * 1.1 }, "label", w.name));
    drawCote(group, [w.a, w.b], w.typed ? number(w.typed, 0) : `≈ ${number(w.length, 0)}`, w.typed ? "done" : "", font, centre);
  }
  for (const o of doc?.openings ?? []) {
    const w = walls().find((x) => x.name === o.wall);
    if (!w) continue;
    const at = (cm: number) => [w.a[0] + (w.b[0] - w.a[0]) * cm / w.length, w.a[1] + (w.b[1] - w.a[1]) * cm / w.length], [p, q] = [at(o.pos), at(o.pos + o.width)];
    group.append(node("line", { x1: p[0], y1: p[1], x2: q[0], y2: q[1] }, `opening ${o.type} ${picked === o.id ? "selected" : ""}`));
  }
  const t = open(), last = t?.points.at(-1), draft = node("g", {});
  for (const p of all) group.append(node("circle", { cx: p[0], cy: p[1], r: font * 0.25 }, "vertex"));
  const paint = () => {
    draft.replaceChildren();
    if (!cursor) return;
    const q = tool === "mur" || tool === "cloison" ? snap(cursor) : cursor;
    if (last && (tool === "mur" || tool === "cloison")) drawCote(draft, [last, q], `≈ ${number(gap(last, q) * 100, 0)}`, "focus", font, centre);
    draft.append(node("circle", { cx: q[0], cy: q[1], r: font * 0.4 }, "snap"));
    if (lengthInput && last) lengthInput.placeholder = number(gap(last, q) * 100, 0);
  };
  svg.append(lines, group, draft);
  const at = (x: number, y: number, m = svg.getScreenCTM()!.inverse()) => { const p = new DOMPoint(x, y).matrixTransform(m); return [p.x, p.y]; };
  const set = (box: number[]) => { view = box; svg.setAttribute("viewBox", box.join(" ")); };
  let drag: { m: DOMMatrix; start: number[]; box: number[]; moved: boolean } | null = null;
  svg.addEventListener("wheel", (e) => { e.preventDefault(); const [x, y] = at(e.clientX, e.clientY), [bx, by, bw, bh] = view!, f = e.deltaY > 0 ? 1.2 : 1 / 1.2; set([x - (x - bx) * f, y - (y - by) * f, bw * f, bh * f]); }, { passive: false });
  svg.addEventListener("pointerdown", (e) => { drag = { m: svg.getScreenCTM()!.inverse(), start: [e.clientX, e.clientY], box: view!, moved: false }; });
  svg.addEventListener("pointermove", (e) => {
    if (drag && e.buttons & 1) {
      drag.moved ||= gap(drag.start, [e.clientX, e.clientY]) > 4;
      if (drag.moved) { const [a, b] = [at(drag.start[0], drag.start[1], drag.m), at(e.clientX, e.clientY, drag.m)]; set([drag.box[0] - b[0] + a[0], drag.box[1] - b[1] + a[1], drag.box[2], drag.box[3]]); }
      return;
    }
    cursor = at(e.clientX, e.clientY);
    paint();
  });
  svg.addEventListener("click", (e) => { if (!drag?.moved) run(() => pick(at(e.clientX, e.clientY))); });
  const zoom = (f: number) => { const [bx, by, bw, bh] = view!; set([bx + bw * (1 - f) / 2, by + bh * (1 - f) / 2, bw * f, bh * f]); };
  const bar = el("div", "cote-tools"), box = el("section"), keep = Object.assign(el("input"), { type: "checkbox", checked: square });
  keep.onchange = () => { square = keep.checked; };
  const label = el("label", "check", "");
  label.append(keep, "Angle droit accroché");
  const zooms = el("span", "zoom");
  zooms.append(button("−", "", "Dézoomer", () => zoom(1.25)), button("+", "", "Zoomer", () => zoom(0.8)), button("Tout voir", "", "Revenir au plan entier", () => set(full)));
  bar.append(choice("Outil", TOOLS, tool, (id) => { tool = id; h.render(); }), label, el("span", "spacer"), phoneButton(h.render), zooms);
  box.append(bar, svg);
  overlay.replaceChildren(box);
  overlay.classList.remove("half");
  overlay.hidden = false;
}

function wallRow(w: Wall) {
  const line = el("div", `tagged ${picked === w.name ? "selected" : ""}`), length = Object.assign(el("input"), { inputMode: "decimal", value: w.typed ? String(w.typed).replace(".", ",") : "", placeholder: number(w.length, 0), ariaLabel: `Longueur du mur ${w.name} en cm` });
  length.onchange = () => run(() => send({ op: "length", wall: w.name, value: decimal(length.value) || null }));
  const angle = Object.assign(el("input"), { inputMode: "decimal", value: w.angle == null ? "" : String(Math.round(w.angle)), ariaLabel: `Angle au début du mur ${w.name} en degrés`, disabled: w.angle == null || w.trace.closed && w.k === w.trace.names.length - 1 });
  angle.onchange = () => run(() => send({ op: "angle", wall: w.name, value: decimal(angle.value) }));
  line.append(el("span", "tag", w.name), el("span", "fill", w.trace.cloison ? "Cloison" : "Mur"), length, el("span", "", "cm"), angle, el("span", "", "°"),
    button("×", "remove", `Supprimer le mur ${w.name}`, () => run(() => send({ op: "remove", wall: w.name }))),
    ...w.typed && Math.abs(w.length - w.typed) > 1 ? [el("small", "grow warn", `le tracé donne ${number(w.length, 0)} cm : retouche un mur voisin, sinon la vérification signalera l'écart`)] : []);
  line.onclick = (event) => { if (event.target === line) { picked = picked === w.name ? "" : w.name; h.render(); } };
  return line;
}

function holeRow(o: Hole) {
  const line = el("div", `tagged ${picked === o.id ? "selected" : ""}`), type = el("select");
  type.append(...Object.entries(KINDS).map(([id, label]) => new Option(label, id, false, id === o.type)));
  type.ariaLabel = "Type";
  const field = (key: "pos" | "width" | "height" | "sill", label: string) => Object.assign(el("input"), { inputMode: "decimal", value: String(o[key]).replace(".", ","), ariaLabel: `${label} en cm`, title: label });
  const [pos, width, height, sill] = [field("pos", "Depuis le début du mur"), field("width", "Largeur"), field("height", "Hauteur"), field("sill", "Allège")];
  const save = () => run(() => send({ op: "opening", id: o.id, type: type.value, wall: o.wall, pos: decimal(pos.value), width: decimal(width.value), height: decimal(height.value), sill: decimal(sill.value) }));
  for (const input of [type, pos, width, height, sill]) input.onchange = save;
  line.append(el("span", "tag", o.wall), type, pos, width, el("span", "", "×"), height, ...o.type === "fenetre" ? [el("span", "", "allège"), sill] : [], el("span", "", "cm"),
    button("×", "remove", "Supprimer cette ouverture", () => run(() => send({ op: "remove", opening: o.id }))),
    el("small", "grow", `à ${number(o.pos, 0)} cm du début du mur ${o.wall} · ${number(o.width, 0)} × ${number(o.height, 0)} cm${o.type === "fenetre" ? ` · allège ${number(o.sill, 0)} cm` : ""}`));
  return line;
}

export function buildStep(helpers: Helpers): Node[] {
  h = helpers;
  if (!doc) { run(async () => { await loadBuild(); h.render(); }); return [h.intro("Construire", "Chargement…")]; }
  plan();
  const t = open(), list = walls(), state = JSON.stringify(doc), valid = h.built() > 0 && (applied === null || applied === state);
  lengthInput = Object.assign(el("input", "measure"), { inputMode: "decimal", autocomplete: "off", placeholder: "cm", ariaLabel: "Longueur du mur en cours en cm" });
  angleInput = Object.assign(el("input", "measure"), { inputMode: "decimal", autocomplete: "off", placeholder: "90", ariaLabel: "Angle entre les murs en degrés" });
  lengthInput.onkeydown = angleInput.onkeydown = (event) => { if (event.key === "Enter") run(typed); };
  const current = t ? (() => {
    const box = el("div", "card focus"), fields = el("div", "fields");
    for (const [text, input] of [["Longueur (cm)", lengthInput!], ["Angle entre les murs (°)", angleInput!]] as const) { const label = el("label", "", text); label.append(input); fields.append(label); }
    box.append(el("b", "", `${t.cloison ? "Cloison" : "Mur"} en cours · ${fresh()}`), el("span", "", t.points.length > 1 ? `depuis le coin ${t.names.at(-1)}–${fresh()} · angle vide = suivre la souris` : "depuis le premier angle · la souris donne la direction"), fields,
      h.row(h.primary("Mur suivant", typed), ...t.cloison ? [] : [Object.assign(button("Fermer la pièce", "", "Relier le dernier angle au premier", () => run(() => send({ op: "close", ...decimal(lengthInput!.value) > 0 ? { length: decimal(lengthInput!.value) } : {} }))), { disabled: t.points.length < 3 })],
        button(t.cloison ? "Terminer la cloison" : "Terminer sans fermer", "", "Garder les murs posés", () => run(() => send({ op: "finish" }))), button("Annuler le point", "link", "Retirer le dernier angle", () => run(() => send({ op: "back" })))));
    queueMicrotask(() => lengthInput?.focus());
    return [box];
  })() : [el("p", "status", HINTS[tool])];
  const height = Object.assign(el("input"), { inputMode: "decimal", value: String(doc.height), ariaLabel: "Hauteur sous plafond en cm" });
  height.onchange = () => run(() => send({ op: "height", value: decimal(height.value) }));
  const heightRow = el("div", "tagged");
  heightRow.append(el("span", "fill", "Hauteur sous plafond"), height, el("span", "", "cm"));
  const box = el("div", "walls");
  box.append(...list.map(wallRow));
  const holes = el("div", "walls");
  holes.append(...doc.openings.map(holeRow));
  const phone = phonePanel({ title: "Étape 4 · Construire", mode: "construire", k: 1, segs: [], ring: [], hidden: [], status: {}, todo: [] },
    { changed: () => {}, hide: () => {}, built: () => run(async () => { await loadBuild(); if (applied === null) applied = ""; h.render(); }) }, h.render);
  const validate = () => h.act("Création du plan", async () => { made = await call("construction.apply"); applied = JSON.stringify(doc); await h.finish(); });
  return [h.intro("Construis la pièce", "Pose les angles un par un, au clic ou au clavier. Les longueurs que tu tapes donnent l'échelle : mesure chaque mur au mètre."),
    ...current, ...phone,
    ...list.length ? [el("span", "kicker", `Murs · ${list.length} · longueur (cm) et angle au début du mur`), box] : [],
    ...doc.openings.length ? [el("span", "kicker", `Portes et fenêtres · ${doc.openings.length} · depuis le début du mur, largeur × hauteur`), holes] : [],
    heightRow,
    ...made && valid ? [h.card("C'est fait", `${made.walls.length} mur${made.walls.length > 1 ? "s" : ""} et ${made.openings.length} ouverture${made.openings.length > 1 ? "s" : ""} dans le plan.${made.skipped.length ? ` Non posées : ${made.skipped.join(" ; ")}.` : ""}`)] : [],
    ...valid ? h.extra() : [],
    el("div", "note", "Retoucher : une longueur changée déplace les murs suivants jusqu'au prochain mur parallèle ; un angle changé tourne la suite. Ctrl+Z annule."),
    h.footer(h.back(), ...valid ? [button("Valider à nouveau", "", "Refaire le plan depuis la construction", () => run(validate)), h.primary("Étape suivante", h.next)]
      : [Object.assign(h.primary("Valider la construction", validate), { disabled: !list.length || Boolean(t) })])];
}

export function resetBuild() {
  [doc, view, cursor, picked, applied, made] = [null, null, null, "", null, null];
}

document.addEventListener("keydown", (event) => {
  if (overlay.hidden || !doc || !open() || event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
  if (/^[\d.,]$/.test(event.key) && lengthInput) lengthInput.focus();
  if (event.key === "Escape") run(() => send({ op: "finish" }));
  if (event.key === "Backspace") run(() => send({ op: "back" }));
});
