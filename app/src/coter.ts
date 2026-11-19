import { node } from "./plan2d";
import { button, el } from "./ui";

export type End = { wall: string; s: number; r?: number; edge?: number };
export type Seg = { wall: string; a: number[]; b: number[]; kind: string };
export type Mode = "libre" | "point" | "ligne";
export type Pick = { at: number[]; ends: End[] };
export type Tool = { mode: Mode; first: Pick | null; view: number[] | null };
export type Drawn = { id: string; ends: number[][]; text: string; cls: string };

export const MODES: [Mode, string, string][] = [["libre", "Libre", "2 points n'importe où, chacun suit le mur le plus proche"],
  ["point", "Point", "accroche aux coins, aux bouts de cloison, aux bords des portes et fenêtres"], ["ligne", "Ligne", "2 points n'importe où sur les murs ou les ouvertures"]];
const NOUNS: Record<string, string> = { porte: "la porte", fenetre: "la fenêtre", porte_fenetre: "la porte-fenêtre" };
const sub = (p: number[], q: number[]) => [p[0] - q[0], p[1] - q[1]];
const dot = (u: number[], v: number[]) => u[0] * v[0] + u[1] * v[1];
const gap = (p: number[], q: number[]) => Math.hypot(p[0] - q[0], p[1] - q[1]);
const round = (x: number) => Math.round(x * 1e4) / 1e4;
const along = (s: Seg, t: number) => [s.a[0] + t * (s.b[0] - s.a[0]), s.a[1] + t * (s.b[1] - s.a[1])];
const fraction = (s: Seg, p: number[]) => dot(sub(p, s.a), sub(s.b, s.a)) / Math.max(1e-12, dot(sub(s.b, s.a), sub(s.b, s.a)));
const isWall = (s: Seg) => s.kind === "mur" || s.kind === "cloison";
const host = (segs: Seg[], name: string) => segs.find((s) => isWall(s) && s.wall === name);

export function locate(end: End, segs: Seg[]) {
  const w = host(segs, end.wall);
  if (!w) return null;
  if (end.edge) {
    const best = segs.filter((s) => !isWall(s) && s.wall === end.wall).flatMap((s) => [s.a, s.b]).map((p) => [Math.abs(fraction(w, p) - end.s), p] as const).sort((x, y) => x[0] - y[0])[0];
    return best && best[0] < 0.2 ? best[1] : null;
  }
  const v = sub(w.b, w.a), r = end.r ?? 0;
  return [w.a[0] + end.s * v[0] - r * v[1], w.a[1] + end.s * v[1] + r * v[0]];
}

export function snap(mode: Mode, p: number[], segs: Seg[], tol: number): Pick | null {
  if (mode === "point") {
    const spots = segs.flatMap((s) => [s.a, s.b].map((at, i) => ({ at, end: isWall(s) ? { wall: s.wall, s: i } : { wall: s.wall, s: round(fraction(host(segs, s.wall)!, at)), edge: 1 } })));
    const best = [...spots].sort((x, y) => gap(x.at, p) - gap(y.at, p))[0];
    return best && gap(best.at, p) < tol ? { at: best.at, ends: spots.filter((x) => gap(x.at, best.at) < tol * 0.05).map((x) => x.end) } : null;
  }
  const near = segs.filter((s) => mode === "ligne" || isWall(s)).map((s) => { const t = Math.min(1, Math.max(0, fraction(s, p))); return { s, t, d: gap(along(s, t), p) }; }).sort((x, y) => x.d - y.d)[0];
  if (!near || mode === "ligne" && near.d > tol) return null;
  if (mode === "ligne") { const at = along(near.s, near.t); return { at, ends: [{ wall: near.s.wall, s: round(fraction(host(segs, near.s.wall)!, at)) }] }; }
  const v = sub(near.s.b, near.s.a), q = sub(p, near.s.a), n = dot(v, v);
  return { at: p, ends: [{ wall: near.s.wall, s: round(dot(q, v) / n), r: round((v[0] * q[1] - v[1] * q[0]) / n) }] };
}

export function pair(a: Pick, b: Pick): End[] {
  for (const e of a.ends) for (const f of b.ends) if (e.wall === f.wall) return [e, f];
  return [a.ends[0], b.ends[0]];
}

function describe(end: End, segs: Seg[], ring: string[]) {
  const w = host(segs, end.wall), [word, of] = w?.kind === "cloison" ? ["cloison", "de la cloison"] : ["mur", "du mur"], i = ring.indexOf(end.wall), n = ring.length;
  if (end.edge) return `bord de ${NOUNS[segs.find((s) => !isWall(s) && s.wall === end.wall && [s.a, s.b].some((p) => Math.abs(fraction(w!, p) - end.s) < 0.02))?.kind ?? ""] ?? "l'ouverture"} du mur ${end.wall}`;
  if (end.r) return `près ${of} ${end.wall}`;
  if (end.s !== 0 && end.s !== 1) return `${word} ${end.wall}`;
  if (i < 0) return `${end.s ? "bout" : "départ"} ${of} ${end.wall}`;
  return `coin ${end.s ? `${end.wall}–${ring[(i + 1) % n]}` : `${ring[(i - 1 + n) % n]}–${end.wall}`}`;
}

export function title(ends: End[], segs: Seg[], ring: string[]) {
  const [a, b] = ends;
  if (a.wall === b.wall && !a.edge && !b.edge && !a.r && !b.r && a.s + b.s === 1 && a.s * b.s === 0) return `Longueur ${host(segs, a.wall)?.kind === "cloison" ? "de la cloison" : "du mur"} ${a.wall}`;
  const text = `${describe(a, segs, ring)} → ${describe(b, segs, ring)}`;
  return text[0].toUpperCase() + text.slice(1);
}

export function drawCote(group: Element, ends: number[][], text: string, cls: string, font: number, centre: number[], away = 1.2) {
  const [dx, dy] = sub(ends[1], ends[0]), length = Math.hypot(dx, dy) || 1;
  let [nx, ny] = [-dy / length, dx / length];
  if (nx * ((ends[0][0] + ends[1][0]) / 2 - centre[0]) + ny * ((ends[0][1] + ends[1][1]) / 2 - centre[1]) > 0) [nx, ny] = [-nx, -ny];
  const [p, q] = ends.map(([x, y]) => [x + nx * font * away, y + ny * font * away]);
  const tick = ([x, y]: number[]) => `M${x - nx * font * 0.4} ${y - ny * font * 0.4}L${x + nx * font * 0.4} ${y + ny * font * 0.4}`;
  if (cls.includes("focus")) group.append(node("path", { d: `M${p[0]} ${p[1]}L${q[0]} ${q[1]}` }, "cote glow"));
  group.append(node("path", { d: `M${ends[0][0]} ${ends[0][1]}L${p[0]} ${p[1]}M${ends[1][0]} ${ends[1][1]}L${q[0]} ${q[1]}` }, `cote ext ${cls}`));
  group.append(node("path", { d: `M${p[0]} ${p[1]}L${q[0]} ${q[1]}${tick(p)}${tick(q)}` }, `cote ${cls}`));
  group.append(node("text", { x: (p[0] + q[0]) / 2 + nx * font * 0.9, y: (p[1] + q[1]) / 2 + ny * font * 0.9, "font-size": font }, `cote-label ${cls}`, text));
}

export function coteLayer(svg: SVGSVGElement, tool: Tool, segs: Seg[], drawn: Drawn[], draft: number[][] | null, draftText: string, onPick: (p: Pick) => void) {
  const base = svg.viewBox.baseVal, full = [base.x, base.y, base.width, base.height], font = Math.max(full[2], full[3]) * 0.022;
  const ends = segs.flatMap((s) => [s.a, s.b]), centre = [0, 1].map((i) => ends.reduce((t, p) => t + p[i], 0) / Math.max(1, ends.length));
  const shown = el("span", "zoom-value"), set = (box: number[]) => { tool.view = box; svg.setAttribute("viewBox", box.join(" ")); shown.textContent = `${Math.round(full[2] / box[2] * 100)} %`; };
  set(tool.view ?? full);
  const group = node("g", {}), marker = node("g", {});
  for (const d of drawn) drawCote(group, d.ends, d.text, d.cls, font, centre);
  if (draft) drawCote(group, draft, draftText, "focus", font, centre);
  svg.append(group, marker);
  const point = (x: number, y: number, m = svg.getScreenCTM()!.inverse()) => { const p = new DOMPoint(x, y).matrixTransform(m); return [p.x, p.y]; };
  const tol = () => 14 / (svg.getScreenCTM()?.a || 1);
  let drag: { m: DOMMatrix; start: number[]; box: number[]; moved: boolean } | null = null;
  svg.addEventListener("wheel", (e) => {
    e.preventDefault();
    const [x, y] = point(e.clientX, e.clientY), [bx, by, bw, bh] = tool.view!, f = e.deltaY > 0 ? 1.2 : 1 / 1.2;
    set([x - (x - bx) * f, y - (y - by) * f, bw * f, bh * f]);
  }, { passive: false });
  svg.addEventListener("pointerdown", (e) => { drag = { m: svg.getScreenCTM()!.inverse(), start: [e.clientX, e.clientY], box: tool.view!, moved: false }; });
  svg.addEventListener("pointermove", (e) => {
    if (drag && e.buttons & 1) {
      drag.moved ||= gap(drag.start, [e.clientX, e.clientY]) > 4;
      if (drag.moved) { const [a, b] = [point(drag.start[0], drag.start[1], drag.m), point(e.clientX, e.clientY, drag.m)]; set([drag.box[0] - b[0] + a[0], drag.box[1] - b[1] + a[1], drag.box[2], drag.box[3]]); }
      return;
    }
    const hit = snap(tool.mode, point(e.clientX, e.clientY), segs, tol());
    marker.replaceChildren(...tool.first && hit ? [node("path", { d: `M${tool.first.at[0]} ${tool.first.at[1]}L${hit.at[0]} ${hit.at[1]}` }, "cote")] : [],
      ...[tool.first, hit].filter(Boolean).map((h) => node("circle", { cx: h!.at[0], cy: h!.at[1], r: font * 0.35 }, "snap")));
  });
  svg.addEventListener("click", (e) => {
    if (drag?.moved) return;
    const hit = snap(tool.mode, point(e.clientX, e.clientY), segs, tol());
    if (hit && !(tool.first && gap(hit.at, tool.first.at) < tol())) onPick(hit);
  });
  const zoom = (f: number) => { const [bx, by, bw, bh] = tool.view!; set([bx + bw * (1 - f) / 2, by + bh * (1 - f) / 2, bw * f, bh * f]); };
  const box = el("span", "zoom");
  box.append(button("−", "", "Dézoomer", () => zoom(1.25)), shown, button("+", "", "Zoomer", () => zoom(0.8)), button("Tout voir", "", "Revenir au plan entier", () => set(full)));
  return [box];
}

export function choice(label: string, options: [string, string][], current: string, change: (id: string) => void) {
  const box = el("div", "seg");
  box.setAttribute("role", "radiogroup");
  box.ariaLabel = label;
  box.append(...options.map(([id, text]) => {
    const line = el("label", id === current ? "sel" : ""), input = Object.assign(el("input"), { type: "radio", name: label, checked: id === current });
    input.onchange = () => change(id);
    line.append(input, text);
    return line;
  }));
  return box;
}
