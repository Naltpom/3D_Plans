export type DrawnWall = { name: string; a: number[]; b: number[] };
export type Opening = { type: string; wall: string; t0: number; t1: number; seen?: boolean | null; same?: boolean };
export type Fit = { k: number; turn: number | null; pose: { L: number[][]; c: number[] }; pairs: Record<string, string[]>; gaps: Record<string, number>; height: number | null;
  photo: { id: string; sure: boolean; meuble?: boolean; front?: string | null; wall: string | null; a: number[]; b: number[] }[];
  walls?: DrawnWall[]; extra?: DrawnWall[]; seen?: Record<string, string>; moved?: Record<string, number>; stairs?: (number | null)[] };
export type Corner = { corner: number; clicks: { image: string; u: number; v: number; kind?: string }[]; skipped?: boolean };
export type Stair = { box: number[]; kept?: boolean; flights?: number[][]; treads?: number[]; turn?: { square: number[]; pivot: number[]; winders: number; tour: number } };
export type Rule = { kind: string; walls: string[]; ok: boolean | null };
export type Plan = { image: number[]; angle: number; walls: DrawnWall[]; extra: { a: number[]; b: number[]; inside: boolean; name?: string }[]; openings: Opening[];
  stairs: Stair[]; rules?: Rule[]; legend: { color: string; hex: string; count: number; role: string }[]; ai?: boolean; fit?: Fit | null; points?: Corner[]; refused?: Record<string, string[]> };
type Options = { selected?: string; opening?: number; corner?: number; photo?: boolean; fitted?: boolean; cloud?: number[][] | null; onWall?: (name: string) => void; onOpening?: (i: number) => void };

const NS = "http://www.w3.org/2000/svg";

export function node(tag: string, attributes: Record<string, string | number>, className = "", text = "") {
  const item = document.createElementNS(NS, tag);
  for (const [name, value] of Object.entries(attributes)) item.setAttribute(name, String(value));
  if (className) item.setAttribute("class", className);
  if (text) item.textContent = text;
  return item;
}

const along = (wall: DrawnWall, t: number) => wall.a.map((v, i) => v + t * (wall.b[i] - v));

export function metres(plan: Plan, k: number, [u, v]: number[]) {
  const [u0, v0] = plan.walls[0].a, c = Math.cos(plan.angle), s = Math.sin(plan.angle);
  return [k * (c * (u - u0) + s * (v - v0)), k * (-s * (u - u0) + c * (v - v0))];
}

export const shapeOf = (plan: Plan) => {
  const fitted = new Map([...plan.fit?.walls ?? [], ...plan.fit?.extra ?? []].map((w) => [w.name, w]));
  return { walls: plan.walls.map((w) => fitted.get(w.name) ?? w), extra: plan.extra.map((e) => e.name && fitted.get(e.name) ? { ...e, ...fitted.get(e.name) } : e) };
};

function stairLines(stair: Stair) {
  const lines: number[][] = [], rects = stair.flights ?? [stair.box], upright = rects[0][3] - rects[0][1] >= rects[0][2] - rects[0][0];
  rects.forEach(([u0, v0, u1, v1], i) => {
    const n = stair.treads?.[i] ?? 0, tall = upright === (i === 0);
    for (let k = 1; k < n; k++) lines.push(tall ? [u0, v0 + k * (v1 - v0) / n, u1, v0 + k * (v1 - v0) / n] : [u0 + k * (u1 - u0) / n, v0, u0 + k * (u1 - u0) / n, v1]);
  });
  const turn = stair.turn;
  if (turn) {
    const [u0, v0, u1, v1] = turn.square, [pu, pv] = turn.pivot, far = [pu === u0 ? u1 : u0, pv === v0 ? v1 : v0], side = Math.max(u1 - u0, v1 - v0);
    for (let k = 1; k < turn.winders; k++) {
      const t = Math.tan(k * Math.PI / 2 / turn.winders), [du, dv] = [far[0] - pu, far[1] - pv], hit = t <= 1 ? [du, Math.sign(dv) * Math.abs(du) * t] : [Math.sign(du) * Math.abs(dv) / t, dv];
      lines.push([pu, pv, pu + Math.sign(hit[0]) * Math.min(Math.abs(hit[0]), side), pv + Math.sign(hit[1]) * Math.min(Math.abs(hit[1]), side)]);
    }
  }
  return { rects: turn ? [...rects, turn.square] : rects, lines };
}

export function drawPlan(plan: Plan, options: Options = {}) {
  const { walls, extra } = options.fitted ? shapeOf(plan) : plan;
  const ends = [...walls, ...extra, ...options.photo ? plan.fit?.photo ?? [] : []].flatMap((w) => [w.a, w.b]);
  const xs = ends.map((p) => p[0]), ys = ends.map((p) => p[1]), size = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1);
  const margin = size * 0.1, font = size * 0.045;
  const svg = node("svg", { viewBox: `${Math.min(...xs) - margin} ${Math.min(...ys) - margin} ${Math.max(...xs) - Math.min(...xs) + 2 * margin} ${Math.max(...ys) - Math.min(...ys) + 2 * margin}`,
    role: "img", "aria-label": "Plan de dessus, murs nommés", preserveAspectRatio: "xMidYMid meet" }, "plan") as SVGSVGElement;
  const paired = new Set([...Object.keys(plan.fit?.pairs ?? {}), ...Object.keys(plan.fit?.seen ?? {})]), asked = new Set((plan.rules ?? []).filter((r) => r.ok === null).flatMap((r) => [r.walls[0], r.walls[2]]));
  svg.append(node("polygon", { points: walls.map((w) => w.a.join(",")).join(" ") }, "room"));
  if (options.cloud?.length) svg.append(node("path", { d: options.cloud.map(([u, v]) => `M${u} ${v}h${size / 220}v${size / 220}h${-size / 220}z`).join(""), "aria-hidden": "true" }, "cloud"));
  for (const stair of plan.stairs.filter((s) => s.kept !== false)) {
    const { rects, lines } = stairLines(stair);
    for (const [u0, v0, u1, v1] of rects) svg.append(node("rect", { x: u0, y: v0, width: u1 - u0, height: v1 - v0 }, "stairs"));
    for (const [x1, y1, x2, y2] of lines) svg.append(node("line", { x1, y1, x2, y2 }, "stairs tread"));
  }
  if (options.photo) for (const w of plan.fit?.photo ?? []) svg.append(node("line", { x1: w.a[0], y1: w.a[1], x2: w.b[0], y2: w.b[1] }, `photo ${w.wall ? "" : "loose"} ${w.meuble ? "meuble" : ""}`));
  for (const wall of walls) {
    if (asked.has(wall.name)) svg.append(node("line", { x1: wall.a[0], y1: wall.a[1], x2: wall.b[0], y2: wall.b[1] }, "rule"));
    const line = node("line", { x1: wall.a[0], y1: wall.a[1], x2: wall.b[0], y2: wall.b[1] }, `wall ${wall.name === options.selected ? "selected" : ""} ${options.photo && !paired.has(wall.name) ? "dim" : ""}`);
    if (options.onWall) line.addEventListener("click", () => options.onWall!(wall.name));
    svg.append(line);
  }
  plan.openings.forEach((opening, i) => {
    const wall = walls.find((w) => w.name === opening.wall);
    if (!wall) return;
    const [p, q] = [along(wall, opening.t0), along(wall, opening.t1)];
    const line = node("line", { x1: p[0], y1: p[1], x2: q[0], y2: q[1] }, `opening ${opening.type} ${i === options.opening ? "selected" : ""}`);
    if (options.onOpening) line.addEventListener("click", () => options.onOpening!(i));
    svg.append(line);
  });
  for (const item of extra) {
    const line = node("line", { x1: item.a[0], y1: item.a[1], x2: item.b[0], y2: item.b[1] }, `extra ${item.name ? "kept" : ""} ${item.name === options.selected ? "selected" : ""} ${options.photo && item.name && !paired.has(item.name) ? "dim" : ""}`);
    if (options.onWall && item.name) line.addEventListener("click", () => options.onWall!(item.name!));
    svg.append(line);
  }
  for (const wall of [...walls, ...extra.filter((e) => e.name) as DrawnWall[]]) {
    const [dx, dy] = [wall.b[0] - wall.a[0], wall.b[1] - wall.a[1]], length = Math.hypot(dx, dy) || 1, middle = along(wall, 0.5);
    const small = Math.min(1, Math.max(0.55, length / (0.15 * size))), label = node("text", { x: middle[0] + dy / length * font * small, y: middle[1] - dx / length * font * small, "font-size": font * small }, `label ${options.photo && !paired.has(wall.name) ? "dim" : ""}`, wall.name);
    if (options.onWall) label.addEventListener("click", () => options.onWall!(wall.name));
    svg.append(label);
  }
  if (options.corner != null && walls[options.corner]) {
    const [u, v] = walls[options.corner].a;
    svg.append(node("circle", { cx: u, cy: v, r: font * 0.4 }, "corner"));
  }
  return svg;
}
