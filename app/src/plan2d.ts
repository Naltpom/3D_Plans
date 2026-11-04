export type DrawnWall = { name: string; a: number[]; b: number[] };
export type Opening = { type: string; wall: string; t0: number; t1: number; seen?: boolean | null; same?: boolean };
export type Fit = { k: number; turn: number | null; pose: { L: number[][]; c: number[] }; pairs: Record<string, string[]>; gaps: Record<string, number>; height: number | null;
  photo: { id: string; sure: boolean; meuble?: boolean; front?: string | null; wall: string | null; a: number[]; b: number[] }[] };
export type Corner = { corner: number; clicks: { image: string; u: number; v: number; kind?: string }[]; skipped?: boolean };
export type Plan = { image: number[]; angle: number; walls: DrawnWall[]; extra: { a: number[]; b: number[]; inside: boolean; name?: string }[]; openings: Opening[];
  stairs: { box: number[]; kept?: boolean }[]; legend: { color: string; hex: string; count: number; role: string }[]; ai?: boolean; fit?: Fit | null; points?: Corner[]; refused?: Record<string, string[]> };
type Options = { selected?: string; opening?: number; corner?: number; photo?: boolean; onWall?: (name: string) => void; onOpening?: (i: number) => void };

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

export function drawPlan(plan: Plan, options: Options = {}) {
  const ends = [...plan.walls, ...plan.extra, ...options.photo ? plan.fit?.photo ?? [] : []].flatMap((w) => [w.a, w.b]);
  const xs = ends.map((p) => p[0]), ys = ends.map((p) => p[1]), size = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1);
  const margin = size * 0.1, font = size * 0.045;
  const svg = node("svg", { viewBox: `${Math.min(...xs) - margin} ${Math.min(...ys) - margin} ${Math.max(...xs) - Math.min(...xs) + 2 * margin} ${Math.max(...ys) - Math.min(...ys) + 2 * margin}`,
    role: "img", "aria-label": "Plan de dessus, murs nommés", preserveAspectRatio: "xMidYMid meet" }, "plan") as SVGSVGElement;
  const paired = new Set(Object.keys(plan.fit?.pairs ?? {}));
  svg.append(node("polygon", { points: plan.walls.map((w) => w.a.join(",")).join(" ") }, "room"));
  for (const stair of plan.stairs.filter((s) => s.kept !== false)) {
    const [u0, v0, u1, v1] = stair.box;
    svg.append(node("rect", { x: u0, y: v0, width: u1 - u0, height: v1 - v0 }, "stairs"));
  }
  if (options.photo) for (const w of plan.fit?.photo ?? []) svg.append(node("line", { x1: w.a[0], y1: w.a[1], x2: w.b[0], y2: w.b[1] }, `photo ${w.wall ? "" : "loose"} ${w.meuble ? "meuble" : ""}`));
  for (const wall of plan.walls) {
    const line = node("line", { x1: wall.a[0], y1: wall.a[1], x2: wall.b[0], y2: wall.b[1] }, `wall ${wall.name === options.selected ? "selected" : ""} ${options.photo && !paired.has(wall.name) ? "dim" : ""}`);
    if (options.onWall) line.addEventListener("click", () => options.onWall!(wall.name));
    svg.append(line);
  }
  plan.openings.forEach((opening, i) => {
    const wall = plan.walls.find((w) => w.name === opening.wall);
    if (!wall) return;
    const [p, q] = [along(wall, opening.t0), along(wall, opening.t1)];
    const line = node("line", { x1: p[0], y1: p[1], x2: q[0], y2: q[1] }, `opening ${opening.type} ${i === options.opening ? "selected" : ""}`);
    if (options.onOpening) line.addEventListener("click", () => options.onOpening!(i));
    svg.append(line);
  });
  for (const extra of plan.extra) svg.append(node("line", { x1: extra.a[0], y1: extra.a[1], x2: extra.b[0], y2: extra.b[1] }, `extra ${extra.name ? "kept" : ""}`));
  for (const wall of [...plan.walls, ...plan.extra.filter((e) => e.name) as DrawnWall[]]) {
    const [dx, dy] = [wall.b[0] - wall.a[0], wall.b[1] - wall.a[1]], length = Math.hypot(dx, dy) || 1, middle = along(wall, 0.5);
    const small = Math.min(1, Math.max(0.55, length / (0.15 * size))), label = node("text", { x: middle[0] + dy / length * font * small, y: middle[1] - dx / length * font * small, "font-size": font * small }, `label ${options.photo && !paired.has(wall.name) ? "dim" : ""}`, wall.name);
    if (options.onWall) label.addEventListener("click", () => options.onWall!(wall.name));
    svg.append(label);
  }
  if (options.corner != null && plan.walls[options.corner]) {
    const [u, v] = plan.walls[options.corner].a;
    svg.append(node("circle", { cx: u, cy: v, r: font * 0.4 }, "corner"));
  }
  return svg;
}
