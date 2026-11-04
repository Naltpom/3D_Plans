import { call } from "./host";
import { $, button, cm, el, number, run } from "./ui";
import { update } from "./projects";
import { highlight, setView } from "./view3d";
import type { Plan } from "./plan2d";

type Item = { key: string; group: string; label: string; hint: string; expr: string; ends: number[][]; on: string[]; covers: string[]; value: number; sigma: number | null; conflict: boolean; twin: string | null };
type Measure = { id: string; label: string; value: number; unit: string; resid: number | null; conflict: boolean; key: string | null; ends: number[][] | null };
type Group = { group: string; name: string; worst: number; label: string; unknown: number; ok: number; n: number };
type Part = { id: string; type: string; label: string; noun: string; xy: number[][] };
type State = { candidates: Item[]; measures: Measure[]; groups: Group[]; scales: { axis: string; value: number; sigma: number | null }[]; conflicts: { text: string; cotes: string[] }[];
  threshold: number; supposed: boolean; precise: boolean; elements: Part[] };
export type Helpers = {
  guide: () => { refused?: string[]; plan?: Plan | null }; save: (change: Record<string, unknown>) => Promise<void>; active: () => boolean;
  intro: (title: string, text: string) => HTMLElement; card: (title: string, text: string, ...actions: HTMLButtonElement[]) => HTMLElement; row: (...nodes: Node[]) => HTMLElement;
  footer: (...actions: HTMLButtonElement[]) => HTMLElement; primary: (text: string, action: () => unknown) => HTMLButtonElement; back: () => HTMLButtonElement; render: () => void;
  next: () => Promise<void>; skip: () => Promise<void>;
};

const NS = "http://www.w3.org/2000/svg";
const overlay = $("#plan2d");
let h: Helpers;
let state: State | null = null;
let before: State | null = null;
let last = "";
let focus = "";
let adding = false;
let picks: string[] = [];
let extra: Item | null = null;
let flat = true;

const decimal = (text: string) => Number(text.trim().replace(",", "."));
const refused = () => new Set(h.guide().refused ?? []);
const measured = () => new Set(state!.measures.map((m) => m.key).filter(Boolean));
const plus = (m: number) => `${m > 0 ? "+" : "−"}${cm(Math.abs(m))}`;
const lower = (text: string) => text[0].toLowerCase() + text.slice(1);
const margin = (s: number | null) => s == null ? "inconnue" : `± ${cm(s)}`;

function proposals() {
  const no = refused(), gone = new Set([...no, ...measured()]), byKey = new Map(state!.candidates.map((c) => [c.key, c]));
  const near = (a: Item, b: Item) => [b.ends, [...b.ends].reverse()].some((e) => a.ends.every((p, i) => Math.hypot(p[0] - e[i][0], p[1] - e[i][1]) < 0.3));
  return state!.candidates.filter((c) => !gone.has(c.key) && (!c.twin || gone.has(c.twin) && !(no.has(c.twin) && near(c, byKey.get(c.twin)!)))).slice(0, 4);
}

async function refresh() {
  await update();
  state = await call<State>("mesures.list");
  await h.save({ measures: state.measures.length });
  picks = [];
  extra = null;
  h.render();
}

export async function scaleEnter(helpers: Helpers) {
  h = helpers;
  [before, last, focus, adding, state] = [null, "", "", false, null];
  await refresh();
}

async function submit(item: Item, text: string) {
  const value = decimal(text), model = item.value * 100;
  if (!(value > 0)) throw new Error("Indique la mesure en centimètres, par exemple 531.");
  if (value / model < 0.5 || value / model > 2) throw new Error(`Le plan donne environ ${number(model, 0)} cm : vérifie que la mesure est en centimètres.`);
  await call("cotes.add", { id: item.key, expr: item.expr, value, sigma: 1, unit: "cm", label: item.label });
  [before, last, focus, adding] = [state, `« ${item.label} »`, "", false];
  await refresh();
}

async function refuse(key: string) {
  await h.save({ refused: [...refused(), key] });
  if (focus === key) focus = "";
  h.render();
}

async function remove(m: Measure) {
  await call("cotes.remove", { id: m.id });
  [before, last] = [state, `le retrait de « ${m.label} »`];
  await refresh();
}

async function pick(id: string | null) {
  if (!id || !state?.elements.some((e) => e.id === id)) return;
  picks = picks.includes(id) ? picks.filter((p) => p !== id) : [...picks.slice(0, 1), id];
  extra = picks.length === 2 ? await call<Item>("mesures.between", { a: picks[0], b: picks[1] }) : null;
  if (extra) focus = extra.key;
  h.render();
}

function entry(item: Item, rank: number) {
  const box = el("div", `tagged ${item.key === focus ? "selected" : ""}`);
  box.dataset.key = item.key;
  const text = el("span", "grow");
  text.append(el("b", "", item.label), el("br"), el("small", "", `${item.hint} · le plan dit ≈ ${number(item.value * 100, 0)} cm ${item.sigma == null ? "(position inconnue)" : `± ${number(item.sigma * 100, 0)}`}`));
  if (item.covers.length) text.append(el("br"), el("span", "chip ok", `vaut aussi pour ${item.covers.join(", ")}`));
  if (item.conflict) text.append(el("br"), el("span", "chip warn", "aide à trancher la contradiction"));
  box.append(...rank ? [el("span", "tag", String(rank))] : [], text);
  if (item.key !== focus) {
    box.append(button("Mesurer", "", "Saisir cette mesure", () => { focus = item.key; h.render(); }));
  } else {
    const input = Object.assign(el("input", "measure"), { inputMode: "decimal", placeholder: "cm", autocomplete: "off", ariaLabel: `${item.label} en cm` });
    const ok = h.primary("Valider", () => submit(item, input.value));
    input.onkeydown = (event) => { if (event.key === "Enter") ok.click(); };
    box.append(h.row(input, el("span", "", "cm"), ok));
    setTimeout(() => input.focus());
  }
  if (rank) box.append(button("Je ne peux pas la mesurer", "link", "Retirer cette mesure et en proposer une autre", () => run(() => refuse(item.key))));
  return box;
}

function effect() {
  if (!before || !state) return [];
  const old = new Map(before.candidates.map((c) => [c.key, c.value]));
  const scale = (s: State, axis: string) => s.scales.find((x) => x.axis === axis)?.value;
  const moved = state.candidates.filter((c) => old.has(c.key) && Math.abs(c.value - old.get(c.key)!) > 0.01 && !c.twin).sort((a, b) => Math.abs(b.value - old.get(b.key)!) - Math.abs(a.value - old.get(a.key)!)).slice(0, 3);
  const [s0, s1] = [scale(before, "échelle"), scale(state, "échelle")];
  const parts = [...s0 && s1 && Math.abs(s1 / s0 - 1) > 0.001 ? [`échelle corrigée de ${number((s1 / s0 - 1) * 100)} %`] : [],
    ...moved.map((c) => `${lower(c.label)} ${plus(c.value - old.get(c.key)!)}`)];
  const margins = state.groups.map((g) => { const was = before!.groups.find((b) => b.group === g.group); return was && Math.abs(was.worst - g.worst) > 5e-4 ? `${lower(g.name)} ${margin(was.unknown ? null : was.worst)} → ${margin(g.unknown ? null : g.worst)}` : ""; }).filter(Boolean);
  return [h.card(`Après ${last}`, [parts.length ? `${parts.join(" · ")}.` : "Rien n'a bougé de plus d'1 cm.", margins.length ? `Marges : ${margins.join(", ")}.` : "", state.conflicts.length ? "" : "Aucune contradiction."].filter(Boolean).join(" "))];
}

function precision() {
  const s = state!, box = el("div", s.precise ? "card done" : "card plain"), drawing = s.scales.find((x) => x.axis === "dessin"), photo = s.scales.find((x) => x.axis === "échelle");
  box.append(el("b", "", s.precise ? "Assez précis" : `Précision · assez précis sous ± ${cm(s.threshold)}`));
  for (const g of s.groups) box.append(el("div", "", `${g.name} : ${[g.unknown ? `${g.unknown} inconnue${g.unknown > 1 ? "s" : ""}` : "", g.worst ? `± ${cm(g.worst)} au pire (${lower(g.label)})` : "", `${g.ok} sur ${g.n} sous ± ${cm(s.threshold)}`].filter(Boolean).join(" · ")}`));
  if (photo?.sigma) box.append(el("div", "", `Échelle : ± ${number(photo.sigma / photo.value * 100)} %${s.supposed ? " (supposée)" : ""}`));
  if (drawing && !s.supposed && Math.abs(drawing.value - 1) > 0.02) box.append(el("div", "", `Le dessin était ${number(Math.abs(drawing.value - 1) * 100, 0)} % trop ${drawing.value > 1 ? "petit" : "grand"} : corrigé.`));
  return box;
}

function conflicts() {
  const box = el("div", "alert"), n = state!.conflicts.length;
  box.append(el("b", "", n > 1 ? `${n} contradictions` : "Contradiction"), ...state!.conflicts.map((c) => el("span", "", c.text)));
  return box;
}

function mine() {
  const s = state!;
  if (!s.measures.length) return [];
  const list = s.measures.map((m) => {
    const box = el("div", `tagged ${m.key === focus ? "selected" : ""}`), covers = s.candidates.find((c) => c.key === m.key)?.covers ?? [];
    const text = el("span", "grow");
    text.append(el("b", "", m.label), el("br"), el("small", m.conflict ? "warn" : "", m.resid == null ? "à intégrer" : `écart avec le reste du plan : ${cm(Math.abs(m.resid))}${m.conflict ? " · contradiction" : ""}`));
    if (covers.length) text.append(el("br"), el("span", "chip ok", `vaut aussi pour ${covers.join(", ")}`));
    box.append(text, el("span", "m", `${number(m.value)} ${m.unit}`), button("Retirer", "link", `Retirer la mesure ${m.label}`, () => run(() => remove(m))));
    box.onmouseenter = () => { focus = m.key ?? ""; drawPlan(); };
    return box;
  });
  return [el("span", "kicker", `Mes mesures · ${s.measures.length}`), ...list];
}

function custom() {
  if (!adding) return [button("Ajouter ma mesure", "", "Choisir un ou deux éléments sur le plan ou dans la vue 3D", () => { [adding, picks, extra] = [true, [], null]; h.render(); })];
  const s = state!, names = picks.map((id) => s.elements.find((e) => e.id === id)!);
  const box = el("div", "alert");
  box.append(el("b", "", "Ma mesure"), el("span", "", names.length ? names.map((e, i) => `${i + 1} · ${e.label}`).join("   ") : "Clique un mur, une porte, une fenêtre ou l'escalier sur le plan, ou dans la vue 3D. Un 2e élément : la distance entre les deux."));
  const single = names.length === 1 ? s.candidates.filter((c) => c.on[0] === picks[0] && !c.key.startsWith("entre.")) : [];
  return [box, ...extra ? [entry(extra, 0)] : single.map((c) => entry(c, 0)), button("Annuler", "link", "Fermer Ma mesure", () => { [adding, picks, extra] = [false, [], null]; h.render(); })];
}

export function scaleStep(helpers: Helpers): Node[] {
  h = helpers;
  if (!state) return [h.intro("Des mesures au mètre", "Calcul du plan en cours."), h.footer(h.back())];
  const list = proposals(), s = state, gone = refused().size;
  if (!focus && list.length && !adding) focus = list[0].key;
  drawPlan();
  const toggle = button(flat ? "Voir en 3D" : "Voir le plan 2D", "link start", "Changer d'affichage", () => { flat = !flat; if (!flat) setView("3d"); h.render(); });
  return [h.intro("Des mesures au mètre", "Les photos donnent la forme, pas la taille. Chaque mesure fixe l'échelle et corrige le plan là où elle est prise. Mesure ce que tu peux, dans l'ordre que tu veux."),
    toggle, ...effect(), ...s.precise ? [precision()] : [],
    el("span", "kicker", "Mesures proposées · la plus utile d'abord"), ...list.map((c, i) => entry(c, i + 1)), ...list.length ? [] : [el("p", "", "Plus rien à proposer.")],
    ...custom(), ...s.precise ? [] : [precision()], ...s.conflicts.length ? [conflicts()] : [], ...mine(), ...gone ? [button(`${gone} mesure${gone > 1 ? "s" : ""} écartée${gone > 1 ? "s" : ""} · les reproposer`, "link start", "Remettre les mesures écartées", () => run(async () => { await h.save({ refused: [] }); h.render(); }))] : [],
    h.footer(h.back(), ...s.measures.length ? [] : [button("Je n'ai pas de mètre", "", "Continuer avec une échelle supposée", () => run(h.skip))],
      ...s.measures.length ? [s.precise ? h.primary("Étape suivante", h.next) : button("Passer à Vérifier", "", "Étape suivante", () => run(h.next))] : [])];
}

function svg(tag: string, attributes: Record<string, string | number>, className = "", text = "") {
  const item = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attributes)) item.setAttribute(k, String(v));
  if (className) item.setAttribute("class", className);
  if (text) item.textContent = text;
  return item;
}

function turn(parts: Part[]) {
  const drawn = h.guide().plan?.walls ?? [];
  const angles = drawn.flatMap((w) => parts.filter((p) => p.type === "mur" && p.label === `Mur ${w.name}`).map((p) => Math.atan2(w.b[1] - w.a[1], w.b[0] - w.a[0]) - Math.atan2(p.xy[1][1] - p.xy[0][1], p.xy[1][0] - p.xy[0][0])));
  const t = angles.length ? Math.atan2(angles.reduce((s, a) => s + Math.sin(a), 0), angles.reduce((s, a) => s + Math.cos(a), 0)) : 0;
  return ([x, z]: number[]) => [x * Math.cos(t) - z * Math.sin(t), x * Math.sin(t) + z * Math.cos(t)];
}

function cote(group: Element, key: string | null, ends: number[][], text: string, className: string, font: number, at: (p: number[]) => number[], centre: number[]) {
  let [p, q] = ends.map(at), [dx, dy] = [q[0] - p[0], q[1] - p[1]], length = Math.hypot(dx, dy), [nx, ny] = length ? [-dy / length, dx / length] : [0, -1];
  if (length && (nx * ((p[0] + q[0]) / 2 - centre[0]) + ny * ((p[1] + q[1]) / 2 - centre[1])) < 0) [nx, ny] = [-nx, -ny];
  const along = /^(longueur|largeur|position)\./.test(key ?? "") ? font * 1.6 : 0, stack = group.querySelectorAll(`[data-at="${p.map((v) => v.toFixed(2))}"]`).length;
  [p, q] = [p, q].map(([x, y]) => [x + nx * along, y + ny * along]);
  if (length < 1e-3) group.append(svg("circle", { cx: p[0], cy: p[1], r: font * 0.5, "data-at": `${p.map((v) => v.toFixed(2))}` }, `cote ${className}`));
  else group.append(svg("path", { d: `M${p[0]} ${p[1]}L${q[0]} ${q[1]}` + [p, q].map(([x, y]) => `M${x - nx * font * 0.4} ${y - ny * font * 0.4}L${x + nx * font * 0.4} ${y + ny * font * 0.4}`).join("") }, `cote ${className}`));
  group.append(svg("text", { x: (p[0] + q[0]) / 2 + nx * font, y: (p[1] + q[1]) / 2 + ny * font * (1 + 1.2 * stack), "font-size": font }, `cote-label ${className}`, text));
}

function drawPlan() {
  if (!h.active() || !state) return;
  overlay.classList.remove("half");
  overlay.hidden = !flat;
  highlight(picks, 0xf5c451);
  if (!flat) return;
  const s = state, at = turn(s.elements), points = s.elements.flatMap((e) => e.xy.map(at));
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]), size = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1), m = size * 0.12, font = size * 0.03;
  const plan = svg("svg", { viewBox: `${Math.min(...xs) - m} ${Math.min(...ys) - m} ${Math.max(...xs) - Math.min(...xs) + 2 * m} ${Math.max(...ys) - Math.min(...ys) + 2 * m}`, role: "img",
    "aria-label": "Plan de dessus : murs nommés, mesures saisies et proposées", preserveAspectRatio: "xMidYMid meet" }, "plan scale");
  const centre = [xs.reduce((a, b) => a + b, 0) / xs.length, ys.reduce((a, b) => a + b, 0) / ys.length];
  for (const e of [...s.elements].sort((a, b) => Number(a.type !== "mur") - Number(b.type !== "mur"))) {
    const xy = e.xy.map(at), chosen = picks.includes(e.id) ? "selected" : "";
    const shape = e.type === "escalier" ? svg("polygon", { points: xy.map((p) => p.join(",")).join(" ") }, `stairs ${chosen}`)
      : svg("line", { x1: xy[0][0], y1: xy[0][1], x2: xy[1][0], y2: xy[1][1] }, `${e.type === "mur" ? "wall" : `opening ${e.type}`} ${chosen}`);
    shape.append(svg("title", {}, "", e.label));
    shape.setAttribute("data-id", e.id);
    shape.addEventListener("click", () => run(() => { if (!adding) [adding, picks] = [true, []]; return pick(e.id); }));
    plan.append(shape);
    if (e.type !== "mur") continue;
    const mid = [(xy[0][0] + xy[1][0]) / 2, (xy[0][1] + xy[1][1]) / 2], away = Math.hypot(mid[0] - centre[0], mid[1] - centre[1]) || 1;
    plan.append(svg("text", { x: mid[0] + (mid[0] - centre[0]) / away * font * 1.2, y: mid[1] + (mid[1] - centre[1]) / away * font * 1.2, "font-size": font }, "label", e.label.replace(/^Mur /, "")));
  }
  const group = svg("g", {});
  proposals().forEach((c, i) => cote(group, c.key, c.ends, `${i + 1}`, `todo ${c.key === focus ? "focus" : ""}`, font, at, centre));
  for (const mm of s.measures) if (mm.ends) cote(group, mm.key, mm.ends, `${number(mm.value)}`, `${mm.conflict ? "bad" : "done"} ${mm.key === focus ? "focus" : ""}`, font, at, centre);
  if (extra) cote(group, extra.key, extra.ends, `≈ ${number(extra.value * 100, 0)}`, "todo focus", font, at, centre);
  plan.append(group);
  const section = el("section");
  section.append(el("span", "caption", adding ? "Ma mesure · clique 1 ou 2 éléments" : "Plan · cotes saisies en bleu, proposées en jaune (numéro = ordre d'utilité), contradictions en orange"), plan);
  overlay.replaceChildren(section);
}

document.addEventListener("plans3d:pick", (event) => { if (h?.active() && adding && !flat) run(() => pick((event as CustomEvent<string | null>).detail)); });
