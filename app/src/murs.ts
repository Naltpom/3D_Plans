import { call, host } from "./host";
import { $, button, el, number, run } from "./ui";
import { drawPlan, metres, type Corner, type DrawnWall, type Fit, type Opening, type Plan } from "./plan2d";
import { closeGuided, guidedImage, openGuided, showGuided } from "./photos";
import { update } from "./projects";
import { setView, showWalls } from "./view3d";

type Applied = { walls: string[]; openings: string[]; stairs: string[]; clicks: number; pairs: string[][] | null };
export type Helpers = {
  guide: () => { plan?: Plan | null; sub?: number }; save: (change: Record<string, unknown>, undo?: boolean) => Promise<void>; act: (text: string, task: () => Promise<unknown>) => Promise<void>;
  intro: (title: string, text: string) => HTMLElement; card: (title: string, text: string, ...actions: HTMLButtonElement[]) => HTMLElement; row: (...nodes: Node[]) => HTMLElement;
  footer: (...actions: HTMLButtonElement[]) => HTMLElement; primary: (text: string, action: () => unknown) => HTMLButtonElement; back: () => HTMLButtonElement; render: () => void; paint: () => void;
  sketch: () => { path: string } | null; source: () => string | null; walls: () => { a: number[]; b: number[]; height: number; label: string }[]; finish: () => Promise<void>; classic: () => Promise<void>; next: () => Promise<void>;
};

const SUBS = ["Lire le plan", "Caler sur les photos", "Points à préciser", "Portes et fenêtres", "Valider"];
const ROLES: [string, string][] = [["mur", "Murs"], ["porte", "Portes"], ["fenetre", "Fenêtres"], ["porte_fenetre", "Portes-fenêtres"], ["escalier", "Escalier"], ["ignorer", "Ignorer"]];
const KINDS: Record<string, [string, string]> = { porte: ["Porte", "portes"], fenetre: ["Fenêtre", "fenêtres"], porte_fenetre: ["Porte-fenêtre", "portes-fenêtres"] };
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const overlay = $("#plan2d");

let h: Helpers;
let selected = "";
let chosen = -1;
let renaming = false;
let questions: { corner: number; photos: string[] }[] | null = null;
let current = 0;
let found: Opening[] = [];
let applied: Applied | null = null;
let preparing = false;

const plan = () => h.guide().plan!;
const sub = () => h.guide().sub ?? 1;
const plural = (n: number, word: string, many = `${word}s`) => `${n} ${n > 1 ? many : word}`;
const scale = (p: Plan) => p.fit?.k ?? 6 / Math.max(1, ...[0, 1].map((i) => Math.max(...p.walls.map((w) => w.a[i])) - Math.min(...p.walls.map((w) => w.a[i]))));
const size = (p: Plan, w: DrawnWall) => Math.hypot(w.b[0] - w.a[0], w.b[1] - w.a[1]) * scale(p);
const at = (w: DrawnWall, t: number) => w.a.map((v, i) => v + t * (w.b[i] - v));
const project = (w: DrawnWall, p: number[]) => ((p[0] - w.a[0]) * (w.b[0] - w.a[0]) + (p[1] - w.a[1]) * (w.b[1] - w.a[1])) / Math.max(1e-9, (w.b[0] - w.a[0]) ** 2 + (w.b[1] - w.a[1]) ** 2);
const neighbours = (p: Plan, i: number) => [p.walls[(i - 1 + p.walls.length) % p.walls.length].name, p.walls[i].name];
const entry = (corner: number): Corner => plan().points?.find((c) => c.corner === corner) ?? { corner, clicks: [] };

async function savePlan(next: Plan, change: Record<string, unknown> = {}) {
  await h.save({ plan: next, ...change }, true);
  h.render();
  h.paint();
}

function section(caption: Node | string, content: Node) {
  const box = el("section");
  const text = el("span", "caption");
  text.append(caption);
  box.append(text, content);
  return box;
}

function show(half: boolean, ...parts: Node[]) {
  overlay.replaceChildren(...parts);
  overlay.classList.toggle("half", half);
  overlay.hidden = false;
}

export const hidePlan = () => { overlay.hidden = true; };

const image = () => Object.assign(el("img"), { src: host.photoUrl(h.sketch()!.path), alt: "Plan dessiné à main levée" });

function legendText(p: Plan) {
  const box = el("span", "", `Ton dessin · légende ${p.ai ? "lue par l'IA" : "détectée"} : `);
  p.legend.filter((item) => item.role !== "ignorer").forEach((item, i) => {
    box.append(...i ? [", "] : [], Object.assign(el("b", "", item.color), { style: `color:${item.hex === "#1a1a1a" ? "var(--text)" : item.hex}` }), ` ${ROLES.find(([id]) => id === item.role)?.[1].toLowerCase()}`);
  });
  return box;
}

function pills() {
  const box = el("div", "subs");
  box.append(...SUBS.map((title, i) => {
    const n = i + 1, done = n < sub() || n === 5 && Boolean(applied);
    const pill = button(`4.${n} ${title}${done ? " ✓" : ""}`, n === sub() ? "on" : done ? "ok" : "", `Sous-étape 4.${n}`, () => run(() => goSub(n)));
    pill.disabled = n > 1 && !h.guide().plan || n === 3 && !plan().fit;
    return pill;
  }));
  return box;
}

function tagged(tag: string, text: string, extra = "", on = false, action?: () => void) {
  const line = el(action ? "button" : "div", `tagged ${on ? "selected" : ""}`);
  line.append(el("span", "tag", tag), el("span", "", text), ...extra ? [el("span", "m", extra)] : []);
  if (action) Object.assign(line, { type: "button", onclick: action });
  return line;
}

function free(p: Plan) {
  const used = new Set([...p.walls.map((w) => w.name), ...p.extra.map((e) => e.name)]);
  for (let i = 0; ; i++) {
    const name = LETTERS[i % 26] + (i >= 26 ? String(Math.floor(i / 26) + 1) : "");
    if (!used.has(name)) return name;
  }
}

function describe(p: Plan, wall: DrawnWall) {
  const cx = p.walls.reduce((s, w) => s + w.a[0], 0) / p.walls.length, cy = p.walls.reduce((s, w) => s + w.a[1], 0) / p.walls.length;
  const [dx, dy] = [wall.b[0] - wall.a[0], wall.b[1] - wall.a[1]], middle = at(wall, 0.5), longest = Math.max(...p.walls.map((w) => size(p, w)));
  const side = Math.abs(dy) < 0.36 * Math.abs(dx) ? middle[1] < cy ? "Mur du haut" : "Mur du bas" : Math.abs(dx) < 0.36 * Math.abs(dy) ? middle[0] > cx ? "Mur de droite" : "Mur de gauche" : "Mur en biais";
  const kinds = Object.entries(KINDS).map(([id, [one, many]]) => [p.openings.filter((o) => o.wall === wall.name && o.type === id).length, one.toLowerCase(), many] as const).filter(([n]) => n);
  const span = Math.max(...[0, 1].map((i) => Math.max(...p.walls.map((w) => w.a[i])) - Math.min(...p.walls.map((w) => w.a[i]))));
  const near = p.stairs.some((s) => s.kept !== false && [[s.box[0], s.box[1]], [s.box[2], s.box[1]], [s.box[2], s.box[3]], [s.box[0], s.box[3]]]
    .some((c) => project(wall, c) > 0 && project(wall, c) < 1 && Math.abs((c[0] - wall.a[0]) * dy - (c[1] - wall.a[1]) * dx) / Math.hypot(dx, dy) < 0.04 * span));
  return [size(p, wall) < 0.12 * longest ? `${side} · petit retour` : side, ...kinds.map(([n, one, many]) => n > 1 ? `${n} ${many}` : one), ...near ? ["escalier contre"] : []].join(" · ");
}

function reproject(p: Plan, from: DrawnWall[], to: DrawnWall[]) {
  const openings = p.openings.flatMap((o) => {
    const old = from.find((w) => w.name === o.wall);
    if (!old) return [o];
    const [s, e] = [at(old, o.t0), at(old, o.t1)], middle = at(old, (o.t0 + o.t1) / 2);
    const target = [...to].sort((a, b) => Math.abs(project(a, middle) - 0.5) - Math.abs(project(b, middle) - 0.5))[0];
    if (!target || project(target, middle) < 0 || project(target, middle) > 1) return [];
    const [t0, t1] = [project(target, s), project(target, e)].sort((a, b) => a - b);
    return [{ ...o, wall: target.name, t0: Math.max(0, t0), t1: Math.min(1, t1) }];
  });
  return openings;
}

function intersect(a: DrawnWall, b: DrawnWall) {
  const [d1, d2] = [[a.b[0] - a.a[0], a.b[1] - a.a[1]], [b.b[0] - b.a[0], b.b[1] - b.a[1]]], det = d1[0] * -d2[1] + d2[0] * d1[1];
  if (Math.abs(det) < 1e-9 * Math.hypot(...d1) * Math.hypot(...d2)) return null;
  const t = ((b.a[0] - a.a[0]) * -d2[1] + d2[0] * (b.a[1] - a.a[1])) / det;
  return [a.a[0] + t * d1[0], a.a[1] + t * d1[1]];
}

async function edit(kind: string) {
  const p = plan(), i = p.walls.findIndex((w) => w.name === selected), n = p.walls.length;
  if (i < 0) return;
  const walls = p.walls.map((w) => ({ ...w })), wall = walls[i];
  if (kind === "couper") {
    const middle = at(wall, 0.5), added = { name: free(p), a: middle, b: wall.b };
    walls.splice(i, 1, { ...wall, b: middle }, added);
  } else {
    if (n <= 3) throw new Error("Il faut au moins 3 murs pour une pièce.");
    const j = (i + 1) % n, k = (i - 1 + n) % n;
    if (kind === "fusionner") walls[i] = { ...wall, b: walls[j].b };
    else {
      const corner = intersect(walls[k], walls[j]) ?? at(wall, 0.5);
      walls[k] = { ...walls[k], b: corner };
      walls[j] = { ...walls[j], a: corner };
    }
    walls.splice(kind === "fusionner" ? j : i, 1);
  }
  const start = walls.findIndex((w) => w.name === p.walls[0].name);
  const ring = start > 0 ? [...walls.slice(start), ...walls.slice(0, start)] : walls;
  selected = kind === "supprimer" ? "" : selected;
  await savePlan({ ...p, walls: ring, openings: reproject(p, p.walls, ring), fit: null, points: [] });
}

async function rename(name: string) {
  const p = plan(), next = name.trim().toUpperCase();
  if (!next || next === selected) { renaming = false; return h.render(); }
  if (p.walls.some((w) => w.name === next) || p.extra.some((e) => e.name === next)) throw new Error(`Le nom ${next} est déjà pris.`);
  const swap = (n?: string) => n === selected ? next : n;
  const fit = p.fit && { ...p.fit, pairs: Object.fromEntries(Object.entries(p.fit.pairs).map(([k, v]) => [swap(k)!, v])), gaps: Object.fromEntries(Object.entries(p.fit.gaps).map(([k, v]) => [swap(k)!, v])),
    photo: p.fit.photo.map((w) => ({ ...w, wall: w.wall && swap(w.wall)! })) };
  renaming = false;
  selected = next;
  await savePlan({ ...p, walls: p.walls.map((w) => ({ ...w, name: swap(w.name)! })), openings: p.openings.map((o) => ({ ...o, wall: swap(o.wall)! })), fit });
}

async function read(legend?: Plan["legend"]) {
  const result = await call<Plan>("plan.read_drawing", legend ? { legend } : {});
  [selected, chosen, questions, found, applied] = ["", -1, null, [], null];
  await h.save({ plan: { ...result, fit: null, points: [] }, sub: 1 }, true);
}

async function fit(turn: number | null = null) {
  const result = await call<Fit>("plan.fit", { plan: { ...plan(), fit: null }, turn });
  questions = null;
  await h.save({ plan: { ...plan(), fit: result, points: [] } }, true);
}

async function goSub(n: number) {
  if (n !== 3) closeGuided();
  await h.save({ sub: n });
  if (n === 3) {
    await prepare();
    const q = questions?.[current];
    h.render();
    if (q) await openCorner(q);
    return;
  }
  h.render();
  h.paint();
}

async function prepare() {
  if (preparing || !h.source()) return;
  preparing = true;
  try { await ask(); } finally { preparing = false; }
}

async function ask() {
  const source = h.source(), p = plan(), f = p.fit, n = p.walls.length, paired = new Set(Object.keys(f?.pairs ?? {})), list: typeof questions = [];
  for (let i = 0; f && i < n; i++) {
    const [before, after] = neighbours(p, i);
    if (paired.has(before) && paired.has(after)) continue;
    const [x, z] = metres(p, f.k, p.walls[i].a);
    const photos = await call<string[]>("photos.visible", { source, points_xyz_m: [[x, 0.02, z], [x, 1.2, z]], pose: f.pose, limit: 6 });
    if (photos.length >= 2) list.push({ corner: i, photos });
  }
  questions = list;
  current = Math.max(0, list.findIndex((q) => entry(q.corner).clicks.length < 2 && !entry(q.corner).skipped));
  if (list.every((q) => entry(q.corner).clicks.length >= 2 || entry(q.corner).skipped)) current = list.length;
}

async function openCorner(q: { corner: number; photos: string[] }) {
  const p = plan(), f = p.fit!, [x, z] = metres(p, f.k, p.walls[q.corner].a), [before, after] = neighbours(p, q.corner);
  await openGuided({ anchor: `coin ${before}-${after}`, text: `Clique le coin sol – mur ${before} – mur ${after}, là où le sol rencontre les deux murs.`,
    points: [[x, 0, z], [x, f.height ?? 2.4, z]], photos: q.photos, pose: f.pose, picks: () => entry(q.corner).clicks, pick: (c) => setPick(q.corner, c),
    plan: () => drawPlan(plan(), { corner: q.corner }) });
  h.render();
}

async function setPick(corner: number, click: { image: string; u: number; v: number }) {
  const p = plan(), old = entry(corner), clicks = old.clicks.filter((c) => c.image !== click.image);
  if (clicks.length >= 2) clicks.shift();
  await savePlan({ ...p, points: [...(p.points ?? []).filter((c) => c.corner !== corner), { corner, clicks: [...clicks, click] }] });
}

async function skip(mark: boolean) {
  const q = questions![current], p = plan();
  if (mark) await h.save({ plan: { ...p, points: [...(p.points ?? []).filter((c) => c.corner !== q.corner), { ...entry(q.corner), skipped: true }] } }, true);
  current++;
  const next = questions![current];
  h.render();
  if (next) await openCorner(next);
  else closeGuided();
}

async function hidden() {
  const q = questions![current], shown = guidedImage();
  if (!shown) return;
  q.photos = q.photos.filter((name) => name !== shown);
  const p = plan(), kept = entry(q.corner).clicks.filter((c) => c.image !== shown);
  await h.save({ plan: { ...p, points: [...(p.points ?? []).filter((c) => c.corner !== q.corner), { corner: q.corner, clicks: kept }] } }, true);
  if (q.photos.length + kept.length < 2) return skip(true);
  await openCorner(q);
}

function readFirst() {
  show(false, section("Ton dessin", image()));
  return [h.intro("Lire le plan dessiné", "L'appli relève les murs, les portes, les portes-fenêtres et l'escalier d'après les couleurs du dessin, redresse les traits et met à 90° les angles qui en sont proches."),
    h.footer(h.back(), button("Sans le dessin", "", "Murs des photos ou tracé à la main", () => run(h.classic)), h.primary("Lire le dessin", () => h.act("Lecture du dessin", () => read())))];
}

function readStep() {
  const p = plan(), choose = (name: string) => { selected = selected === name ? "" : name; renaming = false; h.render(); };
  show(false, section(legendText(p), image()), section("Ce que j'ai compris · traits redressés, angles à 90° quand c'est proche", drawPlan(p, { selected, onWall: choose })));
  const kinds = Object.entries(KINDS).map(([id, [one, many]]) => [p.openings.filter((o) => o.type === id).length, one.toLowerCase(), many] as const).filter(([n]) => n);
  const stairs = p.stairs.filter((s) => s.kept !== false).length;
  const text = [plural(p.walls.length, "mur"), ...kinds.map(([n, one, many]) => plural(n, one, many)), ...stairs ? [stairs > 1 ? `${stairs} escaliers` : "un escalier"] : []];
  const longest = [...p.walls].sort((a, b) => size(p, b) - size(p, a))[0]?.name;
  const legend = el("div", "walls");
  legend.append(...p.legend.map((item) => {
    const select = el("select");
    select.append(...ROLES.map(([id, label]) => new Option(label, id, false, id === item.role)));
    select.ariaLabel = `Rôle du ${item.color}`;
    select.onchange = () => run(() => h.act("Relecture du dessin", () => read(p.legend.map((l) => l.color === item.color ? { ...l, role: select.value } : l))));
    const line = el("div", "tagged");
    line.append(Object.assign(el("span", "swatch"), { style: `background:${item.hex}` }), el("span", "", item.color), select);
    return line;
  }));
  const list = el("div", "walls");
  list.append(...p.walls.map((w) => tagged(w.name, describe(p, w), w.name === longest ? "le plus long" : "", w.name === selected, () => choose(w.name))));
  const alerts = p.extra.map((extra, i) => {
    const box = el("div", "alert");
    const near = [...p.walls].sort((a, b) => Math.hypot(...at(a, 0.5).map((v, k) => v - extra.a[k])) - Math.hypot(...at(b, 0.5).map((v, k) => v - extra.a[k])))[0];
    box.append(el("span", "", `${extra.inside ? "Un trait à l'intérieur" : "Un trait qui dépasse"}, près du mur ${near.name} : ${extra.name ? `gardé comme mur ${extra.name}` : "je l'ai ignoré"}.`),
      button(extra.name ? "L'ignorer" : "C'est un mur, le garder", "", "Garder ou ignorer ce trait", () => run(() => savePlan({ ...p, extra: p.extra.map((e, k) => k === i ? { ...e, name: e.name ? undefined : free(p) } : e), fit: null, points: [] }))));
    return box;
  });
  const tools = h.row(...[["Fusionner", "fusionner", "Fusionner avec le mur suivant"], ["Couper", "couper", "Couper en deux"], ["Supprimer", "supprimer", "Supprimer ce mur (Suppr)"]]
    .map(([label, kind, title]) => Object.assign(button(label, "", title, () => run(() => edit(kind))), { disabled: !selected })),
  Object.assign(button("Renommer", "", "Changer le nom de ce mur", () => { renaming = true; h.render(); }), { disabled: !selected }));
  const naming = renaming ? [h.row(Object.assign(el("input"), { value: selected, ariaLabel: "Nouveau nom", maxLength: 4, onkeydown: (event: KeyboardEvent) => {
    if (event.key === "Enter") run(() => rename((event.target as HTMLInputElement).value));
  } }), button("OK", "", "Renommer", () => run(() => rename((document.querySelector<HTMLInputElement>("#step .row input")!).value))))] : [];
  const onward = h.source() ? h.primary("Oui, caler sur les photos", () => h.act("Calage sur les photos", async () => { await fit(); await goSub(2); }))
    : h.primary("Oui, continuer", () => goSub(4));
  return [h.intro("Est-ce bien ta pièce ?", `J'ai lu ${text.join(", ").replace(/, ([^,]*)$/, " et $1")}. Les longueurs ne comptent pas encore : seules la forme et l'ordre des murs servent.`),
    list, ...alerts, tools, ...naming, el("b", "", "Légende des couleurs"), legend,
    h.footer(h.back(), button("Relire le dessin", "", "Relire avec la légende actuelle", () => run(() => h.act("Lecture du dessin", () => read(p.legend)))), onward)];
}

function fitStep() {
  const p = plan(), f = p.fit, choose = (name: string) => { selected = selected === name ? "" : name; h.render(); };
  show(false, section("Superposition · bleu : ton dessin, vert : murs vus sur les photos, pointillés : surfaces photo non appariées", drawPlan(p, { photo: true, selected, onWall: choose })));
  if (!h.source()) return [h.intro("Pas de photos", "La forme du dessin est gardée telle quelle. Une mesure au mètre (étape 5) fixera la taille."), h.footer(button("Retour", "", "4.1", () => run(() => goSub(1))), h.primary("Portes et fenêtres", () => goSub(4)))];
  if (!f) return [h.intro("Caler sur les photos", "Le dessin donne l'ordre des murs, les photos leur position exacte."),
    h.footer(button("Retour", "", "4.1", () => run(() => goSub(1))), h.primary("Caler sur les photos", () => h.act("Calage sur les photos", () => fit())))];
  const names = Object.keys(f.pairs), worst = Math.max(0, ...Object.values(f.gaps));
  const ends = p.walls.map((w) => metres(p, f.k, w.a)), extent = [0, 1].map((i) => Math.max(...ends.map((e) => e[i])) - Math.min(...ends.map((e) => e[i])));
  const list = el("div", "walls");
  list.append(...p.walls.map((w) => tagged(w.name, f.pairs[w.name] ? "vu sur les photos" : "pas vu sur les photos : forme du dessin (± 30 cm)",
    f.pairs[w.name] ? `écart ${number(f.gaps[w.name], 0)} cm` : `${number(size(p, w), 2)} m`, w.name === selected, () => choose(w.name))));
  const turn = (q: number) => () => run(() => h.act("Calage sur les photos", () => fit(((f.turn ?? 0) + q) % 4)));
  return [h.intro(`Calé sur ${plural(names.length, "mur")} vu${names.length > 1 ? "s" : ""} en photo`,
    `Pièce d'environ ${number(extent[0], 1)} × ${number(extent[1], 1)} m. Écart le plus grand entre ton dessin et les photos : ${number(worst, 0)} cm. Les murs non vus gardent les proportions du dessin.`),
  list, h.card("Le sens ne colle pas ?", "Si la pièce est presque symétrique, le calage peut se tromper de sens.", button("Tourner de 90°", "", "Autre sens", turn(1)), button("Tourner de 180°", "", "Autre sens", turn(2))),
  h.footer(button("Retour", "", "4.1", () => run(() => goSub(1))), h.primary("Points à préciser", () => goSub(3)))];
}

function pointsStep() {
  const p = plan();
  if (!questions) {
    run(async () => { await prepare(); if (questions) h.render(); });
    return [h.intro("Points à préciser", "Recherche des coins à cliquer sur les photos…")];
  }
  const q = questions[current];
  show(false, section("Plan · coin à préciser", drawPlan(p, { corner: q?.corner })));
  const done = questions.filter((x) => entry(x.corner).clicks.length >= 2).length;
  if (!q) return [h.intro(questions.length ? `${plural(done, "point précisé", "points précisés")} sur ${questions.length}` : "Aucun point à préciser",
    questions.length ? "Rien n'est appliqué au plan avant la validation (4.5)." : "Tous les coins sont donnés par les murs vus sur les photos."),
  h.footer(button("Retour", "", "4.2", () => run(() => goSub(2))), ...questions.length ? [button("Recommencer", "", "Revoir les points", () => run(async () => { current = 0; h.render(); await openCorner(questions![0]); }))] : [],
    h.primary("Portes et fenêtres", () => goSub(4)))];
  const [before, after] = neighbours(p, q.corner), clicks = entry(q.corner).clicks, open = Boolean(guidedImage());
  const steps = [0, 1].map((i) => {
    const line = el("div", "tagged");
    line.append(el("span", "tag", clicks[i] ? "✓" : String(i + 1)), el("span", "", clicks[i] ? `Photo ${clicks[i].image} · point posé` : i ? "Même coin sur une 2e photo, prise d'un autre angle" : "Clique le coin sur cette photo"));
    return line;
  });
  const next = button("Photo suivante qui montre ce coin", "", "Autre photo", () => run(() => {
    const shown = guidedImage(), k = q.photos.indexOf(shown ?? "");
    return showGuided(q.photos[(k + 1) % q.photos.length]);
  }));
  const box = h.intro(`Clique le coin sol – mur ${before} – mur ${after}`, "Là où le sol rencontre les deux murs, sur cette photo. Un seul clic : un nouveau clic remplace le précédent.");
  box.querySelector(".kicker")!.textContent = `Point ${current + 1} sur ${questions.length}`;
  return [box, ...steps, h.row(...open ? [next, button("Ce coin n'est pas visible", "", "Retirer cette photo", () => run(hidden))] : [h.primary("Ouvrir les photos", () => openCorner(q))]),
    el("div", "note", "Le plan ne change qu'à la fin de l'étape 4.5. Tu peux annuler chaque clic (Ctrl+Z)."),
    h.footer(button("Passer ce point", "", "Ne pas préciser ce coin", () => run(() => skip(true))), Object.assign(h.primary("Point suivant", () => skip(false)), { disabled: clicks.length !== 2 }))];
}

function openingsStep() {
  const p = plan(), choose = (i: number) => { chosen = chosen === i ? -1 : i; h.render(); };
  show(false, section("Portes et fenêtres · clique pour choisir", drawPlan(p, { opening: chosen, onOpening: choose })));
  const change = (i: number, patch: Partial<Opening>) => run(() => savePlan({ ...p, openings: p.openings.map((o, k) => k === i ? { ...o, ...patch } : o) }));
  const list = el("div", "walls");
  list.append(...p.openings.map((o, i) => {
    const wall = p.walls.find((w) => w.name === o.wall)!, length = size(p, wall), line = el("div", `tagged ${i === chosen ? "selected" : ""}`);
    const type = el("select");
    type.append(...Object.entries(KINDS).map(([id, [label]]) => new Option(label, id, false, id === o.type)));
    type.ariaLabel = "Type";
    type.onchange = () => change(i, { type: type.value });
    const width = Object.assign(el("input"), { inputMode: "decimal", value: String(Math.round((o.t1 - o.t0) * length * 100)), ariaLabel: "Largeur en cm" });
    width.onchange = () => {
      const w = Number(width.value.replace(",", ".")) / 100 / length, middle = (o.t0 + o.t1) / 2;
      if (w > 0 && w < 1) change(i, { t0: Math.max(0, middle - w / 2), t1: Math.min(1, middle + w / 2) });
    };
    line.append(el("span", "tag", o.wall), type, width, el("span", "", "cm"), el("span", "m", o.seen === true ? "✓ vu sur les photos" : o.seen === false ? "pas vu sur les photos" : ""),
      button("Supprimer", "", "Supprimer cette ouverture (Suppr)", () => run(() => removeOpening(i))));
    line.onclick = (event) => { if (event.target === line) choose(i); };
    return line;
  }));
  if (!p.openings.length) list.append(el("p", "", "Aucune porte ni fenêtre sur le dessin."));
  const extra = found.map((o) => h.card(`${KINDS[o.type][0]} vue sur les photos, mur ${o.wall}`, `${number((o.t1 - o.t0) * size(p, p.walls.find((w) => w.name === o.wall)!), 2)} m de large, pas dessinée.`,
    button("L'ajouter", "", "Ajouter cette ouverture", () => run(async () => { found = found.filter((x) => x !== o); await savePlan({ ...p, openings: [...p.openings, { ...o, seen: true }] }); }))));
  const stairs = p.stairs.map((s, i) => tagged("≡", s.kept === false ? "Escalier ignoré" : "Escalier du dessin", "", false, () => run(() => savePlan({ ...p, stairs: p.stairs.map((x, k) => k === i ? { ...x, kept: x.kept === false } : x) }))));
  return [h.intro("Portes et fenêtres", "Elles viennent de ton dessin. Change le type ou la largeur, supprime ce qui est faux, puis vérifie sur les photos."), list, ...extra, ...stairs,
    ...p.fit ? [h.row(button("Vérifier sur les photos", "", "Trous dans les points des murs", () => run(() => h.act("Vérification sur les photos", check))))] : [],
    h.footer(button("Retour", "", "4.3", () => run(() => goSub(p.fit ? 3 : 1))), h.primary("Récapitulatif", () => goSub(5)))];
}

async function removeOpening(i: number) {
  const p = plan();
  chosen = -1;
  await savePlan({ ...p, openings: p.openings.filter((_, k) => k !== i) });
}

async function check() {
  const p = plan(), holes = await call<Opening[]>("plan.check_openings", { plan: p });
  const overlaps = (a: Opening, b: Opening) => a.wall === b.wall && Math.min(a.t1, b.t1) > Math.max(a.t0, b.t0);
  found = holes.filter((hole) => !p.openings.some((o) => overlaps(o, hole)));
  await h.save({ plan: { ...p, openings: p.openings.map((o) => ({ ...o, seen: holes.some((hole) => overlaps(o, hole)) })) } }, true);
}

function validStep() {
  const p = plan(), f = p.fit, names = p.walls.map((w) => w.name), points = (p.points ?? []).filter((c) => c.clicks.length === 2).length;
  show(true, section(applied ? "Plan validé" : "Plan à valider", drawPlan(p, { photo: Boolean(f) })));
  const kinds = Object.entries(KINDS).map(([id, [one, many]]) => [p.openings.filter((o) => o.type === id).length, one.toLowerCase(), many] as const).filter(([n]) => n);
  const kept = p.extra.filter((e) => e.name).length, stairs = p.stairs.filter((s) => s.kept !== false).length;
  const cards = [h.card(plural(names.length + kept, "mur"), `${names[0]} à ${names.at(-1)}${kept ? `, plus ${p.extra.filter((e) => e.name).map((e) => e.name).join(", ")}` : ""}, dans l'ordre du dessin.`),
    h.card("Calage", f ? `${plural(Object.keys(f.pairs).length, "mur")} sur les photos (${Object.keys(f.pairs).join(", ")}), écart ${number(Math.max(0, ...Object.values(f.gaps)), 0)} cm au plus.` : "Sans photos : forme du dessin."),
    h.card("Points précisés", points ? `${plural(points, "coin")} cliqué${points > 1 ? "s" : ""} sur 2 photos.` : "Aucun."),
    h.card("Portes et fenêtres", kinds.length ? kinds.map(([n, one, many]) => plural(n, one, many)).join(", ") + (stairs ? ", un escalier" : "") : stairs ? "Un escalier." : "Aucune."),
    ...applied ? [h.card("C'est fait", `${plural(applied.walls.length, "mur")}, ${plural(applied.openings.length, "ouverture")}${applied.stairs.length ? ", un escalier" : ""} créés. Le plan est dans la vue 3D.`)] : []];
  return [h.intro(applied ? "Plan validé" : "Récapitulatif", applied ? "Tu peux encore revenir sur une sous-étape et valider à nouveau : la validation remplace, jamais de double." : "Rien n'a encore changé dans le plan. La validation remplace les murs, portes et fenêtres de l'étape 4."),
    ...cards, el("div", "note", "Ctrl+Z annule la validation."),
    h.footer(button("Retour", "", "4.4", () => run(() => goSub(4))), ...applied ? [button("Valider à nouveau", "", "Remplacer le plan validé", () => run(() => h.act("Validation du plan", validate))), h.primary("Étape suivante", h.next)]
      : [h.primary("Valider le plan", () => h.act("Validation du plan", validate))])];
}

async function validate() {
  applied = await call<Applied>("plan.apply", { plan: plan() });
  await update();
  await h.finish();
  setView("3d");
}

export function fourStep(helpers: Helpers): Node[] {
  h = helpers;
  if (!h.guide().plan) return readFirst();
  return [pills(), ...[readStep, fitStep, pointsStep, openingsStep, validStep][Math.min(Math.max(sub(), 1), 5) - 1]()];
}

export function fourPaint() {
  const p = h?.guide().plan;
  if (!p || sub() !== 5) return showWalls([]);
  if (applied) return showWalls(h.walls().map((w) => ({ a: w.a, b: w.b, height: w.height, tone: "sure", on: false, text: w.label })));
  const k = scale(p);
  showWalls(p.walls.map((w) => ({ a: metres(p, k, w.a), b: metres(p, k, w.b), height: p.fit?.height ?? 2.5, tone: p.fit?.pairs[w.name] ? "sure" : "doubt", on: true, text: `Mur ${w.name}` })));
}

export function resetFour() {
  [selected, chosen, renaming, questions, current, found, applied] = ["", -1, false, null, 0, [], null];
  closeGuided();
}

document.addEventListener("plans3d:guided-closed", () => { if (h?.guide().plan && sub() === 3) h.render(); });
document.addEventListener("plans3d:delete", () => {
  if (!h?.guide().plan || overlay.hidden) return;
  if (sub() === 4 && chosen >= 0) run(() => removeOpening(chosen));
  else if (sub() === 1 && selected) run(() => edit("supprimer"));
});
