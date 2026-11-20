import { call } from "./host";
import { button, el, number, run } from "./ui";
import { denseInfo, photoInfo, photoTask, stopDense } from "./photos";
import { classes, cloudMode } from "./seances";
import { cloudPick } from "./view3d";
import { update } from "./projects";

type Thing = { kind: string; label: string; center: number[]; size: number[]; n: number };
type Edge = { seen: boolean; openings?: { type: string }[] };
type Result = { scene: { place: string }; room: { edges: Edge[] } | null; lights: Thing[]; furniture: Thing[]; closed: boolean; missing: number[]; device?: string; total_s?: number };
type Status = { installed: boolean; download_mb: number; photos: number; segmented: number; dense: boolean; result: Result | null };

const PLACES: Record<string, string> = { dedans: "Dedans", dehors: "Dehors", "dedans et dehors": "Dedans et dehors" };
let status: Status | null = null, last: Result | null = null;
let popup: HTMLElement | null = null;
type Candidate = { id: string; label: string; size: number[]; factor: number };
type Piece = { key: string; kind: string; label: string; size: number[]; n: number; confirmed: string | null; candidates: Candidate[] };
let found: { catalog: boolean; pieces: Piece[] } | null = null, notes: Record<string, string> = {};
const metres = (size: number[]) => size.map((v) => number(v, 2)).join(" × ") + " m";

export const understood = () => status?.result ?? null;

export async function loadUnderstanding() {
  const source = photoInfo().source;
  status = source ? await call<Status>("semantique.status", { source }).catch(() => null) : null;
}

function tags(result: Result) {
  const box = el("div", "ia-tags"), kinds = [...new Set(result.furniture.filter((f) => f.n >= 40).map((f) => f.label.toLowerCase()))];
  const doors = result.room?.edges.flatMap((e) => e.openings ?? []) ?? [];
  const count = (type: string, one: string, many: string) => { const n = doors.filter((o) => o.type === type).length; return n ? [`${n} ${n > 1 ? many : one}`] : []; };
  box.append(...[PLACES[result.scene.place] ?? result.scene.place, ...kinds, ...count("porte", "porte", "portes"), ...count("fenetre", "fenêtre", "fenêtres"),
    ...count("porte_fenetre", "porte-fenêtre", "portes-fenêtres"), ...result.lights.length ? [`${result.lights.length} luminaire${result.lights.length > 1 ? "s" : ""}`] : []].map((t) => el("span", "ia-tag", t)));
  return box;
}

export function understandCard(after: () => Promise<void>) {
  if (!status || photoInfo().busy) return [];
  const analyse = () => run(async () => {
    last = await photoTask("semantique.analyse") as Result | null;
    await loadUnderstanding();
    await after();
  });
  const box = el("div", "card");
  const result = status.result;
  if (!result || status.segmented < status.photos) {
    box.append(el("b", "", "Comprendre la pièce"), el("span", "", "L'IA reconnaît sur chaque photo murs, sol, plafond, portes, fenêtres, meubles et lampes, sur ce PC. Les murs sont ensuite cherchés seulement sur les points « mur »."
      + (status.installed ? "" : ` Premier usage : modèle de ≈ ${status.download_mb} Mo téléchargé une fois.`)),
      button("Analyser les photos", "primary", "Reconnaissance locale des objets sur chaque photo", analyse));
    return [box];
  }
  box.append(el("b", "", result.room ? (result.closed ? "Pièce fermée, comprise par l'IA" : "Pièce fermée par l'IA, un mur non vu") : result.scene.place === "dehors" ? "Dehors : pas de pièce à fermer" : "Pièce comprise par l'IA"),
    tags(result), el("span", "", `Les meubles ne deviennent plus des murs. Couleur du nuage « Objets » : un clic sur un point corrige son groupe.${last?.total_s ? ` Analyse : ${number(last.total_s, 0)} s (${last.device}).` : ""}`));
  const actions = el("div", "row");
  if (result.lights.length) actions.append(button(`Ajouter ${result.lights.length > 1 ? `les ${result.lights.length} luminaires` : "le luminaire"}`, "", "Luminaires vus sur les photos, placés dans l'éclairage", () => run(async () => {
    const done = await call<{ added: number }>("semantique.lumieres", { source: photoInfo().source });
    document.dispatchEvent(new CustomEvent("plans3d:furniture-refresh"));
    actions.replaceChildren(el("small", "", `${done.added} luminaire${done.added > 1 ? "s ajoutés" : " ajouté"} : réglages à l'étape Lumière.`));
  })));
  if (actions.childElementCount) box.append(actions);
  return [box];
}

const AUTO = "plans3d.denseAuto";
const PHASES: Record<string, string> = { redressement: "Préparation des photos", profondeurs: "Profondeur de chaque pixel", fusion: "Fusion des points", plans: "Recherche des murs", "étiquettes": "Étiquettes et murs" };
export const denseAuto = () => localStorage.getItem(AUTO) !== "0";
let live: { bar: HTMLProgressElement; text: HTMLElement } | null = null;

function tick() {
  const info = denseInfo();
  if (!info || !live) return;
  const seconds = (Date.now() - info.started) / 1000, known = info.fraction > 0.02, left = known ? Math.max(1, Math.round(seconds * (1 - info.fraction) / info.fraction / 60)) : 0;
  if (known) live.bar.value = info.fraction;
  const phase = Object.entries(PHASES).find(([key]) => info.message.includes(key))?.[1] ?? "Démarrage";
  live.text.textContent = `${phase} · ${known ? `${Math.round(info.fraction * 100)} % · encore ≈ ${left} min` : `${Math.floor(seconds / 60)} min ${String(Math.floor(seconds % 60)).padStart(2, "0")} s`}`;
}

export function denseCard(start: () => void, minutes: number) {
  const info = denseInfo(), box = el("div", "card");
  if (!info && status?.dense) return [];
  const option = Object.assign(el("input"), { type: "checkbox", checked: denseAuto() }), line = el("label", "check");
  option.onchange = () => localStorage.setItem(AUTO, option.checked ? "1" : "0");
  line.append(option, "Après chaque scan, le lancer tout seul");
  if (!info) {
    live = null;
    box.append(el("b", "", "Nuage dense"), el("span", "", `Plus de points sur les murs (≈ ${minutes} min). Il calcule en fond : tu peux continuer, les murs seront reproposés à la fin.`), line,
      row(button("Calculer le nuage dense", "", "Profondeur de chaque pixel sur une partie des photos, en fond", start)));
    return [box];
  }
  live = { bar: Object.assign(el("progress"), { max: 1, ariaLabel: "Avancement du nuage dense" }), text: el("small") };
  box.setAttribute("role", "status");
  box.append(el("b", "", "Nuage dense en cours, en fond"), el("span", "", "Plus de points sur les murs. Tu peux continuer : les murs seront reproposés à la fin, sans toucher à ceux que tu as validés."),
    live.text, live.bar, line, row(button("Arrêter le nuage dense", "", "Arrêter le calcul du nuage dense", () => run(stopDense))));
  tick();
  return [box];
}

document.addEventListener("plans3d:dense", tick);
window.setInterval(tick, 1000);

export function piecesCard(after: () => void) {
  if (!status?.result || photoInfo().busy) return [];
  const box = el("div", "card"), search = (label: string) => button(label, found ? "" : "primary", "Comparer chaque meuble reconnu aux articles du catalogue IKEA importé", () => run(async () => {
    found = await photoTask("meubles.reconnus") as typeof found;
    after();
  }));
  box.append(el("b", "", "Meubles reconnus → catalogue IKEA"));
  if (!found) {
    box.append(el("span", "", "Chaque meuble reconnu est comparé au catalogue IKEA importé (dimensions et ressemblance). Un meuble confirmé se place dans le plan et donne l'échelle."), search("Retrouver les meubles"));
    return [box];
  }
  if (!found.catalog) {
    box.append(el("span", "", "Le catalogue IKEA n'est pas encore importé sur ce PC."), button("Importer le catalogue IKEA", "primary", "Téléchargé une fois, gardé sur ce PC", () => run(async () => {
      await call("catalog.import_ikea");
      found = await photoTask("meubles.reconnus") as typeof found;
      after();
    })));
    return [box];
  }
  const list = el("div", "pieces");
  for (const piece of found.pieces.filter((p) => Math.max(...p.size) >= 0.3)) {
    const line = el("div", "piece"), select = el("select");
    select.append(...piece.candidates.map((c) => new Option(`${c.label} · ${metres(c.size)}${Math.abs(c.factor - 1) > 0.03 ? ` · échelle ×${number(c.factor, 2)}` : ""}`, c.id)));
    if (piece.confirmed) select.value = piece.confirmed;
    select.ariaLabel = `Article pour : ${piece.label}`;
    const confirm = button(piece.confirmed ? "Confirmé" : "Confirmer", piece.confirmed ? "" : "primary", "Poser cet article dans le plan ; ses dimensions donnent l'échelle", () => run(async () => {
      await call("meubles.confirmer", { source: photoInfo().source, piece: piece.key, item: select.value });
      piece.confirmed = select.value;
      notes[piece.key] = "Posé dans le plan ; ses dimensions donnent l'échelle (± 3 %).";
      await update();
      after();
    }));
    line.append(el("b", "", `${piece.label} · ${metres(piece.size)} mesuré`), el("small", "", `Ressemble à : ${piece.candidates[0]?.label ?? "?"}`), select, row(confirm), ...notes[piece.key] ? [el("small", "", notes[piece.key])] : []);
    list.append(line);
  }
  box.append(list.childElementCount ? list : el("span", "", "Aucun meuble assez grand reconnu."), search("Relancer la comparaison"));
  return [box];
}

function row(...nodes: Node[]) { const box = el("div", "row"); box.append(...nodes); return box; }

function closePopup() { popup?.remove(); popup = null; }

async function correct(point: number[], event: PointerEvent) {
  closePopup();
  if (cloudMode() !== "semantique" || !status?.result) return;
  const all = await classes(), box = el("div", "popover");
  const select = el("select");
  select.append(...all.filter((k) => k.key !== "inconnu").map((k) => new Option(k.label, k.key)));
  box.style.left = `${event.clientX + 8}px`;
  box.style.top = `${event.clientY + 8}px`;
  const label = el("label", "", "Ce groupe de points est en fait");
  label.append(select);
  box.append(label, el("div", "row", ""));
  box.lastElementChild!.append(button("Annuler", "", "Fermer", closePopup), button("Corriger", "primary", "Changer la classe du groupe et recalculer les murs", () => run(async () => {
    const done = await call<{ from: string; points: number }>("semantique.corriger", { source: photoInfo().source, point_xyz_m: point, to: select.value });
    closePopup();
    await loadUnderstanding();
    document.dispatchEvent(new CustomEvent("plans3d:cloud"));
    document.dispatchEvent(new CustomEvent("plans3d:understood", { detail: done }));
  })));
  document.body.append(box);
  popup = box;
  select.focus();
}

cloudPick.on = (point, event) => void run(() => correct(point, event));
cloudPick.when = () => cloudMode() === "semantique" && Boolean(status?.result);
window.addEventListener("keydown", (event) => { if (event.key === "Escape") closePopup(); });
