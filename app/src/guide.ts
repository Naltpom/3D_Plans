import { call, host, onEvent } from "./host";
import { $, button, cm, el, number, run } from "./ui";
import { addSources, openProject, refreshSources, remember, update, type UpdateResult } from "./projects";
import { idle, showHome } from "./home";
import { autoPositions, closeGuided, openGuided, photoInfo, photoTask, previewPose, stopTask } from "./photos";
import { fourPaint, fourStep, hidePlan, resetFour, type Helpers } from "./murs";
import type { Plan } from "./plan2d";
import { highlight, setView, showWalls } from "./view3d";
import { calibrate, controls, mode as drawing, opacity, place, placeHint, placeSketch, stop as stopDrawing, trace, type Sketch } from "./drawing";

type Guide = { type: string; step: number; done: number[]; scale?: number; manual?: boolean; plan?: Plan | null; sub?: number; classic?: boolean };
type Info = { name: string; path: string; guide?: Guide };
type Source = { id: string; path: string; kind: string; obs: number; photos?: number; registered?: number | null; state?: string; error?: string };
type End = "a" | "b";
type Wall = { id: string; label: string; axis: string; a: number[]; b: number[]; length: number; height: number; n: number; photos: number; sure: boolean; reason: string; component: string | null; angles: Record<End, string | null> };
type Proposal = { walls: Wall[]; measure: string | null; dense: boolean; densify_min: number };
type Candidate = { label: string; expr: string; sigma: number };
type Reading = { walls: { points: number[][]; closed: boolean; length: number }[]; openings: { type: string; point: number[] }[];
  cotes: { texte: string; value_cm: number; drawn_m: number | null; ok: boolean; message: string; wall: number | null; edge: number | null }[] } | null;
type Built = { id: string; label: string; source: string; uses: string[]; a: number[]; b: number[]; height: number };
type Cadastre = { house: { walls: number; perimeter: number; points: number[][] } | null; parcel: { area: number } | null; created: string[] } | null;

const TYPES = [
  ["piece", "Une pièce intérieure", "Murs, portes, fenêtres, escalier, à partir de photos et de quelques cotes.", "Pièce intérieure"],
  ["maison", "Une maison et son terrain", "Cadastre, façades, toit, relief.", "Maison et terrain"],
  ["vehicule", "Un véhicule", "Caisse, parois, aménagement.", "Véhicule"],
];
const TIPS: Record<string, string> = {
  piece: "Conseil : 50 à 200 photos, en tournant dans la pièce, avec du recouvrement entre deux photos. Photographie aussi les angles.",
  maison: "Conseil : fais le tour des façades en photo, et ajoute le plan cadastral (PDF ou GeoJSON) et le relief si tu les as.",
  vehicule: "Conseil : photographie l'intérieur paroi par paroi, portes ouvertes, avec du recouvrement entre deux photos.",
};
const FLOWS: Record<string, [string, string, string][]> = {
  piece: [["projet", "Le projet", "Ce que tu reconstruis"], ["sources", "Les sources", "Photos, cotes, plans"], ["positions", "Positions des photos", "Calcul automatique"],
    ["murs", "Les murs", "Valider ce qui est trouvé"], ["echelle", "L'échelle", "Une mesure au mètre"], ["verifier", "Vérifier", "Contradictions, mesure utile"], ["meubler", "Meubler", "Optionnel, une fois le plan prêt"]],
  maison: [["projet", "Le projet", "Ce que tu reconstruis"], ["cadastre", "Le cadastre", "Parcelle et murs extérieurs"], ["relief", "Le relief", "Optionnel : maison sur le terrain"],
    ["toit", "Le toit", "Proposé sur l'emprise"], ["facades", "Les façades", "Photos recalées"], ["verifier", "Vérifier", "Débordements, mesure utile"], ["meubler", "Meubler", "Optionnel"]],
  vehicule: [["projet", "Le projet", "Ce que tu reconstruis"], ["gabarit", "Le gabarit", "Fiche constructeur"], ["positions", "Les photos", "Positions des photos"],
    ["murs", "Le recalage", "Photos sur le gabarit"], ["verifier", "Cotes au mètre", "Vérifier, mesure utile"], ["meubler", "Aménager", "Galerie et agencement"]],
};
const BODY: [string, string, string][] = [["longueur", "Longueur utile", "330"], ["largeur", "Largeur utile", "170"], ["hauteur", "Hauteur utile", "180"],
  ["roue_longueur", "Passage de roue : longueur", "90"], ["roue_largeur", "Passage de roue : saillie", "20"], ["roue_hauteur", "Passage de roue : hauteur", "30"],
  ["roue_position", "Passage de roue : depuis l'arrière", "80"], ["porte_largeur", "Porte latérale : largeur", "100"], ["porte_hauteur", "Porte latérale : hauteur", "130"],
  ["porte_position", "Porte latérale : depuis l'arrière", "60"], ["arriere_largeur", "Portes arrière : largeur", "150"], ["arriere_hauteur", "Portes arrière : hauteur", "160"]];
const PHASES: Record<string, [string, string][]> = {
  "photos.extract": [["points clés", "Repérage des détails sur chaque photo"], ["appariement", "Mise en relation des photos"],
    ["poses", "Calcul de la position de chaque photo"], ["plans", "Recherche des murs, du sol et du plafond"]],
  "photos.densify": [["redressement", "Préparation des photos"], ["profondeurs", "Calcul de la profondeur de chaque pixel"],
    ["fusion", "Fusion en un nuage dense"], ["plans", "Recherche des murs"]],
  "tools.install": [["", "Téléchargement et installation de COLMAP"]],
};
const KINDS: Record<string, string> = { cotes_csv: "Cotes", photos_colmap: "Photos", cadastre_pdf: "Cadastre", cadastre_geojson: "Cadastre", grille_alti: "Relief" };
const MODE = "plans3d.mode";
const OPENINGS: [string, string][] = [["porte", "Porte"], ["fenetre", "Fenêtre"], ["porte_fenetre", "Porte-fenêtre"]];

const rail = $("#rail-steps");
const panel = $("#step");
const modeButton = $<HTMLButtonElement>("#mode-button");

let mode = localStorage.getItem(MODE) ?? "guide";
let info: Info | null = null;
let guide: Guide = { type: "piece", step: 1, done: [] };
let creating = false;
let draft = { type: "piece", name: "" };
let parent = "";
let sources: Source[] = [];
let proposal: Proposal | null = null;
let checked = new Set<string>();
let result: UpdateResult | null = null;
let next: Candidate | null = null;
let phase = 0;
let pending = "";
let fresh = "";
let hover = "";
let moves: { photo: string; wall: string; moved_cm: number | null }[] = [];
let holes: { type: string; label: string; width: number; height: number; sill: number; sigma: number; point: number[] }[] | null = null;
let picked = new Set<number>();
let parts: { id: string; type: string }[] = [];
let roofed: { measured: boolean } | null = null;
let reading: Reading = null;
let kept = new Set<number>();
let vision: { vision_ready?: boolean } | null = null;
let detail = "";
let everything = false;
let fitted: { pairs: string[][]; scale_known: boolean } | null = null;
const shown = () => everything ? walls() : walls().filter((w, i, all) => w.component || checked.has(w.id) || [...all].sort((a, b) => Number(b.sure) - Number(a.sure) || b.n - a.n).indexOf(w) < 6);
let built: Built[] = [];
let land: Cadastre = null;
let sketch: Sketch = null;

const flow = () => FLOWS[guide.type] ?? FLOWS.piece;
const key = (step = guide.step) => flow()[Math.min(Math.max(step, 1), flow().length) - 1][0];
const position = (name: string) => flow().findIndex(([id]) => id === name) + 1;
const plural = (n: number, word: string) => `${number(n, 0)} ${word}${n > 1 ? "s" : ""}`;
const lower = (label: string) => label[0].toLowerCase() + label.slice(1);
const decimal = (text: string) => Number(text.trim().replace(",", "."));
const target = () => `${parent.replace(/[\\/]+$/, "")}\\${draft.name.trim().replace(/[<>:"/\\|?*]+/g, "_")}`;
const photoSource = () => sources.find((s) => s.kind === "photos_colmap");
const walls = () => proposal?.walls ?? [];
const created = () => walls().filter((w) => w.component);
const side = (wall: Wall, end: End) => wall.axis === "z" ? end === "a" ? "bout de gauche" : "bout de droite" : end === "a" ? "bout du haut" : "bout du bas";
const drawn = () => guide.type === "piece" && Boolean(sketch) && !guide.classic;
const missing = () => created().flatMap((w) => (["a", "b"] as End[]).filter((e) => !w.angles[e]).map((e) => [w, e] as [Wall, End]));

function intro(title: string, text: string) {
  const box = el("div", "intro"), step = info && !creating ? guide.step : 1;
  box.append(el("span", "kicker", step === flow().length ? "Optionnel" : `Étape ${step} sur ${flow().length - 1}`), el("h1", "", title), ...text ? [el("p", "", text)] : []);
  return box;
}

function card(title: string, text: string, ...actions: HTMLButtonElement[]) {
  const box = el("div", "card");
  box.append(el("b", "", title), el("span", "", text));
  if (actions.length) box.append(row(...actions));
  return box;
}

function row(...nodes: Node[]) {
  const box = el("div", "row");
  box.append(...nodes);
  return box;
}

function footer(...actions: HTMLButtonElement[]) {
  const box = el("div", "actions");
  box.append(...actions);
  return box;
}

const primary = (text: string, action: () => unknown) => button(text, "primary", text, () => run(action));
const back = () => button("Retour", "", "Étape précédente", () => run(() => go(guide.step - 1)));
const stop = () => button("Arrêter le calcul", "", "Arrêter le calcul en cours", () => run(stopTask));
const onward = (enabled = true, main = true) => Object.assign(button("Étape suivante", main ? "primary" : "", "Passer à l'étape suivante", () => run(() => complete(guide.step))), { disabled: !enabled });

function summary(step: number) {
  const photos = photoSource(), count = created().length;
  const texts: Record<string, string> = {
    projet: TYPES.find(([id]) => id === guide.type)?.[3] ?? "",
    sources: photos ? plural(photos.photos ?? 0, "photo") : sources.length ? plural(sources.length, "source") : sketch ? "croquis" : guide.manual ? "tracé à la main" : "",
    positions: photos?.registered != null ? `${photos.registered} positionnées` : photos ? "" : "sans photos",
    murs: built.length ? `${plural(built.length, "mur")}${count ? ` dont ${count} des photos` : ""}` : "",
    echelle: guide.scale ? `${number(guide.scale)} cm mesurés` : count ? "échelle supposée" : "déjà en mètres",
    verifier: "plan vérifié", cadastre: built.length ? `${plural(built.length, "mur")} extérieurs` : "",
    relief: parts.some((c) => c.type === "terrain") ? "relief importé" : "terrain plat", toit: parts.some((c) => c.type === "toit") ? "toit proposé" : "",
    facades: photos?.registered != null ? `${photos.registered} photos` : "sans photos", gabarit: built.length ? plural(built.length, "paroi") : "",
  };
  return texts[key(step)] || flow()[step - 1][2];
}

function renderRail() {
  $("#rail-title").textContent = info && !creating ? info.name : "Nouveau projet";
  rail.replaceChildren(...flow().flatMap(([, title, hint], i) => {
    const step = i + 1, done = guide.done.includes(step) && Boolean(info) && !creating, current = info && !creating ? guide.step : 1;
    const item = button("", `st ${step === current ? "current" : ""} ${done ? "done" : ""}`, title, () => run(() => go(step)));
    const text = el("span");
    text.append(el("b", "", title), done ? summary(step) : hint);
    item.append(el("span", "n", done ? "✓" : String(step)), text);
    item.disabled = (!info || creating) && step > 1;
    if (step === current) item.ariaCurrent = "step";
    return step === flow().length ? [el("hr"), item] : [item];
  }));
}

function projectStep() {
  const options = el("div", "options");
  options.append(...TYPES.map(([id, title, text]) => {
    const option = button("", "opt", title, () => run(async () => {
      draft.type = id;
      if (info && !creating) await save({ type: id });
      render();
    }));
    option.append(el("b", "", title), text);
    option.ariaPressed = String((info && !creating ? guide.type : draft.type) === id);
    return option;
  }));
  if (info && !creating) return [intro("Ton projet", `${info.name} · ${info.path}`), options,
    footer(button("Changer de projet", "", "Accueil : tous les projets", () => showHome()), onward())];
  const name = Object.assign(el("input"), { value: draft.name, placeholder: "Salon", autocomplete: "off", ariaLabel: "Nom du projet" });
  const where = el("small", "", draft.name.trim() ? `Enregistré dans ${target()}` : "Donne un nom au projet.");
  const create = primary("Créer et continuer", createProject);
  create.disabled = !draft.name.trim();
  name.oninput = () => {
    draft.name = name.value;
    where.textContent = draft.name.trim() ? `Enregistré dans ${target()}` : "Donne un nom au projet.";
    create.disabled = !draft.name.trim();
  };
  name.onkeydown = (event) => { if (event.key === "Enter" && !create.disabled) create.click(); };
  const label = el("label", "", "Nom du projet");
  label.append(name);
  const folder = button("Changer de dossier", "link", "Choisir le dossier parent", () => run(async () => {
    parent = (await host.pickFolder("Dossier où créer le projet")) ?? parent;
    name.dispatchEvent(new Event("input"));
  }));
  const form = el("div", "card plain");
  form.append(label, row(where, folder));
  return [intro("Que veux-tu reconstruire ?", "Le choix règle les éléments proposés ensuite. Il se change plus tard."), options, form,
    footer(button("Annuler", "", "Revenir sans créer de projet", () => info ? run(() => go(guide.step)) : showHome()), create)];
}

async function createProject() {
  idle();
  fresh = draft.type;
  await openProject("project.create", { path: target(), name: draft.name.trim() });
}

function entry(title: string, text: string, status: string, action: HTMLButtonElement) {
  const box = card(title, text, action);
  box.classList.toggle("done", Boolean(status));
  if (status) box.firstElementChild!.append(" · ", el("span", "status", status));
  return box;
}

function sourcesStep() {
  const photos = photoSource(), cadastral = sources.some((s) => s.kind.startsWith("cadastre"));
  const grid = el("div", "grid");
  grid.append(
    entry("Photos", `Un dossier de photos prises en tournant. ${TIPS[guide.type] ?? TIPS.piece}`, photos ? plural(photos.photos ?? 0, "photo") : "",
      button("Choisir un dossier", "", "Ajouter toutes les photos d'un dossier", () => run(async () => {
        const path = await host.pickFolder("Dossier de photos");
        if (path) await addSources([path]);
      }))),
    entry("Plan cadastral", "PDF du cadastre ou fichier GeoJSON : bâtiment principal, parcelle.", cadastral ? "importé" : "",
      button("Choisir un fichier", "", "PDF ou GeoJSON du cadastre", () => run(async () => addSources((await host.pickFiles("Plan cadastral")) ?? [])))),
    entry("Croquis à main levée", "Photo ou scan d'un plan dessiné : il s'affiche en fond pour le décalquer.", sketch ? "ajouté" : "",
      button("Choisir une image", "", "Image JPEG ou PNG du croquis", () => run(async () => {
        const [path] = (await host.pickFiles("Croquis")) ?? [];
        if (path) await call("plan.sketch", { path });
        await loadPlan();
        render();
      }))),
    entry("Construction manuelle", "Rien à importer : tu traces les murs en vue de dessus, au clic ou au clavier.", guide.manual ? "choisie" : "",
      button(guide.manual ? "Annuler ce choix" : "Je trace moi-même", "", "Tracer les murs à l'étape 4", () => run(async () => { await save({ manual: !guide.manual }); render(); }))));
  const list = el("ul", "list");
  list.append(...sources.map((source) => {
    const item = el("li");
    item.append(el("strong", "", KINDS[source.kind] ?? source.kind),
      button("×", "remove", "Retirer cette source", () => run(async () => { await call("sources.remove", { id: source.id }); await refreshSources(); })),
      el("small", "", source.photos != null ? plural(source.photos, "photo") : source.path.split(/[\\/]/).pop()!));
    return item;
  }));
  if (!sources.length) list.append(el("li", "empty", "Aucun fichier pour l'instant : glisse un dossier de photos ou un plan ici."));
  const more = button("Ajouter des cotes ou un relief", "link fit", "CSV de cotes, grille d'altitudes", () => run(async () => addSources((await host.pickFiles("Cotes ou relief")) ?? [])));
  return [intro("D'où viennent les murs ?", "Choisis une ou plusieurs entrées : elles se combinent, un mur tracé se précise ensuite avec des photos ou des cotes."),
    grid, list, more, footer(back(), onward(sources.length > 0 || Boolean(sketch) || Boolean(guide.manual)))];
}

function progress(method: string, fraction: number, message: string, started: number) {
  const steps = PHASES[method] ?? [];
  const found = steps.findIndex(([key]) => key && message.includes(key));
  phase = found < 0 ? phase : found;
  const list = el("ol", "phases");
  list.append(...steps.map(([, text], i) => el("li", i < phase ? "past" : i === phase ? "now" : "", text)));
  const seconds = (Date.now() - started) / 1000, known = Number.isFinite(fraction) && fraction > 0.02;
  const left = known ? Math.max(1, Math.round(seconds * (1 - fraction) / fraction / 60)) : 0;
  const bar = Object.assign(el("progress"), { max: 1, ariaLabel: "Avancement" });
  if (known) bar.value = fraction;
  return [list, bar, el("small", "", `${known ? `${Math.round(fraction * 100)} % · ` : ""}${Math.floor(seconds / 60)} min ${String(Math.floor(seconds % 60)).padStart(2, "0")} s${known ? ` · encore ≈ ${left} min` : ""}`)];
}

function positionsStep() {
  const photo = photoInfo(), source = photoSource();
  if (photo.busy === "tools.install") return [intro("Installation des outils photo", "Tu peux laisser l'appli travailler : le calcul des positions suit."),
    ...progress(photo.busy, photo.fraction, photo.message, photo.started), footer(back())];
  if (photo.busy === "photos.extract") return [intro("Je place tes photos",
    "Chaque cône est une photo placée, les points sont les détails retrouvés sur plusieurs photos. Tu peux tourner la vue pendant le calcul."),
    ...progress(photo.busy, photo.fraction, photo.message, photo.started), footer(back(), stop())];
  if (!source) return [intro("Pas de photos", "Rien à calculer ici : les murs viendront du tracé, du croquis ou du cadastre."), footer(back(), onward())];
  if (!photo.installed) return [intro("Il faut les outils photo", "COLMAP calcule la position des photos. Il s'installe une seule fois (≈ 400 Mo)."),
    footer(back(), primary("Installer les outils photo", autoPositions))];
  if (photo.count) return [intro(`${photo.count} photos positionnées sur ${source.photos}`,
    "Dans la vue, chaque cône bleu est une photo et les points sont les détails retrouvés sur plusieurs photos. Les murs se cachent dans ces points : l'étape suivante les trouve."),
  card("Pour tourner la vue", "Glisse avec le bouton gauche, zoome avec la molette. « Dessus » montre le plan."), footer(back(), onward())];
  const minutes = Math.max(2, Math.round((source.photos ?? 0) * 2.5 / 60));
  phase = -1;
  return [intro("Calculer la position des photos", "L'appli retrouve d'où chaque photo a été prise et place en 3D les détails qu'elles ont en commun."),
    ...progress("photos.extract", NaN, "", Date.now()).slice(0, 1), el("p", "", `Durée estimée : ≈ ${minutes} min pour ${plural(source.photos ?? 0, "photo")}.`),
    ...source.state === "échec" ? [card("Le dernier calcul a échoué", source.error ?? "")] : [],
    footer(back(), primary("Calculer les positions", autoPositions))];
}

function photoWalls() {
  if (!proposal) return [el("p", "", "Recherche des murs sur les photos…")];
  const list = el("div", "walls");
  list.append(...shown().map((wall) => {
    const box = Object.assign(el("input"), { type: "checkbox", checked: checked.has(wall.id) });
    box.onchange = () => {
      if (box.checked) checked.add(wall.id);
      else checked.delete(wall.id);
      render();
      paint();
    };
    const line = el("label", `wall ${wall.sure ? "" : "doubt"}`);
    line.append(box, el("span", "", `${wall.label} · ${wall.reason || `vu sur ${plural(wall.photos, "photo")}`} · ${number(wall.length)} m`), el("span", "m", `${number(wall.n, 0)} pts`));
    line.onmouseenter = () => { hover = wall.id; paint(); };
    line.onmouseleave = () => { hover = ""; paint(); };
    return line;
  }));
  const hidden = walls().length - shown().length;
  if (hidden) list.append(button(`Voir les ${hidden} autre${hidden > 1 ? "s pans incertains" : " pan incertain"}`, "link fit", "Afficher toutes les surfaces trouvées", () => { everything = true; render(); paint(); }));
  const axes = new Set(walls().filter((w) => checked.has(w.id)).map((w) => w.axis)), done = created();
  const same = done.length === checked.size && done.every((w) => checked.has(w.id));
  const validate = button("Valider les murs", same && done.length ? "" : "primary", "Créer les murs cochés", () => run(() => act("Création des murs", validateWalls)));
  validate.disabled = axes.size < 2;
  const doubt = walls().find((w) => !w.sure && !w.component);
  const densify = button(`Densifier (≈ ${proposal.densify_min} min)`, "", "Optionnel : nuage dense, plus de murs trouvés", () => run(densifyCloud));
  const angles = missing();
  return [
    list,
    ...walls().length && axes.size < 2 ? [el("p", "warn", "Coche au moins un mur dans chaque direction.")] : [],
    row(validate),
    ...doubt ? [card(`Le ${lower(doubt.label)} est incertain`,
      `${doubt.reason[0].toUpperCase()}${doubt.reason.slice(1)}. Deux solutions : clique un de ses angles sur 2 photos, ou lance la densification (≈ ${proposal.densify_min} min, optionnelle).`,
      button("Cliquer sur les photos", "", "Valider ce mur et cliquer un de ses angles", () => run(() => act("Création des murs", () => clickWall(doubt)))), ...proposal.dense ? [] : [densify])]
      : proposal.dense ? [] : [card("D'autres murs ?", `La densification cherche plus de points (≈ ${proposal.densify_min} min, optionnelle).`, densify)],
    ...angles.length ? [card("Angles à préciser", `Ces angles n'ont pas été trouvés : clique-les sur 2 photos pour que les longueurs soient justes.`,
      ...angles.map(([wall, end]) => button(`${wall.label}, ${side(wall, end)}`, "", "Cliquer cet angle sur 2 photos", () => run(() => clickAngle(wall, end)))))] : [],
  ];
}

function wallsStep() {
  const photo = photoInfo(), tool = drawing();
  if (photo.busy === "photos.densify") return [intro("Densification en cours", "Elle cherche des points sur les surfaces peu texturées, puis de nouveaux murs."),
    ...progress(photo.busy, photo.fraction, photo.message, photo.started), footer(back(), stop())];
  if (tool === "trace") return [intro("Trace la pièce", "Clique chaque angle dans la vue de dessus : la grille de 5 cm, l'angle droit et les murs existants sont aimantés. Pour une longueur exacte, tape-la en cm puis Entrée. Clique le premier angle pour fermer."),
    ...controls(), ...sketch?.points ? [opacity(sketch)] : []];
  if (tool) return [intro(`Pose : ${OPENINGS.find(([id]) => id === tool)?.[1].toLowerCase()}`, "Clique sur un mur dans la vue : l'ouverture se centre sur le clic. Clique un autre mur pour en poser d'autres."),
    placeHint(), footer(button("Terminer", "primary", "Quitter la pose", stopDrawing))];
  const fromPhotos = photo.count > 0, count = walls().length;
  const house = land?.house, parcel = land?.parcel;
  const parts = [
    intro(fromPhotos && count ? `J'ai trouvé ${count} mur${count > 1 ? "s probables" : " probable"}` : "Les murs", fromPhotos
      ? "Ils viennent des surfaces planes vues sur tes photos. Décoche ceux qui sont faux. Tu peux aussi tracer des murs ou les reprendre du cadastre."
      : "Trace les murs, décalque ton croquis ou reprends-les du cadastre. Une cote au mètre les précisera ensuite."),
    ...guide.type === "piece" ? [sketch ? card("Plan dessiné", "Lis ton dessin pas à pas : murs nommés, calage sur les photos, points à préciser, portes et fenêtres.",
      button("Lire le dessin", "", "Revenir à la lecture du plan dessiné", () => run(async () => { await save({ classic: false }); render(); paint(); })))
      : card("Tu as un plan dessiné ?", "Ajoute son image, même non cotée : l'appli le lit et te guide.", button("Choisir l'image du plan", "", "Image JPEG ou PNG", () => run(async () => {
        const [path] = (await host.pickFiles("Plan dessiné")) ?? [];
        if (path) await call("plan.sketch", { path });
        await save({ classic: false });
        await loadPlan();
        render();
      })))] : [],
    ...fromPhotos && built.some((w) => w.source !== photoInfo().source) ? [fitCard()] : [],
    ...fromPhotos ? photoWalls() : [],
    ...house || parcel ? [card(land!.created.length ? "Cadastre : créé" : "Depuis le cadastre", [house && `Bâtiment principal : ${plural(house.walls, "mur")} extérieurs, ${number(house.perimeter)} m de périmètre`,
      parcel && `parcelle de ${number(parcel.area, 0)} m²`].filter(Boolean).join(" ; ") + ".",
    ...land!.created.length ? [] : [button("Créer les murs et la parcelle", "", "Murs extérieurs du bâtiment principal et parcelle", () => run(() => act("Création depuis le cadastre", fromCadastre)))])] : [],
    card("Tracer à la main", `Clique les angles dans la vue de dessus, longueurs au clavier.${sketch ? sketch.points ? " Le croquis calé s'affiche en fond : décalque-le." : " Cale d'abord le croquis : 2 points et leur distance." : ""}`,
      button("Tracer une pièce", built.length || fromPhotos || house ? "" : "primary", "Tracer des murs dans la vue de dessus", () => trace(() => built, render, refreshPlan)),
      ...sketch ? [button(sketch.points ? "Recaler le croquis" : "Caler le croquis", sketch.points ? "" : "primary", "2 points du croquis et leur distance", () => calibrate(sketch!, async () => { await loadPlan(); setView("dessus"); render(); }))] : []),
    ...sketch?.points ? [opacity(sketch)] : [],
    ...moves.length ? [card("Murs calés sur les photos", moves.map((m) => `${m.wall} → ${m.photo}${m.moved_cm == null ? "" : ` : ${number(m.moved_cm, 0)} cm`}`).join(", ") + ".")] : [],
    ...openingsCard(),
    ...sketchReading(),
    ...built.length ? [card("Portes et fenêtres", "Choisis le type, puis clique sur un mur dans la vue.",
      ...OPENINGS.map(([id, label]) => button(label, "", `Poser : ${label.toLowerCase()}`, () => place(id, () => built, render, refreshPlan))))] : [],
  ];
  return [...parts, footer(back(), onward(built.length > 0, built.length > 0 && (!fromPhotos || created().length > 0)))];
}

function fitCard() {
  const run2 = (turn?: number) => () => run(() => act("Recalage des photos", async () => {
    fitted = await call("photos.fit_walls", { source: photoInfo().source, ...turn ? { turn } : {} });
    await refreshPlan();
    await loadProposal();
  }));
  if (!fitted) return card("Recaler les photos sur tes murs", "Les murs vus sur les photos sont appariés à tes murs tracés : les photos prennent la position et l'échelle du tracé, sans cote de plus.",
    button("Recaler les photos", "primary", "Apparier automatiquement photos et murs tracés", run2()));
  return card(`${fitted.pairs.length} mur${fitted.pairs.length > 1 ? "s appariés" : " apparié"}`, `${fitted.pairs.map(([photo, wall]) => `${photo} → ${wall}`).join(", ")}. ${fitted.scale_known
    ? "Position et échelle des photos viennent du tracé." : "Position calée ; ajoute une cote pour l'échelle."} Pièce symétrique ? Vérifie le sens dans la vue.`,
  button("Tourner de 180°", "", "Autre sens possible", run2(2)), button("Tourner de 90°", "", "Autre sens possible", run2(1)));
}

function sketchReading() {
  if (!sketch?.points) return [];
  if (!reading) return [card("Lire le croquis avec l'IA", "L'IA locale relève les murs, les cotes écrites et les portes du croquis calé ; tu choisis ce que tu gardes.",
    button("Lire le croquis", "primary", "Lecture par le modèle vision local", () => run(() => act("Lecture du croquis par l'IA", readSketch))),
    ...vision && !vision.vision_ready ? [button("Télécharger le modèle vision (≈ 6 Go)", "", "qwen2.5vl:7b, une seule fois", () => run(() => act("Téléchargement du modèle vision", pullVision)))] : [])];
  const box = el("div", "card");
  box.append(el("b", "", "Lu sur le croquis"));
  for (const [i, wall] of reading.walls.entries()) {
    const check = Object.assign(el("input"), { type: "checkbox", checked: kept.has(i) });
    check.onchange = () => { if (check.checked) kept.add(i); else kept.delete(i); paint(); };
    const line = el("label", "wall");
    line.append(check, el("span", "", `Pièce ${i + 1} · ${plural(wall.points.length, "mur")} · ${number(wall.length)} m${wall.closed ? " · fermée" : ""}`));
    box.append(line);
  }
  for (const c of reading.cotes) box.append(el("span", c.ok ? "" : "warn", `${c.ok ? "✓" : "⚠"} « ${c.texte} »${c.ok ? ` : ${number(c.drawn_m ?? 0)} m dessinés, cote gardée` : ` : ${c.message}, ignorée`}`));
  if (reading.openings.length) box.append(el("span", "", `${plural(reading.openings.length, "ouverture")} : ${reading.openings.map((o) => OPENINGS.find(([id]) => id === o.type)?.[1].toLowerCase()).join(", ")}`));
  if (!reading.walls.length) box.append(el("span", "warn", "Aucun mur reconnu : trace-les à la main sur le croquis."));
  box.append(row(button("Créer les murs cochés", "primary", "Murs, cotes « lu » et ouvertures", () => run(() => act("Création des murs du croquis", createRead))),
    button("Oublier la lecture", "", "Revenir au tracé", () => { reading = null; render(); paint(); })));
  return [box];
}

async function readSketch() {
  reading = await call<Reading>("plan.read_sketch");
  kept = new Set(reading!.walls.map((_, i) => i));
}

async function pullVision() {
  await call("ai.pull", { vision: true });
  vision = await call("ai.status");
}

async function createRead() {
  for (const [i, wall] of reading!.walls.entries()) {
    if (!kept.has(i)) continue;
    const lengths = wall.points.map((_, e) => (reading!.cotes.find((c) => c.ok && c.wall === i && c.edge === e)?.value_cm ?? 0) / 100 || null);
    await call("plan.trace", { points: wall.points, closed: wall.closed, lengths, cote_status: "lu", cote_sigma: 2, prior: 0.3 });
  }
  await update();
  for (const opening of reading!.openings) await call("plan.place", { type: opening.type, point: opening.point }).catch(() => {});
  reading = null;
  await refreshPlan();
}

async function refreshPlan(rough: string[] = []) {
  if (rough.length && photoInfo().count) moves = (await call<{ pairs: typeof moves }>("photos.snap_walls", { source: photoInfo().source, walls: rough })).pairs;
  await update();
  await loadPlan();
  render();
}

function openingsCard() {
  if (!photoInfo().count || !built.length) return [];
  if (!holes) return [card("Ouvertures proposées", "Les portes et fenêtres se voient comme des trous dans les points des murs validés.",
    button("Chercher les ouvertures", "", "Trous rectangulaires dans les points des murs", () => run(() => act("Recherche des ouvertures", async () => {
      holes = await call("photos.propose_openings", { source: photoInfo().source });
      picked = new Set(holes!.map((_, i) => i));
    }))))];
  const box = el("div", "card");
  box.append(el("b", "", holes.length ? "Ouvertures proposées" : "Aucune ouverture trouvée"));
  for (const [i, hole] of holes.entries()) {
    const check = Object.assign(el("input"), { type: "checkbox", checked: picked.has(i) });
    check.onchange = () => { if (check.checked) picked.add(i); else picked.delete(i); };
    const line = el("label", "wall");
    line.append(check, el("span", "", `${OPENINGS.find(([id]) => id === hole.type)?.[1]} · ${hole.label} · ${number(hole.width, 2)} × ${number(hole.height, 2)} m ± ${cm(hole.sigma)}${hole.sill > 0.15 ? ` · allège ${number(hole.sill, 2)} m` : ""}`));
    box.append(line);
  }
  if (holes.length) box.append(row(button("Créer les ouvertures cochées", "primary", "Portes et fenêtres", () => run(() => act("Création des ouvertures", async () => {
    for (const [i, hole] of holes!.entries()) if (picked.has(i)) await call("plan.place", { type: hole.type, point: hole.point, width: hole.width, height: hole.height, sill: hole.sill }).catch(() => {});
    holes = null;
    await refreshPlan();
  })))));
  return [box];
}

async function fromCadastre() {
  await call("plan.cadastre", { create: true });
  await refreshPlan();
}

async function loadPlan() {
  if (!info) return;
  [built, land, sketch, parts] = await Promise.all([call<Built[]>("plan.walls"), call<Cadastre>("plan.cadastre"), call<Sketch>("plan.sketch"), call<{ id: string; type: string }[]>("components.list")]);
  if (sketch) await host.allowAssets([sketch.path]).catch(() => {});
  if (sketch?.points && !vision) vision = await call<{ vision_ready?: boolean }>("ai.status").catch(() => null);
  placeSketch(sketch);
  paint();
}

async function loadProposal(reset = false) {
  const source = photoInfo().source;
  proposal = source ? await call<Proposal>("photos.propose_walls", { source }).catch(() => null) : null;
  if (reset || !checked.size) checked = new Set(walls().filter((w) => (created().length ? w.component : w.sure)).map((w) => w.id));
  paint();
}

async function act(text: string, task: () => Promise<unknown>) {
  pending = text;
  render();
  try {
    await task();
  } finally {
    pending = detail = "";
    render();
    paint();
  }
}

async function validateWalls() {
  await call("photos.create_walls", { source: photoInfo().source, walls: [...checked] });
  guide.done = [...guide.done.filter((n) => n < 4), 4];
  delete guide.scale;
  await save();
  await update();
  await loadPlan();
  await loadProposal(true);
  render();
}

async function densifyCloud() {
  render();
  await photoTask("photos.densify");
  await loadProposal();
  render();
}

async function clickWall(wall: Wall) {
  checked.add(wall.id);
  await validateWalls();
  const made = walls().find((w) => w.id === wall.id && w.component);
  if (made) await clickAngle(made, made.angles.a ? "b" : "a");
}

async function clickAngle(wall: Wall, end: End) {
  const [x, z] = wall[end], source = photoInfo().source;
  const photos = await call<string[]>("photos.visible", { source, points_xyz_m: [[x, 0.8, z], [x, 1.6, z]] });
  if (!photos.length) throw new Error("Aucune photo ne montre cet angle : densifie ou ajoute-le en mode expert.");
  await openGuided({ anchor: `${wall.component}:${end}`, points: [[x, 0, z], [x, wall.height, z]],
    text: `Clique l'angle au ${side(wall, end)} du ${lower(wall.label)} (vue de dessus), à n'importe quelle hauteur. Le trait jaune montre où il devrait être.`, photos });
}

const planned = () => guide.plan && photoInfo().count ? [...built].sort((a, b) => Math.hypot(b.b[0] - b.a[0], b.b[1] - b.a[1]) - Math.hypot(a.b[0] - a.a[0], a.b[1] - a.a[1]))[0] : undefined;

function scaleStep() {
  const target = planned(), wall = target ? { ...target, component: target.id } as unknown as Wall : walls().find((w) => w.component && w.component === proposal?.measure), angle = missing()[0];
  if (!target && !created().length && built.length) return [intro("Pas besoin d'échelle", "Tes murs viennent du tracé ou du cadastre : ils sont déjà en mètres. Une cote au mètre les précisera à l'étape suivante."), footer(back(), onward())];
  if (!wall && !created().length) return [intro("Valide d'abord les murs", "L'échelle se mesure sur un mur validé à l'étape 4."), footer(back())];
  if (!wall) return [intro("Il manque un angle", `Pour fixer l'échelle, il faut un mur dont les deux angles sont trouvés. Clique l'angle manquant du ${lower(angle[0].label)} sur 2 photos.`),
    footer(back(), button("Je n'ai pas de mètre", "", "Continuer avec une échelle supposée", () => run(skipScale)), primary("Cliquer cet angle", () => clickAngle(...angle)))];
  const name = lower(wall.label);
  const input = Object.assign(el("input", "measure"), { inputMode: "decimal", value: guide.scale ? String(guide.scale) : "", placeholder: "531", autocomplete: "off" });
  const label = el("label", "", `Longueur du ${name}, en cm`);
  label.append(input);
  const submit = primary("Valider la mesure", () => validateScale(wall, input.value));
  input.onkeydown = (event) => { if (event.key === "Enter") submit.click(); };
  return [intro("Donne une seule mesure", `Les photos donnent les formes, pas la taille. Mesure au mètre le ${name} surligné en jaune, d'un angle à l'autre, à hauteur de taille.`),
    label, card("Résultat attendu après validation", "Toutes les cotes du plan à ± 1 à 2 cm."), el("hr"),
    card("Ensuite : vérifier", "L'appli te dira s'il y a une contradiction, et quelle mesure de plus rendrait le plan plus précis."),
    footer(button("Je n'ai pas de mètre", "", "Continuer avec une échelle supposée", () => run(skipScale)), submit)];
}

async function validateScale(wall: Wall, text: string) {
  const value = decimal(text);
  if (!(value > 0)) throw new Error("Indique la longueur en centimètres, par exemple 531.");
  const measured = await call<{ expr: string; value: number }>("components.measure_expr", { a: `${wall.component}:a`, b: `${wall.component}:b` });
  const ratio = value / 100 / measured.value;
  if (ratio < 0.5 || ratio > 2) throw new Error(`Le modèle donne environ ${cm(measured.value)} : vérifie que la mesure est en centimètres.`);
  await call("cotes.add", { id: "echelle", expr: measured.expr, value, sigma: 1, unit: "cm", label: `Longueur du ${lower(wall.label)}` });
  guide.scale = value;
  await complete(guide.step);
}

async function skipScale() {
  delete guide.scale;
  await complete(guide.step);
}

function checkStep() {
  if (!result) return [intro("Vérification…", "Calcul du plan en cours."), footer(back())];
  const uses = new Set(built.flatMap((w) => w.uses));
  const placed = result.params.filter((p) => (p.axis === "x" || p.axis === "z") && uses.has(p.name));
  const known = placed.filter((p) => p.sigma != null && Number.isFinite(p.sigma)), unknown = placed.filter((p) => !known.includes(p));
  const worst = Math.max(0, ...known.map((p) => p.sigma!)), supposed = result.warnings?.some((w) => w.code === "echelle_supposee");
  const outside = (result.warnings ?? []).filter((w) => (w.code === "hors_parcelle" || w.code === "sur_batiment")
    && parts.find((c) => c.id === (w as { component?: string }).component)?.type !== "parcelle");
  const cards = [
    ...outside.length ? [card("Débordements", outside.map((w) => w.message).join(" ; ") + ".")] : position("cadastre") ? [card("Aucun débordement", "Tout tient dans la parcelle.")] : [],
    supposed ? card("Échelle supposée", "Sans mesure au mètre, les longueurs peuvent être fausses de 10 à 20 %.", ...position("echelle") ? [button("Donner une mesure", "", "Étape de l'échelle", () => run(() => go(position("echelle"))))] : [])
      : card(known.length ? `Murs placés à ± ${cm(worst)}` : "Précision inconnue", unknown.length
        ? `${plural(unknown.length, "valeur")} pas encore connue${unknown.length > 1 ? "s" : ""} : ${unknown.slice(0, 3).map((p) => p.label).join(", ")}. Précise-les à l'étape 4 ou avec une cote.`
        : "Toutes les positions des murs sont déterminées."),
    result.conflicts.length ? card(`${plural(result.conflicts.length, "contradiction")}`, result.conflicts.map((group) => group.map((o) => `« ${o.label || o.id} »`).join(" et ")).join(" ; ")
      + ". Une des valeurs est fausse : vérifie la mesure au mètre, ou décoche le mur concerné à l'étape 4.") : card("Aucune contradiction", "Les photos et les mesures sont d'accord."),
  ];
  if (next && !supposed) {
    const input = Object.assign(el("input", "measure"), { inputMode: "decimal", placeholder: "cm", autocomplete: "off", ariaLabel: "Mesure en cm" });
    const add = button("Ajouter", "", "Ajouter cette mesure et recalculer", () => run(() => addMeasure(next!, input.value)));
    const box = card("Mesure utile ensuite", `${describe(next)} (aujourd'hui ± ${cm(next.sigma)}).`);
    box.append(row(input, add));
    cards.push(box);
  }
  return [intro(unknown.length || supposed ? "Le plan est presque prêt" : "Le plan est prêt", "Voici ce que donnent tes photos et tes mesures."), ...cards,
    footer(back(), button(`${flow().at(-1)![1]} (optionnel)`, "", "Dernière étape", () => run(() => complete(guide.step))), primary("Ouvrir le plan", openPlan))];
}

function describe(candidate: Candidate) {
  const [, comp, axis] = /^(\w+)_([xz])$/.exec(candidate.expr) ?? [];
  const wall = created().find((w) => w.component === comp && w.axis === axis);
  const origin = created().find((w) => w.axis === axis && !result?.params.some((p) => p.name === `${w.component}_${axis}`));
  return wall && origin ? `Distance entre le ${lower(origin.label)} et le ${lower(wall.label)}, en cm` : candidate.label;
}

async function check() {
  result = null;
  render();
  result = await update();
  await loadPlan();
  next = (await call<Candidate[]>("next_best", { k: 1 }).catch(() => []))[0] ?? null;
  if (key() === "verifier") render();
}

async function addMeasure(candidate: Candidate, text: string) {
  const value = decimal(text);
  if (!(value > 0)) throw new Error("Indique la mesure en centimètres.");
  const cotes = await call<{ id: string }[]>("cotes.list");
  let n = 1;
  while (cotes.some((c) => c.id === `m${n}`)) n++;
  await call("cotes.add", { id: `m${n}`, expr: candidate.expr, value, sigma: 1, unit: "cm", label: candidate.label });
  await check();
}

async function openPlan() {
  if (!guide.done.includes(6)) await save({ done: [...guide.done, 6] });
  setMode("expert");
  setView("dessus");
}

function furnishStep() {
  const box = el("div", "furnish");
  box.append($("#gallery"), $("#layout"));
  return [intro("Meubler", "Choisis des meubles dans la galerie, place-les dans la vue, puis demande des agencements."), box, footer(back(), primary("Ouvrir le plan", openPlan))];
}

function pickInto(title: string) {
  return async () => {
    await addSources((await host.pickFiles(title)) ?? []);
    await loadPlan();
    render();
  };
}

function cadastreStep() {
  const has = sources.some((s) => s.kind.startsWith("cadastre")), house = land?.house, parcel = land?.parcel;
  const text = [house && `Bâtiment principal : ${plural(house.walls, "mur")} extérieurs, ${number(house.perimeter)} m de périmètre`, parcel && `parcelle de ${number(parcel.area, 0)} m²`].filter(Boolean).join(" ; ");
  return [intro("Le cadastre", "Le plan cadastral (PDF de cadastre.gouv.fr ou GeoJSON) donne la parcelle et l'emprise de la maison, qui devient les murs extérieurs."),
    card(has ? built.length ? "Murs et parcelle créés" : "Cadastre importé" : "Plan cadastral", has ? `${text || "Aucun bâtiment principal reconnu"}.` : "PDF ou GeoJSON du cadastre.",
      ...has ? [] : [button("Choisir un fichier", "primary", "PDF ou GeoJSON", () => run(pickInto("Plan cadastral")))],
      ...has && (house || parcel) && !land?.created.length ? [button("Créer les murs et la parcelle", "primary", "Murs extérieurs et parcelle", () => run(() => act("Création depuis le cadastre", fromCadastre)))] : []),
    footer(back(), onward(built.length > 0))];
}

function reliefStep() {
  const ground = parts.some((c) => c.type === "terrain");
  return [intro("Le relief", "Optionnel : une grille d'altitudes (ASC, GeoTIFF ou JSON) pose la maison et le jardin sur le vrai terrain."),
    card(ground ? "Relief importé" : "Grille d'altitudes", ground ? "Pose la maison sur le terrain : ses murs partent du point le plus bas de l'emprise." : "Sans grille, le terrain reste plat.",
      ...ground ? [button("Poser la maison sur le terrain", "primary", "Base des murs à l'altitude du terrain", () => run(() => act("Pose sur le terrain", async () => { await call("plan.settle"); await refreshPlan(); })))]
        : [button("Choisir une grille", "", "ASC, GeoTIFF ou JSON", () => run(pickInto("Grille d'altitudes")))]),
    footer(back(), onward())];
}

function roofStep() {
  const done = parts.some((c) => c.type === "toit");
  return [intro("Le toit", "Proposé sur l'emprise : deux pans, faîtage le long du plus grand côté, pente de 35°. Hauteur depuis la BD TOPO si le fichier la donne, sinon à mesurer."),
    done ? card("Toit proposé", roofed && !roofed.measured ? "Hauteur de gouttière à mesurer : ajoute une cote à l'étape Vérifier, ou règle-la en mode expert." : "Hauteurs reprises du cadastre ; ajuste-les en mode expert si besoin.")
      : card("Toit à deux pans", built.length ? "Sur les murs extérieurs de la maison." : "Crée d'abord les murs depuis le cadastre.",
        ...built.length ? [button("Proposer le toit", "primary", "Toit sur l'emprise", () => run(() => act("Création du toit", async () => { roofed = await call("plan.roof"); await refreshPlan(); })))] : []),
    footer(back(), onward(true, done))];
}

function facadeStep() {
  if (photoSource()) return photoInfo().count ? wallsStep() : positionsStep();
  return [intro("Les façades", "Ajoute les photos prises en faisant le tour de la maison : elles se recalent ensuite sur les murs du cadastre."),
    card("Photos des façades", "Un dossier de photos, avec du recouvrement entre deux photos.", button("Choisir un dossier", "primary", "Dossier de photos", () => run(async () => {
      const path = await host.pickFolder("Photos des façades");
      if (path) await addSources([path]);
    }))), footer(back(), onward(true, false))];
}

function bodyStep() {
  if (built.length) return [intro("Le gabarit", "La caisse est créée : ses parois sont des murs, le plancher sert de pièce pour la galerie et l'agencement."),
    card("Gabarit créé", `${plural(built.length, "paroi")} et un plancher ; cotes de la fiche enregistrées en « lu ».`), footer(back(), onward())];
  const form = el("div", "fields");
  const inputs = BODY.map(([name, label, example]) => {
    const input = Object.assign(el("input", "measure"), { name, inputMode: "decimal", placeholder: example, autocomplete: "off" }), box = el("label", "", `${label} (cm)`);
    box.append(input);
    form.append(box);
    return input;
  });
  const side = el("select");
  side.append(new Option("Porte latérale à droite", "droite"), new Option("Porte latérale à gauche", "gauche"));
  return [intro("Le gabarit", "Recopie les cotes utiles de la fiche constructeur, en cm. Laisse vide ce qui n'existe pas."), form, side,
    footer(back(), primary("Créer le gabarit", () => act("Création du gabarit", async () => {
      await call("plan.vehicle", { ...Object.fromEntries(inputs.map((i) => [i.name, decimal(i.value) || 0])), porte_cote: side.value });
      await refreshPlan();
    })))];
}

async function finishPlan() {
  guide.done = [...guide.done.filter((n) => n < 4), 4];
  delete guide.scale;
  await save();
  await loadPlan();
  await loadProposal(true);
}

const helpers: Helpers = {
  guide: () => guide, save: (change, undo) => save(change as Partial<Guide>, undo), act, intro, card, row, footer, primary, back, render, paint, sketch: () => sketch, walls: () => built,
  source: () => photoInfo().count ? photoInfo().source : null, finish: finishPlan, next: () => complete(guide.step),
  classic: async () => { await save({ classic: true }); render(); paint(); },
};

const VIEWS: Record<string, () => Node[]> = { projet: projectStep, sources: sourcesStep, positions: positionsStep, murs: () => drawn() ? fourStep(helpers) : wallsStep(), echelle: scaleStep, verifier: checkStep,
  meubler: furnishStep, cadastre: cadastreStep, relief: reliefStep, toit: roofStep, facades: facadeStep, gabarit: bodyStep };

function paint() {
  const step = mode === "guide" ? key() : "";
  if (step !== "murs" || !drawn()) previewPose(null);
  const measured = walls().filter((w) => w.component && w.component === proposal?.measure), house = land?.created.length ? [] : land?.house?.points ?? [];
  const outline = house.map((a, i) => ({ a, b: house[(i + 1) % house.length], height: 2.5, tone: "sure", on: true, text: i ? "" : "Cadastre" }));
  const read = (reading?.walls ?? []).flatMap((wall, w) => wall.points.slice(0, wall.closed ? undefined : -1).map((a, i) => ({
    a, b: wall.points[(i + 1) % wall.points.length], height: 2.5, tone: "doubt", on: kept.has(w), text: i ? "" : `Croquis ${w + 1}` })));
  showWalls((step === "murs" || step === "facades") && !drawing() ? [...shown().map((w) => ({ ...w, tone: w.id === hover ? "measure" : w.sure ? "sure" : "doubt", on: checked.has(w.id) || w.id === hover, text: checked.has(w.id) || w.id === hover ? w.label : "" })), ...outline, ...read]
    : step === "echelle" ? measured.map((w) => ({ ...w, tone: "measure", on: true, text: "à mesurer" })) : []);
  if (step) highlight(step === "echelle" ? planned()?.id ?? proposal?.measure ?? null : walls().find((w) => w.id === hover)?.component ?? null, 0xf5c451);
}

function render() {
  const guided = mode === "guide";
  if (!guided || key() !== "meubler") $("#sidebar").append($("#gallery"), $("#layout"));
  document.body.classList.toggle("guided", guided);
  document.body.classList.toggle("page", guided && (["projet", "sources"].includes(key()) || !info || creating));
  document.body.classList.toggle("furnish", guided && key() === "meubler");
  modeButton.textContent = guided ? "Mode expert" : "Mode guidé";
  modeButton.title = guided ? "Afficher toute l'interface" : "Revenir au parcours par étapes";
  if (!guided || key() !== "murs" || !drawn() || !info || creating) hidePlan();
  if (!guided) return;
  renderRail();
  const bar = Object.assign(el("progress"), { max: 1, ariaLabel: "Calcul en cours" });
  panel.replaceChildren(...pending ? [intro(pending, detail || "Un instant : l'appli calcule le plan."), bar] : VIEWS[key(info && !creating ? guide.step : 1)]());
}

async function save(change: Partial<Guide> = {}, undo = false) {
  guide = { ...guide, ...change };
  if (!info) return;
  remember({ type: TYPES.find(([id]) => id === guide.type)?.[3], step: `Étape ${guide.step} · ${flow()[guide.step - 1]?.[1] ?? ""}` });
  await call("project.guide", { guide, undo });
}

async function go(step: number) {
  if (!info || step < 1 || step > flow().length) return;
  const forward = step > guide.step;
  creating = false;
  if (drawing()) stopDrawing();
  if (step !== guide.step) closeGuided();
  await save({ step });
  await loadPlan();
  if (["murs", "echelle", "facades"].includes(key(step))) await loadProposal();
  render();
  paint();
  if (mode !== "guide") return;
  if (key(step) !== "projet" && key(step) !== "sources") setView(["positions", "toit", "relief", "verifier", "meubler"].includes(key(step)) ? "3d" : "dessus");
  if (key(step) === "verifier") await check();
  if (forward && key(step) === "positions") void run(autoPositions);
}

async function complete(step: number) {
  await save({ done: [...new Set([...guide.done, step])] });
  await go(Math.min(step + 1, 7));
}

function setMode(value: string) {
  mode = value;
  if (drawing()) stopDrawing();
  localStorage.setItem(MODE, mode);
  if (mode === "guide" && info) return run(() => go(guide.step));
  render();
  paint();
}

async function resume(stored?: Guide) {
  if (stored?.step) return stored;
  const [listed, parts, cotes] = await Promise.all([call<Source[]>("sources.list"), call<{ type: string }[]>("components.list"), call<unknown[]>("cotes.list")]);
  const photos = listed.find((s) => s.kind === "photos_colmap");
  const walled = parts.some((c) => c.type === "mur");
  const done = [1, ...listed.length || walled ? [2] : [], ...photos?.registered || walled && !photos ? [3] : [], ...walled ? [4] : [], ...cotes.length ? [5] : []];
  const type = stored?.type ?? (fresh || "piece");
  fresh = "";
  if (type !== "piece") return { type, done: walled ? [1, 2] : [1], step: walled ? 3 : 2 };
  return { type, done, step: [1, 2, 3, 4, 5, 6].find((n) => !done.includes(n)) ?? 6 };
}

modeButton.onclick = () => setMode(mode === "guide" ? "expert" : "guide");
host.documents().then((folder) => { parent ||= `${folder}\\3D_Plans`; if (!info) render(); }).catch(() => {});

document.addEventListener("plans3d:opened", (event) => run(async () => {
  const opened = (event as CustomEvent<Info | null>).detail;
  [info, creating, proposal, result, next, checked, fitted, everything] = [opened, false, null, null, null, new Set(), null, false];
  resetFour();
  guide = info ? await resume(info.guide) : { type: draft.type, step: 1, done: [] };
  render();
  if (info) await go(guide.step);
}));
document.addEventListener("plans3d:new", () => {
  if (mode !== "guide") return $<HTMLButtonElement>("#new-project-button").click();
  if (drawing()) stopDrawing();
  closeGuided();
  [creating, draft] = [true, { type: "piece", name: "" }];
  render();
});
document.addEventListener("plans3d:sources", (event) => {
  sources = (event as CustomEvent<Source[]>).detail;
  if (mode === "guide" && ["sources", "positions", "cadastre", "relief", "facades"].includes(key())) render();
});
document.addEventListener("plans3d:photos", () => run(async () => {
  if (mode !== "guide" || !info) return;
  const loading = ["murs", "echelle", "facades"].includes(key()) && !proposal && photoInfo().count > 0 && !photoInfo().busy;
  if (loading) await loadProposal();
  if (loading || ["positions", "facades"].includes(key()) || key() === "murs" && photoInfo().busy) render();
}));
document.addEventListener("plans3d:clicked", () => run(async () => {
  if (!["murs", "echelle", "facades"].includes(key())) return;
  await update();
  await loadPlan();
  await loadProposal();
  render();
}));
document.addEventListener("plans3d:scene", () => setTimeout(paint));
onEvent((event) => {
  if (!pending || event.event !== "progress" || !["ai.pull", "plan.read_sketch"].includes(String(event.method))) return;
  detail = String(event.message ?? "");
  render();
});

render();
