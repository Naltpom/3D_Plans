import { call, host } from "./host";
import { $, button, cm, el, number, run } from "./ui";
import { addSources, openProject, recent, refreshSources, update, type UpdateResult } from "./projects";
import { openGuided, photoInfo, photoTask } from "./photos";
import { highlight, setView, showWalls } from "./view3d";
import { calibrate, controls, mode as drawing, opacity, place, placeHint, placeSketch, stop as stopDrawing, trace, type Sketch } from "./drawing";

type Guide = { type: string; step: number; done: number[]; scale?: number; manual?: boolean };
type Info = { name: string; path: string; guide?: Guide };
type Source = { id: string; path: string; kind: string; obs: number; photos?: number; registered?: number | null; state?: string; error?: string };
type End = "a" | "b";
type Wall = { id: string; label: string; axis: string; a: number[]; b: number[]; length: number; height: number; n: number; photos: number; sure: boolean; reason: string; component: string | null; angles: Record<End, string | null> };
type Proposal = { walls: Wall[]; measure: string | null; dense: boolean; densify_min: number };
type Candidate = { label: string; expr: string; sigma: number };
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
const STEPS = [["Le projet", "Ce que tu reconstruis"], ["Les sources", "Photos, cotes, plans"], ["Positions des photos", "Calcul automatique"],
  ["Les murs", "Valider ce qui est trouvé"], ["L'échelle", "Une mesure au mètre"], ["Vérifier", "Contradictions, mesure utile"], ["Meubler", "Optionnel, une fois le plan prêt"]];
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
let everything = false;
let fitted: { pairs: string[][]; scale_known: boolean } | null = null;
const shown = () => everything ? walls() : walls().filter((w, i, all) => w.component || checked.has(w.id) || [...all].sort((a, b) => Number(b.sure) - Number(a.sure) || b.n - a.n).indexOf(w) < 6);
let built: Built[] = [];
let land: Cadastre = null;
let sketch: Sketch = null;

const plural = (n: number, word: string) => `${number(n, 0)} ${word}${n > 1 ? "s" : ""}`;
const lower = (label: string) => label[0].toLowerCase() + label.slice(1);
const decimal = (text: string) => Number(text.trim().replace(",", "."));
const target = () => `${parent.replace(/[\\/]+$/, "")}\\${draft.name.trim().replace(/[<>:"/\\|?*]+/g, "_")}`;
const photoSource = () => sources.find((s) => s.kind === "photos_colmap");
const walls = () => proposal?.walls ?? [];
const created = () => walls().filter((w) => w.component);
const side = (wall: Wall, end: End) => wall.axis === "z" ? end === "a" ? "bout de gauche" : "bout de droite" : end === "a" ? "bout du haut" : "bout du bas";
const missing = () => created().flatMap((w) => (["a", "b"] as End[]).filter((e) => !w.angles[e]).map((e) => [w, e] as [Wall, End]));

function intro(step: number, title: string, text: string) {
  const box = el("div", "intro");
  box.append(el("span", "kicker", step === 7 ? "Optionnel" : `Étape ${step} sur 6`), el("h1", "", title), ...text ? [el("p", "", text)] : []);
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
const onward = (enabled = true, main = true) => Object.assign(button("Étape suivante", main ? "primary" : "", "Passer à l'étape suivante", () => run(() => complete(guide.step))), { disabled: !enabled });

function summary(step: number) {
  const photos = photoSource(), count = created().length;
  return [
    TYPES.find(([id]) => id === guide.type)?.[3] ?? "",
    photos ? plural(photos.photos ?? 0, "photo") : sources.length ? plural(sources.length, "source") : sketch ? "croquis" : guide.manual ? "tracé à la main" : "",
    photos?.registered != null ? `${photos.registered} positionnées` : photos ? "" : "sans photos",
    built.length ? `${plural(built.length, "mur")}${count ? ` dont ${count} des photos` : ""}` : "",
    guide.scale ? `${number(guide.scale)} cm mesurés` : count ? "échelle supposée" : "déjà en mètres",
    "plan vérifié", "",
  ][step - 1] || STEPS[step - 1][1];
}

function renderRail() {
  $("#rail-title").textContent = info && !creating ? info.name : "Nouveau projet";
  rail.replaceChildren(...STEPS.flatMap(([title, hint], i) => {
    const step = i + 1, done = guide.done.includes(step) && Boolean(info) && !creating, current = info && !creating ? guide.step : 1;
    const item = button("", `st ${step === current ? "current" : ""} ${done ? "done" : ""}`, title, () => run(() => go(step)));
    const text = el("span");
    text.append(el("b", "", title), done ? summary(step) : hint);
    item.append(el("span", "n", done ? "✓" : String(step)), text);
    item.disabled = (!info || creating) && step > 1;
    if (step === current) item.ariaCurrent = "step";
    return step === 7 ? [el("hr"), item] : [item];
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
  if (info && !creating) return [intro(1, "Ton projet", `${info.name} · ${info.path}`), options,
    footer(button("Nouveau projet", "", "Créer un autre projet", () => { creating = true; draft = { type: "piece", name: "" }; render(); }),
      button("Ouvrir un autre projet", "", "Choisir le dossier d'un projet", () => run(pick)), onward())];
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
  const places = el("ul", "list");
  places.append(...recent().map((project) => {
    const item = el("li");
    const open = button("", "item", project.path, () => run(() => openProject("project.open", { path: project.path })));
    open.append(el("strong", "", project.name), el("small", "", project.path));
    item.append(open);
    return item;
  }));
  const resume = el("div", "card plain");
  resume.append(el("b", "", "Reprendre un projet"), ...places.children.length ? [places] : [el("span", "", "Aucun projet récent.")],
    row(button("Ouvrir un dossier de projet…", "", "Choisir le dossier d'un projet", () => run(pick))));
  const form = el("div", "card plain");
  form.append(label, row(where, folder));
  return [intro(1, "Que veux-tu reconstruire ?", "Le choix règle les éléments proposés ensuite. Il se change plus tard."), options, form,
    footer(...info ? [button("Annuler", "", "Revenir au projet ouvert", () => { creating = false; render(); })] : [], create), resume];
}

async function pick() {
  const path = await host.pickFolder("Ouvrir un projet (dossier contenant project.db)");
  if (path) await openProject("project.open", { path });
}

async function createProject() {
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
  return [intro(2, "D'où viennent les murs ?", "Choisis une ou plusieurs entrées : elles se combinent, un mur tracé se précise ensuite avec des photos ou des cotes."),
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
  if (photo.busy && photo.busy !== "photos.densify") return [intro(3, photo.busy === "tools.install" ? "Installation des outils photo" : "Calcul des positions en cours",
    "Tu peux laisser l'appli travailler. La vue se remplit à la fin."), ...progress(photo.busy, photo.fraction, photo.message, photo.started), footer(back())];
  if (!source) return [intro(3, "Pas de photos", "Rien à calculer ici : les murs viendront du tracé, du croquis ou du cadastre."), footer(back(), onward())];
  if (!photo.installed) return [intro(3, "Il faut les outils photo", "COLMAP calcule la position des photos. Il s'installe une seule fois (≈ 400 Mo)."),
    footer(back(), primary("Installer les outils photo", () => photoTask("tools.install")))];
  if (photo.count) return [intro(3, `${photo.count} photos positionnées sur ${source.photos}`,
    "Dans la vue, chaque cône bleu est une photo et les points sont les détails retrouvés sur plusieurs photos, en couleur après densification. Les murs se cachent dans ces points : l'étape suivante les trouve."),
  card("Pour tourner la vue", "Glisse avec le bouton gauche, zoome avec la molette. « Dessus » montre le plan."), footer(back(), onward())];
  const minutes = Math.max(2, Math.round((source.photos ?? 0) * 2.5 / 60));
  phase = -1;
  return [intro(3, "Calculer la position des photos", "L'appli retrouve d'où chaque photo a été prise et place en 3D les détails qu'elles ont en commun."),
    ...progress("photos.extract", NaN, "", Date.now()).slice(0, 1), el("p", "", `Durée estimée : ≈ ${minutes} min pour ${plural(source.photos ?? 0, "photo")}.`),
    ...source.state === "échec" ? [card("Le dernier calcul a échoué", source.error ?? "")] : [],
    footer(back(), primary("Calculer les positions", () => photoTask("photos.extract")))];
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
    };
    const line = el("label", `wall ${wall.sure ? "" : "doubt"}`);
    line.append(box, el("span", "", `${wall.label} · ${wall.reason || `vu sur ${plural(wall.photos, "photo")}`} · ${number(wall.length)} m`), el("span", "m", `${number(wall.n, 0)} pts`));
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
  if (photo.busy === "photos.densify") return [intro(4, "Densification en cours", "Elle cherche des points sur les surfaces peu texturées, puis de nouveaux murs."),
    ...progress(photo.busy, photo.fraction, photo.message, photo.started), footer(back())];
  if (tool === "trace") return [intro(4, "Trace la pièce", "Clique chaque angle dans la vue de dessus : la grille de 5 cm, l'angle droit et les murs existants sont aimantés. Pour une longueur exacte, tape-la en cm puis Entrée. Clique le premier angle pour fermer."),
    ...controls(), ...sketch?.points ? [opacity(sketch)] : []];
  if (tool) return [intro(4, `Pose : ${OPENINGS.find(([id]) => id === tool)?.[1].toLowerCase()}`, "Clique sur un mur dans la vue : l'ouverture se centre sur le clic. Clique un autre mur pour en poser d'autres."),
    placeHint(), footer(button("Terminer", "primary", "Quitter la pose", stopDrawing))];
  const fromPhotos = photo.count > 0, count = walls().length;
  const house = land?.house, parcel = land?.parcel;
  const parts = [
    intro(4, fromPhotos && count ? `J'ai trouvé ${count} mur${count > 1 ? "s probables" : " probable"}` : "Les murs", fromPhotos
      ? "Ils viennent des surfaces planes vues sur tes photos. Décoche ceux qui sont faux. Tu peux aussi tracer des murs ou les reprendre du cadastre."
      : "Trace les murs, décalque ton croquis ou reprends-les du cadastre. Une cote au mètre les précisera ensuite."),
    ...fromPhotos && built.some((w) => w.source !== photoInfo().source) ? [fitCard()] : [],
    ...fromPhotos ? photoWalls() : [],
    ...house || parcel ? [card(land!.created.length ? "Cadastre : créé" : "Depuis le cadastre", [house && `Bâtiment principal : ${plural(house.walls, "mur")} extérieurs, ${number(house.perimeter)} m de périmètre`,
      parcel && `parcelle de ${number(parcel.area, 0)} m²`].filter(Boolean).join(" ; ") + ".",
    ...land!.created.length ? [] : [button("Créer les murs et la parcelle", "", "Murs extérieurs du bâtiment principal et parcelle", () => run(() => act("Création depuis le cadastre", fromCadastre)))])] : [],
    card("Tracer à la main", `Clique les angles dans la vue de dessus, longueurs au clavier.${sketch ? sketch.points ? " Le croquis calé s'affiche en fond : décalque-le." : " Cale d'abord le croquis : 2 points et leur distance." : ""}`,
      button("Tracer une pièce", built.length || fromPhotos || house ? "" : "primary", "Tracer des murs dans la vue de dessus", () => trace(() => built, render, refreshPlan)),
      ...sketch ? [button(sketch.points ? "Recaler le croquis" : "Caler le croquis", sketch.points ? "" : "primary", "2 points du croquis et leur distance", () => calibrate(sketch!, async () => { await loadPlan(); setView("dessus"); render(); }))] : []),
    ...sketch?.points ? [opacity(sketch)] : [],
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

async function refreshPlan() {
  await update();
  await loadPlan();
  render();
}

async function fromCadastre() {
  await call("plan.cadastre", { create: true });
  await refreshPlan();
}

async function loadPlan() {
  if (!info) return;
  [built, land, sketch] = await Promise.all([call<Built[]>("plan.walls"), call<Cadastre>("plan.cadastre"), call<Sketch>("plan.sketch")]);
  if (sketch) await host.allowAssets([sketch.path]).catch(() => {});
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
    pending = "";
    render();
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

function scaleStep() {
  const wall = walls().find((w) => w.component && w.component === proposal?.measure), angle = missing()[0];
  if (!created().length && built.length) return [intro(5, "Pas besoin d'échelle", "Tes murs viennent du tracé ou du cadastre : ils sont déjà en mètres. Une cote au mètre les précisera à l'étape suivante."), footer(back(), onward())];
  if (!created().length) return [intro(5, "Valide d'abord les murs", "L'échelle se mesure sur un mur validé à l'étape 4."), footer(back())];
  if (!wall) return [intro(5, "Il manque un angle", `Pour fixer l'échelle, il faut un mur dont les deux angles sont trouvés. Clique l'angle manquant du ${lower(angle[0].label)} sur 2 photos.`),
    footer(back(), button("Je n'ai pas de mètre", "", "Continuer avec une échelle supposée", () => run(skipScale)), primary("Cliquer cet angle", () => clickAngle(...angle)))];
  const name = lower(wall.label);
  const input = Object.assign(el("input", "measure"), { inputMode: "decimal", value: guide.scale ? String(guide.scale) : "", placeholder: "531", autocomplete: "off" });
  const label = el("label", "", `Longueur du ${name}, en cm`);
  label.append(input);
  const submit = primary("Valider la mesure", () => validateScale(wall, input.value));
  input.onkeydown = (event) => { if (event.key === "Enter") submit.click(); };
  return [intro(5, "Donne une seule mesure", `Les photos donnent les formes, pas la taille. Mesure au mètre le ${name} surligné en jaune, d'un angle à l'autre, à hauteur de taille.`),
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
  await complete(5);
}

async function skipScale() {
  delete guide.scale;
  await complete(5);
}

function checkStep() {
  if (!result) return [intro(6, "Vérification…", "Calcul du plan en cours."), footer(back())];
  const uses = new Set(built.flatMap((w) => w.uses));
  const placed = result.params.filter((p) => (p.axis === "x" || p.axis === "z") && uses.has(p.name));
  const known = placed.filter((p) => p.sigma != null && Number.isFinite(p.sigma)), unknown = placed.filter((p) => !known.includes(p));
  const worst = Math.max(0, ...known.map((p) => p.sigma!)), supposed = result.warnings?.some((w) => w.code === "echelle_supposee");
  const cards = [
    supposed ? card("Échelle supposée", "Sans mesure au mètre, les longueurs peuvent être fausses de 10 à 20 %.", button("Donner une mesure", "", "Étape 5", () => run(() => go(5))))
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
  return [intro(6, unknown.length || supposed ? "Le plan est presque prêt" : "Le plan est prêt", "Voici ce que donnent tes photos et tes mesures."), ...cards,
    footer(back(), button("Meubler (optionnel)", "", "Étape 7", () => run(() => complete(6))), primary("Ouvrir le plan", openPlan))];
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
  if (guide.step === 6) render();
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
  return [intro(7, "Meubler", "Choisis des meubles dans la galerie, place-les dans la vue, puis demande des agencements."), box, footer(back(), primary("Ouvrir le plan", openPlan))];
}

const VIEWS = [projectStep, sourcesStep, positionsStep, wallsStep, scaleStep, checkStep, furnishStep];

function paint() {
  const step = mode === "guide" ? guide.step : 0;
  const measured = walls().filter((w) => w.component && w.component === proposal?.measure), house = land?.created.length ? [] : land?.house?.points ?? [];
  const outline = house.map((a, i) => ({ a, b: house[(i + 1) % house.length], height: 2.5, tone: "sure", on: true, text: i ? "" : "Cadastre" }));
  showWalls(step === 4 && !drawing() ? [...shown().map((w) => ({ ...w, tone: w.sure ? "sure" : "doubt", on: checked.has(w.id), text: checked.has(w.id) ? w.label : "" })), ...outline]
    : step === 5 ? measured.map((w) => ({ ...w, tone: "measure", on: true, text: "à mesurer" })) : []);
  if (step) highlight(step === 5 ? proposal?.measure ?? null : null, 0xf5c451);
}

function render() {
  const guided = mode === "guide";
  if (!guided || guide.step !== 7) $("#sidebar").append($("#gallery"), $("#layout"));
  document.body.classList.toggle("guided", guided);
  document.body.classList.toggle("page", guided && (guide.step < 3 || !info || creating));
  document.body.classList.toggle("furnish", guided && guide.step === 7);
  modeButton.textContent = guided ? "Mode expert" : "Mode guidé";
  modeButton.title = guided ? "Afficher toute l'interface" : "Revenir au parcours par étapes";
  if (!guided) return;
  renderRail();
  const bar = Object.assign(el("progress"), { max: 1, ariaLabel: "Calcul en cours" });
  panel.replaceChildren(...pending ? [intro(guide.step, pending, "Un instant : l'appli calcule le plan."), bar] : VIEWS[(info && !creating ? guide.step : 1) - 1]());
}

async function save(change: Partial<Guide> = {}) {
  guide = { ...guide, ...change };
  if (info) await call("project.guide", { guide });
}

async function go(step: number) {
  if (!info || step < 1 || step > 7) return;
  creating = false;
  if (drawing()) stopDrawing();
  await save({ step });
  await loadPlan();
  if (step === 4 || step === 5) await loadProposal();
  render();
  paint();
  if (mode !== "guide") return;
  if (step >= 3 && step <= 5) setView(step === 3 ? "3d" : "dessus");
  if (step === 6) await check();
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
  return { type, done, step: [1, 2, 3, 4, 5, 6].find((n) => !done.includes(n)) ?? 6 };
}

modeButton.onclick = () => setMode(mode === "guide" ? "expert" : "guide");
host.documents().then((folder) => { parent ||= `${folder}\\3D_Plans`; if (!info) render(); }).catch(() => {});

document.addEventListener("plans3d:opened", (event) => run(async () => {
  const opened = (event as CustomEvent<Info | null>).detail;
  [info, creating, proposal, result, next, checked, fitted, everything] = [opened, false, null, null, null, new Set(), null, false];
  guide = info ? await resume(info.guide) : { type: draft.type, step: 1, done: [] };
  render();
  if (info) await go(guide.step);
}));
document.addEventListener("plans3d:sources", (event) => {
  sources = (event as CustomEvent<Source[]>).detail;
  if (mode === "guide" && [2, 3].includes(guide.step)) render();
});
document.addEventListener("plans3d:photos", () => run(async () => {
  if (mode !== "guide" || !info) return;
  const loading = (guide.step === 4 || guide.step === 5) && !proposal && photoInfo().count > 0 && !photoInfo().busy;
  if (loading) await loadProposal();
  if (loading || guide.step === 3 || guide.step === 4 && photoInfo().busy) render();
}));
document.addEventListener("plans3d:clicked", () => run(async () => {
  if (guide.step !== 4 && guide.step !== 5) return;
  await update();
  await loadPlan();
  await loadProposal();
  render();
}));
document.addEventListener("plans3d:scene", () => setTimeout(paint));

render();
