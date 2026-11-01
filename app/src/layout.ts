import { call, onEvent } from "./host";
import { $, button, el, number, run, warn } from "./ui";
import { highlight, setView, showFurniture, type Shape } from "./view3d";

type Ref = string | number;
type Rule = { type: string; objet?: number; objets?: number[]; cible?: Ref; vers?: Ref; mur?: string; distance?: number[]; max?: number; min?: number };
type Placed = { id: number; label: string; rot: number };
type Component = { id: string; type: string; label: string };
type Proposal = { index: number; score: number; ok: boolean; details: { ok: boolean; text: string }[]; shapes: Shape[] };

const KINDS: Record<string, string> = {
  contre_mur: "Contre un mur", face_a: "Face à (vis-à-vis)", pres_de: "Près de", loin_de: "Loin de", aligne: "Alignés", centre: "Au centre de la pièce", orientation: "Orienté vers", angle: "Angle imposé",
};
const FIELDS: Record<string, string[]> = {
  contre_mur: ["objet", "mur"], face_a: ["objet", "cible", "dmin", "dmax"], pres_de: ["objet", "cible", "max"], loin_de: ["objet", "cible", "min"],
  aligne: ["objets"], centre: ["objet"], orientation: ["objet", "vers"], angle: ["objet", "angle"],
};
const LABELS: Record<string, string> = {
  objet: "Objet", objets: "Objets (Ctrl+clic)", cible: "Cible", vers: "Vers", mur: "Mur", dmin: "Distance min (cm)", dmax: "Distance max (cm)", max: "Écart max (cm)", min: "Écart min (cm)", angle: "Angle (°)",
};
const DEFAULTS: Record<string, number> = { dmin: 200, dmax: 300, max: 50, min: 150 };
const NEAREST: Record<string, string> = { fenetre: "une fenêtre", porte: "une porte", porte_fenetre: "une porte-fenêtre", mur: "un mur" };
const LOCKED = "Aperçu : appliquez la proposition ou revenez à l'agencement actuel pour déplacer un objet";
const TARGETS = ["mur", "porte", "porte_fenetre", "fenetre", "escalier", "boite"];

const section = $("#layout");
const ruleList = $("#rule-list");
const form = $<HTMLFormElement>("#rule-form");
const kind = $<HTMLSelectElement>("#rule-type");
const fields = $("#rule-fields");
const propose = $<HTMLButtonElement>("#layout-propose");
const progress = $<HTMLProgressElement>("#layout-progress");
const state = $("#layout-state");
const back = $<HTMLButtonElement>("#layout-current");
const proposalList = $("#proposal-list");

let rules: Rule[] = [];
let objects: Placed[] = [];
let components: Component[] = [];
let proposals: Proposal[] = [];
let previewed: number | null = null;

const de = (text: string) => /^[aeiouyéèêh]/i.test(text) ? `d’${text}` : `de ${text}`;
const cm = (m = 0) => `${Math.round(m * 100)} cm`;
const nameOf = (o: Placed) => objects.filter((p) => p.label === o.label).length > 1 ? `${o.label} ${o.id}` : o.label;
const objectName = (id: unknown) => { const o = objects.find((p) => p.id === Number(id)); return o ? nameOf(o) : `objet ${id} (supprimé)`; };
const targetName = (ref: Ref | undefined) => /^(meuble-)?\d+$/.test(String(ref)) ? objectName(String(ref).replace("meuble-", "")) : NEAREST[ref ?? ""] ?? components.find((c) => c.id === ref)?.label ?? String(ref);

function describe(rule: Rule) {
  const o = objectName(rule.objet);
  if (rule.type === "contre_mur") return `${o} contre ${rule.mur ? targetName(rule.mur) : "un mur"}`;
  if (rule.type === "face_a") return `${o} face à ${targetName(rule.cible)} (${(rule.distance ?? []).map(cm).join(" à ")})`;
  if (rule.type === "pres_de") return `${o} près ${de(targetName(rule.cible))} (≤ ${cm(rule.max)})`;
  if (rule.type === "loin_de") return `${o} loin ${de(targetName(rule.cible))} (≥ ${cm(rule.min)})`;
  if (rule.type === "aligne") return `${(rule.objets ?? []).map(objectName).join(", ")} alignés`;
  if (rule.type === "centre") return `${o} au centre de la pièce`;
  return typeof rule.vers === "number" ? `${o} à ${rule.vers}°` : `${o} orienté vers ${targetName(rule.vers)}`;
}

function renderRules() {
  ruleList.replaceChildren(...rules.map((rule, k) => {
    const item = el("li");
    item.append(el("span", "", describe(rule)), button("×", "remove", "Supprimer la règle", () => save(rules.filter((_, i) => i !== k))));
    return item;
  }));
  if (!rules.length) ruleList.append(el("li", "empty", "Aucune règle"));
}

function renderProposals() {
  back.hidden = previewed === null;
  proposalList.replaceChildren(...proposals.map((p, k) => {
    const item = el("li", k === previewed ? "current" : "");
    const open = button("", "item", "Afficher cette proposition dans la vue", () => preview(k));
    open.append(el("strong", "", `Proposition ${k + 1}`), el("span", `chip ${p.ok ? "ok" : "warn"}`, `score ${p.score}/100`));
    const details = el("ul");
    details.append(...p.details.map((d) => el("li", d.ok ? "ok" : "warn", d.text)));
    item.append(open, button("Appliquer", "action", "Déplacer les objets selon cette proposition", () => apply(p.index)), details);
    return item;
  }));
}

function preview(index: number | null) {
  previewed = index;
  if (state.textContent === LOCKED) state.textContent = "";
  if (index === null) run(async () => document.dispatchEvent(new CustomEvent("plans3d:furniture-refresh", { detail: await call("furniture.check") })));
  else {
    warn("meubles", []);
    document.dispatchEvent(new CustomEvent("plans3d:furniture", { detail: { id: null } }));
    showFurniture(proposals[index].shapes.map((shape) => ({ ...shape, objet: undefined })));
    highlight(null);
    setView("dessus");
  }
  renderProposals();
}

const apply = (index: number) => run(async () => {
  const report = await call("layout.apply", { index });
  proposals = [];
  previewed = null;
  state.textContent = `Proposition ${index + 1} appliquée`;
  renderProposals();
  document.dispatchEvent(new CustomEvent("plans3d:furniture-refresh", { detail: report }));
});

const save = (list: Rule[]) => run(async () => {
  rules = await call<Rule[]>("layout.rules", { rules: list });
  renderRules();
});

function select(name: string, options: [string, string][], multiple = false) {
  const control = Object.assign(el("select"), { name, multiple, size: multiple ? Math.min(6, options.length) : 0 });
  control.append(...options.map(([value, text]) => new Option(text, value)));
  return control;
}

function control(name: string) {
  const pieces = objects.map((o): [string, string] => [String(o.id), nameOf(o)]);
  if (name === "objet" || name === "objets") return select(name, pieces, name === "objets");
  if (name === "mur") return select(name, [["", "n'importe quel mur"], ...components.filter((c) => c.type === "mur").map((c): [string, string] => [c.id, c.label])]);
  if (name === "cible" || name === "vers") {
    const nearest = Object.entries(NEAREST).filter(([type]) => components.some((c) => c.type === type)).map(([type, text]): [string, string] => [type, `${text} (la plus proche)`]);
    return select(name, [...pieces, ...nearest, ...components.filter((c) => TARGETS.includes(c.type)).map((c): [string, string] => [c.id, c.label])]);
  }
  return Object.assign(el("input"), { name, type: "number", step: "1", value: String(DEFAULTS[name] ?? 0), required: true }, name === "angle" ? {} : { min: "0" });
}

function fillForm() {
  fields.replaceChildren(...FIELDS[kind.value].flatMap((name) => {
    const input = control(name);
    input.id = `rule-${name}`;
    return [Object.assign(el("label", "", LABELS[name]), { htmlFor: input.id }), input];
  }));
  const objet = fields.querySelector<HTMLSelectElement>('[name="objet"]'), angle = fields.querySelector<HTMLInputElement>('[name="angle"]');
  if (objet && angle) (objet.onchange = () => { angle.value = String(Math.round(objects.find((o) => o.id === Number(objet.value))?.rot ?? 0)); })();
}

$("#rule-add-button").onclick = () => run(async () => {
  [objects, components] = await Promise.all([call<Placed[]>("furniture.list"), call<Component[]>("components.list")]);
  if (!kind.options.length) kind.append(...Object.entries(KINDS).map(([value, text]) => new Option(text, value)));
  form.hidden = false;
  fillForm();
});
kind.onchange = fillForm;
$("#rule-cancel").onclick = () => { form.hidden = true; };
form.onsubmit = (event) => {
  event.preventDefault();
  const field = (name: string) => form.querySelector<HTMLInputElement & HTMLSelectElement>(`[name="${name}"]`)!;
  const ref = (name: string): Ref => /^\d+$/.test(field(name).value) ? Number(field(name).value) : field(name).value;
  const metres = (name: string) => Number(field(name).value) / 100;
  const rule: Rule = { type: kind.value === "angle" ? "orientation" : kind.value };
  for (const name of FIELDS[kind.value]) {
    if (name === "objet") rule.objet = Number(field(name).value);
    else if (name === "objets") rule.objets = [...field(name).selectedOptions].map((o) => Number(o.value));
    else if (name === "dmin") rule.distance = [metres("dmin"), metres("dmax")];
    else if (name === "max" || name === "min") rule[name] = metres(name);
    else if (name === "cible") rule.cible = ref(name);
    else if (name === "vers") rule.vers = typeof ref(name) === "number" ? `meuble-${ref(name)}` : ref(name);
    else if (name === "angle") rule.vers = Number(field(name).value);
    else if (name === "mur" && field(name).value) rule.mur = field(name).value;
  }
  run(async () => {
    rules = await call<Rule[]>("layout.rules", { rules: [...rules, rule] });
    form.hidden = true;
    renderRules();
  });
};

propose.onclick = () => run(async () => {
  propose.disabled = true;
  progress.hidden = false;
  progress.value = 0;
  state.textContent = "Recherche des agencements…";
  try {
    const found = await call<{ seconds: number; proposals: Proposal[] }>("layout.propose", { n: 3 });
    proposals = found.proposals;
    state.textContent = `${proposals.length} proposition${proposals.length > 1 ? "s" : ""} en ${number(found.seconds, 1)} s`;
    preview(proposals.length ? 0 : null);
  } catch (error) {
    state.textContent = "";
    throw error;
  } finally {
    propose.disabled = false;
    progress.hidden = true;
  }
});
back.onclick = () => preview(null);
document.addEventListener("plans3d:locked", () => { state.textContent = LOCKED; });

onEvent((event) => {
  if (event.event !== "progress" || event.method !== "layout.propose") return;
  progress.value = Number(event.fraction);
  state.textContent = String(event.message ?? "");
});

document.addEventListener("plans3d:opened", (event) => {
  section.hidden = !(event as CustomEvent).detail;
  form.hidden = true;
  proposals = [];
  previewed = null;
  state.textContent = "";
  renderProposals();
  if (!section.hidden) run(async () => {
    [rules, components] = await Promise.all([call<Rule[]>("layout.rules"), call<Component[]>("components.list")]);
    renderRules();
  });
});
document.addEventListener("plans3d:furniture-list", (event) => {
  objects = (event as CustomEvent<Placed[]>).detail;
  previewed = null;
  renderRules();
  renderProposals();
});
