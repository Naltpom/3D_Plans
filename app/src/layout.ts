import { call, onEvent } from "./host";
import { $, button, el, number, run, warn } from "./ui";
import { highlight, setView, showFurniture, type Shape } from "./view3d";

type Ref = string | number;
type Rule = { type: string; objet?: number; objets?: number[]; cible?: Ref; vers?: Ref; mur?: string; distance?: number[]; max?: number; min?: number };
type Placed = { id: number; label: string; rot: number };
type Component = { id: string; type: string; label: string };
type Proposal = { index: number; score: number; ok: boolean; details: { ok: boolean; text: string }[]; shapes: Shape[] };
type Status = { url: string; model: string; key: boolean; local: boolean; installed: boolean; server: boolean; ready: boolean };
type Rejected = { rule: unknown; reason: string };

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
const DEFAULT_MODEL = "qwen2.5:7b-instruct";
const AI_METHODS = ["ai.install", "ai.pull", "ai.rules_from_text", "ai.object_from_text"];

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
const aiForm = $<HTMLFormElement>("#ai-form");
const assistant = $("#assistant");
const aiSettings = $<HTMLFormElement>("#ai-settings");
const aiHelp = $("#ai-help");
const aiInstall = $<HTMLButtonElement>("#ai-install");
const aiProgress = $<HTMLProgressElement>("#ai-progress");
const aiState = $("#ai-state");
const aiList = $("#ai-list");
const input = (id: string) => $<HTMLInputElement>(id);

let rules: Rule[] = [];
let objects: Placed[] = [];
let components: Component[] = [];
let proposals: Proposal[] = [];
let previewed: number | null = null;
let suggested: Rule[] = [];

const plural = (n: number, text: string) => `${n} ${n > 1 ? text.replace(/(\S)(?= |$)/g, "$1s") : text}`;
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

function renderSuggestions(rejected: Rejected[] = []) {
  aiList.replaceChildren(
    ...suggested.map((rule) => {
      const item = el("li"), label = el("label", "check");
      label.append(Object.assign(el("input"), { type: "checkbox", checked: true }), describe(rule));
      item.append(label);
      return item;
    }),
    ...rejected.map((r) => {
      const item = el("li", "rejected");
      item.append(el("span", "", `Rejetée : ${r.reason}`), el("small", "", JSON.stringify(r.rule)));
      return item;
    }));
  $("#ai-actions").hidden = !suggested.length;
}

export async function aiReady() {
  const s = await call<Status>("ai.status");
  const [text, action, method] = s.ready ? ["", "", ""]
    : s.server ? [`Le modèle ${s.model} n'est pas encore sur cet ordinateur.`, `Télécharger le modèle${s.model === DEFAULT_MODEL ? " (~4,7 Go)" : ""}`, "ai.pull"]
    : !s.local ? [`Serveur IA injoignable : ${s.url}`, "", ""]
    : s.installed ? ["Ollama est installé mais ne répond pas.", "Démarrer Ollama", "ai.install"]
    : ["L'assistant utilise Ollama, une IA locale : votre description ne quitte pas cet ordinateur. Ollama n'est pas installé.", "Installer Ollama", "ai.install"];
  aiHelp.hidden = s.ready;
  aiHelp.querySelector("small")!.textContent = text;
  Object.assign(aiInstall, { textContent: action, hidden: !action });
  aiInstall.dataset.method = method;
  return s.ready;
}

export async function busy(task: () => Promise<string>, where?: Element) {
  const buttons = document.querySelectorAll<HTMLButtonElement>("[data-ai]");
  where?.after(assistant);
  buttons.forEach((b) => { b.disabled = true; });
  aiProgress.hidden = false;
  try {
    aiState.textContent = await task();
  } catch (error) {
    aiState.textContent = "";
    throw error;
  } finally {
    buttons.forEach((b) => { b.disabled = false; });
    aiProgress.hidden = true;
  }
}

aiForm.onsubmit = (event) => {
  event.preventDefault();
  run(() => busy(async () => {
    aiProgress.removeAttribute("value");
    aiState.textContent = "Vérification de l'assistant…";
    if (!(await aiReady())) return "";
    [objects, components] = await Promise.all([call<Placed[]>("furniture.list"), call<Component[]>("components.list")]);
    const found = await call<{ rules: Rule[]; rejected: Rejected[]; model: string; seconds: number }>("ai.rules_from_text", { text: $<HTMLTextAreaElement>("#ai-text").value });
    suggested = found.rules;
    renderSuggestions(found.rejected);
    return `${plural(found.rules.length, "règle proposée")} par ${found.model} en ${number(found.seconds, 1)} s${found.rejected.length ? `, ${plural(found.rejected.length, "rejetée")}` : ""}${found.rules.length ? " : cochez celles à garder" : ""}`;
  }, aiSettings));
};
aiInstall.onclick = () => run(() => busy(async () => {
  aiProgress.value = 0;
  await call(aiInstall.dataset.method!);
  return await aiReady() ? "Assistant prêt : relancez votre demande" : "";
}));
$("#ai-add").onclick = () => {
  const boxes = [...aiList.querySelectorAll<HTMLInputElement>("input")], chosen = suggested.filter((_, k) => boxes[k].checked);
  suggested = [];
  renderSuggestions();
  aiState.textContent = `${plural(chosen.length, "règle ajoutée")} au projet`;
  if (chosen.length) save([...rules, ...chosen]);
};
$("#ai-ignore").onclick = () => {
  suggested = [];
  renderSuggestions();
  aiState.textContent = "";
};
$("#ai-settings-button").onclick = () => run(async () => {
  const s = await call<Status>("ai.settings");
  input("#ai-url").value = s.url;
  input("#ai-model").value = s.model;
  Object.assign(input("#ai-key"), { value: "", placeholder: s.key ? "enregistrée (vide = inchangée)" : "aucune (serveur local)" });
  aiSettings.hidden = false;
});
$("#ai-settings-cancel").onclick = () => { aiSettings.hidden = true; };
aiSettings.onsubmit = (event) => {
  event.preventDefault();
  run(async () => {
    const key = input("#ai-key").value;
    await call("ai.settings", { url: input("#ai-url").value, model: input("#ai-model").value, ...key ? { key } : {} });
    aiSettings.hidden = aiHelp.hidden = true;
    aiState.textContent = "Réglages du modèle enregistrés";
  });
};
document.addEventListener("plans3d:locked", () => { state.textContent = LOCKED; });

onEvent((event) => {
  if (event.event === "progress" && AI_METHODS.includes(String(event.method))) {
    if (["ai.install", "ai.pull"].includes(String(event.method))) aiProgress.value = Number(event.fraction);
    aiState.textContent = String(event.message ?? "");
  }
  if (event.event !== "progress" || event.method !== "layout.propose") return;
  progress.value = Number(event.fraction);
  state.textContent = String(event.message ?? "");
});

document.addEventListener("plans3d:opened", (event) => {
  section.hidden = !(event as CustomEvent).detail;
  form.hidden = true;
  proposals = [];
  previewed = null;
  state.textContent = aiState.textContent = "";
  suggested = [];
  aiHelp.hidden = aiSettings.hidden = true;
  aiSettings.after(assistant);
  renderSuggestions();
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
