import { call } from "./host";
import { $, button, cm, el, message, number, run, warn } from "./ui";

type Obs = { id: string; expr: string; value: number; sigma: number; status: string; label: string; resid: number; w: number; source: string; unit: string };
type Cote = { id: string; expr: string; value: number; sigma: number; unit: string; status: string; label: string };
type Conflict = { id: string; w: number; label: string }[];
type Result = {
  params: { name: string; value: number; label: string }[]; obs: Obs[]; conflicts: Conflict[];
  warnings?: { code: string; param?: string; component?: string; meuble?: number; message: string }[];
};
type Candidate = { gain: number; label: string; expr: string; sigma: number };
type Factor = { group: Record<string, string>; factor: number | null; n: number };
type Component = { id: string; label: string; params: string[] };
type Anchor = { ref: string; label: string };

const panel = $("#analysis");
const coteList = $("#cote-list");
const conflictList = $("#conflict-list");
const nextList = $("#next-list");
const factorList = $("#factor-list");
const hint = $("#cote-hint");
const form = $<HTMLFormElement>("#cote-form");
const [compA, anchorA, compB, anchorB, axis] = ["comp-a", "a", "comp-b", "b", "axis"].map((n) => form.elements.namedItem(n) as HTMLSelectElement);
const [exprInput, valueInput, sigmaInput, labelInput] = ["expr", "value", "sigma", "label"].map((n) => form.elements.namedItem(n) as HTMLInputElement);
const unitWarning = Object.assign(el("small", "large"), { hidden: true });
hint.after(unitWarning);

let result: Result | null = null;
let cotes: Cote[] = [];
let candidates: Candidate[] = [];
let factors: Factor[] = [];
let focused: Conflict | null = null;
let slot = 0;
let modelValue: number | null = null;

const decimal = (text: string) => Number(text.trim().replace(",", "."));

function fill(list: HTMLElement, items: HTMLElement[], empty: string) {
  list.replaceChildren(...items);
  if (!items.length) list.append(el("li", "empty", empty));
}

function line(title: string, detail: string, className = "", action?: HTMLButtonElement, tip = "", value?: HTMLElement) {
  const item = Object.assign(el("li", className), { title: tip });
  item.append(el("strong", "", title), ...(value ? [value] : []), ...(action ? [action] : []), el("small", "", detail));
  return item;
}

function amount(value: string, sigma = "") {
  const span = el("span", "value", value);
  if (sigma) span.append(el("span", "muted", ` ± ${sigma}`));
  return span;
}

function removable(id: string) {
  const cote = /^cotes:(.+)$/.exec(id)?.[1];
  return cote ? button("×", "remove", "Supprimer la cote", () => run(async () => {
    await call("cotes.remove", { id: cote });
    await loadCotes();
  })) : undefined;
}

function renderCotes() {
  const kept = (o: Obs) => o.source !== "cotes" || cotes.some((c) => o.id === `cotes:${c.id}`);
  const obs = (result?.obs ?? []).filter(kept).sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
  const flagged = new Set(result?.conflicts.flat().map((c) => c.id));
  const marked = new Set(focused?.map((c) => c.id));
  fill(coteList, [
    ...obs.map((o) => line(o.label || o.id,
      `${o.status} · ${o.source} · écart ${o.resid > 0 ? "+" : ""}${o.unit === "°" ? `${number(o.resid)} °` : cm(o.resid)} (w ${number(o.w)})`,
      `${flagged.has(o.id) ? "conflict" : ""} ${marked.has(o.id) ? "current" : ""}`, removable(o.id), `${o.id} = ${o.expr}`,
      o.unit === "°" ? amount(number(o.value), `${number(o.sigma)} °`) : amount(number(o.value * 100), `${number(o.sigma * 100)} cm`))),
    ...cotes.filter((c) => !obs.some((o) => o.id === `cotes:${c.id}`)).map((c) => line(c.label || c.id,
      `${c.status} · à intégrer : mettre à jour`, "pending", removable(`cotes:${c.id}`), `cotes:${c.id} = ${c.expr}`,
      amount(number(c.value), `${number(c.sigma)} ${c.unit}`))),
  ], "Aucune cote");
  coteList.querySelector(".current")?.scrollIntoView({ block: "nearest" });
}

function renderConflicts() {
  const groups = result?.conflicts ?? [];
  $("#conflicts").hidden = !groups.length;
  conflictList.replaceChildren(...groups.map((group) => {
    const item = el("li", group === focused ? "current" : "");
    item.append(...group.map((c) => {
      const row = Object.assign(el("div"), { title: c.id });
      row.append(el("span", "", c.label || c.id), amount(`w ${c.w > 0 ? "+" : ""}${number(c.w)}`));
      return row;
    }), Object.assign(button("Voir", "small", "Mettre en évidence dans la vue 3D", () => run(() => focus(group === focused ? null : group))),
      { ariaPressed: String(group === focused) }));
    return item;
  }));
}

function render() {
  const params = new Map(result?.params.map((p) => [p.name, p.label || p.name]));
  warn("calcul", (result?.warnings ?? []).filter((w) => w.meuble === undefined)
    .map((w) => ({ ...w, message: w.param ? `${w.message} (${params.get(w.param) ?? w.param})` : w.message })));
  const idle = result ? "Aucune proposition" : "Mettez à jour pour obtenir des propositions";
  fill(nextList, candidates.map((c) => line(c.label, `marge actuelle ± ${cm(c.sigma)} · gain total ${cm(c.gain)}`, "",
    button("J'ai mesuré", "action", c.expr, () => run(() => prefill(c))))), idle);
  fill(factorList, factors.map((f) => line(Object.values(f.group).join(" · "),
    f.factor === null ? `${f.n} obs. · redondance insuffisante` : `σ réel / σ annoncé · ${f.n} obs.`,
    f.factor !== null && f.factor > 1.5 ? "large" : "", undefined, "", amount(f.factor === null ? "—" : `${number(f.factor, 2)} × annoncé`))),
    result ? "Aucun groupe" : "Mettez à jour pour estimer");
  renderConflicts();
  renderCotes();
}

async function focus(group: Conflict | null) {
  focused = group;
  render();
  const ids = new Set(group?.map((c) => c.id));
  const names = new Set(result?.obs.filter((o) => ids.has(o.id)).flatMap((o) => o.expr.match(/[A-Za-z_]\w*/g) ?? []));
  const components = group ? await call<Component[]>("components.list") : [];
  document.dispatchEvent(new CustomEvent("plans3d:highlight", { detail: components.filter((c) => c.params.some((p) => names.has(p))).map((c) => c.id) }));
}

async function loadCotes() {
  cotes = await call<Cote[]>("cotes.list");
  renderCotes();
}

async function openForm() {
  const components = await call<Component[]>("components.list");
  form.reset();
  checkUnit(null);
  for (const select of [compA, compB]) select.replaceChildren(new Option("composant…", ""), ...components.map((c) => new Option(c.label, c.id)));
  anchorA.replaceChildren();
  anchorB.replaceChildren();
  slot = 0;
  hint.textContent = "Cliquez un composant dans la vue pour « De », puis pour « À ».";
  form.hidden = false;
  form.scrollIntoView({ block: "nearest" });
}

async function choose(index: number, id: string) {
  slot = 1 - index;
  const [comp, anchor] = index ? [compB, anchorB] : [compA, anchorA];
  comp.value = id;
  const anchors = id ? await call<Anchor[]>("components.anchors", { id }) : [];
  anchor.replaceChildren(...anchors.map((a) => new Option(a.label, a.ref)));
  await measure();
}

async function measure() {
  if (!anchorA.value || !anchorB.value) return;
  try {
    const measured = await call<{ expr: string; value: number; label: string }>("components.measure_expr", { a: anchorA.value, b: anchorB.value, axis: axis.value });
    exprInput.value = measured.expr;
    labelInput.value = measured.label;
    hint.textContent = `Modèle actuel : ${cm(measured.value)}`;
    checkUnit(measured.value);
  } catch (error) {
    exprInput.value = "";
    hint.textContent = message(error);
    checkUnit(null);
  }
}

function checkUnit(model = modelValue) {
  modelValue = model;
  const factor = Math.abs(Math.log10(Math.abs(decimal(valueInput.value) / 100 / (model ?? NaN))));
  unitWarning.hidden = !(Number.isFinite(factor) && factor > 1);
  unitWarning.textContent = `Unité ? Le modèle donne ${cm(Math.abs(model ?? 0))}, soit un facteur ${number(10 ** factor, 0)}.`;
}

async function prefill(candidate: Candidate) {
  await openForm();
  const values = new Map(result?.params.map((p) => [p.name, p.value]));
  const [a, b] = candidate.expr.split(" - ");
  const model = values.has(a) && (b === undefined || values.has(b)) ? values.get(a)! - (b === undefined ? 0 : values.get(b)!) : null;
  exprInput.value = b !== undefined && model !== null && model < 0 ? `${b} - ${a}` : candidate.expr;
  labelInput.value = candidate.label;
  hint.textContent = `${model === null ? "Marge actuelle :" : `Modèle actuel : ${cm(Math.abs(model))}`} ± ${cm(candidate.sigma)}`;
  checkUnit(model);
  valueInput.focus();
}

compA.onchange = () => run(() => choose(0, compA.value));
compB.onchange = () => run(() => choose(1, compB.value));
anchorA.onchange = anchorB.onchange = axis.onchange = () => run(measure);
valueInput.oninput = () => checkUnit();
$("#add-cote-button").onclick = () => run(openForm);
$("#cote-cancel").onclick = () => { form.hidden = true; };
form.onsubmit = (event) => {
  event.preventDefault();
  run(async () => {
    let n = 1;
    while (cotes.some((c) => c.id === `m${n}`)) n++;
    await call("cotes.add", { id: `m${n}`, expr: exprInput.value.trim(), value: decimal(valueInput.value), sigma: decimal(sigmaInput.value),
      unit: "cm", label: labelInput.value.trim() });
    form.hidden = true;
    await loadCotes();
  });
};

document.addEventListener("plans3d:opened", (event) => {
  panel.hidden = !(event as CustomEvent).detail;
  [result, focused, candidates, factors, cotes] = [null, null, [], [], []];
  form.hidden = true;
  render();
  if (!panel.hidden) run(loadCotes);
});
document.addEventListener("plans3d:updated", (event) => run(async () => {
  [result, focused] = [(event as CustomEvent<Result>).detail, null];
  [cotes, candidates, factors] = await Promise.all([call<Cote[]>("cotes.list"), call<Candidate[]>("next_best", { k: 5 }), call<Factor[]>("variance_factors")]);
  render();
}));
document.addEventListener("plans3d:conflicts", () => run(async () => {
  $("#conflicts").scrollIntoView({ block: "nearest" });
  if (result?.conflicts.length) await focus(result.conflicts[0]);
}));
document.addEventListener("plans3d:pick", (event) => {
  const id = (event as CustomEvent<string | null>).detail;
  if (!form.hidden && id) run(() => choose(slot, id));
});
