import { call } from "./host";
import { $, button, cm, el, number, run } from "./ui";
import { highlight, show, status, type Shape } from "./view3d";

type Field = { name: string; label: string; unit: string; default: number | string };
type Type = { type: string; label: string; options: Record<string, string>; fields: Field[] };
type Component = { id: string; type: string; label: string; params: string[] };
type Anchor = { ref: string; label: string };
type Param = { name: string; init: number; label: string; axis?: string };
type Model = { params: Record<string, { value: number; sigma: number | null }> };

const OPTIONS: Record<string, string[]> = {
  mur: ["sens"], porte: ["mur"], fenetre: ["mur"], porte_fenetre: ["mur"], escalier: ["axe", "sens", "cote", "n"], boite: ["nature"],
};
const CHOICES: Record<string, string[]> = { axe: ["x", "z"], sens: ["1", "-1"], cote: ["1", "-1"] };
const NATURES = ["meuble", "placard", "poteau", "potager", "lit"];

const section = $("#components");
const list = $("#component-list");
const info = $("#info");
const form = $<HTMLFormElement>("#component-form");
const typeSelect = $<HTMLSelectElement>("#component-type");
const idInput = $<HTMLInputElement>("#component-id");
const fields = $("#component-fields");
const natures = Object.assign(el("datalist"), { id: "box-natures" });
natures.append(...NATURES.map((n) => new Option(n, n)));
form.append(natures);

let types: Type[] = [];
let components: Component[] = [];
let shapes: Shape[] = [];
let selected: string | null = null;

const degrees = (x: number) => `${number(x)} °`;
const sigmaText = (sigma: number | null | undefined, unit = cm) => sigma === null ? "σ non déterminé" : sigma === undefined ? "" : `± ${unit(sigma)}`;
const typeLabel = (type: string) => types.find((t) => t.type === type)?.label ?? type;
const labelled = (text: string, input: HTMLInputElement | HTMLSelectElement) => [Object.assign(el("label", "", text), { htmlFor: input.id = `field-${input.name}` }), input];

async function refresh(frame = false) {
  [components, shapes, types] = await Promise.all([
    call<Component[]>("components.list"), call<Shape[]>("components.geometry"), types.length ? types : call<Type[]>("components.types")]);
  show(shapes, frame);
  await select(components.some((c) => c.id === selected) ? selected : null);
}

function renderList() {
  list.replaceChildren(...components.map((component) => {
    const sigma = shapes.find((s) => s.id === component.id)?.sigma_max;
    const item = el("li", component.id === selected ? "current" : "");
    const open = button("", "item", `${typeLabel(component.type)} · ${component.id}`, () => run(() => select(component.id)));
    open.append(el("span", "", component.label), el("span", `value ${status(sigma)}`, sigmaText(sigma)));
    item.append(open, button("×", "remove", "Supprimer le composant", () => run(async () => { await call("components.remove", { id: component.id }); await refresh(); })));
    return item;
  }));
  if (!components.length) list.append(el("li", "empty", "Aucun composant"));
}

async function select(id: string | null) {
  selected = id;
  highlight(id);
  renderList();
  const component = components.find((c) => c.id === id);
  info.hidden = !component;
  if (!component) return;
  const [anchors, params, model] = await Promise.all([
    call<Anchor[]>("components.anchors", { id }), call<Param[]>("params.list"), call<Model>("model").catch(() => null)]);
  if (selected !== id) return;
  const table = el("table");
  for (const name of component.params) {
    const solved = model?.params[name], sigma = solved ? solved.sigma : null, param = params.find((p) => p.name === name);
    const row = table.insertRow();
    row.insertCell().append(param?.label.split(" — ").pop() ?? name, " ", el("small", "", name));
    const unit = param?.axis === "angle" ? degrees : cm;
    row.insertCell().append(el("span", status(sigma), `${unit(solved?.value ?? param?.init ?? NaN)} ${sigmaText(sigma, unit)}`));
  }
  const close = button("×", "remove", "Fermer", () => run(() => select(null)));
  const anchorList = el("ul");
  anchorList.append(...anchors.map((anchor) => {
    const item = el("li");
    item.append(el("code", "", anchor.ref), " ", anchor.label);
    return item;
  }));
  info.replaceChildren(close, el("h3", "", component.label), el("small", "", `${typeLabel(component.type)} · ${component.id}`),
    component.params.length ? table : el("p", "", "Aucun paramètre libre"), el("h4", "", "Ancres"), anchorList);
}

function control(key: string, description = "") {
  const choices = key === "mur" ? components.filter((c) => c.type === "mur").map((c) => c.id) : CHOICES[key];
  if (!choices) {
    const input = Object.assign(el("input"), { name: key, placeholder: description, title: description, value: key === "nature" ? NATURES[0] : "" });
    if (key === "nature") input.setAttribute("list", natures.id);
    return input;
  }
  const dropdown = Object.assign(el("select"), { name: key, title: description });
  dropdown.append(...choices.map((choice) => new Option(choice, choice)));
  return dropdown;
}

function fillForm() {
  const type = types.find((t) => t.type === typeSelect.value)!;
  let n = 1;
  while (components.some((c) => c.id === `${type.type}${n}`)) n++;
  idInput.value = `${type.type}${n}`;
  fields.replaceChildren(
    ...(OPTIONS[type.type] ?? []).flatMap((key) => labelled(key, control(key, type.options[key]))),
    ...type.fields.flatMap((field) => labelled(field.label, Object.assign(el("input", field.unit === "m" ? "cm" : ""), {
      name: field.name, title: `valeur initiale en ${field.unit === "m" ? "cm" : field.unit}, ou expression de paramètres (en ${field.unit})`,
      placeholder: typeof field.default === "string" ? `fixé à ${field.default}` : field.unit === "m" ? cm(field.default) : `${field.default} ${field.unit}` }))));
}

function wallAxis({ x0, z0, x1, z1, ...rest }: Record<string, unknown>) {
  if (x0 !== undefined && x0 === x1) return { ...rest, axe: "z", x: x0, z0, z1 };
  if (z0 === z1) return { ...rest, axe: "x", z: z0, x0, x1 };
  return { ...rest, x0, z0, x1, z1 };
}

$("#add-component-button").onclick = () => {
  if (!typeSelect.options.length) typeSelect.append(...types.map((t) => new Option(t.label, t.type)));
  form.hidden = false;
  fillForm();
  form.scrollIntoView({ block: "nearest" });
};
typeSelect.onchange = fillForm;
$("#component-cancel").onclick = () => { form.hidden = true; };
form.onsubmit = (event) => {
  event.preventDefault();
  run(async () => {
    const options: Record<string, unknown> = {};
    for (const input of fields.querySelectorAll<HTMLInputElement | HTMLSelectElement>("[name]")) {
      const text = input.value.trim(), number = Number(text.replace(",", "."));
      if (text) options[input.name] = Number.isFinite(number) ? number / (input.classList.contains("cm") ? 100 : 1) : text;
    }
    const id = idInput.value.trim();
    await call("components.add", { type: typeSelect.value, id, options: typeSelect.value === "mur" ? wallAxis(options) : options });
    form.hidden = true;
    selected = id;
    await refresh();
  });
};

document.addEventListener("plans3d:opened", (event) => {
  const opened = Boolean((event as CustomEvent).detail);
  section.hidden = !opened;
  form.hidden = true;
  selected = null;
  if (opened) run(() => refresh(true));
});
document.addEventListener("plans3d:updated", () => run(() => refresh()));
document.addEventListener("plans3d:pick", (event) => run(() => select((event as CustomEvent<string | null>).detail)));
