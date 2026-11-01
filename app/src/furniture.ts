import { call } from "./host";
import { $, button, el, number, run, warn } from "./ui";
import { focusPoint, highlight, showFurniture, type Shape } from "./view3d";

type Item = { id: string; label: string; kind: string; size: number[]; source: string };
type Placed = { id: number; ref: string; label: string; kind: string; size: number[]; x: number; z: number; y: number; rot: number };
type Warning = { code: string; message: string; component?: string; meuble?: number };
type Report = { object: Placed | null; warnings: Warning[] };

const SOURCES: Record<string, string> = { gabarit: "gabarit", ikea: "IKEA", perso: "perso" };
const section = $("#gallery");
const results = $("#gallery-list");
const placedList = $("#furniture-list");
const search = $<HTMLInputElement>("#gallery-search");
const kind = $<HTMLSelectElement>("#gallery-kind");
const count = $("#gallery-count");
const tools = $("#furniture-tools");
const angle = $<HTMLInputElement>("#furniture-angle");
const height = $<HTMLInputElement>("#furniture-y");

let placed: Placed[] = [];
let selected: number | null = null;
let query = 0;
let queue: Promise<unknown> = Promise.resolve();

const dims = (size: number[]) => `${[size[0], size[2], size[1]].map((v) => number(v * 100, 0)).join(" × ")} cm`;
const current = () => placed.find((p) => p.id === selected);

async function refresh(report?: Report) {
  const [list, shapes] = await Promise.all([call<Placed[]>("furniture.list"), call<Shape[]>("furniture.geometry")]);
  placed = list;
  showFurniture(shapes);
  document.dispatchEvent(new CustomEvent("plans3d:furniture-list", { detail: placed }));
  if (report) warn("meubles", report.warnings);
  select(current()?.id ?? null);
}

const act = (method: string, params: Record<string, unknown>) => queue = queue.then(() => run(async () => {
  const report = await call<Report>(method, params);
  if (report.object) selected = report.object.id;
  await refresh(report);
}));

function select(id: number | null) {
  selected = id;
  highlight(current()?.ref ?? null);
  render();
}

function render() {
  const obj = current();
  tools.hidden = !obj;
  if (obj) {
    $("#furniture-name").textContent = obj.label;
    angle.value = String(Math.round(obj.rot));
    height.value = String(Math.round(obj.y * 100));
  }
  placedList.replaceChildren(...placed.map((p) => {
    const item = el("li", p.id === selected ? "current" : "");
    const open = button("", "item", p.label, () => select(p.id));
    open.append(el("span", "", p.label), el("small", "", dims(p.size)));
    item.append(open, button("×", "remove", "Supprimer l'objet", () => act("furniture.remove", { id: p.id })));
    return item;
  }));
  if (!placed.length) placedList.append(el("li", "empty", "Aucun objet"));
}

async function find() {
  const ticket = ++query;
  const found = await call<{ total: number; items: Item[] }>("catalog.search", { q: search.value, kind: kind.value || null, limit: 60 });
  if (ticket !== query) return;
  count.textContent = found.total > found.items.length ? `${found.items.length} sur ${found.total} objets` : `${found.total} objet${found.total > 1 ? "s" : ""}`;
  results.replaceChildren(...found.items.map((entry) => {
    const item = Object.assign(el("li"), { title: entry.label });
    item.append(el("strong", "", entry.label), el("small", "", `${dims(entry.size)} · ${SOURCES[entry.source] ?? entry.source}`),
      button("Ajouter", "action", "Placer cet objet au centre de la vue", () => act("furniture.add", { item: entry.id, ...focusPoint() })));
    return item;
  }));
  if (!found.items.length) results.append(el("li", "empty", "Aucun objet"));
}

async function loadKinds() {
  const kinds = await call<{ kind: string; label: string; count: number }[]>("catalog.kinds");
  const chosen = kind.value;
  kind.replaceChildren(new Option("Tous les types", ""), ...kinds.filter((k) => k.count).map((k) => new Option(`${k.label} (${k.count})`, k.kind)));
  kind.value = chosen;
  await find();
}

const turn = (rot: number) => {
  if (selected === null || !Number.isFinite(rot)) return;
  angle.value = String(rot);
  act("furniture.rotate", { id: selected, rot });
};

$("#ikea-button").onclick = () => run(async () => {
  count.textContent = "Téléchargement du catalogue IKEA…";
  const done = await call<{ count: number }>("catalog.import_ikea");
  await loadKinds();
  count.textContent = `Catalogue IKEA importé : ${done.count} objets`;
});
search.oninput = () => run(find);
kind.onchange = () => run(find);
tools.querySelectorAll<HTMLButtonElement>("[data-turn]").forEach((b) => { b.onclick = () => turn(angle.valueAsNumber + Number(b.dataset.turn)); });
angle.onchange = () => turn(angle.valueAsNumber);
height.onchange = () => {
  const obj = current();
  if (obj && Number.isFinite(height.valueAsNumber)) act("furniture.move", { id: obj.id, x: obj.x, z: obj.z, y: height.valueAsNumber / 100, snap_walls: false });
};
$("#furniture-remove").onclick = () => selected !== null && act("furniture.remove", { id: selected });
$("#furniture-close").onclick = () => select(null);

document.addEventListener("plans3d:opened", (event) => {
  section.hidden = !(event as CustomEvent).detail;
  placed = [];
  select(null);
  warn("meubles", []);
  if (section.hidden) showFurniture([]);
  else run(async () => { await act("furniture.check", {}); await loadKinds(); });
});
document.addEventListener("plans3d:updated", (event) => {
  warn("meubles", ((event as CustomEvent).detail.warnings as Warning[] ?? []).filter((w) => w.meuble !== undefined));
});
document.addEventListener("plans3d:pick", () => { selected = null; render(); });
document.addEventListener("plans3d:furniture-refresh", (event) => run(() => refresh((event as CustomEvent<Report | undefined>).detail)));
document.addEventListener("plans3d:furniture", (event) => select(placed.find((p) => p.ref === (event as CustomEvent).detail.id)?.id ?? null));
document.addEventListener("plans3d:furniture-move", (event) => {
  const { id, dx, dz } = (event as CustomEvent<{ id: string; dx: number; dz: number }>).detail, obj = placed.find((p) => p.ref === id);
  if (obj) act("furniture.move", { id: obj.id, x: obj.x + dx, z: obj.z + dz });
});
