import { call } from "./host";
import { $, button, el, number, run } from "./ui";
import { refreshSources } from "./projects";

type Group = { id: string; name: string; start: string; end: string; count: number; excluded: boolean; color: string; thumbs: string[] };
type Groups = { gap_min: number; groups: Group[] };
type Mode = "photo" | "groupe" | "mobilite";

const loaded = new Map<string, Groups>();
const DAYS = ["dim.", "lun.", "mar.", "mer.", "jeu.", "ven.", "sam."];
const hour = (iso: string) => iso.slice(11, 16);
const day = (iso: string) => `${DAYS[new Date(iso).getDay()]} ${iso.slice(8, 10)}/${iso.slice(5, 7)}`;
let mode: Mode = "photo", source = "";

export const cloudMode = () => mode;

function changed(id: string, group: string, excluded: boolean) {
  return run(async () => {
    await call("photos.set_group", { source: id, id: group, excluded });
    await refreshSources();
    document.dispatchEvent(new CustomEvent("plans3d:cloud"));
  });
}

function row(id: string, group: Group, compact: boolean) {
  const item = el("li", group.excluded ? "session off" : "session");
  const name = Object.assign(el("input", "name"), { value: group.name, ariaLabel: "Nom de la séance" });
  name.onchange = () => run(async () => { await call("photos.set_group", { source: id, id: group.id, name: name.value }); await load(id); });
  const used = Object.assign(el("input"), { type: "checkbox", checked: !group.excluded, ariaLabel: "Utiliser cette séance pour le calcul" });
  used.onchange = () => changed(id, group.id, !used.checked);
  const check = el("label", "check");
  check.append(used, compact ? "" : group.excluded ? "Exclue du calcul" : "Utilisée");
  const dot = el("span", "dot");
  dot.style.background = group.excluded ? "var(--strong)" : group.color;
  const span = hour(group.start) === hour(group.end) ? hour(group.start) : `${hour(group.start)} → ${hour(group.end)}`;
  const thumbs = el("span", "thumbs");
  thumbs.ariaHidden = "true";
  thumbs.append(...group.thumbs.map((src) => Object.assign(el("img"), { src, alt: "" })));
  item.append(dot, name, check, el("small", "", `${day(group.start)} · ${span} · ${group.count} photo${group.count > 1 ? "s" : ""}`), thumbs);
  return item;
}

function fill(box: HTMLElement, id: string, groups: Groups, compact: boolean) {
  const excluded = groups.groups.filter((g) => g.excluded).length, photos = groups.groups.reduce((n, g) => n + g.count, 0);
  const gap = Object.assign(el("input"), { type: "number", min: "1", step: "1", value: String(groups.gap_min), ariaLabel: "Écart entre deux séances (min)" });
  gap.onchange = () => run(async () => { await call("photos.set_gap", { source: id, minutes: Number(gap.value) }); await load(id); document.dispatchEvent(new CustomEvent("plans3d:cloud")); });
  const setting = el("label", "gap");
  setting.append(compact ? "écart" : "Nouvelle séance après", gap, compact ? "min" : "min sans photo");
  const head = el("div", "head");
  head.append(el(compact ? "h2" : "strong", "", compact ? "Séances" : "Séances de photos"),
    ...compact ? [] : [el("small", "", `${photos} photos · ${groups.groups.length} séance${groups.groups.length > 1 ? "s" : ""}${excluded ? ` · ${excluded} exclue${excluded > 1 ? "s" : ""}` : ""}`)], setting);
  const list = el("ul", "list sessions");
  list.append(...groups.groups.map((group) => row(id, group, compact)));
  box.replaceChildren(head, list, el("small", "", "Décocher une séance l'exclut du calcul : les positions des photos sont recalculées sans elle."));
}

async function load(id: string) {
  loaded.set(id, await call<Groups>("photos.groups", { source: id }));
  for (const box of document.querySelectorAll<HTMLElement>(`[data-sessions="${CSS.escape(id)}"]`)) fill(box, id, loaded.get(id)!, box.classList.contains("compact"));
  void run(legend);
}

export function sessions(id: string, compact = false) {
  const box = el("div", compact ? "sessions compact" : "sessions card plain");
  box.dataset.sessions = id;
  const known = loaded.get(id);
  if (known) fill(box, id, known, compact);
  else box.append(el("small", "", "Lecture des heures de prise de vue…"));
  void run(() => load(id));
  return box;
}

async function legend() {
  const box = $("#cloud-legend"), groups = loaded.get(source)?.groups ?? [];
  box.hidden = mode === "photo" || $("#cloud-colors").hidden;
  if (box.hidden) return;
  const key = (color: string, text: string) => { const item = el("span", "", text); item.style.setProperty("--key", color); return item; };
  if (mode === "groupe") return box.replaceChildren(...groups.filter((g) => !g.excluded).map((g) => key(g.color, g.name)), key("#eceef1", "plusieurs séances"));
  const stats = await call<{ mobile_share: number }>("photos.mobility", { source }).catch(() => null);
  box.replaceChildren(key("#8b95a3", "fixe : vu pareil par toutes les séances"), key("#e8953a", `mobile${stats ? ` (${number(stats.mobile_share * 100, 0)} %)` : ""} : présent dans une séance, vu vide dans une autre`));
}

export function showColors(id: string | null, visible: boolean) {
  if (id) source = id;
  $("#cloud-colors").hidden = !visible;
  void run(() => id && !loaded.has(id) ? load(id) : legend());
}

function paint() {
  for (const b of document.querySelectorAll<HTMLButtonElement>("#cloud-colors button")) b.ariaPressed = String(b.dataset.color === mode);
}

for (const [value, text] of [["photo", "Photo"], ["groupe", "Séance"], ["mobilite", "Fixe / mobile"]] as const) {
  const b = button(text, "small", `Couleur du nuage : ${text.toLowerCase()}`, () => { mode = value; paint(); document.dispatchEvent(new CustomEvent("plans3d:cloud")); });
  b.dataset.color = value;
  $("#cloud-colors").append(b);
}
paint();
