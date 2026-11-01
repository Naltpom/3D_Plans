import { call, host, onEvent } from "./host";
import { $, el, number, report, run as runTask } from "./ui";

type Project = { name: string; path: string };
type Source = { id: string; path: string; kind: string; extracted: boolean; obs: number; photos?: number; registered?: number | null; state?: string; error?: string };
type UpdateResult = { chi2: number; dof: number; params: { sigma: number | null }[]; conflicts: unknown[][] };

const RECENT = "plans3d.recent";
const KINDS: Record<string, string> = { cotes_csv: "Cotes", photos_colmap: "Photos", cadastre_pdf: "Cadastre PDF", cadastre_geojson: "Cadastre GeoJSON", grille_alti: "Terrain" };
const projectList = $("#project-list");
const sourceList = $("#source-list");
const sourcesPanel = $("#sources");
const newForm = $<HTMLFormElement>("#new-project-form");
const nameInput = $<HTMLInputElement>("#new-project-name");
const updateButton = $<HTMLButtonElement>("#update-button");
const progress = $<HTMLProgressElement>("#progress");
const status = $("#status");
const banner = $("#banner");

let current: Project | null = null;
let updating = false;
let parent = "";

const recent = (): Project[] => JSON.parse(localStorage.getItem(RECENT) ?? "[]");
const saveRecent = (list: Project[]) => localStorage.setItem(RECENT, JSON.stringify(list.slice(0, 12)));
const plural = (n: number, word: string) => `${n} ${word}${n > 1 ? "s" : ""}`;
const target = () => `${parent.replace(/[\\/]+$/, "")}\\${nameInput.value.trim().replace(/[<>:"/\\|?*]+/g, "_")}`;

const run = (task: () => unknown) => {
  banner.hidden = true;
  return runTask(task);
};

function render() {
  projectList.replaceChildren(...recent().map((project) => {
    const item = el("li", project.path === current?.path ? "current" : "");
    const open = el("button", "item");
    open.append(el("strong", "", project.name), el("small", "", project.path));
    open.title = project.path;
    open.onclick = () => run(() => openProject("project.open", { path: project.path }));
    const forget = el("button", "remove", "×");
    forget.title = "Retirer de la liste";
    forget.onclick = () => { saveRecent(recent().filter((p) => p.path !== project.path)); render(); };
    item.append(open, forget);
    return item;
  }));
  if (!projectList.children.length) projectList.append(el("li", "empty", "Aucun projet"));
  updateButton.disabled = !current || updating;
  sourcesPanel.hidden = !current;
  $("#project-name").textContent = current?.name ?? "";
}

async function openProject(method: string, params: Record<string, unknown>) {
  const info = await call<Project | null>(method, params);
  current = info && { name: info.name, path: info.path };
  if (current) saveRecent([current, ...recent().filter((p) => p.path !== current!.path)]);
  status.textContent = "";
  render();
  document.dispatchEvent(new CustomEvent("plans3d:opened", { detail: current }));
  await refreshSources();
}

async function refreshSources() {
  if (!current) return;
  const sources = await call<Source[]>("sources.list");
  sourceList.replaceChildren(...sources.map((source) => {
    const item = el("li");
    const remove = el("button", "remove", "×");
    remove.title = "Supprimer la source";
    remove.onclick = () => run(async () => { await call("sources.remove", { id: source.id }); await refreshSources(); });
    const details = source.photos == null
      ? [KINDS[source.kind] ?? source.kind, plural(source.obs, "élément"), source.extracted ? "extraite" : "à extraire"]
      : [plural(source.photos, "photo"), source.registered != null && `${source.registered} positionnée${source.registered > 1 ? "s" : ""}`, source.state];
    item.title = source.error ?? source.path;
    item.append(el("strong", "", source.id), remove, el("small", "", details.filter(Boolean).join(" · ")));
    return item;
  }));
  if (!sources.length) sourceList.append(el("li", "empty", "Aucune source"));
}

async function addSources(paths: string[]) {
  if (!current) return report("Ouvrez ou créez un projet avant d'ajouter des sources.");
  const errors: string[] = [];
  for (const path of paths) await call("sources.add_auto", { path }).catch((error: Error) => errors.push(error.message));
  await refreshSources();
  if (errors.length) report(errors.join("\n"));
}

function summarize(result: UpdateResult, seconds: number) {
  const undetermined = result.params.filter((p) => p.sigma == null || !Number.isFinite(p.sigma)).length;
  const conflicts = el("button", "link", result.conflicts.length ? plural(result.conflicts.length, "conflit") : "aucun conflit");
  conflicts.onclick = () => document.dispatchEvent(new CustomEvent("plans3d:conflicts", { detail: result.conflicts }));
  status.replaceChildren(
    `Mis à jour en ${number(seconds, 1)} s · χ²/ddl ${result.dof > 0 ? number(result.chi2 / result.dof, 2) : "—"} · `, conflicts,
    undetermined ? ` · ${plural(undetermined, "paramètre")} non déterminé${undetermined > 1 ? "s" : ""}` : "");
  document.dispatchEvent(new CustomEvent("plans3d:updated", { detail: result }));
}

onEvent((event) => {
  if (event.event === "exit") report(`Le moteur s'est arrêté${event.message ? ` : ${event.message}` : ""}.`);
  if (event.event !== "progress" || event.method !== "update") return;
  progress.value = Number(event.fraction);
  status.textContent = String(event.message ?? "");
});

updateButton.onclick = () => run(async () => {
  updating = true;
  render();
  progress.hidden = false;
  progress.removeAttribute("value");
  status.textContent = "Mise à jour…";
  const start = performance.now();
  const polling = setInterval(() => refreshSources().catch(() => {}), 2000);
  try {
    summarize(await call<UpdateResult>("update"), (performance.now() - start) / 1000);
  } catch (error) {
    status.textContent = "Échec de la mise à jour";
    throw error;
  } finally {
    clearInterval(polling);
    updating = false;
    progress.hidden = true;
    render();
    await refreshSources().catch(() => {});
  }
});

$("#new-project-button").onclick = () => run(async () => {
  parent ||= `${await host.documents()}\\3D_Plans`;
  newForm.hidden = false;
  nameInput.focus();
  nameInput.dispatchEvent(new Event("input"));
});
nameInput.oninput = () => { $("#new-project-path").textContent = target(); };
$("#parent-button").onclick = () => run(async () => {
  parent = (await host.pickFolder("Dossier parent du projet")) ?? parent;
  $("#new-project-path").textContent = target();
});
$("#cancel-button").onclick = () => { newForm.hidden = true; newForm.reset(); };
newForm.onsubmit = (event) => {
  event.preventDefault();
  run(async () => {
    await openProject("project.create", { path: target(), name: nameInput.value.trim() });
    newForm.hidden = true;
    newForm.reset();
  });
};
$("#open-project-button").onclick = () => run(async () => {
  const path = await host.pickFolder("Ouvrir un projet (dossier contenant project.db)");
  if (path) await openProject("project.open", { path });
});
$("#add-source-button").onclick = () => run(async () => addSources((await host.pickFiles("Ajouter des sources")) ?? []));
$("#banner button").onclick = () => { banner.hidden = true; };
host.onDrop((active) => document.body.classList.toggle("dropping", active), (paths) => run(() => addSources(paths)));

render();
run(() => openProject("project.info", {}));
