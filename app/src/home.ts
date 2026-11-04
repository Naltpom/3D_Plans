import { host } from "./host";
import { $, button, el, run } from "./ui";
import { forget, openProject, recent } from "./projects";
import { photoInfo } from "./photos";

const home = $("#home");
const list = $("#home-list");
const back = $<HTMLButtonElement>("#home-back");
const toggle = $<HTMLButtonElement>("#projects-button");

let project: { name: string; path: string } | null = null;

function when(iso?: string) {
  if (!iso) return "—";
  const date = new Date(iso), days = Math.round((new Date().setHours(0, 0, 0, 0) - new Date(iso).setHours(0, 0, 0, 0)) / 864e5);
  const time = date.toLocaleTimeString("fr-FR", { timeStyle: "short" });
  return days === 0 ? `aujourd'hui, ${time}` : days === 1 ? `hier, ${time}` : date.toLocaleDateString("fr-FR", { dateStyle: "medium" });
}

export function idle() {
  if (photoInfo().busy) throw new Error("Un calcul de photos est en cours : arrête-le ou attends la fin avant de changer de projet.");
}

export function showHome(on = true) {
  home.hidden = !on;
  document.body.classList.toggle("home", on);
  toggle.ariaPressed = String(on);
  if (on) render();
}

async function choose(path: string | null) {
  if (!path) return;
  if (path === project?.path) return showHome(false);
  idle();
  await openProject("project.open", { path });
}

function render() {
  back.hidden = !project;
  back.textContent = `Revenir à ${project?.name}`;
  list.replaceChildren(...recent().map((p) => {
    const item = el("li", p.path === project?.path ? "current" : "");
    const open = button("", "item cols", p.path, () => run(() => choose(p.path)));
    const name = el("span", "name"), title = el("strong", "", p.name);
    if (p.path === project?.path) title.append(el("span", "chip ok", "ouvert"));
    name.append(title, el("small", "", p.path));
    open.append(name, el("span", "", p.type ?? "—"), el("span", "", p.step ?? "—"), el("span", "", when(p.opened)));
    item.append(open, button("×", "remove", "Retirer de la liste (le dossier reste sur le disque)", () => { forget(p.path); render(); }));
    return item;
  }));
  if (!list.children.length) list.append(el("li", "empty", "Aucun projet récent."));
}

toggle.onclick = () => showHome(Boolean(home.hidden) || !project);
$("#home-link").onclick = () => showHome();
$("#project-name").onclick = back.onclick = () => showHome(false);
$("#home-open").onclick = () => run(async () => choose(await host.pickFolder("Ouvrir un projet (dossier contenant project.db)")));
$("#home-new").onclick = () => {
  showHome(false);
  document.dispatchEvent(new Event("plans3d:new"));
};
window.addEventListener("keydown", (event) => { if (event.key === "Escape" && !home.hidden && project) showHome(false); });
document.addEventListener("plans3d:opened", (event) => {
  project = (event as CustomEvent<{ name: string; path: string } | null>).detail;
  showHome(!project);
});
showHome();
