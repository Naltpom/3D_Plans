import { call, onEvent } from "./host";
import { $, run, typing } from "./ui";
import { openProject, update } from "./projects";

const undoButton = $<HTMLButtonElement>("#undo-button");
const redoButton = $<HTMLButtonElement>("#redo-button");
const pickTools = $("#pick-tools");
const DELETABLE = ["mur", "porte", "fenetre", "porte_fenetre", "escalier", "boite", "sol", "plafond", "toit", "parcelle"];

let picked: { id: string; label: string } | null = null;

async function travel(method: string) {
  if ((await call<{ done: boolean }>(method)).done) await openProject("project.info", {});
}

async function pick(id: string | null) {
  const found = id ? (await call<{ id: string; type: string; label: string }[]>("components.list")).find((c) => c.id === id && DELETABLE.includes(c.type)) : null;
  picked = found ?? null;
  pickTools.hidden = !picked;
  $("#pick-name").textContent = picked?.label ?? "";
}

async function removePicked() {
  if (!picked) return;
  await call("components.remove", { id: picked.id });
  await pick(null);
  await update();
}

undoButton.onclick = () => run(() => travel("history.undo"));
redoButton.onclick = () => run(() => travel("history.redo"));
$("#pick-remove").onclick = () => run(removePicked);
$("#pick-close").onclick = () => run(() => pick(null));
onEvent((event) => {
  if (event.event !== "history") return;
  undoButton.disabled = !Number(event.undo);
  redoButton.disabled = !Number(event.redo);
});
document.addEventListener("plans3d:pick", (event) => run(() => pick((event as CustomEvent<string | null>).detail)));
document.addEventListener("plans3d:opened", () => run(() => pick(null)));
window.addEventListener("keydown", (event) => {
  if (typing(event.target)) return;
  const key = event.key.toLowerCase();
  if ((event.ctrlKey || event.metaKey) && key === "z" && !event.shiftKey) undoButton.click();
  else if ((event.ctrlKey || event.metaKey) && (key === "y" || key === "z" && event.shiftKey)) redoButton.click();
  else if (event.key === "Delete" && picked) run(removePicked);
  else if (event.key === "Delete" && !$("#furniture-tools").hidden) $<HTMLButtonElement>("#furniture-remove").click();
  else if (event.key === "Delete") document.dispatchEvent(new CustomEvent("plans3d:delete"));
  else return;
  event.preventDefault();
});
