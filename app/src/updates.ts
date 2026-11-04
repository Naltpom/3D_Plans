import { getVersion } from "@tauri-apps/api/app";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { relaunch } from "@tauri-apps/plugin-process";
import { $, message, report } from "./ui";

type Update = { version: string; notes: string };

const dialog = $<HTMLDialogElement>("#update");
const about = $<HTMLDialogElement>("#about");
const bar = $<HTMLProgressElement>("#update-progress");
const state = $("#update-state");
const choices = [$<HTMLButtonElement>("#update-later"), $<HTMLButtonElement>("#update-now"), $<HTMLButtonElement>("#update-close")];
const REPO = "https://github.com/Naltpom/3D_Plans";

let found: Update | null = null;
let current = "";
let installing = false;

function offer() {
  $("#update-title").textContent = `Version ${found!.version}`;
  $("#update-current").textContent = `Tu utilises la version ${current}.`;
  $("#update-notes").textContent = found!.notes;
  $("#update-notes").hidden = !found!.notes;
  state.textContent = "Téléchargement, installation, puis l'appli se relance seule. Ton projet reste tel quel.";
  bar.hidden = true;
  for (const choice of choices) choice.hidden = false;
  if (!dialog.open) dialog.showModal();
}

async function check(manual: boolean) {
  try {
    found = await invoke<Update | null>("update_check");
  } catch (error) {
    if (manual) report(`Recherche de mise à jour impossible : ${message(error)}`);
    return;
  }
  if (found) {
    about.close();
    offer();
  } else if (manual) $("#about-state").textContent = "3D_Plans est à jour.";
}

async function install() {
  installing = true;
  for (const choice of choices) choice.hidden = true;
  bar.hidden = false;
  bar.removeAttribute("value");
  state.textContent = `Téléchargement de la version ${found!.version}…`;
  const mb = (bytes: number) => (bytes / 1e6).toLocaleString("fr-FR", { maximumFractionDigits: 0 });
  const stop = await listen<{ done: number; total: number | null }>("update-progress", ({ payload }) => {
    if (!payload.total) return;
    bar.value = payload.done / payload.total;
    state.textContent = `Téléchargement : ${Math.round(bar.value * 100)} % · ${mb(payload.done)} / ${mb(payload.total)} Mo`;
  });
  try {
    await invoke("update_install");
    state.textContent = "Relance…";
    await relaunch();
  } catch (error) {
    report(`Mise à jour impossible : ${message(error)}`);
    offer();
  } finally {
    installing = false;
    stop();
  }
}

dialog.addEventListener("cancel", (event) => { if (installing) event.preventDefault(); });
$("#update-later").onclick = $("#update-close").onclick = () => dialog.close();
$("#update-now").onclick = () => void install();

getVersion().then((version) => {
  current = version;
  $("#version").textContent = `v${version}`;
  $("#about-version").textContent = `Version ${version}`;
  Object.assign($<HTMLAnchorElement>("#about-notes"), { href: `${REPO}/releases/tag/v${version}` });
  Object.assign($<HTMLAnchorElement>("#about-notice"), { href: `${REPO}/blob/main/NOTICE` });
});
$("#about-button").onclick = () => about.showModal();
$("#about-close").onclick = () => about.close();
$("#about-check").onclick = () => { $("#about-state").textContent = "Recherche…"; void check(true); };
setTimeout(() => void check(false), 3000);
