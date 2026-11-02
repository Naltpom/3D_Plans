import { getVersion } from "@tauri-apps/api/app";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { relaunch } from "@tauri-apps/plugin-process";
import { $, button, el, message, report } from "./ui";

type Update = { version: string; notes: string };

const banner = $("#update");
const about = $<HTMLDialogElement>("#about");
const REPO = "https://github.com/Naltpom/3D_Plans";

let found: Update | null = null;

function offer() {
  banner.hidden = false;
  banner.replaceChildren(el("span", "", `Version ${found!.version} disponible`), button("Mettre à jour", "primary small", "Télécharger, installer et relancer", () => void install()),
    button("×", "remove", "Plus tard", () => { banner.hidden = true; }));
}

async function check(manual: boolean) {
  try {
    found = await invoke<Update | null>("update_check");
  } catch (error) {
    if (manual) report(`Recherche de mise à jour impossible : ${message(error)}`);
    return;
  }
  if (found) offer();
  else if (manual) $("#about-state").textContent = "3D_Plans est à jour.";
}

async function install() {
  const bar = Object.assign(el("progress"), { max: 1, ariaLabel: "Téléchargement de la mise à jour" });
  banner.replaceChildren(el("span", "", `Téléchargement de la version ${found!.version}…`), bar);
  const stop = await listen<{ done: number; total: number | null }>("update-progress", ({ payload }) => { if (payload.total) bar.value = payload.done / payload.total; });
  try {
    await invoke("update_install");
    await relaunch();
  } catch (error) {
    report(`Mise à jour impossible : ${message(error)}`);
    offer();
  } finally {
    stop();
  }
}

getVersion().then((version) => {
  $("#version").textContent = `v${version}`;
  $("#about-version").textContent = `Version ${version}`;
  Object.assign($<HTMLAnchorElement>("#about-notes"), { href: `${REPO}/releases/tag/v${version}` });
  Object.assign($<HTMLAnchorElement>("#about-notice"), { href: `${REPO}/blob/main/NOTICE` });
});
$("#about-button").onclick = () => about.showModal();
$("#about-close").onclick = () => about.close();
$("#about-check").onclick = () => { $("#about-state").textContent = "Recherche…"; void check(true); };
setTimeout(() => void check(false), 3000);
