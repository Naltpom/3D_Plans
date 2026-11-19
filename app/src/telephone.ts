import qrcode from "qrcode-generator";
import { call, onEvent } from "./host";
import { node } from "./plan2d";
import { button, el, number, run } from "./ui";
import type { End, Seg } from "./coter";

type Link = { kind: string; host: string; url: string };
type Todo = { key: string; label: string; hint: string; plan: number; trace: End[] | null; expr?: string; ends?: number[][] | null };
export type Scene = { title: string; k: number; segs: Seg[]; ring: string[]; hidden: string[]; status: Record<string, { text: string; conflict?: boolean; ends?: number[][] | null }>; todo: Todo[] };
export type Handlers = { changed: (id: string, label: string) => void; hide: (id: string, shown: boolean) => void };

let links: Link[] | null = null;
let which = 0;
let clients = 0;
let last = "";
let sent = "";
let seen = false;
let on: Handlers | null = null;
const chip = el("span", "chip");

const paint = () => { chip.className = `chip ${clients ? "ok" : ""}`; chip.textContent = clients ? clients > 1 ? `${clients} téléphones connectés` : "Téléphone connecté" : "En attente du téléphone"; };

async function start() {
  links = (await call<{ links: Link[] }>("phone.start")).links;
  [which, clients, last, sent] = [0, 0, "", ""];
}

function stop() {
  links = null;
  return call("phone.stop");
}

export function phoneButton(render: () => void) {
  return Object.assign(button("Saisir sur le téléphone", "", "Ouvrir une page de saisie des cotes sur le téléphone, par un QR code", () => run(async () => { if (links) await stop(); else await start(); render(); })),
    { ariaPressed: String(Boolean(links)) });
}

function qr(url: string) {
  const code = qrcode(0, "M");
  code.addData(url);
  code.make();
  const n = code.getModuleCount();
  let d = "";
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) if (code.isDark(r, c)) d += `M${c} ${r}h1v1h-1z`;
  const svg = node("svg", { viewBox: `-4 -4 ${n + 8} ${n + 8}`, role: "img", "aria-label": "QR code : adresse de la page et code d'accès" }, "qr");
  svg.append(node("path", { d }));
  return svg;
}

export function phonePanel(scene: Scene, handlers: Handlers, render: () => void): HTMLElement[] {
  [seen, on] = [true, handlers];
  if (!links) return [];
  const json = JSON.stringify(scene);
  if (json !== sent) { sent = json; void call("phone.scene", { scene }).catch(() => { sent = ""; }); }
  const box = el("section", "phone"), head = el("div", "head"), link = links[which], other = links[1 - which];
  paint();
  head.append(el("b", "", "Saisir sur le téléphone"), chip);
  box.append(head);
  if (!link) {
    box.append(el("div", "alert", "Aucun réseau : connecte le PC au Wi-Fi de la maison ou ouvre Tailscale, puis rouvre ce panneau."));
  } else {
    const text = el("div", "text");
    text.append(el("span", "", "Scanne avec l'appareil photo du téléphone, puis va mesurer dans la pièce."), el("span", "kicker", link.kind === "tailscale" ? "Adresse Tailscale" : "Adresse Wi-Fi"),
      el("span", "m", link.host), el("small", "", "Le code d'accès n'est que dans le QR. Il change à chaque ouverture."));
    box.append(qr(link.url), text);
  }
  const foot = el("div", "foot");
  if (last) foot.append(el("span", "", `Dernière saisie : ${last}`));
  if (other) foot.append(button(other.kind === "wifi" ? `Pas de Tailscale sur le téléphone ? QR du Wi-Fi de la maison (${other.host.split(":")[0]})` : `QR Tailscale (${other.host.split(":")[0]})`, "link", "Changer d'adresse", () => { which = 1 - which; render(); }));
  box.append(foot, button("Arrêter", "", "Arrêter la saisie sur le téléphone", () => run(async () => { await stop(); render(); })));
  return [box];
}

export function phoneCheck() {
  if (links && !seen) void stop();
  seen = false;
}

onEvent((event) => {
  if (event.event !== "phone" || !links) return;
  if (typeof event.clients === "number") { clients = event.clients; paint(); }
  if (event.changed != null) {
    last = event.removed ? `${event.label} retirée` : `${event.label} · ${number(Number(event.value))} cm`;
    on?.changed(String(event.changed), String(event.label));
  }
  if (event.hide != null) on?.hide(String(event.hide), Boolean(event.shown));
});
