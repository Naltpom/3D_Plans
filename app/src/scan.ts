import { call, onEvent } from "./host";
import { qr } from "./telephone";
import { button, el, number, run } from "./ui";

type Link = { kind: string; host: string; url: string };
type Phone = { passage?: number; paused?: boolean; gardees?: number; floues?: number; doublons?: number; freq?: number; attente?: number; resolution?: string };
type State = { id: string; mode: string; active: boolean; final: string; received: number; placed: number; passages: Record<string, number>; phone: Phone; computing: boolean;
  ago_s: number | null; duration_s: number | null; colmap_mb: number; engine_mb: number; gaps: string; turn?: number; error: string };

let link: Link | null = null, note: { text: string; url?: string } | null = null, state: State | null = null, showQr = false, open = false, redraw: () => void = () => {};

const plural = (n: number, word: string) => `${number(n, 0)} ${word}${n > 1 ? "s" : ""}`;
const go = (mb: number) => `${number(mb / 1024, 1)} Go`;

export async function openScan(render: () => void) {
  const started = await call<{ links: Link[]; tailscale: typeof note }>("phone.start", { page: "scan" });
  [link, note, state, showQr, open, redraw] = [started.links[0] ?? null, started.tailscale, null, false, true, render];
}

export function closeScan() {
  if (!open) return;
  [link, note, state, open] = [null, null, null, false];
  void call("phone.stop").catch(() => {});
}

export const scanFinal = () => state?.final === "calcul";

export function scanTitle() {
  const passage = state?.phone.passage;
  return !state?.received ? "Scanner au téléphone" : scanFinal() ? "Calcul final des positions" : `Scan au téléphone${passage ? ` · passage ${passage}` : ""}`;
}

function qrBox() {
  const box = el("section", "phone"), head = el("div", "head");
  head.append(el("b", "", "Page Scanner du téléphone"), el("span", `chip ${state?.phone.passage ? "ok" : ""}`, state?.phone.passage ? "Téléphone connecté" : "En attente du téléphone"));
  box.append(head);
  if (note) {
    const alert = el("div", "alert");
    alert.append(el("span", "", `${note.text} La caméra du téléphone n'est donnée qu'en HTTPS, par Tailscale.`));
    if (note.url) alert.append(Object.assign(el("a", "m", note.url), { href: note.url, target: "_blank", rel: "noreferrer" }));
    box.append(alert);
  }
  if (link) {
    const text = el("div", "text");
    text.append(el("span", "", "Scanne avec l'appareil photo du téléphone : la page Scanner s'ouvre et demande la caméra."), el("span", "kicker", "Adresse Tailscale · HTTPS"),
      el("span", "m", link.host), el("small", "", "Le code d'accès n'est que dans le QR. Il change à chaque ouverture."));
    box.append(qr(link.url), text);
  }
  return box;
}

export function scanPanel(): Node[] {
  const s = state, phone = s?.phone ?? {}, thrown = (phone.floues ?? 0) + (phone.doublons ?? 0);
  if (!s?.received && !phone.passage) return [qrBox()];
  const stats = el("div", "stats");
  for (const [value, label] of [[s?.received ?? 0, "reçues"], [s?.placed ?? 0, "placées"], [thrown, "jetées sur le tél."]] as [number, string][]) {
    const box = el("div", "stat");
    box.append(el("b", "", number(value, 0)), el("span", "", label));
    stats.append(box);
  }
  const lines = el("div", "lines");
  const passages = Object.entries(s?.passages ?? {}).map(([k, n]) => `passage ${k} : ${number(n, 0)}`).join(" · ");
  lines.append(el("span", "", `${phone.paused ? "En pause sur le téléphone · " : ""}${s?.mode === "objet" ? "Objet" : "Pièce"}${phone.freq ? ` · ${phone.freq} img/s au plus` : ""}${phone.resolution ? ` · ${phone.resolution.replace("Caméra arrière · ", "")}` : ""}`));
  if (passages) lines.append(el("span", "", passages.charAt(0).toUpperCase() + passages.slice(1)));
  if (thrown) lines.append(el("span", "", `Jetées : ${plural(phone.floues ?? 0, "floue")}, ${number(phone.doublons ?? 0, 0)} trop proches de la précédente.`));
  if (phone.attente) lines.append(el("span", "", `${plural(phone.attente, "image")} en attente d'envoi sur le téléphone.`));
  lines.append(el("span", "", s?.computing ? "Calcul en cours…" : s?.ago_s == null ? "Premier calcul dès 5 images reçues." :
    `Dernier calcul : il y a ${s.ago_s} s · ${number(s.duration_s ?? 0, 0)} s · mémoire ${go(s.colmap_mb)} (COLMAP), ${go(s.engine_mb)} (appli).`));
  if (!s?.computing && !scanFinal()) lines.append(el("span", "", "Prochain : toutes les 30 s, ou dès 20 nouvelles images."));
  if (s?.gaps) lines.append(el("span", "warn", s.mode === "objet" ? `Tour : ${s.turn ?? 0}° · ${s.gaps}` : `Peu vu : ${s.gaps}`));
  if (s?.error) lines.append(el("span", "warn", s.error));
  const toggle = button(showQr ? "Masquer le QR" : "Afficher le QR de la page Scanner", "link", "Afficher ou masquer le QR", () => { showQr = !showQr; redraw(); });
  return [stats, lines, ...showQr ? [qrBox()] : [], toggle];
}

export const scanButtons = () => scanFinal() ? [] : [
  button("Arrêter le scan", "", "Arrêter le scan sans calculer les positions finales", () => run(() => call("scan.stop"))),
  Object.assign(button("Recalculer maintenant", "primary", "Relancer le calcul des positions sans attendre", () => run(() => call("scan.now"))), { disabled: !state?.received || state.computing })];

onEvent((event) => {
  if (event.event !== "scan" || !open) return;
  state = event as unknown as State;
  redraw();
});
