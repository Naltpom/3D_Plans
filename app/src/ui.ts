export const $ = <T extends Element = HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
export const el = <K extends keyof HTMLElementTagNameMap>(tag: K, className = "", text = "") =>
  Object.assign(document.createElement(tag), { className, textContent: text });
export const button = (text: string, className: string, title: string, onclick: () => void) =>
  Object.assign(el("button", className, text), { type: "button", title, onclick, ariaLabel: text === "×" ? title : null });
export const number = (x: number, digits = 1) => x.toLocaleString("fr-FR", { maximumFractionDigits: digits });
export const cm = (m: number) => `${number(m * 100)} cm`;
export const message = (error: unknown) => error instanceof Error ? error.message : String(error);

export function report(error: unknown) {
  const text = message(error), stale = text.includes("cliquez sur « Mettre à jour »");
  $("#banner span").replaceChildren(text,
    ...stale ? [" ", button("Mettre à jour", "link", "Lancer la mise à jour", () => $<HTMLButtonElement>("#update-button").click())] : []);
  $("#banner").hidden = false;
}

export const run = (task: () => unknown) => Promise.resolve().then(task).catch(report);

type Warning = { message: string; component?: string };
const warnings = new Map<string, Warning[]>();

export function warn(group: string, list: Warning[]) {
  const all = [...warnings.set(group, list).values()].flat(), band = $("#warnings");
  band.hidden = !all.length;
  band.replaceChildren(...all.map((w) => {
    const item = el("div", "", w.message);
    if (w.component) item.append(" ", button("voir", "link", "Mettre en évidence dans la vue 3D",
      () => document.dispatchEvent(new CustomEvent("plans3d:highlight", { detail: [w.component] }))));
    return item;
  }));
}
