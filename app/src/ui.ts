export const $ = <T extends Element = HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
export const el = <K extends keyof HTMLElementTagNameMap>(tag: K, className = "", text = "") =>
  Object.assign(document.createElement(tag), { className, textContent: text });
export const button = (text: string, className: string, title: string, onclick: () => void) =>
  Object.assign(el("button", className, text), { type: "button", title, onclick });
export const number = (x: number, digits = 1) => x.toLocaleString("fr-FR", { maximumFractionDigits: digits });
export const cm = (m: number) => `${number(m * 100)} cm`;
export const message = (error: unknown) => error instanceof Error ? error.message : String(error);

export function report(error: unknown) {
  $("#banner span").textContent = message(error);
  $("#banner").hidden = false;
}

export const run = (task: () => unknown) => Promise.resolve().then(task).catch(report);
