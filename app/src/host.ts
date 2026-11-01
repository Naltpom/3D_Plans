import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { documentDir } from "@tauri-apps/api/path";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save } from "@tauri-apps/plugin-dialog";

const SOURCE_EXTENSIONS = ["csv", "jpg", "jpeg", "heic", "png", "pdf", "geojson", "json", "asc", "tif", "tiff"];

export async function call<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
  try {
    return await invoke<T>("engine_call", { method, params });
  } catch (error) {
    throw new Error(String(error));
  }
}

export function onEvent(cb: (event: Record<string, unknown>) => void): void {
  void listen<Record<string, unknown>>("engine-event", ({ payload }) => cb(payload));
}

export const host = {
  photoUrl: (path: string) => convertFileSrc(path),
  allowAssets: (paths: string[]) => invoke<void>("allow_assets", { paths }),
  documents: documentDir,
  pickFolder: (title: string) => open({ directory: true, title }),
  pickPng: (title: string, defaultPath: string) => save({ title, defaultPath, filters: [{ name: "Image PNG", extensions: ["png"] }] }),
  pickFiles: (title: string) => open({ multiple: true, title, filters: [{ name: "Sources", extensions: SOURCE_EXTENSIONS }] }),
  onDrop: (hover: (active: boolean) => void, drop: (paths: string[]) => void) =>
    getCurrentWebview().onDragDropEvent(({ payload }) => {
      hover(payload.type === "enter" || payload.type === "over");
      if (payload.type === "drop") drop(payload.paths);
    }),
};
