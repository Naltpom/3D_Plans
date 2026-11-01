import * as THREE from "three";
import { GradientEquirectTexture, WebGLPathTracer } from "three-gpu-pathtracer";
import { call, host } from "./host";
import { $, el, number, run } from "./ui";
import { GLASS, hooks, view, type Lamp } from "./view3d";

type Sun = { lat: number; lon: number; source: string; hauteur: number; azimut: number; direction: [number, number, number]; direct: number; diffus: number };

const CUT = ["mur", "porte", ...GLASS];
const ABOVE = ["plafond", "toit"];
const COMPASS = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"];
const GLOW = 0.15;
const RHO = 0.5;
const NIGHT = new THREE.Color(0x0d1424), DAY = new THREE.Color(0xa9c9ea), EDIT = new THREE.Color(0x101318);
const year = new Date().getFullYear();

const section = $("#lighting");
const real = $<HTMLInputElement>("#light-real");
const day = $<HTMLInputElement>("#light-day");
const hour = $<HTMLInputElement>("#light-hour");
const ev = $<HTMLInputElement>("#light-ev");
const lat = $<HTMLInputElement>("#light-lat");
const lon = $<HTMLInputElement>("#light-lon");
const lampList = $("#lamp-list");
const tools = $("#render-tools");
const passes = $("#render-state");
const pause = $<HTMLButtonElement>("#render-pause");

const sun = new THREE.DirectionalLight(0xffffff, 0);
const sky = new THREE.HemisphereLight(0xf4f1ea, 0xb8ab98, 0);
const ghost = new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false, side: THREE.DoubleSide });
const env = new GradientEquirectTexture(256);
const center = new THREE.Vector3();
Object.assign(env, { exponent: 0.6 });
env.topColor.set(0xc4d7ef);
env.bottomColor.set(0x8d8a7a);
env.update();
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0005;
sun.shadow.normalBias = 0.02;
view.scene.add(sun, sun.target, sky);

let state: Sun | null = null;
let tracer: WebGLPathTracer | null = null;
let tracing = false;
let paused = false;
let ticket = 0;

const meshes = () => view.model.children.filter((o): o is THREE.Mesh => o instanceof THREE.Mesh);
const clock = (h: number) => `${Math.floor(h)} h ${String(Math.round(h % 1 * 60)).padStart(2, "0")}`;
const degrees = (v: number, plus: string, minus: string) => `${number(Math.abs(v), 4)}° ${v < 0 ? minus : plus}`;

function apply() {
  const on = real.checked, box = new THREE.Box3().setFromObject(view.model);
  if (box.isEmpty()) box.set(new THREE.Vector3(-2, 0, -2), new THREE.Vector3(2, 2.5, 2));
  const size = box.getSize(new THREE.Vector3()), radius = box.getBoundingSphere(new THREE.Sphere()).radius + 1;
  box.getCenter(center);
  let lumens = 0;
  view.furniture.traverse((o) => {
    const lamp = o.userData.lamp as Lamp | undefined;
    if (!lamp) return;
    if (o instanceof THREE.Light) {
      o.visible = on && lamp.allume;
      lumens += o.visible ? lamp.lumens : 0;
    } else if (o instanceof THREE.Mesh) {
      o.userData.glow = new THREE.Color(lamp.couleur).multiplyScalar(!lamp.allume ? 0 : on ? GLOW * lamp.luminance : 0.6);
      Object.assign(o.userData.material, { castShadow: false }).emissive.copy(o.userData.glow);
    }
  });
  const toSun = new THREE.Vector3(...state?.direction ?? [0, 1, 0]);
  let flux = lumens;
  for (const o of meshes()) {
    const glass = GLASS.includes(o.userData.type), material = o.userData.material as THREE.MeshStandardMaterial, opacity = on ? glass ? 0.15 : 1 : o.userData.opacity;
    if (material.opacity !== opacity) Object.assign(material, { opacity, transparent: opacity < 1, depthWrite: opacity === 1, needsUpdate: true });
    o.material = material;
    if (!state || !glass) continue;
    const { width, height } = (o.geometry as THREE.BoxGeometry).parameters, outward = new THREE.Vector3(Math.sin(o.rotation.y), 0, Math.cos(o.rotation.y));
    if (outward.dot(o.position.clone().sub(center)) < 0) outward.negate();
    flux += width * height * (state.direct * Math.max(outward.dot(toSun), 0) + state.diffus / 2);
  }
  for (const group of [view.model, view.furniture]) group.traverse((o) => { if (o instanceof THREE.LineSegments) o.visible = !on; });
  for (const o of view.helpers) o.visible = !on;
  view.neutral.visible = !on;
  const alt = THREE.MathUtils.degToRad(state?.hauteur ?? -10);
  sun.visible = on && alt > 0;
  sky.visible = on;
  if (state && alt > 0) {
    sun.intensity = state.direct;
    sun.color.setHSL(0.09, Math.min(1, 0.9 * Math.exp(-6 * alt)), 0.5 + 0.5 * Math.min(1, 3 * Math.sin(alt)));
    sun.position.copy(center).addScaledVector(new THREE.Vector3(...state.direction), 2 * radius);
    sun.target.position.copy(center);
    Object.assign(sun.shadow.camera, { left: -radius, right: radius, top: radius, bottom: -radius, near: 0.1, far: 4 * radius });
    sun.shadow.camera.updateProjectionMatrix();
  }
  const average = flux / (2 * size.x * size.z + 2 * (size.x + size.z) * size.y) / (1 - RHO);
  sky.intensity = average;
  view.renderer.toneMapping = on ? THREE.ACESFilmicToneMapping : THREE.NoToneMapping;
  view.renderer.toneMappingExposure = on ? 2 ** ev.valueAsNumber / Math.max(average, 10) : 1;
  view.scene.background = on ? new THREE.Color().lerpColors(NIGHT, DAY, THREE.MathUtils.clamp(alt * 5 + 0.5, 0, 1)) : EDIT;
  if (tracing) trace();
}

function cutaway() {
  const cam = view.camera.position, on = real.checked;
  let changed = false;
  for (const o of meshes()) {
    const type = o.userData.type, normal = new THREE.Vector3(Math.sin(o.rotation.y), 0, Math.cos(o.rotation.y));
    if (ABOVE.includes(type) && !o.geometry.boundingBox) o.geometry.computeBoundingBox();
    const hide = on && (CUT.includes(type) ? Math.sign(normal.dot(cam.clone().sub(o.position))) !== Math.sign(normal.dot(center.clone().sub(o.position)))
      : ABOVE.includes(type) && cam.y > o.position.y + o.geometry.boundingBox!.min.y);
    const material = hide ? ghost : o.userData.material;
    changed ||= o.material !== material;
    o.material = material;
  }
  return changed;
}

function trace() {
  tracer ??= Object.assign(new WebGLPathTracer(view.renderer), { minSamples: 1, renderDelay: 0, fadeDuration: 0 });
  cutaway();
  for (const o of meshes()) o.visible = o.material !== ghost && !GLASS.includes(o.userData.type);
  const radiance = Math.max(state?.diffus ?? 0, 1) / Math.PI;
  Object.assign(view.scene, { environment: env, background: env, environmentIntensity: radiance, backgroundIntensity: radiance });
  tracer.setScene(view.scene, view.camera);
}

function stop() {
  tracing = false;
  tools.hidden = true;
  for (const o of meshes()) o.visible = true;
  view.scene.environment = null;
  apply();
}

function renderLamps() {
  const lamps = view.furniture.children.filter((o) => o instanceof THREE.Light && o.userData.lamp);
  lampList.replaceChildren(...lamps.map((o) => {
    const { id, label, lamp } = o.userData as { id: string; label: string; lamp: Lamp }, item = el("li"), name = el("label", "check"), settings = el("div", "row");
    const set = (options: Record<string, number>) => run(async () => document.dispatchEvent(new CustomEvent("plans3d:furniture-refresh",
      { detail: await call("furniture.options", { id: Number(id.replace("meuble-", "")), options }) })));
    const field = (key: string, value: number, unit: string, max: number) => {
      const input = Object.assign(el("input"), { type: "number", value: String(value), min: "0", max: String(max), step: "1", required: true, ariaLabel: `${label} (${unit})` });
      input.onchange = () => input.validity.valid && set({ [key]: input.valueAsNumber });
      const wrap = el("label", "unit");
      wrap.append(input, unit);
      return wrap;
    };
    const toggle = Object.assign(el("input"), { type: "checkbox", checked: lamp.allume, title: "Interrupteur", ariaLabel: `Allumer ${label}` });
    const type = Object.assign(el("select"), { ariaLabel: `Type de ${label}` });
    type.append(new Option("Ponctuel", "0"), new Option("Spot", "1"));
    type.value = String(Number(lamp.spot));
    toggle.onchange = () => set({ allume: Number(toggle.checked) });
    type.onchange = () => set({ spot: Number(type.value) });
    name.append(toggle, label);
    settings.append(field("lumens", lamp.lumens, "lm", 20000), field("kelvin", lamp.kelvin, "K", 10000));
    item.append(name, type, settings);
    return item;
  }));
  if (!lamps.length) lampList.append(el("li", "empty", "Aucun luminaire : ajoutez-en depuis la Galerie"));
}

function labels() {
  $("#light-day-label").textContent = new Date(year, 0, day.valueAsNumber).toLocaleDateString("fr-FR", { day: "numeric", month: "long" });
  $("#light-hour-label").textContent = clock(hour.valueAsNumber);
  $("#light-ev-label").textContent = `${ev.valueAsNumber > 0 ? "+" : ""}${number(ev.valueAsNumber)} IL`;
}

function renderSun(s: Sun) {
  $("#light-where").hidden = s.source === "cadastre";
  lat.value = String(s.lat);
  lon.value = String(s.lon);
  $("#light-place").textContent = s.source === "cadastre" ? `Lieu du cadastre : ${degrees(s.lat, "N", "S")}, ${degrees(s.lon, "E", "O")}`
    : s.source === "défaut" ? "Lieu par défaut (Paris) : précisez la latitude et la longitude du projet" : "Lieu du projet";
  $("#light-sun").textContent = s.hauteur > 0 ? `Soleil à ${number(s.hauteur, 0)}° au-dessus de l'horizon, azimut ${number(s.azimut, 0)}° (${COMPASS[Math.round(s.azimut / 45) % 8]})` : "Soleil couché";
}

const update = () => run(async () => {
  labels();
  const t = ++ticket, quand = new Date(year, 0, day.valueAsNumber, 0, Math.round(hour.valueAsNumber * 60)).toISOString();
  const found = await call<Sun>("lumiere.soleil", { quand });
  if (t !== ticket) return;
  state = found;
  renderSun(found);
  apply();
});

const now = new Date();
day.value = String(Math.round((Date.UTC(year, now.getMonth(), now.getDate()) - Date.UTC(year, 0, 1)) / 864e5) + 1);
hour.value = String(Math.round((now.getHours() + now.getMinutes() / 60) * 4) / 4);
day.oninput = hour.oninput = update;
ev.oninput = () => { labels(); apply(); };
real.onchange = () => tracing && !real.checked ? stop() : apply();
lat.onchange = lon.onchange = () => lat.validity.valid && lon.validity.valid && lat.value && lon.value && run(async () => {
  await call("lumiere.lieu", { lat: lat.valueAsNumber, lon: lon.valueAsNumber });
  await update();
});
$("#render-button").onclick = () => {
  real.checked = tracing = true;
  paused = false;
  pause.textContent = "Arrêter";
  tools.hidden = false;
  apply();
};
pause.onclick = () => {
  paused = !paused;
  pause.textContent = paused ? "Reprendre" : "Arrêter";
};
$("#render-exit").onclick = stop;
$("#render-export").onclick = () => run(async () => {
  const path = await host.pickPng("Exporter le rendu réaliste", "rendu.png");
  if (!path) return;
  const done = await call<{ path: string }>("lumiere.export", { path, data: view.renderer.domElement.toDataURL("image/png") });
  $("#status").textContent = `Rendu enregistré : ${done.path}`;
});

hooks.frame = () => {
  if (!tracing || !tracer) {
    if (real.checked) cutaway();
    view.renderer.render(view.scene, view.camera);
    return;
  }
  tracer.pausePathTracing = paused;
  tracer.renderSample();
  passes.textContent = tracer.samples < 1 ? "préparation (jusqu'à 1 min)…" : `${Math.floor(tracer.samples)} passe${tracer.samples >= 2 ? "s" : ""}`;
};
view.controls.addEventListener("change", () => { if (tracing && tracer) cutaway() ? trace() : tracer.updateCamera(); });
new ResizeObserver(() => { if (tracing) tracer?.updateCamera(); }).observe(view.renderer.domElement);

document.addEventListener("plans3d:scene", () => { renderLamps(); apply(); });
document.addEventListener("plans3d:opened", (event) => {
  section.hidden = !(event as CustomEvent).detail;
  if (tracing) stop();
  if (!section.hidden) update();
});
