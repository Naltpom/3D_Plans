"use strict";
const FILTRE = (() => {
  const SIG_W = 32, SIG_H = 24, NET_MIN = 25, NET_RATIO = 0.6, ECART_MIN = 0.12, ECART_MAX = 1.0, SOMBRE = 45, RECENT = 48;

  const luma = (rgba, n) => { const g = new Float32Array(n); for (let i = 0, j = 0; i < n; i++, j += 4) g[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2]; return g; };

  function nettete(g, w, h) {
    let s = 0, s2 = 0, n = 0;
    for (let y = 1; y < h - 1; y++) for (let x = 1, i = y * w + 1; x < w - 1; x++, i++) {
      const v = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - w] - g[i + w];
      s += v; s2 += v * v; n++;
    }
    return n ? s2 / n - (s / n) ** 2 : 0;
  }

  function signature(g, w, h) {
    const sig = new Float32Array(SIG_W * SIG_H);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) sig[Math.min(SIG_H - 1, Math.floor(y * SIG_H / h)) * SIG_W + Math.min(SIG_W - 1, Math.floor(x * SIG_W / w))] += g[y * w + x];
    let m = 0, d = 0;
    for (const v of sig) m += v / sig.length;
    for (const v of sig) d += (v - m) ** 2 / sig.length;
    const k = 1 / Math.max(Math.sqrt(d), 1e-6);
    return sig.map((v) => (v - m) * k);
  }

  const ecart = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s / a.length; };
  const mediane = (v) => { const s = [...v].sort((a, b) => a - b); return s.length ? s[s.length >> 1] : 0; };
  const moyenne = (g) => { let s = 0; for (const v of g) s += v; return s / g.length; };

  class Filtre {
    constructor(freq = 2) { this.freq = freq; this.recent = []; this.best = null; this.debut = null; this.dernier = null; this.hist = []; this.gardees = 0; this.floues = 0; this.doublons = 0; }

    offre(t, g, w, h) {
      const image = { t, net: nettete(g, w, h), sig: signature(g, w, h), lum: moyenne(g) };
      this.recent = [...this.recent.slice(1 - RECENT), image.net];
      const mieux = !this.best || image.net > this.best.net;
      if (mieux) this.best = image;
      if (this.debut === null) this.debut = t;
      if (t - this.debut < 1000 / this.freq) return { mieux, decision: null };
      const decision = this.decide();
      [this.best, this.debut] = [null, t];
      return { mieux, decision };
    }

    decide() {
      const b = this.best, floue = b.net < Math.max(NET_MIN, NET_RATIO * mediane(this.recent)), e = this.dernier ? ecart(b.sig, this.dernier) : Infinity;
      const raison = floue ? "floue" : e < ECART_MIN ? "doublon" : null;
      if (!raison) { this.dernier = b.sig; this.gardees++; } else if (floue) this.floues++; else this.doublons++;
      this.hist = [...this.hist.slice(-3), raison ?? "gardee"];
      const n = (r) => this.hist.filter((x) => x === r).length;
      const conseil = b.lum < SOMBRE ? "lumiere" : n("floue") >= 2 || !raison && e > ECART_MAX && this.gardees > 1 ? "ralentis" : n("doublon") >= 4 ? "bouge" : null;
      return { garde: !raison, raison, conseil, net: b.net, ecart: e };
    }
  }

  return { luma, nettete, signature, ecart, Filtre };
})();
if (typeof module !== "undefined") module.exports = FILTRE;
