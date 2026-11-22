/* Cloud removal explorer: recorded outputs of the model's test run, compared and analysed in the browser. */
(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const f1 = new Intl.NumberFormat("en", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const f2 = new Intl.NumberFormat("en", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const f3 = new Intl.NumberFormat("en", { minimumFractionDigits: 3, maximumFractionDigits: 3 });
  const VIEWS = { output: "Model output", truth: "Ground truth", unc: "Uncertainty, brighter is less sure", err: "Error against truth, brighter is worse" };
  const S = 512;
  let data, cur = null, view = "output", flagPct = 0, colors = {};
  const cache = new Map();

  function token(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
  function rgbOf(c) { const k = document.createElement("canvas"); k.width = k.height = 1; const x = k.getContext("2d"); x.fillStyle = c; x.fillRect(0, 0, 1, 1); return [...x.getImageData(0, 0, 1, 1).data.slice(0, 3)]; }
  function img(src) { return new Promise((ok, no) => { const i = new Image(); i.onload = () => ok(i); i.onerror = () => no(new Error(src)); i.src = src; }); }
  function pixels(im) { const c = document.createElement("canvas"); c.width = c.height = S; const x = c.getContext("2d"); x.drawImage(im, 0, 0, S, S); return x.getImageData(0, 0, S, S); }

  /* ---------- tooltip ---------- */
  const tip = $("#tip");
  function showTip(html, x, y) { tip.innerHTML = html; tip.hidden = false; const r = tip.getBoundingClientRect(); tip.style.left = `${Math.max(8, Math.min(innerWidth - r.width - 8, x + 14))}px`; tip.style.top = `${y - r.height - 12 < 8 ? y + 16 : y - r.height - 12}px`; }
  const hideTip = () => { tip.hidden = true; };

  /* ---------- load ---------- */
  async function boot() {
    colors.lo = rgbOf(token("--color-paper")); colors.hi = rgbOf(token("--color-accent")); colors.ink = rgbOf(token("--color-ink")); colors.flag = rgbOf(token("--color-accent"));
    try { data = await fetch("data/explorer.json").then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }); }
    catch { $("#stage-state").textContent = "The results did not load. Reload the page to try again."; return; }
    buildCases(); buildScatter(); buildSummary(); bindStage();
    const q = new URLSearchParams(location.search);
    const v = q.get("view"); if (v && VIEWS[v]) view = v;
    const start = data.cases.find((c) => c.slug === q.get("scene")) || data.cases.find((c) => c.slug === "rice2-30") || data.cases[0];
    await pick(start);
  }

  async function loadCase(c) {
    if (cache.has(c.slug)) return cache.get(c.slug);
    const base = `img/cases/${c.slug}`;
    const [cl, out, tr, un] = await Promise.all(["cloudy.webp", "output.webp", "truth.webp", "unc.png"].map((s) => img(`${base}-${s}`)));
    const P = { cloudy: pixels(cl), output: pixels(out), truth: pixels(tr), uncRaw: pixels(un) };
    const n = S * S, unc = new Uint8Array(n), err = new Float32Array(n);
    let maxErr = 0;
    for (let i = 0; i < n; i++) {
      unc[i] = P.uncRaw.data[i * 4];
      const e = (Math.abs(P.output.data[i * 4] - P.truth.data[i * 4]) + Math.abs(P.output.data[i * 4 + 1] - P.truth.data[i * 4 + 1]) + Math.abs(P.output.data[i * 4 + 2] - P.truth.data[i * 4 + 2])) / 3;
      err[i] = e; if (e > maxErr) maxErr = e;
    }
    // Pearson correlation between predicted uncertainty and actual error, per pixel
    let su = 0, se = 0; for (let i = 0; i < n; i++) { su += unc[i]; se += err[i]; }
    const mu = su / n, me = se / n; let cov = 0, vu = 0, ve = 0;
    for (let i = 0; i < n; i++) { const a = unc[i] - mu, b = err[i] - me; cov += a * b; vu += a * a; ve += b * b; }
    const sorted = Uint8Array.from(unc).sort();
    const res = { ...P, unc, err, maxErr, corr: cov / Math.sqrt(vu * ve || 1), sorted };
    cache.set(c.slug, res); return res;
  }

  /* ---------- the stage ---------- */
  function ramp(v) { const t = Math.min(1, Math.max(0, v)); return [0, 1, 2].map((k) => colors.lo[k] + (colors.hi[k] - colors.lo[k]) * t); }
  function drawRight(P) {
    const out = new ImageData(S, S), d = out.data, n = S * S;
    if (view === "output" || view === "truth") d.set(P[view].data);
    else if (view === "unc") { for (let i = 0; i < n; i++) { const [r, g, b] = ramp((P.unc[i] / 255) ** 0.8); d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = 255; } }
    else { const m = Math.max(24, P.maxErr * 0.6); for (let i = 0; i < n; i++) { const [r, g, b] = ramp((P.err[i] / m) ** 0.7); d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = 255; } }
    if (flagPct > 0 && (view === "output" || view === "truth")) {
      const cut = P.sorted[Math.min(n - 1, Math.floor(n * (1 - flagPct / 100)))];
      for (let i = 0; i < n; i++) if (P.unc[i] >= cut && P.unc[i] > 0) { d[i * 4] = d[i * 4] * 0.35 + colors.flag[0] * 0.65; d[i * 4 + 1] = d[i * 4 + 1] * 0.35 + colors.flag[1] * 0.65; d[i * 4 + 2] = d[i * 4 + 2] * 0.35 + colors.flag[2] * 0.65; }
    }
    $("#right").getContext("2d").putImageData(out, 0, 0);
    $("#right-label").textContent = VIEWS[view];
  }
  async function pick(c) {
    cur = c;
    document.querySelectorAll(".case").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.slug === c.slug)));
    const st = $("#stage-state"); st.hidden = false; st.textContent = "Loading the scene";
    try {
      const P = await loadCase(c); if (cur !== c) return;
      $("#left").getContext("2d").putImageData(P.cloudy, 0, 0);
      drawRight(P); st.hidden = true;
      $("#r-psnr").textContent = f2.format(c.psnr);
      $("#r-psnr-in").textContent = `${f2.format(c.psnrIn)} dB, so a gain of ${f1.format(c.psnr - c.psnrIn)} dB`;
      $("#r-ssim").textContent = `${f3.format(c.ssimIn)} then ${f3.format(c.ssim)}`;
      $("#r-cloud").textContent = c.cloud != null ? `${Math.round(c.cloud * 100)}% of the scene` : "No mask in RICE1, haze covers it all";
      $("#r-corr").textContent = `${f2.format(P.corr)}; the test run logged ${f2.format(c.corr)}`;
      writeURL();
    } catch { st.hidden = false; st.textContent = "This scene's images did not load. Pick another or reload."; }
  }
  function writeURL() { history.replaceState(null, "", `?scene=${cur.slug}&view=${view}`); }
  function buildCases() {
    $("#case-list").innerHTML = data.cases.map((c) => `<li><button class="case" type="button" role="option" aria-selected="false" data-slug="${c.slug}"><img src="img/cases/${c.slug}-cloudy.webp" alt="" width="44" height="44"><span><b>${c.label}</b><small>${c.set}, scene ${c.name}: ${f1.format(c.psnrIn)} to ${f1.format(c.psnr)} dB</small></span></button></li>`).join("");
    $("#case-list").addEventListener("click", (e) => { const b = e.target.closest(".case"); if (b) pick(data.cases.find((c) => c.slug === b.dataset.slug)); });
  }
  function bindStage() {
    document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
      view = b.dataset.view;
      document.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
      if (cur && cache.has(cur.slug)) drawRight(cache.get(cur.slug)); writeURL();
    }));
    const inner = $("#stage-inner"), split = $("#split");
    const set = (pct) => { pct = Math.max(0, Math.min(100, pct)); inner.style.setProperty("--split", `${pct}%`); split.value = String(Math.round(pct)); };
    let drag = false;
    const at = (e) => { const r = inner.getBoundingClientRect(); set(((e.clientX - r.left) / r.width) * 100); };
    inner.addEventListener("pointerdown", (e) => { drag = true; inner.setPointerCapture(e.pointerId); at(e); split.focus({ preventScroll: true }); });
    inner.addEventListener("pointermove", (e) => { if (drag) at(e); });
    inner.addEventListener("pointerup", () => { drag = false; });
    split.addEventListener("input", () => set(+split.value));
    const flag = $("#flag");
    flag.addEventListener("input", () => {
      flagPct = +flag.value; $("#flag-value").textContent = flagPct ? `top ${flagPct}%` : "off";
      if (flagPct && view !== "output" && view !== "truth") document.querySelector('.seg [data-view="output"]').click();
      else if (cur && cache.has(cur.slug)) drawRight(cache.get(cur.slug));
    });
    document.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-selected", String(x.dataset.view === view)));
  }

  /* ---------- the test set ---------- */
  function buildScatter() {
    const W = 640, H = 460, L = 52, R = 16, T = 16, B = 48, lo = 5, hi = 55;
    const xs = (v) => L + ((v - lo) / (hi - lo)) * (W - L - R), ys = (v) => T + (H - T - B) - ((v - lo) / (hi - lo)) * (H - T - B);
    const ticks = [10, 20, 30, 40, 50];
    const grid = ticks.map((t) => `<line x1="${L}" x2="${W - R}" y1="${ys(t)}" y2="${ys(t)}"/><line y1="${T}" y2="${H - B}" x1="${xs(t)}" x2="${xs(t)}"/>`).join("");
    const ax = ticks.map((t) => `<text x="${xs(t)}" y="${H - B + 18}" text-anchor="middle">${t}</text><text x="${L - 8}" y="${ys(t) + 4}" text-anchor="end">${t}</text>`).join("");
    const pts = data.test.map((t, i) => `<circle class="pt ${t.set === "RICE1" ? "r1" : "r2"}" cx="${xs(Math.max(lo, t.psnrIn)).toFixed(1)}" cy="${ys(Math.min(hi, t.psnr)).toFixed(1)}" r="5" data-i="${i}" tabindex="0" role="button" aria-label="${t.set} image ${t.name}: input ${f1.format(t.psnrIn)} dB, output ${f1.format(t.psnr)} dB"/>`).join("");
    $("#scatter").innerHTML = `<svg viewBox="0 0 ${W} ${H}"><g class="grid">${grid}</g>
      <line class="same" x1="${xs(lo)}" y1="${ys(lo)}" x2="${xs(hi)}" y2="${ys(hi)}"/><text class="lab" x="${xs(47)}" y="${ys(47) + 18}" text-anchor="middle" transform="rotate(-35 ${xs(47)} ${ys(47) + 18})">no change</text>
      <g class="axis">${ax}<text class="title" x="${(L + W - R) / 2}" y="${H - 8}" text-anchor="middle">PSNR of the cloudy input, dB</text><text class="title" x="14" y="${(T + H - B) / 2}" text-anchor="middle" transform="rotate(-90 14 ${(T + H - B) / 2})">PSNR of the output, dB</text></g>
      <g>${pts}</g></svg><ul class="legend"><li><i class="r1"></i>RICE1, haze and thin cloud, ${data.summary.RICE1.n} images</li><li><i class="r2"></i>RICE2, thick cloud, ${data.summary.RICE2.n} images</li></ul>`;
    const improved = data.test.filter((t) => t.psnr > t.psnrIn).length;
    $("#d-test").textContent = `Each point is one held-out image: the PSNR of the cloudy input across, the model's output up. ${improved} of ${data.test.length} sit above the line, so the output is closer to the clear scene than the input was.`;
    const svg = $("#scatter svg");
    svg.addEventListener("pointermove", (e) => { const p = e.target.closest(".pt"); if (!p) return hideTip(); const t = data.test[+p.dataset.i]; showTip(`${t.set} image ${t.name}<br>PSNR ${f1.format(t.psnrIn)} to <b>${f1.format(t.psnr)}</b> dB<br>SSIM ${f3.format(t.ssimIn)} to <b>${f3.format(t.ssim)}</b>`, e.clientX, e.clientY); });
    svg.addEventListener("pointerleave", hideTip);
    const choose = (p) => { svg.querySelectorAll(".pt.on").forEach((x) => x.classList.remove("on")); p.classList.add("on"); peek(data.test[+p.dataset.i]); };
    svg.addEventListener("click", (e) => { const p = e.target.closest(".pt"); if (p) choose(p); });
    svg.addEventListener("keydown", (e) => { const p = e.target.closest(".pt"); if (p && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); choose(p); } });
  }
  function peek(t) {
    const caseMatch = data.cases.find((c) => c.key === `${t.set}-${t.name}`);
    $("#peek").innerHTML = `<h3>${t.set}, image ${t.name}</h3>
      <div class="pair"><figure><img src="img/test/${t.slug}.webp" alt="Model output for ${t.set} image ${t.name}" width="384" height="384"><figcaption>Output</figcaption></figure>
      ${t.hasUnc ? `<figure><img src="img/test/${t.slug}-unc.webp" alt="Uncertainty map for ${t.set} image ${t.name}" width="384" height="384"><figcaption>Uncertainty, magma scale</figcaption></figure>` : ""}</div>
      <dl><dt>PSNR in, out</dt><dd>${f1.format(t.psnrIn)}, ${f1.format(t.psnr)} dB</dd><dt>SSIM in, out</dt><dd>${f3.format(t.ssimIn)}, ${f3.format(t.ssim)}</dd>${t.cloud != null ? `<dt>Cloud cover</dt><dd>${Math.round(t.cloud * 100)}%</dd>${t.psnrCloud != null ? `<dt>PSNR under cloud</dt><dd>${f1.format(t.psnrCloud)} dB</dd>` : ""}` : ""}</dl>
      ${caseMatch ? `<button class="seg-link" type="button" data-slug="${caseMatch.slug}">Open it in the comparison</button>` : `<p class="muted">The repository keeps this image's output and uncertainty, not its cloudy input.</p>`}`;
    const b = $("#peek .seg-link"); if (b) b.addEventListener("click", () => { pick(caseMatch); $("#compare").scrollIntoView({ behavior: "smooth" }); });
  }
  function buildSummary() {
    $("#t-summary").innerHTML = Object.entries(data.summary).map(([k, s]) => `<tr><th scope="row">${k === "RICE1" ? "RICE1, haze and thin cloud" : "RICE2, thick cloud and shadow"}</th><td class="n">${s.n}</td><td class="n">${f2.format(s.psnrIn)} dB</td><td class="n">${f2.format(s.psnr)} dB</td><td class="n">${f3.format(s.ssimIn)}</td><td class="n">${f3.format(s.ssim)}</td><td class="n">${f1.format(s.ms)} ms</td></tr>`).join("");
  }

  boot();
})();
