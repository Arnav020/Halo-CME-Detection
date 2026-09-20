/* =====================================================================
   Halo CME Detection — frontend
   ===================================================================== */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const FEATURE_META = {
    alpha_proton_ratio: {
      title: "Alpha–proton ratio",
      formula: "Nα / Np",
      unit: "",
      desc: "Helium abundance relative to protons. Elevated in CME ejecta.",
    },
    vp_std_15min: {
      title: "Speed variability",
      formula: "σ(Vp) · 15 min",
      unit: "km/s",
      desc: "Rolling standard deviation of proton speed. Rises with shock turbulence.",
    },
    alpha_over_vpstd: {
      title: "Alpha over Vp std",
      formula: "(Nα/Np) / σ(Vp)",
      unit: "",
      desc: "Composition per unit turbulence. High values mark coherent ejecta cores.",
    },
    alpha_tp_ratio: {
      title: "Alpha–temperature ratio",
      formula: "Nα / Tp",
      unit: "",
      desc: "Alpha density against proton temperature. Cool, dense material scores high.",
    },
  };

  const SERIES_META = {
    proton_density: { title: "Proton density", symbol: "Np", unit: "cm⁻³" },
    proton_speed: { title: "Proton speed", symbol: "Vp", unit: "km s⁻¹" },
    proton_temperature: { title: "Proton temperature", symbol: "Tp", unit: "eV" },
    alpha_density: { title: "Alpha density", symbol: "Nα", unit: "cm⁻³" },
  };

  const ESTIMATOR_NAMES = {
    rf: ["Random forest", "RandomForestClassifier"],
    xgb: ["XGBoost", "XGBClassifier"],
    lr: ["Logistic regression", "LogisticRegression"],
  };

  /* ------------------------------------------------------------------
     Formatting helpers
     ------------------------------------------------------------------ */
  const fmtNum = (v, digits = 3) => {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    const a = Math.abs(v);
    if (a !== 0 && (a < 0.001 || a >= 1e6)) return v.toExponential(2);
    if (a < 1) return v.toFixed(4);
    if (a < 100) return v.toFixed(digits);
    return v.toFixed(1);
  };
  const fmtPct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
  const fmtDate = (iso) => {
    if (!iso) return "–";
    const d = new Date(iso);
    return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false });
  };
  const fmtDuration = (hours) => {
    if (hours === null || hours === undefined) return "–";
    if (hours < 48) return `${hours.toFixed(1)} h`;
    return `${(hours / 24).toFixed(2)} d`;
  };
  const el = (tag, attrs = {}, children = []) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v;
      else if (k === "text") node.textContent = v;
      else node.setAttribute(k, v);
    }
    for (const c of [].concat(children)) if (c) node.append(c);
    return node;
  };
  const svgEl = (tag, attrs = {}) => {
    const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  };

  /* ------------------------------------------------------------------
     Navigation
     ------------------------------------------------------------------ */
  const initNav = () => {
    const links = $$("[data-nav]");
    const toggle = $("#navToggle");
    const list = $("#navLinks");

    toggle.addEventListener("click", () => {
      const open = list.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", String(open));
    });
    links.forEach((a) => a.addEventListener("click", () => {
      list.classList.remove("is-open");
      toggle.setAttribute("aria-expanded", "false");
    }));

    // Brand link: go to the true page top (the #top anchor sits below the sticky nav).
    $(".brand").addEventListener("click", (e) => {
      e.preventDefault();
      window.scrollTo({ top: 0, behavior: REDUCED ? "auto" : "smooth" });
      history.replaceState(null, "", window.location.pathname);
      links.forEach((a) => a.classList.remove("is-active"));
    });

    const sections = links.map((a) => $(a.getAttribute("href"))).filter(Boolean);
    const obs = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        links.forEach((a) => a.classList.toggle("is-active", a.getAttribute("href") === `#${e.target.id}`));
      });
    }, { rootMargin: "-35% 0px -55% 0px" });
    sections.forEach((s) => obs.observe(s));

    // Health check
    const status = $("#navStatus");
    fetch("/api/health")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d) => {
        status.classList.add("is-online");
        $(".status-text", status).textContent = `${d.model || "Model"} online`;
      })
      .catch(() => {
        status.classList.add("is-offline");
        $(".status-text", status).textContent = "Model offline";
      });
  };

  /* ------------------------------------------------------------------
     Scroll reveal + count-up
     ------------------------------------------------------------------ */
  const countUp = (node) => {
    const target = parseFloat(node.dataset.count);
    const decimals = parseInt(node.dataset.decimals || "0", 10);
    const suffix = node.dataset.suffix || "";
    if (REDUCED) { node.textContent = target.toFixed(decimals) + suffix; return; }
    const start = performance.now();
    const dur = 1400;
    const tick = (t) => {
      const p = Math.min(1, (t - start) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      node.textContent = (target * eased).toFixed(decimals) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };

  const initReveal = () => {
    $$(".reveal").forEach((n) => { if (n.dataset.delay) n.style.setProperty("--delay", `${n.dataset.delay}ms`); });
    const obs = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        // Reveal when entering the viewport, or if already scrolled past (hash landings).
        if (!e.isIntersecting && e.boundingClientRect.bottom > 0) return;
        e.target.classList.add("in");
        $$("[data-count]", e.target).forEach((c) => { if (!c.dataset.done) { c.dataset.done = "1"; countUp(c); } });
        if (e.target.dataset.count && !e.target.dataset.done) { e.target.dataset.done = "1"; countUp(e.target); }
        obs.unobserve(e.target);
      });
    }, { threshold: 0, rootMargin: "0px 0px -8% 0px" });
    $$(".reveal").forEach((n) => obs.observe(n));
    // count-ups that live outside .reveal containers
    $$("[data-count]").forEach((c) => { if (!c.closest(".reveal")) obs.observe(c); });
  };

  /* ------------------------------------------------------------------
     Hero visual — Sun, solar wind, ejecta, L1 vantage
     ------------------------------------------------------------------ */
  const initSun = () => {
    const canvas = $("#sunCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let W = 0, H = 0, dpr = 1;
    const wind = [];
    const ejecta = [];
    let burstTimer = 2.5;
    let last = performance.now();

    const resize = () => {
      dpr = Math.min(2, window.devicePixelRatio || 1);
      const r = canvas.getBoundingClientRect();
      W = r.width; H = r.height;
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const sun = () => ({ x: W * 0.22, y: H * 0.5, r: Math.min(W, H) * 0.13 });
    const l1 = () => ({ x: W * 0.84, y: H * 0.5 });

    const spawnWind = () => {
      const s = sun();
      const a = (Math.random() - 0.5) * Math.PI * 1.2;
      wind.push({
        x: s.x + Math.cos(a) * (s.r + 4),
        y: s.y + Math.sin(a) * (s.r + 4),
        vx: Math.cos(a) * (28 + Math.random() * 26),
        vy: Math.sin(a) * (28 + Math.random() * 26),
        life: 0, ttl: 5 + Math.random() * 3, size: 0.8 + Math.random() * 1.2,
      });
    };
    for (let i = 0; i < 90; i++) { spawnWind(); const p = wind[wind.length - 1]; const t = Math.random() * 4; p.x += p.vx * t; p.y += p.vy * t; p.life = t; }

    const spawnBurst = () => {
      const s = sun();
      const base = (Math.random() - 0.5) * 0.5; // mostly Earth-directed
      for (let i = 0; i < 70; i++) {
        const a = base + (Math.random() - 0.5) * 0.9;
        const sp = 46 + Math.random() * 34;
        ejecta.push({
          x: s.x + Math.cos(a) * (s.r + 2), y: s.y + Math.sin(a) * (s.r + 2),
          vx: Math.cos(a) * sp, vy: Math.sin(a) * sp,
          life: 0, ttl: 6.5, size: 1.2 + Math.random() * 1.8,
        });
      }
    };

    const drawSun = (t) => {
      const s = sun();
      // corona glow
      const g = ctx.createRadialGradient(s.x, s.y, s.r * 0.6, s.x, s.y, s.r * 3.2);
      g.addColorStop(0, "rgba(232,164,64,0.32)");
      g.addColorStop(0.5, "rgba(232,164,64,0.08)");
      g.addColorStop(1, "rgba(232,164,64,0)");
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(s.x, s.y, s.r * 3.2, 0, Math.PI * 2); ctx.fill();

      // rotating dashed rings
      for (let i = 0; i < 3; i++) {
        ctx.save();
        ctx.translate(s.x, s.y);
        ctx.rotate((REDUCED ? 0 : t * 0.00008 * (i % 2 ? -1 : 1)) + i);
        ctx.beginPath();
        ctx.setLineDash([3, 9 + i * 4]);
        ctx.strokeStyle = `rgba(240,200,140,${0.28 - i * 0.07})`;
        ctx.lineWidth = 1;
        ctx.arc(0, 0, s.r * (1.45 + i * 0.42), 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      // disc
      const d = ctx.createRadialGradient(s.x - s.r * 0.35, s.y - s.r * 0.35, s.r * 0.1, s.x, s.y, s.r);
      d.addColorStop(0, "#ffd58a");
      d.addColorStop(0.55, "#e8a24a");
      d.addColorStop(1, "#b9711f");
      ctx.fillStyle = d;
      ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "rgba(255,225,170,0.5)"; ctx.lineWidth = 1;
      ctx.stroke();
    };

    const drawL1 = () => {
      const p = l1();
      const s = sun();
      // orbit guide
      ctx.beginPath();
      ctx.setLineDash([2, 6]);
      ctx.strokeStyle = "rgba(200,200,190,0.18)";
      ctx.lineWidth = 1;
      ctx.moveTo(s.x + s.r * 3.3, s.y);
      ctx.lineTo(p.x - 14, p.y);
      ctx.stroke();
      ctx.setLineDash([]);
      // spacecraft marker
      ctx.beginPath(); ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(240,236,228,0.55)"; ctx.lineWidth = 1; ctx.stroke();
      ctx.beginPath(); ctx.arc(p.x, p.y, 3.2, 0, Math.PI * 2);
      ctx.fillStyle = "#f0ece4"; ctx.fill();
      ctx.font = "600 10px 'JetBrains Mono', monospace";
      ctx.fillStyle = "rgba(240,236,228,0.8)";
      ctx.textAlign = "center";
      ctx.fillText("SWIS", p.x, p.y - 18);
    };

    const step = (p, dt) => { p.x += p.vx * dt; p.y += p.vy * dt; p.life += dt; };

    const frame = (now) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      ctx.clearRect(0, 0, W, H);

      // faint star field
      ctx.fillStyle = "rgba(240,236,228,0.06)";
      for (let i = 0; i < 40; i++) {
        const x = ((i * 977) % 1000) / 1000 * W;
        const y = ((i * 613) % 1000) / 1000 * H;
        ctx.fillRect(x, y, 1.2, 1.2);
      }

      drawSun(now);

      if (!REDUCED) {
        if (wind.length < 130 && Math.random() < 0.6) spawnWind();
        burstTimer -= dt;
        if (burstTimer <= 0) { spawnBurst(); burstTimer = 7 + Math.random() * 4; }
      }

      const s = sun();
      const p1 = l1();
      for (let i = wind.length - 1; i >= 0; i--) {
        const p = wind[i];
        if (!REDUCED) step(p, dt);
        const fade = Math.min(1, p.life * 2) * Math.max(0, 1 - p.life / p.ttl);
        ctx.fillStyle = `rgba(157,183,201,${0.55 * fade})`;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2); ctx.fill();
        if (p.life > p.ttl || p.x > W + 10 || p.y < -10 || p.y > H + 10) { wind.splice(i, 1); }
      }
      for (let i = ejecta.length - 1; i >= 0; i--) {
        const p = ejecta[i];
        step(p, dt);
        const fade = Math.min(1, p.life * 3) * Math.max(0, 1 - p.life / p.ttl);
        ctx.fillStyle = `rgba(232,164,64,${0.85 * fade})`;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2); ctx.fill();
        // impact ring when passing L1
        if (Math.abs(p.x - p1.x) < 2 && Math.abs(p.y - p1.y) < 26) {
          ctx.beginPath(); ctx.arc(p1.x, p1.y, 16, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(232,164,64,0.35)"; ctx.lineWidth = 1.5; ctx.stroke();
        }
        if (p.life > p.ttl || p.x > W + 10) ejecta.splice(i, 1);
      }
      void s;

      drawL1();
      if (!REDUCED) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  };

  /* ------------------------------------------------------------------
     Upload + prediction
     ------------------------------------------------------------------ */
  const state = { file: null, result: null };

  const initUpload = () => {
    const dz = $("#dropzone");
    const input = $("#fileInput");
    const fileLabel = $("#dropzoneFile");
    const analyse = $("#analyseBtn");
    const reset = $("#resetBtn");
    const err = $("#uploadError");

    const setFile = (f) => {
      if (!f) return;
      if (!/\.csv$/i.test(f.name)) { showError("Only .csv files are accepted."); return; }
      state.file = f;
      fileLabel.textContent = `${f.name} · ${(f.size / 1024).toFixed(1)} KB`;
      fileLabel.hidden = false;
      dz.classList.add("has-file");
      analyse.disabled = false;
      reset.hidden = false;
      hideError();
    };
    const showError = (msg) => { err.textContent = msg; err.hidden = false; };
    const hideError = () => { err.hidden = true; err.textContent = ""; };

    dz.addEventListener("click", () => input.click());
    dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
    input.addEventListener("change", () => setFile(input.files[0]));

    ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("is-over"); }));
    ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("is-over"); }));
    dz.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

    const sample = $("#sampleBtn");
    sample.addEventListener("click", async () => {
      sample.disabled = true;
      try {
        const res = await fetch("/static/samples/swis_window_2025-06-14.csv");
        if (!res.ok) throw new Error("Sample file unavailable.");
        const blob = await res.blob();
        setFile(new File([blob], "swis_window_2025-06-14.csv", { type: "text/csv" }));
      } catch (e) {
        showError(e.message || "Could not load the sample file.");
      } finally {
        sample.disabled = false;
      }
    });

    reset.addEventListener("click", () => {
      state.file = null; state.result = null;
      input.value = "";
      fileLabel.hidden = true; dz.classList.remove("has-file");
      analyse.disabled = true; reset.hidden = true;
      hideError();
      const results = $("#results");
      results.hidden = true;
    });

    analyse.addEventListener("click", async () => {
      if (!state.file) return;
      hideError();
      analyse.classList.add("is-loading");
      analyse.disabled = true;
      $(".btn-label", analyse).textContent = "Analysing";
      const fd = new FormData();
      fd.append("file", state.file);
      try {
        const res = await fetch("/api/predict", { method: "POST", body: fd });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
        state.result = data;
        renderResults(data);
      } catch (e) {
        showError(e.message || "Something went wrong while analysing the file.");
      } finally {
        analyse.classList.remove("is-loading");
        analyse.disabled = false;
        $(".btn-label", analyse).textContent = "Run detection";
      }
    });
  };

  /* ------------------------------------------------------------------
     Results rendering
     ------------------------------------------------------------------ */
  const renderResults = (d) => {
    const results = $("#results");
    results.hidden = false;
    results.classList.remove("is-entering");
    void results.offsetWidth; // restart animation
    results.classList.add("is-entering");

    renderVerdict(d);
    renderVotes(d);
    renderDataset(d);
    renderFeatures(d);
    renderSeries(d);
    renderPreview(d);

    requestAnimationFrame(() => {
      results.scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "start" });
    });
  };

  const renderVerdict = (d) => {
    const card = $("#verdictCard");
    const isCME = d.prediction === "CME";
    card.classList.toggle("is-cme", isCME);
    card.classList.toggle("is-quiet", !isCME);
    card.style.setProperty("--gauge-color", isCME ? "var(--ember)" : "var(--sage)");

    $("#verdictFile").textContent = d.filename || "";
    $("#verdictEyebrow").textContent = isCME ? "Classification — positive" : "Classification — negative";
    $("#verdictTitle").textContent = isCME ? "Halo CME detected" : "No CME signature";
    $("#verdictDesc").textContent = isCME
      ? "The window exhibits plasma composition and variability consistent with CME ejecta. Space-weather alert protocols are recommended."
      : "Plasma parameters are consistent with ambient solar wind. No Halo CME signature was found in this window.";
    $("#verdictThreshold").textContent = fmtPct(d.threshold, 0);
    const margin = d.probability - d.threshold;
    $("#verdictMargin").textContent = `${margin >= 0 ? "+" : ""}${(margin * 100).toFixed(1)} pts`;

    // gauge
    const fill = $("#gaugeFill");
    const len = fill.getTotalLength();
    fill.style.strokeDasharray = `${len}`;
    fill.style.strokeDashoffset = `${len}`;
    // threshold tick
    const theta = Math.PI - Math.PI * d.threshold;
    const tick = $("#gaugeThreshold");
    tick.setAttribute("x1", 100 + 68 * Math.cos(theta)); tick.setAttribute("y1", 105 - 68 * Math.sin(theta));
    tick.setAttribute("x2", 100 + 92 * Math.cos(theta)); tick.setAttribute("y2", 105 - 92 * Math.sin(theta));

    requestAnimationFrame(() => requestAnimationFrame(() => {
      fill.style.strokeDashoffset = `${len * (1 - d.probability)}`;
    }));

    // animated readout
    const readout = $("#gaugeValue");
    if (REDUCED) { readout.textContent = fmtPct(d.probability); return; }
    const start = performance.now();
    const tickFn = (t) => {
      const p = Math.min(1, (t - start) / 1400);
      const eased = 1 - Math.pow(1 - p, 3);
      readout.textContent = fmtPct(d.probability * eased);
      if (p < 1) requestAnimationFrame(tickFn);
    };
    requestAnimationFrame(tickFn);
  };

  const renderVotes = (d) => {
    const list = $("#voteList");
    list.innerHTML = "";
    const votes = d.votes && d.votes.length ? d.votes : [];
    if (!votes.length) {
      list.append(el("li", { class: "card-sub", text: "Per-estimator probabilities unavailable." }));
      return;
    }
    votes.forEach((v) => {
      const meta = ESTIMATOR_NAMES[v.name] || [v.name, ""];
      const over = v.probability >= d.threshold;
      const row = el("li", { class: "vote-row", style: `--vote-color:${over ? "var(--ember)" : "var(--sage)"};--threshold:${d.threshold}` }, [
        el("span", { class: "vote-name", html: `${meta[0]}<small>${meta[1] || v.name}</small>` }),
        el("span", { class: "vote-val", text: fmtPct(v.probability) }),
        el("span", { class: "vote-bar" }, [el("i")]),
      ]);
      list.append(row);
      requestAnimationFrame(() => requestAnimationFrame(() => { $("i", row).style.width = `${v.probability * 100}%`; }));
    });
  };

  const renderDataset = (d) => {
    const ds = d.dataset || {};
    const dl = $("#datasetMeta");
    dl.innerHTML = "";
    const rows = [
      ["Rows", ds.rows != null ? ds.rows.toLocaleString() : "–"],
      ["Columns", ds.columns ? ds.columns.length : "–"],
      ["Start", fmtDate(ds.start)],
      ["End", fmtDate(ds.end)],
      ["Span", fmtDuration(ds.span_hours)],
      ["Cadence", ds.cadence_seconds != null ? `${Math.round(ds.cadence_seconds / 60)} min` : "–"],
    ];
    rows.forEach(([k, v]) => dl.append(el("div", {}, [el("dt", { text: k }), el("dd", { class: "mono", text: String(v) })])));
  };

  const renderFeatures = (d) => {
    const grid = $("#featureGrid");
    grid.innerHTML = "";
    Object.entries(d.features || {}).forEach(([key, val], i) => {
      const m = FEATURE_META[key] || { title: key, formula: "", unit: "", desc: "" };
      const card = el("article", { class: "clay clay--flat feature-card", style: `animation-delay:${i * 70}ms` }, [
        el("span", { class: "formula mono", text: m.formula }),
        el("h4", { text: m.title }),
        el("span", { class: "feature-val", html: `${fmtNum(val)}${m.unit ? `<small>${m.unit}</small>` : ""}` }),
        el("p", { class: "feature-desc", text: m.desc }),
      ]);
      grid.append(card);
    });
  };

  const renderPreview = (d) => {
    const table = $("#previewTable");
    table.innerHTML = "";
    const rows = d.preview || [];
    if (!rows.length) return;
    const cols = Object.keys(rows[0]);
    const thead = el("thead", {}, [el("tr", {}, cols.map((c) => el("th", { text: c })))]);
    const tbody = el("tbody", {}, rows.map((r) => el("tr", {}, cols.map((c) => {
      const v = r[c];
      const text = typeof v === "number" ? fmtNum(v, 3) : (v == null ? "–" : String(v));
      return el("td", { text });
    }))));
    table.append(thead, tbody);
  };

  /* ------------------------------------------------------------------
     Time-series small multiples (dependency-free SVG)
     ------------------------------------------------------------------ */
  let seriesCache = null;
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(drawAllSeries, 180);
  });

  const renderSeries = (d) => {
    seriesCache = d.series;
    const ds = d.dataset || {};
    $("#seriesSub").textContent = `${fmtDate(ds.start)} → ${fmtDate(ds.end)} · ${(d.series.timestamp || []).length} plotted samples`;
    drawAllSeries();
  };

  const drawAllSeries = () => {
    if (!seriesCache) return;
    const grid = $("#seriesGrid");
    grid.innerHTML = "";
    const ts = (seriesCache.timestamp || []).map((t) => new Date(t).getTime());
    Object.keys(SERIES_META).forEach((key, i) => {
      const values = seriesCache[key] || [];
      const m = SERIES_META[key];
      const clean = values.filter((v) => v != null && Number.isFinite(v));
      const min = Math.min(...clean), max = Math.max(...clean);
      const mean = clean.reduce((a, b) => a + b, 0) / (clean.length || 1);
      const card = el("article", { class: "clay clay--flat series-card", style: `animation-delay:${i * 70}ms` }, [
        el("div", { class: "series-head" }, [
          el("span", { class: "series-title", html: `${m.title} <span class="series-unit">${m.symbol}</span>` }),
          el("span", { class: "series-unit", text: m.unit }),
        ]),
        el("div", { class: "series-stats", html: `<span>min <b>${fmtNum(min)}</b></span><span>mean <b>${fmtNum(mean)}</b></span><span>max <b>${fmtNum(max)}</b></span>` }),
      ]);
      grid.append(card);
      const width = Math.max(280, card.clientWidth - 40);
      const svg = buildChart(ts, values, `grad-${key}`, i, width);
      card.append(svg);
    });
  };

  const buildChart = (ts, values, gradId, idx, W = 520) => {
    const H = 170;
    const pad = { l: 46, r: 10, t: 10, b: 24 };
    const svg = svgEl("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: W, height: H });
    const defs = svgEl("defs");
    const grad = svgEl("linearGradient", { id: gradId, x1: 0, y1: 0, x2: 0, y2: 1 });
    grad.append(svgEl("stop", { offset: "0", "stop-color": "rgba(217,146,42,0.32)" }));
    grad.append(svgEl("stop", { offset: "1", "stop-color": "rgba(217,146,42,0)" }));
    defs.append(grad);
    svg.append(defs);

    const pts = ts.map((t, i) => [t, values[i]]).filter((p) => p[1] != null && Number.isFinite(p[1]));
    if (pts.length < 2) return svg;
    const x0 = pts[0][0], x1 = pts[pts.length - 1][0];
    let y0 = Math.min(...pts.map((p) => p[1])), y1 = Math.max(...pts.map((p) => p[1]));
    if (y0 === y1) { y0 -= 1; y1 += 1; }
    const yPad = (y1 - y0) * 0.08;
    const nonNeg = y0 >= 0;
    y0 -= yPad; y1 += yPad;
    if (nonNeg && y0 < 0) y0 = 0;
    const sx = (x) => pad.l + ((x - x0) / (x1 - x0 || 1)) * (W - pad.l - pad.r);
    const sy = (y) => pad.t + (1 - (y - y0) / (y1 - y0)) * (H - pad.t - pad.b);

    // grid + y labels
    for (let i = 0; i <= 3; i++) {
      const yv = y0 + ((y1 - y0) * i) / 3;
      const y = sy(yv);
      svg.append(svgEl("line", { class: "grid-line", x1: pad.l, x2: W - pad.r, y1: y, y2: y }));
      const t = svgEl("text", { class: "axis-text", x: pad.l - 6, y: y + 3, "text-anchor": "end" });
      t.textContent = fmtNum(yv, 1);
      svg.append(t);
    }
    // x labels
    const nx = W > 460 ? 4 : 2;
    for (let i = 0; i < nx; i++) {
      const xv = x0 + ((x1 - x0) * i) / (nx - 1);
      const t = svgEl("text", { class: "axis-text", x: sx(xv), y: H - 6, "text-anchor": i === 0 ? "start" : i === nx - 1 ? "end" : "middle" });
      t.textContent = new Date(xv).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", hour12: false });
      svg.append(t);
    }

    const lineD = pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)} ${sy(p[1]).toFixed(1)}`).join(" ");
    const areaD = `${lineD} L${sx(x1).toFixed(1)} ${sy(y0)} L${sx(x0).toFixed(1)} ${sy(y0)} Z`;
    svg.append(svgEl("path", { class: "area", d: areaD, fill: `url(#${gradId})` }));
    const line = svgEl("path", { class: "line", d: lineD });
    svg.append(line);

    // hover
    const cursor = svgEl("line", { class: "cursor", y1: pad.t, y2: H - pad.b, x1: 0, x2: 0 });
    const dot = svgEl("circle", { class: "cursor-dot", r: 3.5 });
    const tip = svgEl("text", { class: "tip", "text-anchor": "middle" });
    svg.append(cursor, dot, tip);

    const onMove = (e) => {
      const r = svg.getBoundingClientRect();
      const px = ((e.clientX - r.left) / r.width) * W;
      const xv = x0 + ((px - pad.l) / (W - pad.l - pad.r)) * (x1 - x0);
      let lo = 0, hi = pts.length - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (pts[mid][0] < xv) lo = mid + 1; else hi = mid; }
      const near = lo > 0 && Math.abs(pts[lo - 1][0] - xv) < Math.abs(pts[lo][0] - xv) ? pts[lo - 1] : pts[lo];
      const cx = sx(near[0]), cy = sy(near[1]);
      cursor.setAttribute("x1", cx); cursor.setAttribute("x2", cx);
      dot.setAttribute("cx", cx); dot.setAttribute("cy", cy);
      const tx = Math.min(W - 60, Math.max(pad.l + 50, cx));
      tip.setAttribute("x", tx); tip.setAttribute("y", Math.max(pad.t + 10, cy - 10));
      tip.textContent = `${fmtNum(near[1])}  ·  ${new Date(near[0]).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false })}`;
      svg.classList.add("is-hover");
    };
    svg.addEventListener("mousemove", onMove);
    svg.addEventListener("mouseleave", () => svg.classList.remove("is-hover"));

    // draw animation (needs to be in DOM to measure)
    requestAnimationFrame(() => {
      if (REDUCED) return;
      const len = line.getTotalLength();
      line.style.setProperty("--len", len);
      line.style.animationDelay = `${idx * 120}ms`;
      line.classList.add("is-drawing");
    });
    return svg;
  };

  /* ------------------------------------------------------------------
     Boot
     ------------------------------------------------------------------ */
  document.addEventListener("DOMContentLoaded", () => {
    initNav();
    initReveal();
    initSun();
    initUpload();
  });
})();
