/* Looped 17M Explainer. Every number shown is recorded from the real models by `rh17m loop-explainer`. */
(() => {
  "use strict";
  const DATA = "data/";
  const S = {
    meta: null, rec: null, cache: new Map(), model: "looped", loopIdx: 1, temp: null,
    stage: null, head: "mean", focus: 0, query: null, space: null, playing: null,
  };
  const $ = (id) => document.getElementById(id);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const tip = $("tip");
  const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "–" : (+v).toFixed(d));
  const pct = (v) => `${(100 * v).toFixed(1)}%`;

  // ------------------------------------------------------------------ helpers
  function softmax(xs, t) {
    const m = Math.max(...xs);
    const e = xs.map((x) => Math.exp((x - m) / t));
    const z = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / z);
  }
  function shortLabel(text, key) {
    if (key === "hold") return "Hold: build nothing";
    const m = /air route (.+?) \(\d+ people\) to (.+?) \(\d+ people\), (\d+) tiles(?:, estimated cost (£[\d.,]+[kM]?))?/.exec(text);
    if (m) return `${m[1]} → ${m[2]} · ${m[3]} tiles`;
    return text.length > 46 ? `${text.slice(0, 45)}…` : text;
  }
  function showTip(html, ev) {
    tip.innerHTML = html;
    tip.style.opacity = 1;
    const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
    if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  }
  const hideTip = () => { tip.style.opacity = 0; };
  function valueColor() {
    const v = S.rec.value, lo = Math.min(...v), hi = Math.max(...v);
    const interp = d3.interpolateRgbBasis([css("--ramp-1"), css("--ramp-2"), css("--ramp-3"), css("--ramp-4")]);
    return (k) => interp((v[k] - lo) / Math.max(hi - lo, 1e-9));
  }
  function heatInterp() {
    return d3.interpolateRgbBasis([css("--heat-0"), css("--ramp-1"), css("--ramp-2"), css("--ramp-3"), css("--ramp-4")]);
  }
  const kindLetter = (k) => (k === "full_attention" ? "F" : "S");
  const kindColor = (k) => css(k === "full_attention" ? "--full" : "--slide");
  const kindSoft = (k) => css(k === "full_attention" ? "--full-soft" : "--slide-soft");

  // ------------------------------------------------------------------ the current view of the model
  function view() {
    const rec = S.rec, types = S.meta.config.layer_types;
    if (S.model === "original") {
      const m = rec.models.original;
      return {
        probe: m.probe, xy: m.xy, temperature: S.meta.temperature.original,
        stages: [{ name: "embeddings", kind: "embed", layer: null, loop: 0 },
          ...types.map((k, i) => ({ name: `L${i}`, kind: k, layer: i, loop: 0, weights: `layer ${i}` }))],
        attention: true, modelKey: "original",
      };
    }
    const n = S.meta.loop_counts[S.loopIdx];
    const l = rec.loops[String(n)];
    const stages = [{ name: "embeddings", kind: "embed", layer: null, loop: 0 },
      { name: "L0", kind: types[0], layer: 0, loop: 0, weights: "layer 0" }];
    for (let r = 1; r <= n; r += 1) {
      for (let j = 0; j < 3; j += 1) {
        stages.push({ name: `B${j + 1}`, kind: types[1 + j], layer: n === 2 ? stages.length - 1 : null, loop: r,
          weights: `block layer ${j + 1}`, slot: j });
      }
    }
    return { probe: l.probe, xy: l.xy, temperature: S.meta.temperature.looped, stages, attention: n === 2, modelKey: "looped", loops: n };
  }

  // ------------------------------------------------------------------ flow (bump chart through the layers)
  function drawFlow() {
    const v = view(), rec = S.rec, K = rec.keys.length;
    const host = $("flow");
    const width = Math.max(host.clientWidth, 330 + 290 + v.stages.length * 40);
    const rowH = 38, top = 74, height = top + K * rowH + 26;
    const left = 330, right = width - 290;
    const x = d3.scalePoint().domain(d3.range(v.stages.length)).range([left + 30, right - 30]);
    const step = v.stages.length > 1 ? x(1) - x(0) : 60;
    const cardW = Math.max(14, Math.min(40, step * 0.5));
    const probs = v.probe.map((row) => softmax(row, S.temp));
    const ranks = v.probe.map((row) => {
      const order = d3.range(K).sort((a, b) => row[b] - row[a]);
      const r = new Array(K);
      order.forEach((k, i) => { r[k] = i; });
      return r;
    });
    const y = (s, k) => top + ranks[s][k] * rowH + rowH / 2;
    const color = valueColor();
    const best = rec.keys.indexOf(rec.best);
    const last = v.stages.length - 1;
    const final = probs[last];
    const pick = d3.maxIndex(final);

    let svg = d3.select(host).select("svg");
    if (svg.empty()) {
      svg = d3.select(host).append("svg");
      svg.append("g").attr("class", "g-loops");
      svg.append("g").attr("class", "g-cards");
      svg.append("g").attr("class", "g-ribbons");
      svg.append("g").attr("class", "g-left");
      svg.append("g").attr("class", "g-right");
    }
    svg.attr("width", width).attr("height", height).attr("viewBox", `0 0 ${width} ${height}`);
    const t = svg.transition().duration(650).ease(d3.easeCubicInOut);

    // loop brackets above the cards
    const groups = [];
    if (S.model === "looped") {
      for (let r = 1; r <= v.loops; r += 1) {
        const idx = v.stages.map((s, i) => (s.loop === r ? i : -1)).filter((i) => i >= 0);
        groups.push({ r, x0: x(idx[0]) - cardW / 2 - 4, x1: x(idx[idx.length - 1]) + cardW / 2 + 4 });
      }
    }
    svg.select(".g-loops").selectAll("g.loop").data(groups, (d) => d.r).join(
      (enter) => {
        const g = enter.append("g").attr("class", "loop").attr("opacity", 0);
        g.append("path");
        g.append("text").attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600);
        return g;
      },
      (update) => update,
      (exit) => exit.transition(t).attr("opacity", 0).remove(),
    ).call((g) => {
      g.transition(t).attr("opacity", 1);
      g.select("path").transition(t)
        .attr("d", (d) => `M${d.x0},${top - 30} V${top - 36} H${d.x1} V${top - 30}`)
        .attr("fill", "none").attr("stroke", css("--accent")).attr("stroke-width", 1.4);
      g.select("text").transition(t).attr("x", (d) => (d.x0 + d.x1) / 2).attr("y", top - 42)
        .attr("fill", css("--accent"))
        .text((d) => (d.r === 1 ? "loop 1 · the shared block" : `loop ${d.r} · same weights`));
    });

    // layer cards
    const cards = svg.select(".g-cards").selectAll("g.layer-card").data(v.stages, (d, i) => `${S.model}-${i}`).join(
      (enter) => {
        const g = enter.append("g").attr("class", "layer-card").attr("opacity", 0);
        g.append("rect").attr("class", "body");
        g.append("text").attr("class", "name").attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600);
        g.append("text").attr("class", "kind").attr("text-anchor", "middle").attr("font-size", 10);
        return g;
      },
      (update) => update,
      (exit) => exit.transition(t).attr("opacity", 0).remove(),
    );
    cards.classed("sel", (d, i) => S.stage === i)
      .on("click", (ev, d) => { selectStage(v.stages.indexOf(d)); })
      .on("mousemove", (ev, d) => {
        const desc = d.kind === "embed" ? "Token embeddings + norm. 12.9M of the 16.8M parameters live in this table."
          : d.kind === "full_attention" ? "Full attention: every option token reads the whole prefix and its own branch."
            : `Sliding attention: reads only positions within ±${S.meta.config.window}.`;
        const reuse = d.loop > 1 ? `<div class="muted">Reuses the weights of ${d.weights}, loop 1.</div>` : "";
        showTip(`<b>${d.kind === "embed" ? "Embeddings" : `${d.name} · ${d.kind === "full_attention" ? "full" : "sliding"}`}</b>${d.loop ? ` <span class="muted">loop ${d.loop}</span>` : ""}<div>${desc}</div>${reuse}<div class="muted">Click to inspect.</div>`, ev);
      })
      .on("mouseleave", hideTip);
    cards.transition(t).attr("opacity", 1);
    cards.select("rect.body").transition(t)
      .attr("x", (d, i) => x(i) - cardW / 2).attr("y", top - 22).attr("width", cardW).attr("height", K * rowH + 30).attr("rx", 8)
      .attr("fill", (d) => (d.kind === "embed" ? css("--surface-2") : kindSoft(d.kind)))
      .attr("stroke", (d) => (d.kind === "embed" ? css("--line-2") : kindColor(d.kind)))
      .attr("stroke-width", (d, i) => (S.stage === i ? 2.5 : 1))
      .attr("stroke-dasharray", (d) => (d.loop > 1 ? "4 3" : null));
    cards.select("text.name").transition(t).attr("x", (d, i) => x(i)).attr("y", top - 8)
      .attr("fill", (d) => (d.kind === "embed" ? css("--ink-2") : kindColor(d.kind)))
      .text((d) => (d.kind === "embed" ? (step > 50 ? "emb" : "E") : d.name));
    cards.select("text.kind").transition(t).attr("x", (d, i) => x(i)).attr("y", top + K * rowH + 2)
      .attr("fill", css("--muted")).text((d) => (d.kind === "embed" ? "" : kindLetter(d.kind)));

    // ribbons: one segment per pair of stages, width = probe probability at that stage
    const segs = [];
    for (let k = 0; k < K; k += 1) {
      segs.push({ k, s: -1, d: d3.linkHorizontal()({ source: [left, top + ranks[0][k] * rowH + rowH / 2], target: [x(0), y(0, k)] }), w: probs[0][k] });
      for (let s = 0; s < last; s += 1) {
        segs.push({ k, s, d: d3.linkHorizontal()({ source: [x(s), y(s, k)], target: [x(s + 1), y(s + 1, k)] }), w: (probs[s][k] + probs[s + 1][k]) / 2 });
      }
      segs.push({ k, s: last, d: d3.linkHorizontal()({ source: [x(last), y(last, k)], target: [right, y(last, k)] }), w: probs[last][k] });
    }
    const widthOf = (p) => 1.5 + 40 * p;
    svg.select(".g-ribbons").selectAll("path.ribbon").data(segs, (d) => `${d.k}:${d.s}`).join(
      (enter) => enter.append("path").attr("class", "ribbon").attr("opacity", 0).attr("d", (d) => d.d),
      (update) => update,
      (exit) => exit.transition(t).attr("opacity", 0).remove(),
    ).attr("data-k", (d) => d.k)
      .on("mousemove", (ev, d) => { hot(d.k); optionTip(d.k, ev, v, probs, ranks); })
      .on("mouseleave", () => { hot(null); hideTip(); })
      .on("click", (ev, d) => setFocus(d.k))
      .transition(t).attr("opacity", 0.9).attr("d", (d) => d.d)
      .attr("stroke", (d) => color(d.k)).attr("stroke-width", (d) => widthOf(d.w));

    // left labels: options in the order of the embedding-stage probe
    const leftRows = svg.select(".g-left").selectAll("g.opt-label").data(d3.range(K), (k) => k).join((enter) => {
      const g = enter.append("g").attr("class", "opt-label");
      g.append("circle").attr("r", 5);
      g.append("text").attr("class", "lab").attr("text-anchor", "end").attr("font-size", 12);
      return g;
    });
    leftRows.on("mousemove", (ev, k) => { hot(k); optionTip(k, ev, v, probs, ranks); })
      .on("mouseleave", () => { hot(null); hideTip(); }).on("click", (ev, k) => setFocus(k));
    leftRows.transition(t).attr("transform", (k) => `translate(0,${top + ranks[0][k] * rowH + rowH / 2})`);
    leftRows.select("circle").attr("cx", left - 6).attr("fill", (k) => color(k));
    leftRows.select("text.lab").attr("x", left - 16).attr("dy", "0.35em").attr("fill", css("--ink"))
      .attr("font-weight", (k) => (k === best ? 700 : 400))
      .text((k) => shortLabel(rec.options[k].text, rec.keys[k]));

    // right: final choice distribution and the simulator's value
    const right0 = right + 10;
    const head = svg.select(".g-right").selectAll("g.head").data([0]).join((enter) => {
      const g = enter.append("g").attr("class", "head");
      g.append("text").attr("class", "h1");
      g.append("text").attr("class", "h2");
      return g;
    });
    head.select(".h1").attr("x", right0).attr("y", top - 30).attr("font-size", 11).attr("font-weight", 600).attr("fill", css("--ink-2")).text("P(pick)");
    head.select(".h2").attr("x", right0 + 150).attr("y", top - 30).attr("font-size", 11).attr("font-weight", 600).attr("fill", css("--ink-2")).text("sim value");
    const maxP = Math.max(...final, 0.3);
    const rows = svg.select(".g-right").selectAll("g.prob").data(d3.range(K), (k) => k).join((enter) => {
      const g = enter.append("g").attr("class", "prob");
      g.append("rect").attr("class", "track");
      g.append("rect").attr("class", "bar");
      g.append("text").attr("class", "pv");
      g.append("text").attr("class", "sv");
      g.append("text").attr("class", "flag");
      return g;
    });
    rows.on("mousemove", (ev, k) => { hot(k); optionTip(k, ev, v, probs, ranks); })
      .on("mouseleave", () => { hot(null); hideTip(); }).on("click", (ev, k) => setFocus(k));
    rows.transition(t).attr("transform", (k) => `translate(${right0},${y(last, k)})`);
    rows.select(".track").attr("x", 0).attr("y", -5).attr("height", 10).attr("width", 86).attr("rx", 5).attr("fill", css("--surface-2"));
    rows.select(".bar").attr("x", 0).attr("y", -5).attr("height", 10).attr("rx", 5).attr("fill", (k) => color(k))
      .transition(t).attr("width", (k) => Math.max(2, (86 * final[k]) / maxP));
    rows.select(".pv").attr("x", 94).attr("dy", "0.35em").attr("font-size", 11.5).attr("font-variant-numeric", "tabular-nums")
      .attr("fill", css("--ink")).attr("font-weight", (k) => (k === pick ? 700 : 400)).text((k) => pct(final[k]));
    rows.select(".sv").attr("x", 150).attr("dy", "0.35em").attr("font-size", 11.5).attr("fill", css("--ink-2"))
      .attr("font-variant-numeric", "tabular-nums").text((k) => (rec.value[k] >= 0 ? "+" : "") + fmt(rec.value[k]));
    rows.select(".flag").attr("x", 196).attr("dy", "0.35em").attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", (k) => (k === best ? css("--good") : css("--accent")))
      .text((k) => [k === pick ? "pick" : "", k === best ? "best" : ""].filter(Boolean).join(" · "));

    const agree = pick === best;
    $("flow-note").innerHTML = `Ribbon height at each layer = the option's rank by the <b>probe</b> (final norm + choice head applied to that layer's stream); ribbon thickness = probe probability at temperature ${fmt(S.temp)}. `
      + `${S.model === "looped" ? `Looped model at <b>${v.loops} loop${v.loops > 1 ? "s" : ""}</b> (${1 + 3 * v.loops} layers run, 4 layers of weights). ` : "Original model, 7 layers of weights. "}`
      + `It picks <b>${shortLabel(rec.options[pick].text, rec.keys[pick])}</b>; the simulator's best is <b>${shortLabel(rec.options[best].text, rec.keys[best])}</b>${agree ? " (agree)." : "."}`;
  }

  function optionTip(k, ev, v, probs, ranks) {
    const rec = S.rec, last = v.stages.length - 1;
    const path = ranks.map((r) => r[k] + 1).join(" → ");
    showTip(`<b>${shortLabel(rec.options[k].text, rec.keys[k])}</b><div class="muted mono">${rec.keys[k]}</div>
      <div style="margin:4px 0">${rec.options[k].text}</div>
      <div class="row"><span>P(pick), final</span><b>${pct(probs[last][k])}</b></div>
      <div class="row"><span>simulator value</span><b>${fmt(rec.value[k], 3)}</b></div>
      <div class="row"><span>rank per layer</span><span class="mono">${path}</span></div>`, ev);
  }
  function hot(k) {
    const svg = d3.select("#flow svg");
    svg.classed("dim", k != null);
    svg.selectAll("path.ribbon").classed("hot", (d) => d.k === k);
  }

  // ------------------------------------------------------------------ inspector
  function selectStage(i) {
    S.stage = i;
    S.query = null;
    d3.selectAll("#flow g.layer-card").classed("sel", (d, j) => j === i)
      .select("rect.body").attr("stroke-width", (d, j) => (j === i ? 2.5 : 1));
    drawInspector();
  }
  function setFocus(k) {
    S.focus = k;
    S.query = null;
    drawFocusSeg();
    drawInspector();
  }
  async function loadMap(path) {
    if (S.cache.has(path)) return S.cache.get(path);
    const img = new Image();
    img.src = DATA + path;
    await img.decode();
    const c = document.createElement("canvas");
    c.width = img.width;
    c.height = img.height;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    const rgba = ctx.getImageData(0, 0, img.width, img.height).data;
    const gray = new Uint8Array(img.width * img.height);
    for (let i = 0; i < gray.length; i += 1) gray[i] = rgba[4 * i];
    const out = { size: img.height, width: img.width, gray };
    S.cache.set(path, out);
    return out;
  }
  function weightAt(map, head, row, col) {
    const n = map.size;
    if (head === "mean") {
      let s = 0;
      for (let h = 0; h < 4; h += 1) { const g = map.gray[row * map.width + h * n + col] / 255; s += g * g; }
      return s / 4;
    }
    const g = map.gray[row * map.width + head * n + col] / 255;
    return g * g;
  }

  function drawHeadSeg() {
    const seg = d3.select("#head-seg");
    seg.selectAll("button").data(["mean", 0, 1, 2, 3]).join("button")
      .text((h) => (h === "mean" ? "all heads" : `head ${h + 1}`))
      .classed("on", (h) => h === S.head)
      .on("click", (ev, h) => { S.head = h; drawHeadSeg(); drawInspector(); });
  }
  function drawFocusSeg() {
    const rec = S.rec, color = valueColor();
    const seg = d3.select("#focus-seg");
    seg.selectAll("button").data(d3.range(rec.keys.length)).join("button")
      .html((k) => `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color(k)};margin-right:5px"></span>${rec.keys[k] === "hold" ? "hold" : k + 1}`)
      .attr("title", (k) => shortLabel(rec.options[k].text, rec.keys[k]))
      .classed("on", (k) => k === S.focus)
      .on("click", (ev, k) => setFocus(k));
  }

  async function drawInspector() {
    const v = view(), rec = S.rec;
    if (S.stage == null || S.stage >= v.stages.length) S.stage = v.stages.length - 1;
    const st = v.stages[S.stage];
    const P = rec.sizes.prefix, branch = rec.sizes.branch;
    const starts = branch.reduce((acc, b, i) => { acc.push(i ? acc[i - 1] + branch[i - 1] : 0); return acc; }, []);
    $("insp-eyebrow").textContent = S.model === "original" ? "Original model · layer inspector" : `Looped model · ${v.loops} loop${v.loops > 1 ? "s" : ""} · layer inspector`;
    if (st.kind === "embed") {
      $("insp-title").textContent = "Embeddings";
      $("insp-sub").textContent = "Click a layer card in the flow to inspect its attention and MLP.";
    } else {
      const reuse = st.loop > 1 ? ` · runs the same weights as loop 1 (${st.weights})` : st.loop === 1 ? ` · ${st.weights} of the shared block` : "";
      $("insp-title").textContent = `${st.name} · ${st.kind === "full_attention" ? "full attention" : `sliding attention (±${S.meta.config.window})`}${st.loop ? ` · loop ${st.loop}` : ""}`;
      $("insp-sub").textContent = `${st.kind === "full_attention" ? "Every option token reads the whole prefix (instruction + state) and its own branch, never another option." : "Each token reads only keys within ±64 positions, so option tokens (which sit after the 270-token prefix) cannot see the instruction."}${reuse}`;
    }
    const layer = st.layer == null ? null : rec.models[v.modelKey].layers[st.layer];
    const ok = st.kind !== "embed" && v.attention && layer;
    for (const id of ["map-caption", "reader-caption"]) $(id).textContent = "";
    if (!ok) {
      const msg = st.kind === "embed" ? "No attention in the embedding stage."
        : "Attention was recorded for the original and the exported 2-loop model. Set Loops to 2 to inspect this layer.";
      const ctx = $("map").getContext("2d");
      ctx.clearRect(0, 0, $("map").width, $("map").height);
      $("map-overlay").innerHTML = `<div style="position:absolute;inset:0;display:grid;place-items:center;text-align:center;padding:30px;color:${css("--muted")}">${msg}</div>`;
      $("reader").innerHTML = `<span class="muted">${msg}</span>`;
      $("share").innerHTML = "";
      $("updates").innerHTML = "";
      $("neurons").innerHTML = "";
      return;
    }
    const map = await loadMap(layer.png);
    renderMap(map, P, starts, branch, st);
    renderReader(map, P, starts, branch, st);
    renderShare(layer);
    renderUpdates(layer, starts, branch);
    renderNeurons(layer);
  }

  function renderMap(map, P, starts, branch, st) {
    const n = map.size, canvas = $("map");
    canvas.width = n;
    canvas.height = n;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(n, n);
    const interp = heatInterp();
    const lut = d3.range(256).map((i) => d3.rgb(interp(i / 255)));
    // sqrt(weight) per cell, then a colour scale that saturates at the 99th percentile of the non-zero cells:
    // full-attention rows spread over ~300 keys, so raw weights would all sit near the bottom of the scale.
    const vals = new Float32Array(n * n);
    for (let r = 0; r < n; r += 1) {
      for (let c = 0; c < n; c += 1) {
        let g;
        if (S.head === "mean") {
          let s = 0;
          for (let h = 0; h < 4; h += 1) { const q = map.gray[r * map.width + h * n + c] / 255; s += q * q; }
          g = Math.sqrt(s / 4);
        } else g = map.gray[r * map.width + S.head * n + c] / 255;
        vals[r * n + c] = g;
      }
    }
    const nz = Array.from(vals.filter((x) => x > 0)).sort((a, b) => a - b);
    const top = nz.length ? nz[Math.floor(0.99 * (nz.length - 1))] : 1;
    for (let i = 0; i < n * n; i += 1) {
      const col = lut[Math.min(255, Math.round((255 * vals[i]) / top))], o = 4 * i;
      img.data[o] = col.r; img.data[o + 1] = col.g; img.data[o + 2] = col.b; img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    S.mapTop = top * top;
    const sec = S.rec.prefix.sections;
    const lines = [];
    sec.slice(1).forEach((s) => lines.push({ at: s.start, strong: false }));
    lines.push({ at: P, strong: true });
    starts.slice(1).forEach((s) => lines.push({ at: P + s, strong: false }));
    const fy = P + starts[S.focus], fh = branch[S.focus];
    const muted = css("--muted"), ink = css("--ink-2"), acc = css("--accent");
    $("map-overlay").innerHTML = `<svg viewBox="0 0 ${n} ${n}" preserveAspectRatio="none" width="100%" height="100%">
      ${lines.map((l) => `<line x1="${l.at}" x2="${l.at}" y1="0" y2="${n}" stroke="${l.strong ? ink : muted}" stroke-opacity="${l.strong ? 0.9 : 0.4}" vector-effect="non-scaling-stroke" stroke-width="${l.strong ? 1.2 : 0.6}"/>
        <line y1="${l.at}" y2="${l.at}" x1="0" x2="${n}" stroke="${l.strong ? ink : muted}" stroke-opacity="${l.strong ? 0.9 : 0.4}" vector-effect="non-scaling-stroke" stroke-width="${l.strong ? 1.2 : 0.6}"/>`).join("")}
      <rect x="0" y="${fy}" width="${n}" height="${fh}" fill="none" stroke="${acc}" stroke-width="1.6" vector-effect="non-scaling-stroke"/>
      ${S.query != null ? `<line x1="0" x2="${n}" y1="${S.query + 0.5}" y2="${S.query + 0.5}" stroke="${acc}" stroke-width="1.6" vector-effect="non-scaling-stroke"/>` : ""}
    </svg>`;
    $("map-caption").textContent = `${n} × ${n} tokens · ${S.head === "mean" ? "mean of 4 heads" : `head ${S.head + 1}`} · colour saturates at weight ${S.mapTop.toFixed(3)} · box = option ${S.rec.keys[S.focus] === "hold" ? "hold" : S.focus + 1}`;
    const tokenAt = (i) => {
      if (i < P) {
        const s = sec.find((q) => i >= q.start && i < q.end);
        return { text: S.rec.prefix.tokens[i], where: s ? s.name : "prefix" };
      }
      const j = i - P;
      let k = 0;
      while (k + 1 < starts.length && starts[k + 1] <= j) k += 1;
      return { text: S.rec.branches[k][j - starts[k]], where: `option ${S.rec.keys[k] === "hold" ? "hold" : k + 1}` };
    };
    canvas.onmousemove = (ev) => {
      const b = canvas.getBoundingClientRect();
      const c = Math.floor(((ev.clientX - b.left) / b.width) * n), r = Math.floor(((ev.clientY - b.top) / b.height) * n);
      if (r < 0 || c < 0 || r >= n || c >= n) return;
      const q = tokenAt(r), k = tokenAt(c), w = weightAt(map, S.head, r, c);
      showTip(`<div class="row"><span class="muted">reader</span><span><b class="mono">${JSON.stringify(q.text)}</b> <span class="muted">${q.where}</span></span></div>
        <div class="row"><span class="muted">reads</span><span><b class="mono">${JSON.stringify(k.text)}</b> <span class="muted">${k.where}</span></span></div>
        <div class="row"><span class="muted">weight</span><b>${w < 1e-4 ? (w === 0 ? "0 (masked or ~0)" : w.toExponential(1)) : w.toFixed(4)}</b></div>
        <div class="muted">Click to show this reader's attention as text.</div>`, ev);
    };
    canvas.onmouseleave = hideTip;
    canvas.onclick = (ev) => {
      const b = canvas.getBoundingClientRect();
      const r = Math.floor(((ev.clientY - b.top) / b.height) * n);
      S.query = r;
      if (r >= P) {
        const j = r - P;
        let k = 0;
        while (k + 1 < starts.length && starts[k + 1] <= j) k += 1;
        S.focus = k;
        drawFocusSeg();
      }
      renderMap(map, P, starts, branch, st);
      renderReader(map, P, starts, branch, st);
    };
  }

  function renderReader(map, P, starts, branch, st) {
    const rec = S.rec, n = map.size, k = S.focus;
    const rows = S.query != null ? [S.query] : d3.range(P + starts[k], P + starts[k] + branch[k]);
    const w = new Float64Array(n);
    rows.forEach((r) => { for (let c = 0; c < n; c += 1) w[c] += weightAt(map, S.head, r, c) / rows.length; });
    // Reachability from the mask: sliding layers see keys within the window of a reader's own position.
    const pos = (i) => (i < P ? i : P + (() => { const j = i - P; let q = 0; while (q + 1 < starts.length && starts[q + 1] <= j) q += 1; return j - starts[q]; })());
    const win = S.meta.config.window;
    const readerOption = S.query == null ? k : (S.query >= P ? (() => { const j = S.query - P; let q = 0; while (q + 1 < starts.length && starts[q + 1] <= j) q += 1; return q; })() : null);
    const reachable = (c) => {
      const isPrefix = c < P;
      if (!isPrefix) {
        if (readerOption == null) return false;
        const j = c - P;
        if (j < starts[readerOption] || j >= starts[readerOption] + branch[readerOption]) return false;
      }
      if (st.kind === "full_attention") return true;
      return rows.some((r) => {
        const rp = r < P ? r : P + (r - P - starts[readerOption]);
        const cp = c < P ? c : P + (c - P - starts[readerOption]);
        return Math.abs(rp - cp) <= win;
      });
    };
    void pos;
    const max = Math.max(1e-9, ...Array.from(w));
    const acc = d3.rgb(css("--accent"));
    const span = (text, c, cls = "") => {
      const a = Math.min(1, Math.sqrt(w[c] / max));
      const can = reachable(c);
      const esc = text.replace(/&/g, "&amp;").replace(/</g, "&lt;");
      return `<span class="tk ${cls} ${can ? "" : "out"}" data-c="${c}" style="background:rgba(${acc.r},${acc.g},${acc.b},${can ? (0.08 + 0.75 * a).toFixed(3) : 0})" title="${w[c].toFixed(5)}">${esc}</span>`;
    };
    let html = "";
    for (const s of rec.prefix.sections) {
      html += `<span class="sec">${s.name} (${s.end - s.start} tokens)</span>`;
      for (let c = s.start; c < s.end; c += 1) html += span(rec.prefix.tokens[c], c, rows.includes(c) ? "self" : "");
    }
    const ko = readerOption == null ? k : readerOption;
    html += `<span class="sec">own branch: option ${rec.keys[ko] === "hold" ? "hold" : ko + 1} (${branch[ko]} tokens)</span>`;
    rec.branches[ko].forEach((tok, j) => { const c = P + starts[ko] + j; html += span(tok, c, rows.includes(c) ? "self" : ""); });
    $("reader").innerHTML = html;
    const who = S.query != null ? `token ${JSON.stringify(S.query < P ? rec.prefix.tokens[S.query] : rec.branches[ko][S.query - P - starts[ko]])}` : `option ${rec.keys[k] === "hold" ? "hold" : k + 1}`;
    $("reader-name").textContent = who;
    $("reader-caption").textContent = `(${S.head === "mean" ? "all heads" : `head ${S.head + 1}`}; darker = more attention; faded = outside the mask${S.query == null ? "; mean over the option's tokens" : ""})`;
  }

  function renderShare(layer) {
    const rec = S.rec, K = rec.keys.length;
    const names = [...rec.prefix.sections.map((s) => s.name), "own branch"];
    const cols = [css("--ink-2"), css("--ramp-2"), css("--accent")];
    const data = d3.range(K).map((k) => {
      const parts = S.head === "mean" ? names.map((_, j) => d3.mean(layer.share[k], (h) => h[j])) : layer.share[k][S.head];
      return { k, parts };
    });
    const w = $("share").clientWidth || 400, rh = 20, h = K * rh + 30;
    const x = d3.scaleLinear([0, 1], [96, w - 10]);
    const svg = d3.select("#share").selectAll("svg").data([0]).join("svg").attr("class", "chart").attr("width", w).attr("height", h);
    svg.selectAll("g.leg").data([0]).join("g").attr("class", "leg").html(names.map((nm, j) => `<rect x="${96 + j * 110}" y="2" width="10" height="10" rx="2" fill="${cols[j]}"/><text x="${110 + j * 110}" y="11">${nm}</text>`).join(""));
    const rows = svg.selectAll("g.r").data(data, (d) => d.k).join((e) => { const g = e.append("g").attr("class", "r"); g.append("text").attr("class", "lab"); return g; })
      .attr("transform", (d) => `translate(0,${24 + d.k * rh})`);
    rows.select("text.lab").attr("x", 88).attr("y", 12).attr("text-anchor", "end").attr("font-weight", (d) => (d.k === S.focus ? 700 : 400))
      .text((d) => (rec.keys[d.k] === "hold" ? "hold" : `option ${d.k + 1}`));
    rows.selectAll("rect.p").data((d) => { let acc0 = 0; return d.parts.map((p, j) => { const o = { x0: acc0, p, j, k: d.k }; acc0 += p; return o; }); })
      .join("rect").attr("class", "p").attr("y", 2).attr("height", rh - 6).attr("fill", (d) => cols[d.j])
      .on("mousemove", (ev, d) => showTip(`<b>${names[d.j]}</b>: ${pct(d.p)} of option ${d.k + 1}'s attention`, ev)).on("mouseleave", hideTip)
      .transition().duration(400).attr("x", (d) => x(d.x0)).attr("width", (d) => Math.max(0, x(d.x0 + d.p) - x(d.x0) - 1));
  }

  function renderUpdates(layer, starts, branch) {
    const rec = S.rec, K = rec.keys.length;
    const mean = (arr, k) => d3.mean(arr.slice(starts[k], starts[k] + branch[k]));
    const data = d3.range(K).map((k) => ({ k, a: mean(layer.attn_update.options, k), m: mean(layer.mlp_update.options, k) }));
    const w = $("updates").clientWidth || 400, rh = 20, h = K * rh + 30;
    const max = d3.max(data, (d) => Math.max(d.a, d.m));
    const x = d3.scaleLinear([0, max], [96, w - 80]);
    const ca = css("--full"), cm = css("--slide");
    const svg = d3.select("#updates").selectAll("svg").data([0]).join("svg").attr("class", "chart").attr("width", w).attr("height", h);
    svg.selectAll("g.leg").data([0]).join("g").attr("class", "leg").html(`<rect x="96" y="2" width="10" height="10" rx="2" fill="${ca}"/><text x="110" y="11">attention update</text><rect x="230" y="2" width="10" height="10" rx="2" fill="${cm}"/><text x="244" y="11">MLP update</text>`);
    const rows = svg.selectAll("g.r").data(data, (d) => d.k).join((e) => {
      const g = e.append("g").attr("class", "r");
      g.append("text").attr("class", "lab");
      g.append("rect").attr("class", "a");
      g.append("rect").attr("class", "m");
      g.append("text").attr("class", "val");
      return g;
    }).attr("transform", (d) => `translate(0,${24 + d.k * rh})`);
    rows.select(".lab").attr("x", 88).attr("y", 12).attr("text-anchor", "end").attr("font-weight", (d) => (d.k === S.focus ? 700 : 400))
      .text((d) => (rec.keys[d.k] === "hold" ? "hold" : `option ${d.k + 1}`));
    rows.select(".a").attr("x", 96).attr("y", 2).attr("height", 7).attr("rx", 2).attr("fill", ca).transition().duration(400).attr("width", (d) => x(d.a) - 96);
    rows.select(".m").attr("x", 96).attr("y", 10).attr("height", 7).attr("rx", 2).attr("fill", cm).transition().duration(400).attr("width", (d) => x(d.m) - 96);
    rows.select(".val").attr("y", 13).attr("font-size", 10.5).transition().duration(400).attr("x", (d) => x(Math.max(d.a, d.m)) + 4).text((d) => `${fmt(d.a, 1)} / ${fmt(d.m, 1)}`);
  }

  function renderNeurons(layer) {
    const rec = S.rec, K = rec.keys.length, U = layer.neurons.index.length;
    const w = $("neurons").clientWidth || 800, cw = Math.min(54, (w - 110) / U), ch = 18, h = K * ch + 34;
    const m = d3.max(layer.neurons.per_option.flat(), (x) => Math.abs(x));
    const interp = d3.interpolateRgbBasis(["#2a78d6", css("--surface-2"), "#e34948"]);
    const svg = d3.select("#neurons").selectAll("svg").data([0]).join("svg").attr("class", "chart").attr("width", w).attr("height", h);
    svg.selectAll("text.u").data(layer.neurons.index).join("text").attr("class", "u").attr("x", (d, j) => 100 + j * cw + cw / 2).attr("y", 12)
      .attr("text-anchor", "middle").attr("font-size", 10.5).text((d) => `#${d}`);
    svg.selectAll("text.l").data(d3.range(K)).join("text").attr("class", "l").attr("x", 92).attr("y", (k) => 22 + k * ch + ch / 2 + 4)
      .attr("text-anchor", "end").attr("font-weight", (k) => (k === S.focus ? 700 : 400)).text((k) => (rec.keys[k] === "hold" ? "hold" : `option ${k + 1}`));
    const cells = [];
    layer.neurons.per_option.forEach((row, k) => row.forEach((val, j) => cells.push({ k, j, val })));
    svg.selectAll("rect.c").data(cells).join("rect").attr("class", "c").attr("x", (d) => 100 + d.j * cw + 1).attr("y", (d) => 22 + d.k * ch + 1)
      .attr("width", cw - 2).attr("height", ch - 2).attr("rx", 3)
      .on("mousemove", (ev, d) => showTip(`unit <b>#${layer.neurons.index[d.j]}</b> · option ${d.k + 1}<div>mean activation <b>${fmt(d.val, 3)}</b></div>`, ev)).on("mouseleave", hideTip)
      .transition().duration(400).attr("fill", (d) => interp(0.5 + d.val / (2 * m)));
  }

  // ------------------------------------------------------------------ representation space
  function drawSpace() {
    const rec = S.rec, color = valueColor(), best = rec.keys.indexOf(rec.best);
    const a = rec.models.original.xy, n = S.meta.loop_counts[S.loopIdx], b = rec.loops[String(n)].xy;
    // Scale to the layer stages; the embeddings sit far away and would squeeze everything else into a corner.
    const all = [...a.slice(1).flat(), ...Object.values(rec.loops).flatMap((l) => l.xy.slice(1).flat())];
    const xs = d3.extent(all, (p) => p[0]), ys = d3.extent(all, (p) => p[1]);
    const maxStage = Math.max(a.length, b.length) - 1;
    const slider = $("stage");
    slider.max = maxStage;
    if (S.space == null || S.space > maxStage) S.space = maxStage;
    slider.value = S.space;
    const panel = (host, pts, title, names) => {
      const w = $(host).clientWidth || 500, h = Math.round(Math.min(380, w * 0.68));
      const x = d3.scaleLinear(xs, [20, w - 20]).nice(), y = d3.scaleLinear(ys, [h - 24, 30]).nice();
      const s = Math.min(S.space, pts.length - 1);
      const svg = d3.select(`#${host}`).selectAll("svg").data([0]).join("svg").attr("class", "chart").attr("width", w).attr("height", h);
      svg.selectAll("rect.bg").data([0]).join("rect").attr("class", "bg").attr("width", w).attr("height", h).attr("rx", 12).attr("fill", css("--surface-2"));
      svg.selectAll("clipPath").data([0]).join("clipPath").attr("id", `clip-${host}`).selectAll("rect").data([0]).join("rect").attr("width", w).attr("height", h).attr("rx", 12);
      svg.attr("clip-path", `url(#clip-${host})`);
      if (s === 0) svg.selectAll("text.off").data([0]).join("text").attr("class", "off").attr("x", w - 14).attr("y", 20).attr("text-anchor", "end").attr("fill", css("--muted")).text("embeddings lie off-panel; press Play");
      else svg.selectAll("text.off").remove();
      svg.selectAll("text.t").data([0]).join("text").attr("class", "t").attr("x", 14).attr("y", 20).attr("font-weight", 600).attr("fill", css("--ink")).text(`${title} · ${names[s]}`);
      const line = d3.line().curve(d3.curveCatmullRom.alpha(0.5)).x((p) => x(p[0])).y((p) => y(p[1]));
      svg.selectAll("path.trail").data(d3.range(rec.keys.length)).join("path").attr("class", "trail").attr("fill", "none")
        .attr("stroke", (k) => color(k)).attr("stroke-opacity", 0.45).attr("stroke-width", 1.4)
        .attr("d", (k) => line(pts.slice(0, s + 1).map((row) => row[k])));
      svg.selectAll("circle.dot").data(d3.range(rec.keys.length)).join("circle").attr("class", "dot").attr("r", 7)
        .attr("fill", (k) => color(k)).attr("stroke", css("--surface")).attr("stroke-width", 2)
        .on("mousemove", (ev, k) => showTip(`<b>${shortLabel(rec.options[k].text, rec.keys[k])}</b><div>simulator value ${fmt(rec.value[k], 3)}</div>`, ev)).on("mouseleave", hideTip)
        .transition().duration(450).attr("cx", (k) => x(pts[s][k][0])).attr("cy", (k) => y(pts[s][k][1]));
      svg.selectAll("circle.best").data([best]).join("circle").attr("class", "best").attr("r", 12).attr("fill", "none")
        .attr("stroke", css("--ink")).attr("stroke-width", 1.6).attr("pointer-events", "none")
        .transition().duration(450).attr("cx", (k) => x(pts[s][k][0])).attr("cy", (k) => y(pts[s][k][1]));
    };
    const namesA = ["embeddings", ...S.meta.config.layer_types.map((_, i) => `after L${i}`)];
    const namesB = ["embeddings", "after L0"];
    for (let r = 1; r <= n; r += 1) for (let j = 1; j <= 3; j += 1) namesB.push(`after B${j}, loop ${r}`);
    panel("space-a", a, "Original", namesA);
    panel("space-b", b, `Looped, ${n} loop${n > 1 ? "s" : ""}`, namesB);
    $("stage-name").textContent = `stage ${S.space} of ${maxStage}`;
  }

  // ------------------------------------------------------------------ story charts
  function chartSvg(host, h) {
    const w = $(host).clientWidth || 600;
    const svg = d3.select(`#${host}`).selectAll("svg").data([0]).join("svg").attr("class", "chart").attr("width", w).attr("height", h);
    return { svg, w, h };
  }
  let simAligned = false;
  function drawSim() {
    const ws = S.meta.weight_similarity, m = simAligned ? ws.aligned : ws.raw;
    const { svg, w } = chartSvg("viz-sim", 190);
    const cw = Math.min(90, (w - 110) / ws.names.length), ch = 42;
    const vmax = Math.max(...ws.aligned.flat());
    const interp = heatInterp();
    svg.selectAll("text.n").data(ws.names).join("text").attr("class", "n").attr("x", (d, j) => 110 + j * cw + cw / 2).attr("y", 14).attr("text-anchor", "middle").text((d) => d);
    svg.selectAll("text.r").data([0, 1, 2]).join("text").attr("class", "r").attr("x", 100).attr("y", (j) => 24 + j * ch + ch / 2 + 4).attr("text-anchor", "end").text((j) => `layer ${j + 1} vs ${j + 4}`);
    const cells = m.flatMap((row, i) => row.map((v, j) => ({ i, j, v })));
    const g = svg.selectAll("g.c").data(cells, (d) => `${d.i}-${d.j}`).join((e) => { const q = e.append("g").attr("class", "c"); q.append("rect"); q.append("text"); return q; });
    g.select("rect").attr("x", (d) => 110 + d.j * cw + 2).attr("y", (d) => 24 + d.i * ch + 2).attr("width", cw - 4).attr("height", ch - 4).attr("rx", 6)
      .transition().duration(900).attr("fill", (d) => interp(Math.max(0, d.v) / vmax));
    g.select("text").attr("x", (d) => 110 + d.j * cw + cw / 2).attr("y", (d) => 24 + d.i * ch + ch / 2 + 4).attr("text-anchor", "middle")
      .attr("font-weight", 600).attr("fill", (d) => (d.v / vmax > 0.55 ? css("--surface") : css("--ink")))
      .transition().duration(900)
      .tween("text", function (d) { const node = this, from = +node.textContent || 0, i = d3.interpolateNumber(from, d.v); return (t) => { node.textContent = i(t).toFixed(2); }; });
    svg.selectAll("text.cap").data([0]).join("text").attr("class", "cap").attr("x", 110).attr("y", 24 + 3 * ch + 18)
      .text(simAligned ? "after the exact re-gauge: same function, comparable coordinates" : "as trained (LayerNorm gains folded in): cosine similarity ≈ 0");
    $("sim-toggle").textContent = simAligned ? "Show as trained" : "Apply the re-gauge";
  }
  function drawHeads() {
    const heads = S.meta.heads;
    const { svg, w } = chartSvg("viz-heads", 220);
    const size = Math.min(160, (w - 40) / 3), c = size / 4;
    const vmax = Math.max(...heads.flatMap((h) => h.similarity.flat()));
    const interp = heatInterp();
    heads.forEach((h, p) => {
      const ox = 20 + p * (size + 20), oy = 34;
      const g = svg.selectAll(`g.p${p}`).data([0]).join("g").attr("class", `p${p}`);
      g.selectAll("text.t").data([0]).join("text").attr("class", "t").attr("x", ox + size / 2).attr("y", 16).attr("text-anchor", "middle").attr("font-weight", 600).text(`layer ${p + 1} ↔ ${p + 4}`);
      const cells = h.similarity.flatMap((row, a) => row.map((v, b) => ({ a, b, v })));
      const cg = g.selectAll("g.cell").data(cells).join((e) => { const q = e.append("g").attr("class", "cell"); q.append("rect"); q.append("text"); return q; });
      cg.select("rect").attr("x", (d) => ox + d.b * c + 1).attr("y", (d) => oy + d.a * c + 1).attr("width", c - 2).attr("height", c - 2).attr("rx", 4).attr("fill", (d) => interp(d.v / vmax));
      cg.select("text").attr("x", (d) => ox + d.b * c + c / 2).attr("y", (d) => oy + d.a * c + c / 2 + 4).attr("text-anchor", "middle").attr("font-size", 10.5)
        .attr("fill", (d) => (d.v / vmax > 0.55 ? css("--surface") : css("--ink"))).text((d) => d.v.toFixed(2));
      g.selectAll("rect.m").data(h.match.map((b, a) => ({ a, b }))).join("rect").attr("class", "m").attr("fill", "none").attr("stroke", css("--slide")).attr("stroke-width", 2.5).attr("rx", 4)
        .attr("x", (d) => ox + d.b * c).attr("y", (d) => oy + d.a * c).attr("width", c).attr("height", c);
      g.selectAll("text.ax").data([0]).join("text").attr("class", "ax").attr("x", ox + size / 2).attr("y", oy + size + 16).attr("text-anchor", "middle").attr("font-size", 10.5).text(`heads of layer ${p + 4} →`);
    });
  }
  const neuronImgs = new Map();
  let neuronPair = 0;
  async function drawNeurons() {
    const seg = d3.select("#neuron-pair");
    seg.selectAll("button").data([0, 1, 2]).join("button").text((j) => `layer ${j + 1} ↔ ${j + 4}`).classed("on", (j) => j === neuronPair)
      .on("click", (ev, j) => { neuronPair = j; drawNeurons(); });
    const info = S.meta.neurons[neuronPair];
    const get = async (p) => { if (!neuronImgs.has(p)) neuronImgs.set(p, await loadMap(p)); return neuronImgs.get(p); };
    const [raw, al] = await Promise.all([get(info.raw), get(info.aligned)]);
    const mix = +$("neuron-mix").value, n = raw.size, canvas = $("viz-neurons");
    canvas.width = n; canvas.height = n;
    const ctx = canvas.getContext("2d"), img = ctx.createImageData(n, n);
    const interp = heatInterp(), lut = d3.range(256).map((i) => d3.rgb(interp(i / 255)));
    for (let i = 0; i < n * n; i += 1) {
      const g = Math.round((1 - mix) * raw.gray[i] + mix * al.gray[i]);
      const col = lut[g];
      img.data[4 * i] = col.r; img.data[4 * i + 1] = col.g; img.data[4 * i + 2] = col.b; img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    $("neuron-caption").textContent = `${info.size} × ${info.size} matching cost (brighter = better partners) · ${info.flipped} of ${info.size} gate signs flipped`;
  }
  const VARIANT = { copy_first: "copy layers 1–3", copy_last: "copy layers 4–6", mean_raw: "mean, unaligned", mean_aligned: "mean, aligned", regmean: "RegMean", lsq1: "RegMean + 1 sweep", lsq2: "RegMean + 2 sweeps", lsq3: "RegMean + 3 sweeps", lsq4: "RegMean + 4 sweeps" };
  function drawVariants() {
    const rows = Object.entries(S.meta.variants).map(([k, v]) => ({ k, ...v })).sort((a, b) => a.kl - b.kl);
    const rh = 28, { svg, w } = chartSvg("viz-variants", rows.length * rh + 40);
    const x = d3.scaleLinear([0, d3.max(rows, (d) => d.kl) * 1.1], [150, w - 60]);
    svg.selectAll("g.ax").data([0]).join("g").attr("class", "ax grid").attr("transform", `translate(0,${rows.length * rh + 6})`).call(d3.axisBottom(x).ticks(5).tickSize(-(rows.length * rh)));
    const g = svg.selectAll("g.r").data(rows, (d) => d.k).join((e) => { const q = e.append("g").attr("class", "r"); q.append("text").attr("class", "l"); q.append("rect"); q.append("text").attr("class", "v"); return q; })
      .attr("transform", (d, i) => `translate(0,${i * rh})`);
    g.select(".l").attr("x", 142).attr("y", rh / 2 + 4).attr("text-anchor", "end").attr("font-weight", (d) => (d.k === S.meta.chosen ? 700 : 400)).text((d) => VARIANT[d.k] || d.k);
    g.select("rect").attr("x", 150).attr("y", 5).attr("height", rh - 10).attr("rx", 4)
      .attr("fill", (d) => (d.k === S.meta.chosen ? css("--accent") : d.data_free ? css("--line-2") : css("--ramp-2")))
      .on("mousemove", (ev, d) => showTip(`<b>${VARIANT[d.k]}</b>${d.data_free ? " <span class='muted'>(no data)</span>" : ""}<div class="row"><span>KL to the original</span><b>${fmt(d.kl, 3)}</b></div><div class="row"><span>top-1 agreement</span><b>${pct(d.top1)}</b></div><div class="row"><span>Spearman</span><b>${fmt(d.spearman, 3)}</b></div>`, ev))
      .on("mouseleave", hideTip)
      .transition().duration(800).attr("width", (d) => x(d.kl) - 150);
    g.select(".v").attr("y", rh / 2 + 4).transition().duration(800).attr("x", (d) => x(d.kl) + 6).text((d) => fmt(d.kl, 3));
  }
  function drawStream() {
    const se = S.meta.stream_error, names = { copy_first: "copy layers 1–3", regmean: "RegMean", lsq4: "chosen (RegMean + 4 sweeps)" };
    const cols = { copy_first: css("--line-2"), regmean: css("--ramp-2"), lsq4: css("--accent") };
    const { svg, w } = chartSvg("viz-stream", 260);
    const x = d3.scaleLinear([0, 6], [50, w - 240]), y = d3.scaleLinear([0, d3.max(Object.values(se).flat()) * 1.1], [230, 20]);
    svg.selectAll("g.ya").data([0]).join("g").attr("class", "ya grid").attr("transform", "translate(50,0)").call(d3.axisLeft(y).ticks(5).tickSize(-(w - 290)));
    svg.selectAll("g.xa").data([0]).join("g").attr("class", "xa axis").attr("transform", "translate(0,230)").call(d3.axisBottom(x).ticks(7).tickFormat((d) => `L${d}`));
    svg.selectAll("rect.loop2").data([0]).join("rect").attr("class", "loop2").attr("x", x(3.5)).attr("y", 20).attr("width", x(6) - x(3.5)).attr("height", 210).attr("fill", css("--accent")).attr("opacity", 0.06);
    svg.selectAll("text.loop2").data([0]).join("text").attr("class", "loop2").attr("x", x(4.75)).attr("y", 34).attr("text-anchor", "middle").text("loop 2");
    const line = d3.line().x((d, i) => x(i)).y((d) => y(d));
    const keys = Object.keys(names);
    svg.selectAll("path.s").data(keys).join("path").attr("class", "s").attr("fill", "none").attr("stroke", (k) => cols[k]).attr("stroke-width", 2.5).attr("d", (k) => line(se[k]));
    svg.selectAll("text.lab").data(keys).join("text").attr("class", "lab").attr("x", x(6) + 8).attr("y", (k) => y(se[k][6]) + 4).attr("fill", (k) => cols[k]).attr("font-weight", 600)
      .text((k) => `${names[k]} ${se[k][6].toFixed(2)}`);
    svg.selectAll("text.yl").data([0]).join("text").attr("class", "yl").attr("x", 50).attr("y", 12).text("relative error of the residual stream (option tokens)");
  }
  function drawLoops() {
    const fp = S.meta.fixed_point, n = fp.kl.length;
    const { svg, w } = chartSvg("viz-loops", 250);
    const half = (w - 40) / 2;
    const x1 = d3.scalePoint(d3.range(1, n + 1), [40, half - 10]).padding(0.3), y1 = d3.scaleLinear([0, d3.max(fp.kl) * 1.1], [210, 24]);
    svg.selectAll("g.y1").data([0]).join("g").attr("class", "y1 grid").attr("transform", "translate(40,0)").call(d3.axisLeft(y1).ticks(4).tickSize(-(half - 50)));
    svg.selectAll("g.x1").data([0]).join("g").attr("class", "x1 axis").attr("transform", "translate(0,210)").call(d3.axisBottom(x1));
    svg.selectAll("path.kl").data([fp.kl]).join("path").attr("class", "kl").attr("fill", "none").attr("stroke", css("--accent")).attr("stroke-width", 2.5)
      .attr("d", d3.line().x((d, i) => x1(i + 1)).y((d) => y1(d)));
    svg.selectAll("circle.kl").data(fp.kl).join("circle").attr("class", "kl").attr("r", 4.5).attr("fill", css("--accent")).attr("cx", (d, i) => x1(i + 1)).attr("cy", (d) => y1(d))
      .on("mousemove", (ev, d) => showTip(`KL to the original: <b>${fmt(d, 3)}</b>`, ev)).on("mouseleave", hideTip);
    svg.selectAll("text.t1").data([0]).join("text").attr("class", "t1").attr("x", 40).attr("y", 12).text("KL to the original, by loops");
    svg.selectAll("text.x1l").data([0]).join("text").attr("class", "x1l").attr("x", half / 2 + 15).attr("y", 244).attr("text-anchor", "middle").text("loops");
    const ox = half + 40, x2 = d3.scaleBand(d3.range(2, n + 1), [ox + 30, w - 10]).padding(0.25), y2 = d3.scaleLinear([0, d3.max(fp.change) * 1.1], [210, 24]);
    svg.selectAll("g.y2").data([0]).join("g").attr("class", "y2 grid").attr("transform", `translate(${ox + 30},0)`).call(d3.axisLeft(y2).ticks(4).tickSize(-(w - ox - 40)));
    svg.selectAll("g.x2").data([0]).join("g").attr("class", "x2 axis").attr("transform", "translate(0,210)").call(d3.axisBottom(x2));
    svg.selectAll("rect.ch").data(fp.change).join("rect").attr("class", "ch").attr("x", (d, i) => x2(i + 2)).attr("width", x2.bandwidth()).attr("rx", 3)
      .attr("y", (d) => y2(d)).attr("height", (d) => 210 - y2(d)).attr("fill", css("--ramp-2"))
      .on("mousemove", (ev, d) => showTip(`relative change from the previous loop: <b>${fmt(d, 3)}</b>`, ev)).on("mouseleave", hideTip);
    svg.selectAll("text.t2").data([0]).join("text").attr("class", "t2").attr("x", ox + 30).attr("y", 12).text("change made by each extra loop");
    svg.selectAll("text.x2l").data([0]).join("text").attr("class", "x2l").attr("x", (ox + 30 + w) / 2).attr("y", 244).attr("text-anchor", "middle").text("loop added");
  }
  function drawRegret() {
    const r = S.meta.results, rows = ["val", "test"].map((s) => ({ s, ...r[s].looped_minus_original.regret, maps: r[s].maps }));
    const { svg, w } = chartSvg("viz-regret", 170);
    const lim = Math.max(...rows.map((d) => Math.max(Math.abs(d.lo), Math.abs(d.hi)))) * 1.25;
    const x = d3.scaleLinear([-lim, lim], [80, w - 30]);
    svg.selectAll("g.xa").data([0]).join("g").attr("class", "xa grid").attr("transform", "translate(0,130)").call(d3.axisBottom(x).ticks(6).tickSize(-110));
    svg.selectAll("line.z").data([0]).join("line").attr("class", "z").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 18).attr("y2", 130).attr("stroke", css("--ink-2")).attr("stroke-dasharray", "3 3");
    const g = svg.selectAll("g.r").data(rows).join((e) => { const q = e.append("g").attr("class", "r"); q.append("text").attr("class", "l"); q.append("line"); q.append("circle"); q.append("text").attr("class", "v"); return q; })
      .attr("transform", (d, i) => `translate(0,${45 + i * 50})`);
    g.select(".l").attr("x", 70).attr("y", 4).attr("text-anchor", "end").attr("font-weight", 600).text((d) => d.s);
    g.select("line").attr("x1", (d) => x(d.lo)).attr("x2", (d) => x(d.hi)).attr("stroke", css("--accent")).attr("stroke-width", 3).attr("stroke-linecap", "round");
    g.select("circle").attr("cx", (d) => x(d.diff)).attr("r", 6).attr("fill", css("--accent")).attr("stroke", css("--surface")).attr("stroke-width", 2);
    g.select(".v").attr("x", (d) => x(d.diff)).attr("y", -12).attr("text-anchor", "middle").attr("fill", css("--ink")).text((d) => `${d.diff >= 0 ? "+" : ""}${fmt(d.diff, 3)} [${fmt(d.lo, 3)}, ${fmt(d.hi, 3)}] · ${d.maps} maps`);
    svg.selectAll("text.cap").data([0]).join("text").attr("class", "cap").attr("x", 80).attr("y", 164).text("regret, looped minus original (← looped better · looped worse →)");
  }

  // ------------------------------------------------------------------ wiring
  function drawAll() {
    drawFlow();
    drawHeadSeg();
    drawFocusSeg();
    drawInspector();
    drawSpace();
  }
  function drawStory() { drawSim(); drawHeads(); drawNeurons(); drawVariants(); drawStream(); drawLoops(); drawRegret(); }
  function syncControls() {
    d3.selectAll("#model-seg button").classed("on", function () { return this.dataset.model === S.model; });
    $("loops-ctl").classList.toggle("disabled", S.model !== "looped");
    $("loops-val").textContent = S.meta.loop_counts[S.loopIdx];
    $("temp").value = S.temp;
    $("temp-val").textContent = fmt(S.temp);
  }
  async function loadDecision(id) {
    S.rec = await (await fetch(`${DATA}${id}.json`)).json();
    S.focus = S.rec.keys.indexOf(S.rec.best);
    S.query = null;
    S.stage = null;
    S.space = null;
    const q = "Which air route should the company build now, to maximise its net worth two years from now? Building nothing is an option.";
    $("question").innerHTML = `<span class="q-label">The question the model answers, for one real decision (${S.rec.task}, ${S.rec.keys.length} options, ${S.rec.sizes.prefix + S.rec.sizes.options} tokens)</span>${q}`;
    drawAll();
  }
  async function init() {
    S.meta = await (await fetch(`${DATA}meta.json`)).json();
    S.temp = S.meta.temperature.looped;
    const labels = ["Both models pick the best route", "The two models disagree", "Holding is best; neither picks it"];
    d3.select("#decision").selectAll("option").data(S.meta.decisions).join("option").attr("value", (d) => d.id)
      .text((d, i) => `${labels[i] || d.id} · ${d.options} options`);
    $("decision").onchange = (ev) => loadDecision(ev.target.value);
    d3.selectAll("#model-seg button").on("click", function () {
      S.model = this.dataset.model;
      S.temp = S.meta.temperature[S.model];
      S.stage = null;
      syncControls();
      drawAll();
    });
    $("loops").max = S.meta.loop_counts.length - 1;
    $("loops").oninput = (ev) => { S.loopIdx = +ev.target.value; S.stage = null; syncControls(); drawAll(); };
    $("temp").oninput = (ev) => { S.temp = +ev.target.value; syncControls(); drawFlow(); };
    $("stage").oninput = (ev) => { S.space = +ev.target.value; drawSpace(); };
    $("play").onclick = () => {
      if (S.playing) { clearInterval(S.playing); S.playing = null; $("play").textContent = "▶ Play"; return; }
      S.space = 0;
      $("play").textContent = "❚❚ Pause";
      S.playing = setInterval(() => {
        const max = +$("stage").max;
        if (S.space >= max) { clearInterval(S.playing); S.playing = null; $("play").textContent = "▶ Play"; return; }
        S.space += 1;
        drawSpace();
      }, 650);
    };
    $("sim-toggle").onclick = () => { simAligned = !simAligned; drawSim(); };
    $("neuron-mix").oninput = () => drawNeurons();
    $("theme").onclick = () => {
      const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
      document.documentElement.dataset.theme = dark ? "light" : "dark";
      d3.select("#flow svg").remove();
      drawAll();
      drawStory();
    };
    $("legend").innerHTML = `<span><i style="background:${css("--full")}"></i>full attention</span><span><i style="background:${css("--slide")}"></i>sliding ±${S.meta.config.window}</span><span><i style="border:1.5px dashed ${css("--accent")};background:none"></i>reused weights</span><span>simulator value<span class="ramp"></span>high</span>`;
    syncControls();
    await loadDecision(S.meta.decisions[0].id);
    drawStory();
    let t;
    addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => { drawAll(); drawStory(); }, 150); });
  }
  init().catch((e) => { console.error(e); document.body.insertAdjacentHTML("afterbegin", `<pre style="color:red;padding:20px">${e}</pre>`); });
})();
