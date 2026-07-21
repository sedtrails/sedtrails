"use strict";
/* ═════════════════════════ Forcing tab ═════════════════════════
   netCDF forcing viewer as a user-managed LAYER STACK: each layer shows one
   variable — a scalar map, a vector-derived map (magnitude/x/y/direction) or
   instanced quiver arrows — with its own colormap/range/opacity, drawn
   bottom→top in user order. Time is driven by the shared global clock
   (state.clock); the tab keeps a per-timestep fine-step slider.
   The layers draw inside the shared scene (between the model layer and the
   particles), so they also serve as background for Seeding/Viewer/Connectivity. */

const ForcingTab = (() => {
  const F = {
    id: null, meta: null,
    mesh: null,      // {posB, idxB, nTris, nVerts, nodeIdx, nNodes, nodeBuf, gathered}
    show: true,      // master toggle for the whole forcing stack
    layers: [],      // bottom→top; see addLayer() for the per-layer model
    nextId: 1,
    t: 0,            // current timestep index
    timeDays: null,  // Float64Array abs epoch-days per timestep (clock link)
    lastK: 0,
    cache: new Map(), vcache: new Map(), dcache: new Map(),
    inflight: new Map(),
  };
  const CACHE_MAX = 30;
  const VEC_STYLES = [['mag', 'magnitude map'], ['x', 'x map'], ['y', 'y map'],
                      ['dir', 'direction map'], ['arrows', 'arrows']];

  /* ── GL programs ────────────────────────────────────────────────────────── */
  const FRC_VS = `#version 300 es
  precision highp float;
  layout(location=0) in vec2 aPos;
  layout(location=1) in float aVal;
  uniform vec4 uView;
  uniform vec2 uOff;
  out float vVal;
  void main() { vVal = aVal; gl_Position = vec4((aPos + uOff) * uView.xy + uView.zw, 0.0, 1.0); }`;
  const FRC_FS = `#version 300 es
  precision mediump float;
  in float vVal;
  uniform vec2 uClim;
  uniform float uAlpha;
  uniform sampler2D uRamp;
  out vec4 frag;
  void main() {
    if (vVal < -1e29) discard;
    float f = clamp((vVal - uClim.x) / max(uClim.y - uClim.x, 1e-12), 0.0, 1.0);
    vec4 c = texture(uRamp, vec2(f, 0.5));
    frag = vec4(c.rgb * uAlpha, uAlpha);
  }`;

  const ARR_VS = `#version 300 es
  precision highp float;
  layout(location=0) in vec2 aTemplate;   // arrow template in unit space
  layout(location=1) in vec2 aAnchor;     // per-instance node position (world, abs)
  layout(location=2) in vec2 aUV;         // per-instance vector
  uniform vec4 uView;
  uniform vec2 uOff, uCanvas;
  uniform float uScale, uMagRef;
  uniform sampler2D uRamp;
  uniform int uColorMag;
  uniform vec3 uColor;
  out vec4 vColor;
  void main() {
    float mag = length(aUV);
    if (mag <= 1e-12 || aAnchor.x < -1e29 || aUV.x < -1e29) {
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vColor = vec4(0.0); return;
    }
    vec2 dir = aUV / mag;
    float lenPx = clamp(mag / max(uMagRef, 1e-12) * uScale, 3.0, 3.0 * uScale);
    vec2 t = aTemplate * lenPx;
    vec2 px = vec2(t.x * dir.x - t.y * dir.y, t.x * dir.y + t.y * dir.x);
    vec2 clip = (aAnchor + uOff) * uView.xy + uView.zw;
    gl_Position = vec4(clip + px / (uCanvas * 0.5), 0.0, 1.0);
    vec3 col = (uColorMag == 1)
      ? texture(uRamp, vec2(clamp(mag / max(uMagRef * 1.5, 1e-12), 0.0, 1.0), 0.5)).rgb
      : uColor;
    vColor = vec4(col, 0.9);
  }`;
  const ARR_FS = `#version 300 es
  precision mediump float;
  in vec4 vColor; out vec4 frag;
  void main() { frag = vec4(vColor.rgb * vColor.a, vColor.a); }`;

  let progFrc = null, progArr = null, arrTemplate = null;
  function ensurePrograms() {
    if (progFrc) return;
    progFrc = compile(FRC_VS, FRC_FS);
    progArr = compile(ARR_VS, ARR_FS);
    // arrow as 3 line segments: shaft + two head strokes (unit length along +x)
    arrTemplate = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, arrTemplate);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      0, 0, 1, 0,          // shaft
      1, 0, 0.68, 0.16,    // head upper
      1, 0, 0.68, -0.16,   // head lower
    ]), gl.STATIC_DRAW);
  }

  /* ── layer model ────────────────────────────────────────────────────────── */
  function addLayer(sel) {          // sel = {kind:'scalar', varName} | {kind:'vector', pair}
    const L = {
      id: F.nextId++, kind: sel.kind,
      varName: sel.varName || null, pair: sel.pair || null,
      style: sel.kind === 'vector' ? 'mag' : 'map',
      cmap: sel.kind === 'vector' ? 'turbo' : 'viridis',
      alpha: 0.9, range: [0, 1], userRange: false, visible: true, ready: false,
      stride: 4, scale: 40, scaleF: 0.5, colorMag: true, magP98: 1, color: '#ffffff',
      gl: {vao: null, valB: null, vecBuf: null, arrVAO: null, arrStride: 0},
      pending: null, _rangeUI: null,
    };
    F.layers.push(L);
    return L;
  }

  function layerLabel(L) {
    return L.kind === 'scalar' ? L.varName : L.pair.label;
  }
  function layerKey(L) {
    return L.kind === 'scalar' ? 's|' + L.varName : 'v|' + L.pair.x + '|' + L.style;
  }
  function isMapStyle(L) { return !(L.kind === 'vector' && L.style === 'arrows'); }

  function freeLayerGL(L) {
    const g = L.gl;
    if (g.vao) gl.deleteVertexArray(g.vao);
    if (g.valB) gl.deleteBuffer(g.valB);
    if (g.arrVAO) gl.deleteVertexArray(g.arrVAO);
    if (g.vecBuf) gl.deleteBuffer(g.vecBuf);
    L.gl = {vao: null, valB: null, vecBuf: null, arrVAO: null, arrStride: 0};
    L.ready = false;
  }
  function removeLayer(L) {
    freeLayerGL(L);
    F.layers = F.layers.filter(x => x !== L);
  }

  function ensureMapGL(L) {
    if (L.gl.vao) return;
    const valB = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, valB);
    gl.bufferData(gl.ARRAY_BUFFER, F.mesh.nVerts * 4, gl.DYNAMIC_DRAW);
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, F.mesh.posB);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, valB);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 4, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, F.mesh.idxB);
    gl.bindVertexArray(null);
    L.gl.vao = vao; L.gl.valB = valB;
  }
  function ensureArrGL(L) {
    if (L.gl.arrVAO) return;
    const vecBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vecBuf);
    gl.bufferData(gl.ARRAY_BUFFER, F.mesh.nNodes * 8, gl.DYNAMIC_DRAW);
    const arrVAO = gl.createVertexArray();
    gl.bindVertexArray(arrVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, arrTemplate);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
    gl.enableVertexAttribArray(1);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribDivisor(1, 1);
    gl.vertexAttribDivisor(2, 1);
    gl.bindVertexArray(null);
    L.gl.arrVAO = arrVAO; L.gl.vecBuf = vecBuf; L.gl.arrStride = 0;
  }

  /* ── open a forcing file ────────────────────────────────────────────────── */
  async function open(path, base) {
    const r = await apiJson('/api/forcing/open', {path, base: base || null});
    F.id = r.id;
    F.meta = r.meta;
    F.cache.clear(); F.vcache.clear(); F.dcache.clear();
    for (const L of F.layers) freeLayerGL(L);
    F.mesh = null;
    F.t = Math.min(F.t, Math.max(0, F.meta.n_timesteps - 1));
    buildTimeDays();
    $('frc-path').value = F.meta.path;
    const src = $('frc-src');
    if (src) src.textContent = F.meta.path + '  (from Settings → inputs → data)';
    localStorage.setItem('stgui_last_forcing', F.meta.path);

    if (!state.world0) state.world0 = [F.meta.extent[0], F.meta.extent[2]];
    if (!state.runs.length) {           // fit view to the forcing domain
      fitView({meta: {extent: F.meta.extent}});
    }

    if (r.mesh) await loadForcingMesh();
    else if (r.mesh_job) {
      $('frc-info').textContent = 'building cell mesh…';
      await pollJob(r.mesh_job);
      await loadForcingMesh();
    }
    // drop layers whose variable/pair no longer exists; default layer if empty
    const varOk = n => F.meta.variables.some(v => v.name === n);
    F.layers = F.layers.filter(L =>
      L.kind === 'scalar' ? varOk(L.varName)
                          : F.meta.vector_pairs.some(p => p.x === L.pair.x));
    if (!F.layers.length) {
      // defaults: bed level as a topo map + white arrows of the first vector pair
      const scalars = F.meta.variables.filter(v => !v.is_vector_comp);
      const bed = scalars.find(v => /bed/i.test(v.name)) ||
                  scalars.find(v => /depth/i.test(v.name)) || scalars[0];
      if (bed) {
        const L = addLayer({kind: 'scalar', varName: bed.name});
        if (/bed|depth/i.test(bed.name)) L.cmap = 'topo';
      }
      if (F.meta.vector_pairs.length) {
        const A = addLayer({kind: 'vector', pair: F.meta.vector_pairs[0]});
        A.style = 'arrows';
        A.colorMag = false;               // uniform white arrows
      }
    }
    refreshPanel();
    if (typeof updateClockRange === 'function') {
      updateClockRange();
      if (F.timeDays && state.clock.t < F.timeDays[0]) state.clock.t = F.timeDays[0];
      updateTimelineUI();
    }
    await updateLayers();
    if (typeof LayersPanel !== 'undefined') LayersPanel.refresh();
    state.dirty = true;
  }

  function buildTimeDays() {
    const times = F.meta.times || [];
    let firstValid = -1;
    const td = new Float64Array(times.length);
    for (let i = 0; i < times.length; i++) {
      const v = Date.parse(String(times[i]).replace(' ', 'T') + ':00Z') / 86400000;
      td[i] = v;
      if (isFinite(v) && firstValid < 0) firstValid = i;
    }
    if (firstValid < 0) { F.timeDays = null; return; }
    for (let i = 0; i < firstValid; i++) td[i] = td[firstValid];
    for (let i = firstValid + 1; i < td.length; i++)
      if (!isFinite(td[i])) td[i] = td[i - 1];
    F.timeDays = td;
    F.lastK = 0;
  }

  async function pollJob(jobId) {
    for (;;) {
      const j = await apiJson('/api/job/' + jobId);
      if (j.status === 'done') return;
      if (j.status === 'error') throw new Error(j.message);
      $('frc-info').textContent = `building cell mesh… ${Math.round((j.progress || 0) * 100)}%`;
      await new Promise(res => setTimeout(res, 400));
    }
  }

  async function loadForcingMesh() {
    ensurePrograms();
    const buf = await (await api(`/api/forcing/${F.id}/mesh`)).arrayBuffer();
    const nl = new Uint8Array(buf).indexOf(10);
    const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 0, nl)));
    let o = nl + 1;
    const xy = new Float32Array(buf.slice(o, o + head.n_verts * 8)); o += head.n_verts * 8;
    const nodeIdx = new Uint32Array(buf.slice(o, o + head.n_verts * 4)); o += head.n_verts * 4;
    const tris = new Uint32Array(buf.slice(o, o + head.n_tris * 12)); o += head.n_tris * 12;
    const nodes = new Float32Array(buf.slice(o, o + head.n_nodes * 8));

    const posB = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posB);
    gl.bufferData(gl.ARRAY_BUFFER, xy, gl.STATIC_DRAW);
    const idxB = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxB);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, tris, gl.STATIC_DRAW);

    // quiver: per-instance anchor positions (static, decimated later via stride)
    const nodeBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, nodeBuf);
    gl.bufferData(gl.ARRAY_BUFFER, nodes, gl.STATIC_DRAW);

    F.mesh = {posB, idxB, nTris: head.n_tris, nVerts: head.n_verts, nodeIdx,
              nNodes: head.n_nodes, nodeBuf, cellP01: head.cell_p01 || null,
              gathered: new Float32Array(head.n_verts)};
    $('frc-info').textContent =
      `${fmtInt(F.meta.n_nodes)} nodes · ${fmtInt(F.meta.n_timesteps)} timesteps` +
      (F.meta.times.length ? ` · ${F.meta.times[0]} → ${F.meta.times[F.meta.times.length - 1]}` : '');
  }

  /* ── data fetching (LRU cache + in-flight dedup + prefetch) ─────────────── */
  function lru(map, key, val) {
    map.set(key, val);
    if (map.size > CACHE_MAX) map.delete(map.keys().next().value);
  }

  function fetchField(varName, t) {
    const key = varName + '|' + t;
    if (F.cache.has(key)) return Promise.resolve(F.cache.get(key));
    if (F.inflight.has(key)) return F.inflight.get(key);
    const p = (async () => {
      const r = await api(`/api/forcing/${F.id}/field?var=${encodeURIComponent(varName)}&t=${t}`);
      const rng = (r.headers.get('X-Data-Range') || '0,1').split(',').map(Number);
      const arr = new Float32Array(await r.arrayBuffer());
      arr.range = rng;
      lru(F.cache, key, arr);
      return arr;
    })().finally(() => F.inflight.delete(key));
    F.inflight.set(key, p);
    return p;
  }

  function fetchVector(pair, t) {
    const key = 'v|' + pair.x + '|' + t;
    if (F.vcache.has(key)) return Promise.resolve(F.vcache.get(key));
    if (F.inflight.has(key)) return F.inflight.get(key);
    const p = (async () => {
      const r = await api(`/api/forcing/${F.id}/vector?x=${encodeURIComponent(pair.x)}&y=${encodeURIComponent(pair.y)}&t=${t}`);
      const arr = new Float32Array(await r.arrayBuffer());
      arr.p98 = parseFloat(r.headers.get('X-Mag-P98') || '1');
      lru(F.vcache, key, arr);
      return arr;
    })().finally(() => F.inflight.delete(key));
    F.inflight.set(key, p);
    return p;
  }

  /* magnitude / x / y / direction slab derived from an interleaved vector slab */
  function deriveSlab(pair, t, style, vec) {
    const key = pair.x + '|' + t + '|' + style;
    if (F.dcache.has(key)) return F.dcache.get(key);
    const n = F.mesh.nNodes;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const u = vec[2 * i], v = vec[2 * i + 1];
      if (u < -1e29 || v < -1e29) { out[i] = -1e30; continue; }
      out[i] = style === 'x' ? u
             : style === 'y' ? v
             : style === 'mag' ? Math.hypot(u, v)
             : Math.atan2(v, u) * 180 / Math.PI;     // dir, math convention: CCW from east
    }
    lru(F.dcache, key, out);
    return out;
  }

  function defaultRange(L) {
    const p = L.magP98 || 1;
    if (L.style === 'mag') return [0, p * 1.25];
    if (L.style === 'x' || L.style === 'y') return [-p, p];
    if (L.style === 'dir') return [-180, 180];
    return L.range;
  }

  /* ── upload the current timestep of every visible layer to the GPU ──────── */
  function updateLayers(only) {
    if (!F.mesh) { state.dirty = true; return Promise.resolve(); }
    const todo = F.layers.filter(L => L.visible && (!only || L === only));
    return Promise.all(todo.map(L => uploadLayer(L, F.t)));
  }

  /* loading indicator (statusbar chip) */
  let nLoading = 0;
  function loadStart() {
    nLoading++;
    const s = $('st-frc');
    if (s) s.style.display = '';
  }
  function loadEnd() {
    nLoading = Math.max(0, nLoading - 1);
    if (!nLoading) {
      const s = $('st-frc');
      if (s) s.style.display = 'none';
    }
  }

  async function uploadLayer(L, t) {
    const tok = layerKey(L) + '|' + t;
    L.pending = tok;
    const wasReady = L.ready;
    loadStart();
    try {
      if (L.kind === 'scalar') {
        const vals = await fetchField(L.varName, t);
        if (L.pending !== tok) return;
        if (!L.userRange && vals.range) setRange(L, vals.range[0], vals.range[1]);
        uploadMap(L, vals);
      } else {
        const vec = await fetchVector(L.pair, t);
        if (L.pending !== tok) return;
        L.magP98 = vec.p98 || 1;
        if (L.style === 'arrows') {
          ensureArrGL(L);
          gl.bindBuffer(gl.ARRAY_BUFFER, L.gl.vecBuf);
          gl.bufferSubData(gl.ARRAY_BUFFER, 0, vec);
        } else {
          if (!L.userRange) { const r = defaultRange(L); setRange(L, r[0], r[1]); }
          uploadMap(L, deriveSlab(L.pair, t, L.style, vec));
        }
      }
      L.ready = true;
      state.dirty = true;
      if (!wasReady) refreshLayersPanel();   // card leaves its "loading…" state
      if (t + 1 < F.meta.n_timesteps) {      // prefetch for smooth scrubbing
        if (L.kind === 'scalar') fetchField(L.varName, t + 1).catch(() => {});
        else fetchVector(L.pair, t + 1).catch(() => {});
      }
    } catch (e) { toast('forcing: ' + e.message, true); }
    finally { loadEnd(); }
  }

  function setRange(L, lo, hi) {
    L.range = [lo, hi];
    if (L._rangeUI) L._rangeUI(lo, hi);
  }

  function uploadMap(L, nodeVals) {
    ensureMapGL(L);
    const g = F.mesh.gathered, ni = F.mesh.nodeIdx;
    for (let i = 0; i < g.length; i++) g[i] = nodeVals[ni[i]];
    gl.bindBuffer(gl.ARRAY_BUFFER, L.gl.valB);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, g);
  }

  /* ── clock sync: follow the shared playbar (state.clock.t, abs days) ────── */
  function syncClock() {
    const td = F.timeDays;
    if (!td || !td.length) return null;
    const t = state.clock.t;
    let k = F.lastK;
    const inSlot = i => td[i] <= t && (i + 1 >= td.length || t < td[i + 1]);
    if (!(k >= 0 && k < td.length && inSlot(k))) {
      if (k + 1 < td.length && inSlot(k + 1)) k = k + 1;          // playback fast path
      else if (t <= td[0]) k = 0;
      else if (t >= td[td.length - 1]) k = td.length - 1;
      else {
        let lo = 0, hi = td.length - 1;
        while (hi - lo > 1) { const m = (lo + hi) >> 1; if (td[m] <= t) lo = m; else hi = m; }
        k = lo;
      }
    }
    F.lastK = k;
    if (k !== F.t) {
      F.t = k;
      const tl = $('frc-time'); if (tl) tl.value = k;
      updateTimeLabel();
      return updateLayers();
    }
    return null;
  }

  /* set the clock to an absolute day and resolve when the upload finished
     (used by the video export loop so frames aren't stale) */
  async function setTime(absDay) {
    state.clock.t = absDay;
    const p = syncClock();
    if (p) await p;
  }

  /* ── draw (called from drawScene, after the model layer) ────────────────── */
  function drawLayer() {
    syncClock();
    if (!F.mesh || !state.world0 || !F.show) return;
    const off = [-state.world0[0], -state.world0[1]];
    for (const L of F.layers) {
      if (!L.visible || !L.ready) continue;
      if (L.kind === 'vector' && L.style === 'arrows') drawQuivers(L, off);
      else drawMap(L, off);
    }
  }

  function drawMap(L, off) {
    if (!L.gl.vao) return;
    gl.useProgram(progFrc.prog);
    gl.uniform4fv(progFrc.u.uView, viewUniform());
    gl.uniform2fv(progFrc.u.uOff, off);
    gl.uniform2f(progFrc.u.uClim, L.range[0], L.range[1]);
    gl.uniform1f(progFrc.u.uAlpha, L.alpha);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, cmapTex(L.cmap));
    gl.uniform1i(progFrc.u.uRamp, 0);
    gl.bindVertexArray(L.gl.vao);
    gl.drawElements(gl.TRIANGLES, F.mesh.nTris * 3, gl.UNSIGNED_INT, 0);
    gl.bindVertexArray(null);
  }

  function drawQuivers(L, off) {
    if (!L.gl.arrVAO) return;
    gl.bindVertexArray(L.gl.arrVAO);
    if (L.gl.arrStride !== L.stride) {   // decimate by advancing `stride` nodes per instance
      gl.bindBuffer(gl.ARRAY_BUFFER, F.mesh.nodeBuf);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 8 * L.stride, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, L.gl.vecBuf);
      gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 8 * L.stride, 0);
      L.gl.arrStride = L.stride;
    }
    gl.useProgram(progArr.prog);
    gl.uniform4fv(progArr.u.uView, viewUniform());
    gl.uniform2fv(progArr.u.uOff, off);
    gl.uniform2f(progArr.u.uCanvas, canvas.width, canvas.height);
    // world-anchored arrow length: a p98-magnitude arrow spans cell_p01·50^f
    // meters (log scale, f = scale slider 0..1) — arrows zoom with the map
    const p01 = F.mesh.cellP01 || 100;
    const lenW = p01 * Math.pow(50, L.scaleF == null ? 0.5 : L.scaleF);
    gl.uniform1f(progArr.u.uScale, lenW * state.view.ppm);
    gl.uniform1f(progArr.u.uMagRef, L.magP98);
    gl.uniform1i(progArr.u.uColorMag, L.colorMag ? 1 : 0);
    const rgb = hex2rgb(L.color || '#ffffff');
    gl.uniform3f(progArr.u.uColor, rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, cmapTex(L.cmap));
    gl.uniform1i(progArr.u.uRamp, 0);
    gl.drawArraysInstanced(gl.LINES, 0, 6, Math.floor(F.mesh.nNodes / L.stride));
    gl.bindVertexArray(null);
  }

  /* ── panel UI ───────────────────────────────────────────────────────────── */
  function refreshPanel() {
    refreshAddChoices();
    refreshLayersPanel();
    const tl = $('frc-time');
    tl.max = F.meta ? Math.max(0, F.meta.n_timesteps - 1) : 0;
    tl.value = F.t;
    updateTimeLabel();
  }

  function refreshAddChoices() {
    const sel = $('frc-add-var');
    sel.innerHTML = '';
    if (!F.meta) return;
    const ogV = document.createElement('optgroup');
    ogV.label = 'vector fields';
    for (const p of F.meta.vector_pairs) {
      const op = document.createElement('option');
      op.value = 'v:' + p.x;
      op.textContent = p.label;
      ogV.appendChild(op);
    }
    if (ogV.children.length) sel.appendChild(ogV);
    const ogS = document.createElement('optgroup');
    ogS.label = 'scalars';
    for (const v of F.meta.variables.filter(v => !v.is_vector_comp)) {
      const op = document.createElement('option');
      op.value = 's:' + v.name;
      op.textContent = v.name + (v.units ? ` [${v.units}]` : '');
      op.title = v.long_name;
      ogS.appendChild(op);
    }
    if (ogS.children.length) sel.appendChild(ogS);
  }

  function refreshLayersPanel() {
    const box = $('frc-layers');
    box.innerHTML = '';
    if (!F.layers.length) {
      box.innerHTML = '<div class="hint">no layers — add one below</div>';
    }
    // GIS convention: list top layer first (reverse of draw order)
    for (let i = F.layers.length - 1; i >= 0; i--) box.appendChild(buildLayerCard(F.layers[i]));
    if (typeof LayersPanel !== 'undefined') LayersPanel.refresh();
  }

  function buildLayerCard(L) {
    const card = document.createElement('div');
    card.className = 'rule';
    const rerender = () => { refreshLayersPanel(); state.dirty = true; };

    // ── header: eye · label · reorder/remove tools ──
    const head = document.createElement('div');
    head.style.cssText = 'display:flex;align-items:center;gap:6px;width:100%';
    const eye = document.createElement('button');
    eye.className = 'eye' + (L.visible ? '' : ' off');
    eye.textContent = '👁';
    eye.onclick = () => {
      L.visible = !L.visible;
      eye.classList.toggle('off', !L.visible);
      if (L.visible && !L.ready) updateLayers(L);
      state.dirty = true;
      if (typeof LayersPanel !== 'undefined') LayersPanel.refresh();
    };
    head.appendChild(eye);
    const lbl = document.createElement('b');
    lbl.textContent = layerLabel(L);
    lbl.style.cssText = 'font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap';
    head.appendChild(lbl);
    const tools = document.createElement('span');
    tools.className = 'tools';
    const mk = (txt, title, fn) => {
      const b = document.createElement('button');
      b.className = 'mini'; b.textContent = txt; b.title = title; b.onclick = fn;
      tools.appendChild(b);
    };
    const idx = () => F.layers.indexOf(L);
    mk('▲', 'draw later (on top)', () => {
      const i = idx();
      if (i < F.layers.length - 1) {
        [F.layers[i], F.layers[i + 1]] = [F.layers[i + 1], F.layers[i]];
        rerender();
      }
    });
    mk('▼', 'draw earlier (below)', () => {
      const i = idx();
      if (i > 0) { [F.layers[i], F.layers[i - 1]] = [F.layers[i - 1], F.layers[i]]; rerender(); }
    });
    mk('✕', 'remove layer', () => { removeLayer(L); rerender(); });
    head.appendChild(tools);
    card.appendChild(head);

    const row = (label, ...els) => {
      const r = document.createElement('div');
      r.className = 'row small';
      r.style.width = '100%';
      if (label) {
        const s = document.createElement('span');
        s.className = 'dim'; s.style.width = '58px'; s.textContent = label;
        r.appendChild(s);
      }
      for (const el of els) r.appendChild(el);
      card.appendChild(r);
      return r;
    };
    const mkSel = (opts, val, fn) => {
      const s = document.createElement('select');
      s.className = 'grow';
      for (const [v, txt] of opts) {
        const op = document.createElement('option');
        op.value = v; op.textContent = txt;
        s.appendChild(op);
      }
      s.value = val;
      s.onchange = () => fn(s.value);
      return s;
    };

    if (!L.ready) {
      const ld = document.createElement('span');
      ld.className = 'dim small';
      ld.textContent = '⏳ loading…';
      head.insertBefore(ld, tools);
    }

    // ── variable: ONE list holding vector fields AND scalars, so a layer can
    //    switch between the two kinds in place ──
    const varSel = document.createElement('select');
    varSel.className = 'grow';
    const addOpts = (lbl, items) => {
      if (!items.length) return;
      const og = document.createElement('optgroup');
      og.label = lbl;
      for (const [v, txt] of items) {
        const op = document.createElement('option');
        op.value = v; op.textContent = txt;
        og.appendChild(op);
      }
      varSel.appendChild(og);
    };
    addOpts('vector fields', F.meta.vector_pairs.map(p => ['v:' + p.x, p.label]));
    addOpts('scalars', F.meta.variables.filter(v => !v.is_vector_comp)
      .map(v => ['s:' + v.name, v.name + (v.units ? ` [${v.units}]` : '')]));
    varSel.value = L.kind === 'scalar' ? 's:' + L.varName : 'v:' + L.pair.x;
    varSel.onchange = () => {
      const v = varSel.value;
      freeLayerGL(L);                    // kind may change — rebuild GL state
      L.userRange = false;
      if (v.startsWith('v:')) {
        L.kind = 'vector';
        L.pair = F.meta.vector_pairs.find(p => p.x === v.slice(2));
        if (L.style === 'map') L.style = 'mag';
        if (L.cmap === 'viridis') L.cmap = 'turbo';
      } else {
        L.kind = 'scalar';
        L.varName = v.slice(2);
        L.style = 'map';
        if (L.cmap === 'phase' || L.cmap === 'RdBu' || L.cmap === 'turbo') L.cmap = 'viridis';
      }
      updateLayers(L).then(rerender);
    };
    row('variable', varSel);
    if (L.kind === 'vector') {
      // ── style (vector layers only) ──
      row('style', mkSel(VEC_STYLES, L.style, v => {
        L.style = v;
        L.userRange = false;
        L.ready = false;
        if (v === 'dir') L.cmap = 'phase';
        else if (v === 'x' || v === 'y') L.cmap = 'RdBu';
        else if (L.cmap === 'phase' || L.cmap === 'RdBu') L.cmap = 'turbo';
        updateLayers(L).then(rerender);
      }));
    }

    if (isMapStyle(L)) {
      // ── colormap (direction is locked to the cyclic phase map) ──
      if (L.style !== 'dir') {
        row('colormap', mkSel(Object.keys(CMAPS).filter(c => c !== 'phase').map(c => [c, c]),
          L.cmap, v => { L.cmap = v; ramp.style.background = cmapCss(v); state.dirty = true; }));
      }
      // ── range ──
      const inLo = document.createElement('input');
      const inHi = document.createElement('input');
      for (const el of [inLo, inHi]) { el.type = 'number'; el.step = 'any'; el.style.width = '74px'; }
      const to = document.createElement('span'); to.className = 'dim'; to.textContent = 'to';
      const rst = document.createElement('button');
      rst.className = 'mini'; rst.textContent = '↺'; rst.title = 'reset to data range';
      const setUI = (lo, hi) => {
        inLo.value = +lo.toPrecision(4); inHi.value = +hi.toPrecision(4);
        lblLo.textContent = +lo.toPrecision(3); lblHi.textContent = +hi.toPrecision(3);
      };
      L._rangeUI = setUI;
      inLo.onchange = inHi.onchange = () => {
        L.userRange = true;
        L.range = [parseFloat(inLo.value), parseFloat(inHi.value)];
        lblLo.textContent = inLo.value; lblHi.textContent = inHi.value;
        state.dirty = true;
      };
      rst.onclick = () => { L.userRange = false; updateLayers(L); };
      const rrow = row('range', inLo, to, inHi, rst);
      if (L.style === 'dir') rrow.classList.add('grey');
      // ── opacity ──
      const op = document.createElement('input');
      op.type = 'range'; op.min = '0.15'; op.max = '1'; op.step = '0.05'; op.value = L.alpha;
      op.oninput = () => { L.alpha = parseFloat(op.value); state.dirty = true; };
      row('opacity', op);
      // ── mini colorbar ──
      const ramp = document.createElement('div');
      ramp.className = 'ramp';
      ramp.style.cssText = 'height:8px;border-radius:3px;flex:1;background:' + cmapCss(L.cmap);
      const lblLo = document.createElement('span'); lblLo.className = 'dim small';
      const lblHi = document.createElement('span'); lblHi.className = 'dim small';
      row('', lblLo, ramp, lblHi);
      setUI(L.range[0], L.range[1]);
    } else {
      // ── arrows controls ──
      const dn = document.createElement('input');
      dn.type = 'range'; dn.min = '0'; dn.max = '4'; dn.step = '1';
      dn.value = String(Math.max(0, Math.min(4, Math.round(Math.log2(L.stride || 4)))));
      const dnLbl = document.createElement('span');
      dnLbl.className = 'dim small';
      const dTxt = () => L.stride === 1 ? 'all' : '1 / ' + L.stride;
      dnLbl.textContent = dTxt();
      dn.oninput = () => {
        L.stride = 2 ** parseInt(dn.value, 10);
        dnLbl.textContent = dTxt();
        state.dirty = true;
      };
      row('density', dn, dnLbl);
      const sc = document.createElement('input');
      sc.type = 'range'; sc.min = '0'; sc.max = '1'; sc.step = '0.01';
      sc.value = String(L.scaleF == null ? 0.5 : L.scaleF);
      sc.title = 'arrow size on the map — smallest ≈ the finest model cells, ' +
                 'largest = 50× that (log scale)';
      sc.oninput = () => { L.scaleF = parseFloat(sc.value); state.dirty = true; };
      row('scale', sc);
      const cm = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox'; cb.checked = L.colorMag;
      cb.onchange = () => { L.colorMag = cb.checked; state.dirty = true; rerender(); };
      cm.appendChild(cb);
      cm.appendChild(document.createTextNode(' color by magnitude'));
      row('', cm);
      if (L.colorMag) {
        row('colormap', mkSel(Object.keys(CMAPS).filter(c => c !== 'phase').map(c => [c, c]),
          L.cmap, v => { L.cmap = v; state.dirty = true; }));
      } else if (typeof mkColorBtn === 'function') {
        row('color', mkColorBtn(() => L.color || '#ffffff',
          c => { L.color = c; state.dirty = true; }, 'arrow color'));
      }
    }
    return card;
  }

  function updateTimeLabel() {
    $('frc-tlabel').textContent = F.meta && F.meta.times[F.t] ?
      `${F.meta.times[F.t]}  (${F.t + 1}/${F.meta.n_timesteps})` : '–';
  }

  /* fine-step controls drive the SHARED clock (playbar follows) */
  function gotoStep(k) {
    if (!F.meta) return;
    k = Math.max(0, Math.min(F.meta.n_timesteps - 1, k));
    if (F.timeDays) {
      state.clock.playing = false;
      if (typeof refreshPlayBtn === 'function') refreshPlayBtn();
      state.clock.t = F.timeDays[k];
      if (typeof updateTimelineUI === 'function') updateTimelineUI();
      syncClock();
    } else {
      F.t = k;
      updateTimeLabel();
      updateLayers();
    }
    $('frc-time').value = F.t;
    state.dirty = true;
  }

  function wire() {
    $('frc-open').onclick = async () => {
      try {
        const r = await apiJson('/api/pickfile?kind=nc');
        if (r.paths && r.paths[0]) await open(r.paths[0]);
      } catch (e) { toast(String(e), true); }
    };
    $('frc-load').onclick = async () => {
      const p = $('frc-path').value.trim();
      if (!p) return toast('enter a netCDF path first', true);
      try { await open(p); } catch (e) { toast(String(e), true); }
    };
    $('frc-add').onclick = () => {
      const v = $('frc-add-var').value;
      if (!v || !F.meta) return;
      if (v.startsWith('v:')) {
        const pair = F.meta.vector_pairs.find(p => p.x === v.slice(2));
        if (pair) addLayer({kind: 'vector', pair});
      } else {
        addLayer({kind: 'scalar', varName: v.slice(2)});
      }
      refreshLayersPanel();
      updateLayers();
    };
    const eye = (id, get, set) => {
      const b = $(id);
      const upd = () => b.classList.toggle('off', !get());
      b.onclick = ev => { ev.preventDefault(); ev.stopPropagation(); set(!get()); upd(); };
      upd();
    };
    eye('eye-frc', () => F.show, v => {
      F.show = v; state.dirty = true;
      if (typeof LayersPanel !== 'undefined') LayersPanel.refresh();
    });
    $('frc-time').oninput = () => gotoStep(parseInt($('frc-time').value, 10));
    $('frc-step-b').onclick = () => gotoStep(F.t - 1);
    $('frc-step-f').onclick = () => gotoStep(F.t + 1);
  }

  let wired = false;
  /* open the forcing file referenced by the loaded config (called on config
     load from the Settings tab, and lazily on first entry of a map tab) */
  async function autoOpenFromConfig() {
    if (!wired) { wire(); wired = true; }
    const cfg = typeof SettingsTab !== 'undefined' ? SettingsTab.getConfig() : null;
    const cfgPath = cfg ? SettingsTab.getPath() : null;
    const dataRel = cfg && cfg.inputs && cfg.inputs.data;
    if (!dataRel) return false;
    if (F.meta && F.meta.raw_input === dataRel) return true;   // already open
    const base = cfgPath ? cfgPath.replace(/[\\/][^\\/]+$/, '') : null;
    try {
      await open(dataRel, base);
      F.meta.raw_input = dataRel;
      return true;
    } catch (e) { toast('forcing: ' + e.message, true); return false; }
  }

  async function enter() {
    if (!wired) { wire(); wired = true; }
    try {
      // always re-sync with the config: the Settings tab's inputs.data is the
      // single source of the forcing file (no-op when unchanged)
      if (!(await autoOpenFromConfig()) && !F.id) {
        const last = localStorage.getItem('stgui_last_forcing');
        if (last) await open(last);
      }
    } catch (e) { /* nothing auto-openable yet */ }
  }

  // no own tab anymore: the forcing sections live in the Viewer tab, whose
  // enter hook (app.js) calls ForcingTab.enter()
  return {drawLayer, open, state: F, enter, autoOpenFromConfig, setTime,
          updateLayers, refreshLayersPanel};
})();
