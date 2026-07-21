"use strict";
/* ═════════════════════════ polygon registry ═════════════════════════ */
/* closed shapes only (polygons & boxes) — these are usable for classification,
   filter rules and connectivity; points/transects live in the same store but
   are excluded here */
function allPolys() {
  return state.polygons
    .filter(p => !p.type || p.type === 'polygon' || p.type === 'bbox')
    .map(p => ({key: p.name, verts: p.verts, name: p.name, color: p.color}));
}
function ensurePolyColors() {
  let changed = false;
  state.polygons.forEach((p, i) => {
    if (!p.type) p.type = 'polygon';           // legacy entries predate types
    if (!p.color) { p.color = PALETTE[i % PALETTE.length]; changed = true; }
  });
  if (changed) savePolygons();
  rebuildPaletteTex();
}
function polyColorOf(key) {
  const p = allPolys().find(q => q.key === key);
  return p && p.color ? p.color : '#d14b4b';
}
/* area-weighted polygon centroid (world coords); vertex mean when degenerate —
   used by the connectivity network layer (render.js) */
function polyCentroid(p) {
  const v = p.verts;
  if (!v || !v.length) return null;
  let a = 0, cx = 0, cy = 0;
  for (let i = 0; i < v.length; i++) {
    const [x0, y0] = v[i], [x1, y1] = v[(i + 1) % v.length];
    const cr = x0 * y1 - x1 * y0;
    a += cr; cx += (x0 + x1) * cr; cy += (y0 + y1) * cr;
  }
  if (Math.abs(a) < 1e-9)
    return [v.reduce((s, q) => s + q[0], 0) / v.length,
            v.reduce((s, q) => s + q[1], 0) / v.length];
  return [cx / (3 * a), cy / (3 * a)];
}

/* ═════════════════════════ run loading ═════════════════════════ */
function makeRun(meta, stride) {
  const N = Math.floor((meta.n_particles + stride - 1) / stride);
  const T = meta.n_timesteps;
  const run = {
    id: meta.id, meta, stride, N, T,
    // reference_date may carry a time-of-day ("2025-08-03 21:40:00")
    epoch: (d => Date.parse(d.length > 10 ? d.replace(' ', 'T') + 'Z'
                                          : d + 'T00:00:00Z'))
           (String(meta.reference_date).trim()) / 86400000,
    frames: new Array(T).fill(null),
    burFrames: null, burLoading: false,
    flagFrames: null, flagLoading: false,
    loaded: 0,
    attrs: {origin: new Uint8Array(N).fill(255), final: new Uint8Array(N).fill(255),
            first: new Uint16Array(N).fill(SENT), visited: new Uint32Array(N),
            at0: new Uint32Array(N), atEnd: new Uint32Array(N)},
    visible: true,
    tint: SIM_TINTS[state.runs.length % SIM_TINTS.length],
    classifying: false,
    counts: 0,
    distMax: 0,
  };
  const q = meta.quant;
  run.quantU = [q.sx, q.sy, q.xmin - state.world0[0], q.ymin - state.world0[1]];

  run.posA = gl.createBuffer(); run.posB = gl.createBuffer();
  run.burA = gl.createBuffer(); run.burB = gl.createBuffer();
  run.attrB = gl.createBuffer();
  for (const b of [run.posA, run.posB]) {
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, N * 4, gl.DYNAMIC_DRAW);
  }
  for (const b of [run.burA, run.burB]) {
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, N * 2, gl.DYNAMIC_DRAW);
  }
  run.flgA = gl.createBuffer(); run.flgB = gl.createBuffer();
  for (const b of [run.flgA, run.flgB]) {   // default 1 = mobile (no fade)
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(N).fill(1), gl.DYNAMIC_DRAW);
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, run.attrB);
  gl.bufferData(gl.ARRAY_BUFFER, N * 16, gl.STATIC_DRAW);

  const idx = new Uint32Array(N);
  for (let i = 0; i < N; i++) idx[i] = i;
  let seed = 1234567;
  for (let i = N - 1; i > 0; i--) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    const j = seed % (i + 1);
    const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
  }
  run.shuffle = idx;
  run.ebo = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, run.ebo);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);

  run.vaos = [null, null].map((_, half) => {
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const cur = half === 0 ? run.posA : run.posB;
    const nxt = half === 0 ? run.posB : run.posA;
    gl.bindBuffer(gl.ARRAY_BUFFER, cur);
    gl.enableVertexAttribArray(0); gl.vertexAttribIPointer(0, 2, gl.UNSIGNED_SHORT, 4, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, nxt);
    gl.enableVertexAttribArray(1); gl.vertexAttribIPointer(1, 2, gl.UNSIGNED_SHORT, 4, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, run.attrB);
    gl.enableVertexAttribArray(2); gl.vertexAttribIPointer(2, 2, gl.UNSIGNED_BYTE, 16, 0);
    gl.enableVertexAttribArray(3); gl.vertexAttribIPointer(3, 1, gl.UNSIGNED_SHORT, 16, 2);
    gl.enableVertexAttribArray(6); gl.vertexAttribIPointer(6, 1, gl.UNSIGNED_INT, 16, 4);
    gl.enableVertexAttribArray(7); gl.vertexAttribIPointer(7, 1, gl.UNSIGNED_INT, 16, 8);
    gl.enableVertexAttribArray(8); gl.vertexAttribIPointer(8, 1, gl.UNSIGNED_INT, 16, 12);
    const bcur = half === 0 ? run.burA : run.burB;
    const bnxt = half === 0 ? run.burB : run.burA;
    gl.bindBuffer(gl.ARRAY_BUFFER, bcur);
    gl.enableVertexAttribArray(4); gl.vertexAttribIPointer(4, 1, gl.UNSIGNED_SHORT, 2, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, bnxt);
    gl.enableVertexAttribArray(5); gl.vertexAttribIPointer(5, 1, gl.UNSIGNED_SHORT, 2, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, half === 0 ? run.flgA : run.flgB);
    gl.enableVertexAttribArray(9); gl.vertexAttribIPointer(9, 1, gl.UNSIGNED_BYTE, 1, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, run.ebo);
    gl.bindVertexArray(null);
    return vao;
  });
  run.bufFrame = [-1, -1];
  uploadAttrs(run);

  run.segVBO = gl.createBuffer();
  run.segVAO = gl.createVertexArray();
  gl.bindVertexArray(run.segVAO);
  gl.bindBuffer(gl.ARRAY_BUFFER, run.segVBO);
  gl.enableVertexAttribArray(1); gl.vertexAttribIPointer(1, 4, gl.UNSIGNED_SHORT, 12, 0);
  gl.vertexAttribDivisor(1, 1);
  gl.enableVertexAttribArray(2); gl.vertexAttribIPointer(2, 2, gl.UNSIGNED_SHORT, 12, 8);
  gl.vertexAttribDivisor(2, 1);
  gl.bindVertexArray(null);
  run.segCount = 0;
  run.pathsDirty = true;
  return run;
}

function uploadAttrs(run) {
  const N = run.N;
  const buf = new ArrayBuffer(N * 16);
  const u8 = new Uint8Array(buf);
  const u16 = new Uint16Array(buf);
  const u32 = new Uint32Array(buf);
  for (let i = 0; i < N; i++) {
    u8[i*16]   = run.attrs.origin[i];
    u8[i*16+1] = run.attrs.final[i];
    u16[i*8+1] = run.attrs.first[i];
    u32[i*4+1] = run.attrs.visited[i];
    u32[i*4+2] = run.attrs.at0[i];
    u32[i*4+3] = run.attrs.atEnd[i];
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, run.attrB);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, u8);
}

async function loadRun(bundleId, stride, name) {
  if (state.runs.some(r => r.id === bundleId)) { toast('Simulation already loaded'); return; }
  const meta = await apiJson(`/data/${bundleId}/meta.json`);
  if (!state.world0) state.world0 = [meta.quant.xmin, meta.quant.ymin];
  if (name) meta.run_name = name;
  const run = makeRun(meta, stride);
  state.runs.push(run);

  $('modal-open').style.display = 'none';
  updateClockRange();
  updateTimelineUI();
  if (state.runs.length === 1) fitView(run);
  refreshSimList(); refreshRules();
  streamPositions(run);
  requestClassify(run);
  loadMesh(run);
  if (rulesNeedBurial() || state.fade.bur) streamBurial(run);
  if (state.fade.imm) streamFlags(run);
  state.dirty = true;
}

async function streamPositions(run) {
  const fullN = run.meta.n_particles;
  const frameBytes = fullN * 4;
  const resp = await fetch(`/data/${run.id}/positions.bin`);
  if (!resp.ok) { toast(`positions.bin failed for ${run.meta.run_name}`, true); return; }
  const reader = resp.body.getReader();
  const carry = new Uint8Array(frameBytes);
  let carryLen = 0, t = 0;
  try {
    while (t < run.T) {
      const {done, value} = await reader.read();
      if (done) break;
      let o = 0;
      while (o < value.length && t < run.T) {
        const take = Math.min(frameBytes - carryLen, value.length - o);
        carry.set(value.subarray(o, o + take), carryLen);
        carryLen += take; o += take;
        if (carryLen === frameBytes) {
          const full = new Uint16Array(carry.buffer, 0, fullN * 2);
          let fr;
          if (run.stride === 1) {
            fr = new Uint16Array(fullN * 2);
            fr.set(full);
          } else {
            fr = new Uint16Array(run.N * 2);
            for (let i = 0, j = 0; j < run.N; i += run.stride, j++) {
              fr[j*2] = full[i*2]; fr[j*2+1] = full[i*2+1];
            }
          }
          run.frames[t] = fr;
          run.loaded = ++t;
          carryLen = 0;
          if ((t & 7) === 0 || t === run.T) { refreshSimList(); state.dirty = true; }
        }
      }
    }
  } catch (e) {
    toast(`Loading ${run.meta.run_name} failed: ${e.message}. Try a larger particle stride.`, true);
  }
  refreshSimList(); updateLoadStatus(); updateCounts(); state.dirty = true;
}

async function streamBurial(run) {
  if (run.burFrames || run.burLoading || !run.meta.has_burial) return;
  run.burLoading = true;
  const fullN = run.meta.n_particles;
  const frameBytes = fullN * 2;
  try {
    run.burFrames = new Array(run.T).fill(null);
    const resp = await fetch(`/data/${run.id}/burial.bin`);
    if (!resp.ok) throw new Error('burial.bin not available');
    const reader = resp.body.getReader();
    const carry = new Uint8Array(frameBytes);
    let carryLen = 0, t = 0;
    while (t < run.T) {
      const {done, value} = await reader.read();
      if (done) break;
      let o = 0;
      while (o < value.length && t < run.T) {
        const take = Math.min(frameBytes - carryLen, value.length - o);
        carry.set(value.subarray(o, o + take), carryLen);
        carryLen += take; o += take;
        if (carryLen === frameBytes) {
          const full = new Uint16Array(carry.buffer, 0, fullN);
          const fr = new Uint16Array(run.N);
          if (run.stride === 1) fr.set(full);
          else for (let i = 0, j = 0; j < run.N; i += run.stride, j++) fr[j] = full[i];
          run.burFrames[t] = fr; t++;
          carryLen = 0;
        }
      }
    }
    // robust auto color range: p98 of the final burial field — burial_max is
    // often dominated by a few extreme cells, flattening all real signal
    const lastFr = [...run.burFrames].reverse().find(f => f);
    if (lastFr) {
      const vals = [];
      const st = Math.max(1, Math.floor(lastFr.length / 20000));
      for (let i = 0; i < lastFr.length; i += st)
        if (lastFr[i] !== SENT) vals.push(lastFr[i]);
      vals.sort((a, b) => a - b);
      if (vals.length)
        run.burP98 = Math.max(vals[Math.floor(vals.length * 0.98)] / 65534
                              * (run.meta.burial_max || 0.01), 0.01);
    }
    run.bufFrame = [-1, -1];
    updateColorbars(); refreshScaleRows();
    state.dirty = true;
  } catch (e) {
    run.burFrames = null;
    toast(`Burial data: ${e.message}`, true);
  }
  run.burLoading = false;
}

async function streamFlags(run) {
  if (run.flagFrames || run.flagLoading) return;
  if (!run.meta.has_flags) {
    toast(`${run.meta.run_name}: no particle status in this bundle — re-open the simulation once to extract it`, true);
    return;
  }
  run.flagLoading = true;
  const fullN = run.meta.n_particles;
  try {
    run.flagFrames = new Array(run.T).fill(null);
    const resp = await fetch(`/data/${run.id}/flags.bin`);
    if (!resp.ok) throw new Error('flags.bin not available');
    const reader = resp.body.getReader();
    const carry = new Uint8Array(fullN);
    let carryLen = 0, t = 0;
    while (t < run.T) {
      const {done, value} = await reader.read();
      if (done) break;
      let o = 0;
      while (o < value.length && t < run.T) {
        const take = Math.min(fullN - carryLen, value.length - o);
        carry.set(value.subarray(o, o + take), carryLen);
        carryLen += take; o += take;
        if (carryLen === fullN) {
          const fr = new Uint8Array(run.N);
          if (run.stride === 1) fr.set(carry);
          else for (let i = 0, j = 0; j < run.N; i += run.stride, j++) fr[j] = carry[i];
          run.flagFrames[t] = fr; t++;
          carryLen = 0;
        }
      }
    }
    run.bufFrame = [-1, -1];
    state.dirty = true;
  } catch (e) {
    run.flagFrames = null;
    toast(`Status data: ${e.message}`, true);
  }
  run.flagLoading = false;
}

/* ═════════════════════════ classification ═════════════════════════ */
const requestClassifyAll = debounce(() => state.runs.forEach(requestClassify), 500);

async function requestClassify(run) {
  if (run.classifying) { run.classifyPending = true; return; }
  const polys = allPolys().map(p => ({key: p.key, verts: p.verts}));
  if (!polys.length) {
    run.attrs.origin.fill(255); run.attrs.final.fill(255);
    run.attrs.first.fill(SENT); run.attrs.visited.fill(0);
    run.attrs.at0.fill(0); run.attrs.atEnd.fill(0);
    uploadAttrs(run); afterClassify();
    return;
  }
  run.classifying = true; refreshClassifyStatus();
  const progTimer = setInterval(async () => {
    try {
      const p = await apiJson('/api/classify/progress?bundle=' + run.id);
      if (p.T) $('classify-status').innerHTML =
        `<b style="color:var(--warn)">⟳ classifying ${p.n} new polygon(s) — ` +
        `day ${p.t + 1}/${p.T} · filters use the previous polygons until done</b>`;
    } catch (e) {}
  }, 1500);
  const tStart = performance.now();
  try {
    const r = await api('/api/classify', {bundle_id: run.id, polygons: polys});
    const buf = new Uint8Array(await r.arrayBuffer());
    const nl = buf.indexOf(10);
    const header = JSON.parse(new TextDecoder().decode(buf.subarray(0, nl)));
    const N = header.n;
    let o = nl + 1;
    const slice32 = start => new Uint32Array(
      buf.buffer.slice(buf.byteOffset + start, buf.byteOffset + start + 4*N));
    const fOrigin = buf.subarray(o, o + N);
    const fFinal = buf.subarray(o + N, o + 2*N);
    const fFirst = new Uint16Array(buf.buffer.slice(buf.byteOffset + o + 2*N, buf.byteOffset + o + 4*N));
    const fVisited = slice32(o + 4*N);
    const fAt0 = slice32(o + 8*N);
    const fAtEnd = slice32(o + 12*N);
    const s = run.stride;
    for (let i = 0, j = 0; j < run.N; i += s, j++) {
      run.attrs.origin[j] = fOrigin[i];
      run.attrs.final[j] = fFinal[i];
      run.attrs.first[j] = fFirst[i];
      run.attrs.visited[j] = fVisited[i];
      run.attrs.at0[j] = fAt0[i];
      run.attrs.atEnd[j] = fAtEnd[i];
    }
    uploadAttrs(run);
    run.pathsDirty = true;
    if (performance.now() - tStart > 3000)
      toast('Polygon classification updated — filters now reflect the new polygons');
    // (no auto-switch to polygon coloring — "in their simulation color" is
    // the default appearance now)
  } catch (e) {
    toast(`Classification failed: ${e.message}`, true);
  }
  clearInterval(progTimer);
  run.classifying = false;
  afterClassify();
  if (run.classifyPending) {           // polygon set changed meanwhile —
    run.classifyPending = false;       // collapse all edits into one re-run
    requestClassify(run);
  }
}

function afterClassify() {
  refreshClassifyStatus(); updateCounts();
  state.runs.forEach(r => { r.pathsDirty = true; });
  state.dirty = true;
}

function refreshClassifyStatus() {
  const busy = state.runs.some(r => r.classifying);
  $('classify-status').innerHTML = busy
    ? '<b style="color:var(--warn)">⟳ classifying particles… rules use the previous polygons until done</b>'
    : allPolys().length ? `${allPolys().length} polygon(s) classified` : '';
}

/* ═════════════════════════ rule engine (CPU mirror) ═════════════════════════ */
function compileCond(cond, poly, thresh, keys, run) {
  if (cond === 'always' || !cond) return {c: 0, a: 0, t: 0};
  if (cond === 'bur_ge' || cond === 'bur_lt')
    return {c: cond === 'bur_ge' ? 4 : 5, a: 0,
            t: Math.max(0, thresh || 0) / run.meta.burial_max * 65534};
  const c = cond === 'start' ? 1 : cond === 'end' ? 2 : 3;
  let a;
  if (poly === '*any*') a = -1;
  else if (poly === '*none*') a = -2;
  else { a = keys.indexOf(poly); if (a < 0) return null; }
  return {c, a, t: 0};
}

function compiledRules(run) {
  const keys = allPolys().map(p => p.key);
  const out = [];
  for (const rule of state.rules) {
    if (rule.sim !== '*' && rule.sim !== run.id) continue;
    if (out.length >= MAX_RULES) break;
    const c1 = compileCond(rule.cond, rule.poly, rule.thresh, keys, run);
    const c2 = rule.cond2 ? compileCond(rule.cond2, rule.poly2, rule.thresh2, keys, run)
                          : {c: 0, a: 0, t: 0};
    if (!c1 || !c2) continue;
    const action = rule.action === 'hide' ? 0 : rule.action === 'color' ? 1
      : rule.colorby === 'burial' ? 2 : rule.colorby === 'age' ? 3 : 4;
    out.push({c1, c2, action, rgb: hex2rgb(rule.color || '#888888')});
  }
  return out;
}

function condEval(run, cc, i, bur) {
  const mask = cc.c === 1 ? run.attrs.at0[i] : cc.c === 2 ? run.attrs.atEnd[i]
             : run.attrs.visited[i];
  switch (cc.c) {
    case 0: return true;
    case 4: return bur >= cc.t;
    case 5: return bur < cc.t;
    default:
      if (cc.a === -1) return mask !== 0;
      if (cc.a === -2) return mask === 0;
      return (mask & (1 << cc.a)) !== 0;
  }
}

function evalVisible(run, compiled, i, bur) {
  for (const r of compiled)
    if (condEval(run, r.c1, i, bur) && condEval(run, r.c2, i, bur))
      return r.action !== 0;
  return state.defaultApp.action !== 'hide';
}

function updateCounts() {
  let total = 0;
  const {k} = state.runs.length ? frameAt(state.runs[0], state.clock.t) : {k: 0};
  for (const run of state.runs) {
    if (!run.visible || !run.loaded) { run.counts = 0; continue; }
    const compiled = compiledRules(run);
    const kk = Math.min(k, run.loaded - 1);
    const bf = run.burFrames && run.burFrames[kk];
    let c = 0;
    for (let i = 0; i < run.N; i++)
      if (evalVisible(run, compiled, i, bf ? bf[i] : 0)) c++;
    run.counts = c;
    total += c * run.stride;
  }
  state.selCount = total;
  $('st-sel').innerHTML = `<b>${fmtInt(total)}</b> selected`;
  updateColorbars();
}

function rulesNeedBurial() {
  return state.rules.some(r => ['bur_ge', 'bur_lt'].includes(r.cond)
      || ['bur_ge', 'bur_lt'].includes(r.cond2)
      || (r.action === 'colorby' && r.colorby === 'burial'))
    || (state.defaultApp.action === 'colorby' && state.defaultApp.colorby === 'burial')
    || (state.paths.show && state.paths.mode === 'burial');
}

function onRulesChanged() {
  if (rulesNeedBurial()) state.runs.forEach(streamBurial);
  updateCounts(); refreshScaleRows(); schedulePathRebuild(); state.dirty = true;
}

/* ═════════════════════════ color scales & colorbars ═════════════════════════ */
function autoRange(prop) {
  const spanDays = Math.max(1, Math.round(state.clock.t1 - state.clock.t0));
  const burP98 = Math.max(...state.runs.map(r => r.burP98 || 0), 0);
  const burMax = burP98 > 0 ? burP98
               : Math.max(0.01, ...state.runs.map(r => r.meta.burial_max || 0));
  const distMax = Math.max(...state.runs.map(r => r.distMax || 0), 1);
  return prop === 'burial' ? [0, burMax]
       : prop === 'dist' ? [0, distMax]
       : [0, spanDays];
}
function effRange(prop) {
  const [lo, hi] = state.ranges[prop] || [null, null];
  const [alo, ahi] = autoRange(prop);
  return [lo === null || lo === undefined ? alo : lo,
          hi === null || hi === undefined ? ahi : hi];
}
function activePointProps() {
  const used = new Set();
  if (state.defaultApp.action === 'colorby') used.add(state.defaultApp.colorby);
  for (const r of state.rules) if (r.action === 'colorby') used.add(r.colorby);
  return used;
}

const PROP_INFO = {
  burial: {title: 'burial depth (m)', unit: 'm', dec: 2},
  age: {title: 'age (days)', unit: 'd', dec: 0},
  first: {title: 'first polygon visit (days)', unit: 'd', dec: 0},
  dist: {title: 'distance travelled (km)', unit: 'km', dec: 1},
};

function cmapCss(name) {
  const stops = CMAPS[name] || CMAPS.viridis;
  return 'linear-gradient(90deg,' + stops.map((c, i) =>
    `rgb(${c[0]},${c[1]},${c[2]}) ${(i / (stops.length - 1) * 100).toFixed(0)}%`).join(',') + ')';
}

function fmtVal(v, prop) {
  const d = PROP_INFO[prop].dec;
  return prop === 'dist' ? (v / 1000).toFixed(d) : v.toFixed(d);
}

function updateColorbars() {
  const el = $('colorbars');
  el.innerHTML = '';
  const add = (tag, prop, cmap) => {
    const [lo, hi] = effRange(prop);
    const div = document.createElement('div');
    div.className = 'cbar';
    div.innerHTML = `<div class="title">${tag} · ${PROP_INFO[prop].title}</div>
      <div class="ramp" style="background:${cmapCss(cmap)}"></div>
      <div class="lbl"><span>${fmtVal(lo, prop)}</span><span>${fmtVal(hi, prop)}</span></div>`;
    el.appendChild(div);
  };
  for (const prop of activePointProps()) add('points', prop, state.pointCmap);
  if (state.paths.show && state.paths.mode !== 'solid')
    add('paths', state.paths.mode, state.paths.cmap);
}

function mkScaleRow(labelTxt, prop, cmapGet, cmapSet) {
  const div = document.createElement('div');
  div.className = 'scalerow';
  div.innerHTML = `<span class="dim">${labelTxt}</span>`;
  const sel = document.createElement('select');
  for (const name of Object.keys(CMAPS)) {
    const o = document.createElement('option');
    o.value = name; o.textContent = name;
    if (name === cmapGet()) o.selected = true;
    sel.appendChild(o);
  }
  sel.onchange = () => { cmapSet(sel.value); updateColorbars(); state.dirty = true; };
  div.appendChild(sel);
  const [alo, ahi] = effRange(prop);
  const mkNum = (idx, val) => {
    const inp = document.createElement('input');
    inp.type = 'number'; inp.step = 'any';
    inp.value = fmtVal(val, prop);
    inp.onchange = () => {
      let v = parseFloat(inp.value);
      if (prop === 'dist') v *= 1000;
      state.ranges[prop][idx] = isFinite(v) ? v : null;
      updateColorbars(); state.dirty = true;
    };
    return inp;
  };
  const lo = mkNum(0, alo), hi = mkNum(1, ahi);
  const dash = document.createElement('span'); dash.className = 'dim'; dash.textContent = '–';
  const unit = document.createElement('span'); unit.className = 'dim';
  unit.textContent = PROP_INFO[prop].unit;
  const auto = document.createElement('button'); auto.className = 'mini'; auto.textContent = 'auto';
  auto.onclick = () => {
    state.ranges[prop] = [null, null];
    refreshScaleRows(); updateColorbars(); state.dirty = true;
  };
  div.append(lo, dash, hi, unit, auto);
  return div;
}

function refreshScaleRows() {
  const el = $('pointscale');
  el.innerHTML = '';
  for (const prop of activePointProps())
    el.appendChild(mkScaleRow('scale · ' + PROP_INFO[prop].title.split(' (')[0], prop,
      () => state.pointCmap, v => { state.pointCmap = v; }));
  const pe = $('pathscale');
  pe.innerHTML = '';
  if (state.paths.show && state.paths.mode !== 'solid')
    pe.appendChild(mkScaleRow('scale', state.paths.mode,
      () => state.paths.cmap, v => { state.paths.cmap = v; }));
}

/* ═════════════════════════ paths ═════════════════════════ */
function rebuildPaths(run) {
  run.pathsDirty = false;
  const compiled = compiledRules(run);
  const {k} = frameAt(run, state.clock.t);
  const kk = Math.min(k, Math.max(run.loaded - 1, 0));
  const bf = run.burFrames && run.burFrames[kk];
  const limit = Math.ceil(run.N / state.decim);
  const sel = [];
  for (let s = 0; s < limit; s++) {
    const i = run.shuffle[s];
    if (evalVisible(run, compiled, i, bf ? bf[i] : 0)) sel.push(i);
  }
  // cap the number of drawn trajectories: each is ~T CPU-built segments, so
  // unlimited paths of a large selection would freeze the tab
  const segBudget = softwareGL ? 600000 : 3000000;
  const maxPaths = Math.max(50, Math.floor(segBudget / Math.max(run.T - 1, 1)));
  const step = Math.max(1, Math.ceil(sel.length / maxPaths));
  const chosen = [];
  for (let i = 0; i < sel.length; i += step) chosen.push(sel[i]);
  run.pathsShown = chosen.length;
  run.pathsTotal = sel.length;

  const T = run.loaded;
  const mode = state.paths.mode;
  const q = run.meta.quant;
  const maxSegs = Math.max(1, chosen.length * Math.max(T - 1, 1));
  const seg = new Uint16Array(maxSegs * 6);
  let ns = 0, distMax = 0;
  const segDist = mode === 'dist' ? new Float32Array(maxSegs) : null;

  for (const p of chosen) {
    let px = -1, py = -1, cum = 0;
    for (let t = 0; t < T; t++) {
      const fr = run.frames[t];
      if (!fr) break;
      const x = fr[p*2], y = fr[p*2+1];
      if (x === SENT) { px = -1; continue; }
      if (px >= 0) {
        const o = ns * 6;
        seg[o] = px; seg[o+1] = py; seg[o+2] = x; seg[o+3] = y;
        seg[o+4] = t;
        if (mode === 'burial') {
          const b = run.burFrames && run.burFrames[t] ? run.burFrames[t][p] : 0;
          seg[o+5] = b === SENT ? 0 : b;
        } else if (mode === 'age') {
          seg[o+5] = Math.round(t / Math.max(run.T - 1, 1) * 65534);
        } else if (mode === 'dist') {
          cum += Math.hypot((x - px) * q.sx, (y - py) * q.sy);
          segDist[ns] = cum;
        }
        ns++;
      }
      px = x; py = y;
    }
    if (cum > distMax) distMax = cum;
  }
  if (mode === 'dist' && ns) {
    const f = 65534 / Math.max(distMax, 1e-9);
    for (let i = 0; i < ns; i++) seg[i*6+5] = Math.round(segDist[i] * f);
  }
  run.distMax = distMax;
  gl.bindBuffer(gl.ARRAY_BUFFER, run.segVBO);
  gl.bufferData(gl.ARRAY_BUFFER, seg.subarray(0, ns * 6), gl.DYNAMIC_DRAW);
  run.segCount = ns;
  run.pathLoadedAt = run.loaded;
  updateColorbars();
  updatePathsNote();
}

function updatePathsNote() {
  let shown = 0, tot = 0;
  for (const r of state.runs) { shown += r.pathsShown || 0; tot += r.pathsTotal || 0; }
  $('paths-note').textContent = !state.paths.show || !tot ? ''
    : shown < tot ? `showing ${fmtInt(shown)} of ${fmtInt(tot)} filtered paths (auto-capped for speed)`
    : `all ${fmtInt(tot)} filtered paths`;
}
const schedulePathRebuild = debounce(() => {
  state.runs.forEach(r => { r.pathsDirty = true; });
  state.dirty = true;
}, 250);

