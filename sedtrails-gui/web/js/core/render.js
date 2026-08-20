"use strict";
/* ═════════════════════════ rendering ═════════════════════════ */
function resize() {
  if (state.exporting) return;
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
    state.dirty = true;
  }
}

function viewUniform() {
  const {cx, cy, ppm} = state.view;
  const sx = 2 * ppm / canvas.width, sy = 2 * ppm / canvas.height;
  return [sx, sy, -cx * sx, -cy * sy];
}

function fitView(run) {
  const e = run.meta.extent;
  const w = e[1] - e[0], h = e[3] - e[2];
  state.view.cx = (e[0] + e[1]) / 2 - state.world0[0];
  state.view.cy = (e[2] + e[3]) / 2 - state.world0[1];
  state.view.ppm = Math.min(canvas.width / w, canvas.height / h) * 0.94;
}

function screenToWorld(px, py) {
  const dpr = window.devicePixelRatio || 1;
  const x = (px * dpr - canvas.width / 2) / state.view.ppm + state.view.cx;
  const y = (canvas.height / 2 - py * dpr) / state.view.ppm + state.view.cy;
  return [x, y];
}
function screenToRd(px, py) {
  const [lx, ly] = screenToWorld(px, py);
  return state.world0 ? [lx + state.world0[0], ly + state.world0[1]] : [lx, ly];
}

function uploadFrame(run, k, slot) {
  const fr = run.frames[k];
  if (!fr) return false;
  gl.bindBuffer(gl.ARRAY_BUFFER, slot === 0 ? run.posA : run.posB);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, fr);
  if (run.burFrames && run.burFrames[k]) {
    gl.bindBuffer(gl.ARRAY_BUFFER, slot === 0 ? run.burA : run.burB);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, run.burFrames[k]);
  }
  if (run.flagFrames && run.flagFrames[k]) {
    gl.bindBuffer(gl.ARRAY_BUFFER, slot === 0 ? run.flgA : run.flgB);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, run.flagFrames[k]);
  }
  run.bufFrame[slot] = k;
  return true;
}

function setRuleUniforms(prog, run) {
  const compiled = compiledRules(run);
  const ri = new Int32Array(MAX_RULES * 4);
  const rc = new Float32Array(MAX_RULES * 4);
  const rt = new Float32Array(MAX_RULES * 2);
  compiled.forEach((r, i) => {
    ri[i*4] = r.c1.c; ri[i*4+1] = r.c1.a; ri[i*4+2] = r.c2.c; ri[i*4+3] = r.c2.a;
    rc[i*4] = r.rgb[0]/255; rc[i*4+1] = r.rgb[1]/255; rc[i*4+2] = r.rgb[2]/255;
    rc[i*4+3] = r.action;
    rt[i*2] = r.c1.t; rt[i*2+1] = r.c2.t;
  });
  gl.uniform1i(prog.u.uNRules, compiled.length);
  gl.uniform4iv(prog.u.uRuleI, ri);
  gl.uniform4fv(prog.u.uRuleC, rc);
  gl.uniform2fv(prog.u.uRuleT, rt);
  const d = state.defaultApp;
  const defAction = d.action === 'hide' ? 0 : d.action === 'color' ? 1
    : d.action === 'colorby' ? (d.colorby === 'burial' ? 2 : d.colorby === 'age' ? 3 : 4)
    : d.action === 'palette-origin' ? 5 : d.action === 'palette-final' ? 6 : 7;
  gl.uniform1i(prog.u.uDefAction, defAction);
  const dc = hex2rgb(d.color || NEUTRAL);
  gl.uniform3f(prog.u.uDefColor, dc[0]/255, dc[1]/255, dc[2]/255);
  const tc = hex2rgb(run.tint);
  gl.uniform3f(prog.u.uTint, tc[0]/255, tc[1]/255, tc[2]/255);
}

function drawPoints(run, frac, kCur, kNext) {
  let vaoIdx;
  if (run.bufFrame[0] === kCur && run.bufFrame[1] === kNext) vaoIdx = 0;
  else if (run.bufFrame[1] === kCur && run.bufFrame[0] === kNext) vaoIdx = 1;
  else if (run.bufFrame[1] === kCur) { uploadFrame(run, kNext, 0); vaoIdx = 1; }
  else if (run.bufFrame[0] === kNext) { uploadFrame(run, kCur, 1); vaoIdx = 1; }
  else if (run.bufFrame[1] === kNext) { uploadFrame(run, kCur, 0); vaoIdx = 0; }
  else { uploadFrame(run, kCur, 0); uploadFrame(run, kNext, 1); vaoIdx = 0; }
  if (run.bufFrame[vaoIdx] !== kCur) return;

  const sizePx = state.pointSize * (window.devicePixelRatio || 1);
  gl.useProgram(progPt.prog);
  gl.uniform4fv(progPt.u.uQuant, run.quantU);
  gl.uniform4fv(progPt.u.uView, viewUniform());
  gl.uniform1f(progPt.u.uFrac, kNext === kCur ? 0 : frac);
  gl.uniform1f(progPt.u.uPointSize, sizePx);
  gl.uniform1f(progPt.u.uRound, sizePx >= 2.0 ? 1 : 0);
  gl.uniform1f(progPt.u.uAlpha, state.alpha);
  gl.uniform1f(progPt.u.uBurMax, run.meta.burial_max || 0.01);
  const fadeQ = state.fade.bur && run.meta.has_burial
    ? Math.max(state.fade.depth, 0.001) / (run.meta.burial_max || 0.01) * 65534 : 0;
  gl.uniform3f(progPt.u.uFade, fadeQ,
               state.fade.imm && run.flagFrames ? 1 : 0, state.fade.alpha);
  const td = run.meta.time_days;
  gl.uniform1f(progPt.u.uNowDays, Math.max(0, state.clock.t - run.epoch - td[0]));
  gl.uniform2fv(progPt.u.uVRbur, effRange('burial'));
  gl.uniform2fv(progPt.u.uVRage, effRange('age'));
  gl.uniform2fv(progPt.u.uVRfirst, effRange('first'));
  setRuleUniforms(progPt, run);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, cmapTex(state.pointCmap));
  gl.uniform1i(progPt.u.uCmap, 0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, paletteTex);
  gl.uniform1i(progPt.u.uPalette, 1);
  gl.bindVertexArray(run.vaos[vaoIdx]);
  gl.drawElements(gl.POINTS, Math.ceil(run.N / state.decim), gl.UNSIGNED_INT, 0);
  gl.bindVertexArray(null);
}

function drawPolyFill(p, rgb) {
  // stencil parity fill — works for concave polygons, no triangulation needed
  const w0 = state.world0;
  const n = p.verts.length;
  const arr = new Float32Array(n * 2);
  let bx0 = Infinity, bx1 = -Infinity, by0 = Infinity, by1 = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = p.verts[i][0] - w0[0], y = p.verts[i][1] - w0[1];
    arr[i*2] = x; arr[i*2+1] = y;
    bx0 = Math.min(bx0, x); bx1 = Math.max(bx1, x);
    by0 = Math.min(by0, y); by1 = Math.max(by1, y);
  }
  gl.enable(gl.STENCIL_TEST);
  gl.clear(gl.STENCIL_BUFFER_BIT);
  gl.colorMask(false, false, false, false);
  gl.stencilFunc(gl.ALWAYS, 0, 0xff);
  gl.stencilOp(gl.KEEP, gl.KEEP, gl.INVERT);
  gl.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
  gl.drawArrays(gl.TRIANGLE_FAN, 0, n);
  gl.colorMask(true, true, true, true);
  gl.stencilFunc(gl.EQUAL, 1, 1);
  gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
  const quad = new Float32Array([bx0, by0, bx1, by0, bx0, by1, bx1, by1]);
  gl.bufferData(gl.ARRAY_BUFFER, quad, gl.DYNAMIC_DRAW);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
  gl.uniform4f(progFlat.u.uColor, rgb[0]/255, rgb[1]/255, rgb[2]/255, 0.25);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.disable(gl.STENCIL_TEST);
}

/* WebGL lineWidth > 1 is unsupported on Windows/ANGLE — thick lines are drawn
   as per-segment quads (screen-space width) plus round joint points that hide
   the segment seams. Expects progFlat active, the scratch buffer bound and
   uColor set; leaves uRound at 0. */
function strokePath(arr, n, closed, pxWidth) {
  const dpr = window.devicePixelRatio || 1;
  const hw = pxWidth * dpr / state.view.ppm / 2;          // half-width, world units
  const segs = closed ? n : n - 1;
  const quads = new Float32Array(segs * 12);
  let o = 0;
  for (let i = 0; i < segs; i++) {
    const j = (i + 1) % n;
    const x0 = arr[i*2], y0 = arr[i*2+1], x1 = arr[j*2], y1 = arr[j*2+1];
    let dx = x1 - x0, dy = y1 - y0;
    const L = Math.hypot(dx, dy) || 1;
    const nx = -dy / L * hw, ny = dx / L * hw;
    quads[o++] = x0+nx; quads[o++] = y0+ny; quads[o++] = x0-nx; quads[o++] = y0-ny;
    quads[o++] = x1+nx; quads[o++] = y1+ny;
    quads[o++] = x1+nx; quads[o++] = y1+ny; quads[o++] = x0-nx; quads[o++] = y0-ny;
    quads[o++] = x1-nx; quads[o++] = y1-ny;
  }
  gl.bufferData(gl.ARRAY_BUFFER, quads, gl.DYNAMIC_DRAW);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
  gl.drawArrays(gl.TRIANGLES, 0, quads.length / 2);
  gl.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
  gl.uniform1f(progFlat.u.uPtSize, pxWidth * dpr);
  gl.uniform1f(progFlat.u.uRound, 1);
  gl.drawArrays(gl.POINTS, 0, n);
  gl.uniform1f(progFlat.u.uRound, 0);
}

function drawPolyOutlines() {
  if (!state.world0) return;
  gl.useProgram(progFlat.prog);
  gl.uniform4fv(progFlat.u.uView, viewUniform());
  gl.uniform1f(progFlat.u.uPtSize, 7);
  if (!drawPolyOutlines.buf) drawPolyOutlines.buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, drawPolyOutlines.buf);
  gl.enableVertexAttribArray(0);
  const w0 = state.world0;
  for (const p of state.polygons) {
    const sel = state.editSel && state.editSel.name === p.name;
    if (!sel && (!state.showOutlines || p.hidden)) continue;
    const type = p.type || 'polygon';
    const rgb = hex2rgb(p.color || '#445566');
    if (sel && type !== 'point' && type !== 'transect') drawPolyFill(p, rgb);
    const arr = new Float32Array(p.verts.length * 2);
    for (let i = 0; i < p.verts.length; i++) {
      arr[i*2] = p.verts[i][0] - w0[0]; arr[i*2+1] = p.verts[i][1] - w0[1];
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, drawPolyOutlines.buf);
    gl.uniform4f(progFlat.u.uColor, rgb[0]/255, rgb[1]/255, rgb[2]/255, 1.0);
    if (type !== 'point' && p.verts.length > 1)
      strokePath(arr, p.verts.length, type !== 'transect', sel ? 3 : 2);
    if (type === 'point' || type === 'transect') {   // endpoints / release points
      gl.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
      gl.uniform1f(progFlat.u.uPtSize, 7);
      gl.uniform1f(progFlat.u.uRound, 1);
      gl.drawArrays(gl.POINTS, 0, p.verts.length);
      gl.uniform1f(progFlat.u.uRound, 0);
    }
    if (sel) {   // edit handles: the object's own color, larger, round
      gl.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
      gl.uniform1f(progFlat.u.uPtSize, 11);
      gl.uniform1f(progFlat.u.uRound, 1);
      gl.drawArrays(gl.POINTS, 0, p.verts.length);
      gl.uniform1f(progFlat.u.uRound, 0);
      gl.uniform1f(progFlat.u.uPtSize, 7);
    }
  }
  if (state.draft.length) {   // always visible while drawing
    const arr = new Float32Array(state.draft.length * 2);
    for (let i = 0; i < state.draft.length; i++) {
      arr[i*2] = state.draft[i][0] - w0[0]; arr[i*2+1] = state.draft[i][1] - w0[1];
    }
    gl.uniform4f(progFlat.u.uColor, 0.85, 0.55, 0.1, 1.0);
    if (state.draft.length > 1) strokePath(arr, state.draft.length, false, 2);
    gl.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
    gl.uniform1f(progFlat.u.uPtSize, 7);
    gl.uniform1f(progFlat.u.uRound, 1);
    gl.drawArrays(gl.POINTS, 0, state.draft.length);
    gl.uniform1f(progFlat.u.uRound, 0);
  }
}

/* ── connectivity network (Pearson et al. 2021 style): one node per polygon
   centroid (size ∝ departures), directed edges i→j whose width scales with the
   matrix value, colored by the SOURCE polygon. Data = state.connectivity (set
   by the Connectivity tab); toggled from the Layers panel. ── */
function drawConnectivity() {
  const con = state.connectivity;
  if (!state.showConnectivity || !con || !state.world0) return;
  if (typeof allPolys !== 'function') return;
  const polys = allPolys();
  const w0 = state.world0;
  const cents = con.labels.map(l => {
    const p = polys.find(q => q.key === l);
    return p ? polyCentroid(p) : null;      // deleted polygons: node/edges skipped
  });
  const n = con.labels.length;
  let vmax = 0, rmax = 1;
  for (let i = 0; i < n; i++) {
    rmax = Math.max(rmax, con.row_counts[i] || 0);
    for (let j = 0; j < n; j++) if (i !== j) vmax = Math.max(vmax, con.matrix[i][j]);
  }
  if (!vmax) return;                        // no off-diagonal transport at all

  gl.useProgram(progFlat.prog);
  gl.uniform4fv(progFlat.u.uView, viewUniform());
  if (!drawConnectivity.buf) drawConnectivity.buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, drawConnectivity.buf);
  gl.enableVertexAttribArray(0);
  const dpr = window.devicePixelRatio || 1;
  const pxw = dpr / state.view.ppm;         // world units per CSS pixel

  for (let i = 0; i < n; i++) {
    const ci = cents[i];
    if (!ci) continue;
    const rgb = hex2rgb(polyColorOf(con.labels[i]));
    gl.uniform4f(progFlat.u.uColor, rgb[0]/255, rgb[1]/255, rgb[2]/255, 0.78);
    for (let j = 0; j < n; j++) {
      const v = con.matrix[i][j], cj = cents[j];
      if (i === j || !cj || !v || v / vmax < 0.02) continue;   // declutter floor
      const wpx = 1.5 + 6.5 * v / vmax;
      let dx = cj[0] - ci[0], dy = cj[1] - ci[1];
      const L = Math.hypot(dx, dy) || 1;
      dx /= L; dy /= L;
      const nx = -dy * 3 * pxw, ny = dx * 3 * pxw;   // separate i→j from j→i
      const ah = (5 + wpx * 1.6) * pxw;              // arrowhead length
      const tipX = cj[0] - w0[0] + nx - dx * 10 * pxw;
      const tipY = cj[1] - w0[1] + ny - dy * 10 * pxw;
      const bX = tipX - dx * ah, bY = tipY - dy * ah;
      const shaft = new Float32Array([
        ci[0] - w0[0] + nx + dx * 8 * pxw, ci[1] - w0[1] + ny + dy * 8 * pxw, bX, bY]);
      strokePath(shaft, 2, false, wpx);
      const ux = -dy * ah * 0.55, uy = dx * ah * 0.55;
      const tri = new Float32Array([tipX, tipY, bX + ux, bY + uy, bX - ux, bY - uy]);
      gl.bufferData(gl.ARRAY_BUFFER, tri, gl.DYNAMIC_DRAW);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
  }
  for (let i = 0; i < n; i++) {             // nodes on top of the edges
    const ci = cents[i];
    if (!ci) continue;
    const rgb = hex2rgb(polyColorOf(con.labels[i]));
    const pt = new Float32Array([ci[0] - w0[0], ci[1] - w0[1]]);
    gl.bufferData(gl.ARRAY_BUFFER, pt, gl.DYNAMIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
    gl.uniform1f(progFlat.u.uPtSize,
                 (9 + 9 * Math.sqrt((con.row_counts[i] || 0) / rmax)) * dpr);
    gl.uniform1f(progFlat.u.uRound, 1);
    gl.uniform4f(progFlat.u.uColor, rgb[0]/255, rgb[1]/255, rgb[2]/255, 1.0);
    gl.drawArrays(gl.POINTS, 0, 1);
  }
  gl.uniform1f(progFlat.u.uRound, 0);
}

function drawPathsFBO(clock) {
  ensureOffFBO();
  gl.bindFramebuffer(gl.FRAMEBUFFER, off.fbo);
  gl.viewport(0, 0, off.w, off.h);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  for (const run of state.runs) {
    if (!run.visible || !run.loaded) continue;
    if (run.pathsDirty || run.pathLoadedAt !== run.loaded) rebuildPaths(run);
    if (!run.segCount) continue;
    const {k, frac} = frameAt(run, clock.t);
    const mode = state.paths.mode;
    const pmax = mode === 'age' ? Math.max(run.T - 1, 1)
               : mode === 'dist' ? Math.max(run.distMax, 1e-9)
               : (run.meta.burial_max || 0.01);
    gl.useProgram(progSeg.prog);
    gl.uniform4fv(progSeg.u.uQuant, run.quantU);
    gl.uniform4fv(progSeg.u.uView, viewUniform());
    gl.uniform2f(progSeg.u.uCanvas, canvas.width, canvas.height);
    gl.uniform1f(progSeg.u.uWidthPx, state.paths.width * (window.devicePixelRatio || 1));
    gl.uniform1f(progSeg.u.uNow, k + frac);
    gl.uniform1f(progSeg.u.uPmax, pmax);
    gl.uniform2fv(progSeg.u.uVR, effRange(mode === 'solid' ? 'age' : mode));
    gl.uniform1i(progSeg.u.uSolid, mode === 'solid' ? 1 : 0);
    const pc = hex2rgb(state.paths.color);
    gl.uniform3f(progSeg.u.uSolidColor, pc[0]/255, pc[1]/255, pc[2]/255);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, cmapTex(state.paths.cmap));
    gl.uniform1i(progSeg.u.uCmap, 0);
    gl.bindVertexArray(run.segVAO);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, run.segCount);
    gl.bindVertexArray(null);
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.useProgram(progBlit.prog);
  gl.uniform1f(progBlit.u.uAlpha, state.paths.alpha);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, off.tex);
  gl.uniform1i(progBlit.u.uTex, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, screenQuad);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

function drawScene() {
  const clock = state.clock;
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.disable(gl.DEPTH_TEST);
  gl.clearColor(0.906, 0.918, 0.933, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  drawTiles();
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  drawMeshLayer();
  if (typeof ForcingTab !== 'undefined') ForcingTab.drawLayer();   // forcing field + quivers
  if (typeof SeedingTab !== 'undefined') SeedingTab.drawLayer();   // seed-point preview
  if (state.paths.show && state.showParticles) drawPathsFBO(clock);
  if (state.showPoints && state.showParticles) for (const run of state.runs) {
    if (!run.visible || !run.loaded) continue;
    const {k, frac} = frameAt(run, clock.t);
    const kk = Math.min(k, run.loaded - 1);
    const kNext = Math.min(k + 1, run.loaded - 1);
    drawPoints(run, frac, kk, kNext);
  }
  drawPolyOutlines();
  drawConnectivity();
}

let lastTime = performance.now(), fpsEMA = 0;
function render(now) {
  requestAnimationFrame(render);
  if (state.exporting) return;
  resize();
  const dt = Math.min(0.1, (now - lastTime) / 1000);
  lastTime = now;

  const clock = state.clock;
  if (clock.playing) {
    clock.t += clock.speed * dt;
    const front = loadedFrontier();
    if (clock.t >= Math.min(clock.t1, front)) clock.t = Math.min(clock.t1, front);
    if (clock.t >= clock.t1) { clock.playing = false; refreshPlayBtn(); }
    updateTimelineUI();
    state.dirty = true;
  }
  if (!state.dirty) { updateFps(now, true); return; }
  state.dirty = state.clock.playing;
  drawScene();
  updateFps(now, false);
}

function updateFps(now, idle) {
  if (!updateFps.last) updateFps.last = now;
  const dt = now - updateFps.last;
  updateFps.last = now;
  if (!idle)
    fpsEMA = fpsEMA ? fpsEMA * 0.92 + (1000 / Math.max(dt, 0.01)) * 0.08 : 1000 / Math.max(dt, 0.01);
  if (!updateFps.tick || now - updateFps.tick > 500) {
    updateFps.tick = now;
    $('st-fps').innerHTML = `<b>${state.clock.playing ? Math.round(fpsEMA) : '–'}</b> fps`;
    updateLoadStatus();
  }
}

function updateLoadStatus() {
  let tot = 0, got = 0;
  for (const r of state.runs) { tot += r.T; got += r.loaded; }
  const el = $('st-load');
  if (tot && got < tot) {
    el.style.display = '';
    el.innerHTML = `loading <b>${Math.round(got / tot * 100)}%</b>`;
  } else el.style.display = 'none';
}

