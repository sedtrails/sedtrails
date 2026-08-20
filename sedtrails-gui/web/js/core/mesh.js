"use strict";
/* ═════════════════════════ model-layer mesh ═════════════════════════ */
async function loadMesh(run) {
  const m = run.meta.mesh;
  if (!m || state.mesh) return;
  try {
    const buf = await (await fetch(`/data/${run.id}/${m.file}`)).arrayBuffer();
    const N = m.n_nodes, M = m.n_tris;
    let o = 0;
    const xy = new Float32Array(buf, o, N * 2); o += N * 8;
    const bed = new Float32Array(buf, o, N);    o += N * 4;
    const dz = new Float32Array(buf, o, N);     o += N * 4;
    const tris = new Uint32Array(buf, o, M * 3);

    const posB = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posB);
    gl.bufferData(gl.ARRAY_BUFFER, xy, gl.STATIC_DRAW);
    const idxB = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxB);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, tris, gl.STATIC_DRAW);

    const mkVao = vals => {
      const vb = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vb);
      gl.bufferData(gl.ARRAY_BUFFER, vals, gl.STATIC_DRAW);
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, posB);
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, vb);
      gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 4, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxB);
      gl.bindVertexArray(null);
      return vao;
    };
    state.mesh = {
      runId: run.id,
      off: [run.quantU[2], run.quantU[3]],
      nIdx: M * 3,
      vaoBed: mkVao(bed),
      vaoDz: mkVao(dz),
      climBed: m.clim_bed,
      climDz: m.clim_dz,
      hasDz: m.has_dz,
      rampBed: makeRampTexture(m.cmap_bed.filter((_, i) => i % 16 === 0)),
      rampDz: makeRampTexture(m.cmap_dz.filter((_, i) => i % 16 === 0)),
    };
    // model layer stays OFF by default — the Background group only carries the
    // base map now; bathymetry is shown through the Forcing layers instead
    refreshModelLayers();
    state.dirty = true;
  } catch (e) {
    toast('Model layer failed to load: ' + e.message, true);
  }
}

function refreshModelLayers() {
  const sel = $('sel-model');
  const cur = state.modelLayer;
  sel.innerHTML = '<option value="bed">bathymetry</option>' +
    (state.mesh && state.mesh.hasDz ? '<option value="dz">bed-level change</option>' : '');
  sel.value = cur;
  if (sel.value !== cur) { sel.value = 'bed'; state.modelLayer = 'bed'; }
  syncVis();
  if (typeof LayersPanel !== 'undefined') LayersPanel.refresh();
}

function drawMeshLayer() {
  const m = state.mesh;
  if (!m || !state.showModel) return;
  const useDz = state.modelLayer === 'dz';
  gl.useProgram(progMesh.prog);
  gl.uniform4fv(progMesh.u.uView, viewUniform());
  gl.uniform2fv(progMesh.u.uOff, m.off);
  gl.uniform2fv(progMesh.u.uClim, useDz ? m.climDz : m.climBed);
  gl.uniform1f(progMesh.u.uAlpha, state.meshAlpha);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, useDz ? m.rampDz : m.rampBed);
  gl.uniform1i(progMesh.u.uRamp, 0);
  gl.bindVertexArray(useDz ? m.vaoDz : m.vaoBed);
  gl.drawElements(gl.TRIANGLES, m.nIdx, gl.UNSIGNED_INT, 0);
  gl.bindVertexArray(null);
}

