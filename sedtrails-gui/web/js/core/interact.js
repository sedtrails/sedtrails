"use strict";
/* ═════════════════════════ canvas interaction ═════════════════════════ */
let drag = null;

/* vertex / edge hit-testing for the edit tool (shared with the Objects panel) */
const MIN_VERTS = {polygon: 3, bbox: 3, transect: 2, point: 1};
function hitTol() { return 10 / state.view.ppm * (window.devicePixelRatio || 1); }
function nearestVert(p, ax, ay, tol) {
  let best = -1, bestD = tol * tol;
  p.verts.forEach((v, i) => {
    const d = (v[0] - ax) ** 2 + (v[1] - ay) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  });
  return best;
}
function nearestSegment(p, ax, ay, tol) {
  const type = p.type || 'polygon';
  if (type === 'point') return null;               // point sets have no edges
  const n = p.verts.length;
  const segs = type === 'transect' ? n - 1 : n;    // closed shapes include the closing edge
  let best = null, bestD = tol * tol;
  for (let i = 0; i < segs; i++) {
    const a = p.verts[i], b = p.verts[(i + 1) % n];
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const L2 = dx * dx + dy * dy;
    if (!L2) continue;
    const t = Math.max(0, Math.min(1, ((ax - a[0]) * dx + (ay - a[1]) * dy) / L2));
    const x = a[0] + t * dx, y = a[1] + t * dy;
    const d = (x - ax) ** 2 + (y - ay) ** 2;
    if (d < bestD) { bestD = d; best = {i, x, y}; }
  }
  return best;
}
function editTarget() {
  return state.tool === 'edit' && state.editSel && state.world0
    ? state.polygons.find(x => x.name === state.editSel.name) : null;
}

canvas.addEventListener('mousedown', e => {
  if (state.tool === 'draw') return;
  const p = e.button === 0 ? editTarget() : null;
  if (p) {
    const [ax, ay] = screenToRd(e.clientX, e.clientY);
    const tol = hitTol();
    const best = nearestVert(p, ax, ay, tol);
    if (best >= 0) { drag = {vert: {p, i: best}}; return; }
    if (e.altKey && (p.type || 'polygon') === 'point') {   // Alt-click: add a point
      p.verts.push([ax, ay]);
      drag = {vert: {p, i: p.verts.length - 1}};
      state.dirty = true;
      return;
    }
    const seg = nearestSegment(p, ax, ay, tol);
    if (seg) {   // click near an edge: insert a vertex there and drag it right away
      p.verts.splice(seg.i + 1, 0, [seg.x, seg.y]);
      drag = {vert: {p, i: seg.i + 1}};
      state.dirty = true;
      return;
    }
  }
  drag = {pan: {x: e.clientX, y: e.clientY, cx: state.view.cx, cy: state.view.cy}};
  canvas.classList.add('panning');
});
window.addEventListener('mousemove', e => {
  if (state.world0) {
    const [ax, ay] = screenToRd(e.clientX, e.clientY);
    $('coords').textContent = `${Math.round(ax)}, ${Math.round(ay)} RD`;
  }
  if (!drag) {
    const p = editTarget();   // hover feedback over a movable point
    if (p) {
      const [ax, ay] = screenToRd(e.clientX, e.clientY);
      canvas.classList.toggle('vtx', nearestVert(p, ax, ay, hitTol()) >= 0);
    } else if (canvas.classList.contains('vtx')) canvas.classList.remove('vtx');
    return;
  }
  if (drag.pan) {
    const dpr = window.devicePixelRatio || 1;
    state.view.cx = drag.pan.cx - (e.clientX - drag.pan.x) * dpr / state.view.ppm;
    state.view.cy = drag.pan.cy + (e.clientY - drag.pan.y) * dpr / state.view.ppm;
    state.dirty = true;
  } else if (drag.vert) {
    drag.vert.p.verts[drag.vert.i] = screenToRd(e.clientX, e.clientY);
    state.dirty = true;
  }
});
window.addEventListener('mouseup', () => {
  if (drag && drag.vert) onPolygonsChanged();
  drag = null;
  canvas.classList.remove('panning');
});
canvas.addEventListener('contextmenu', e => {
  const p = editTarget();
  if (!p) return;                            // not editing — normal context menu
  e.preventDefault();
  const [ax, ay] = screenToRd(e.clientX, e.clientY);
  const i = nearestVert(p, ax, ay, hitTol());
  if (i < 0) return;
  const type = p.type || 'polygon';
  if (p.verts.length <= (MIN_VERTS[type] || 3)) {
    toast(`a ${type} needs at least ${MIN_VERTS[type]} point(s)`, true);
    return;
  }
  p.verts.splice(i, 1);
  onPolygonsChanged();
});
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const f = Math.exp(-e.deltaY * 0.0013);
  const [wx, wy] = screenToWorld(e.clientX, e.clientY);
  state.view.ppm *= f;
  const [wx2, wy2] = screenToWorld(e.clientX, e.clientY);
  state.view.cx += wx - wx2;
  state.view.cy += wy - wy2;
  state.dirty = true;
}, {passive: false});
canvas.addEventListener('click', e => {
  if (state.tool !== 'draw' || !state.world0) return;
  state.draft.push(screenToRd(e.clientX, e.clientY));
  // two-click shapes finish themselves; others keep the coord table live
  const m = state.drawMode || 'polygon';
  if ((m === 'transect' || m === 'bbox') && state.draft.length === 2) finishDraft();
  else if (typeof ObjectsPanel !== 'undefined') ObjectsPanel.refresh();
  state.dirty = true;
});
canvas.addEventListener('dblclick', e => {
  if (state.tool === 'draw') {
    e.preventDefault();
    if (state.draft.length > 1) state.draft.pop();   // drop the dblclick's extra vertex
    finishDraft();
  }
});
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'Escape') {
    if (state.tool === 'draw') {
      state.draft = []; state.tool = 'pan'; refreshToolButtons(); state.dirty = true;
      if (typeof ObjectsPanel !== 'undefined') ObjectsPanel.refresh();
    } else if (state.tool === 'edit') {
      state.editSel = null; state.tool = 'pan'; refreshToolButtons(); state.dirty = true;
      if (typeof ObjectsPanel !== 'undefined') ObjectsPanel.refresh();
    }
    closeSwatchPop();
    document.querySelectorAll('.modal-back').forEach(m => m.style.display = 'none');
    return;
  }
  // playback shortcuts work everywhere (the playbar is global); polygon-draw
  // finish works wherever the draw tool is armed
  if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
  else if (e.key === 'Enter' && state.tool === 'draw') finishDraft();
  else if (e.key === 'ArrowRight') stepDays(1);
  else if (e.key === 'ArrowLeft') stepDays(-1);
});
