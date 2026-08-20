"use strict";
/* ═════════════════════════ export (PNG / MP4) ═════════════════════════ */
function expSize() {
  const v = $('exp-res').value;
  if (v === 'current') return [canvas.width, canvas.height];
  const [w, h] = v.split('x').map(Number);
  return [w, h];
}

async function tilesSettled(maxMs) {
  const t0 = performance.now();
  while (performance.now() - t0 < maxMs) {
    drawScene();
    if (tileFetches === 0) { drawScene(); return; }
    await new Promise(r => setTimeout(r, 250));
  }
  drawScene();
}

function canvasBlob() {
  return new Promise(res => canvas.toBlob(res, 'image/png'));
}

async function withExportCanvas(w, h, fn, vp) {
  const save = {w: canvas.width, h: canvas.height, ppm: state.view.ppm,
                cx: state.view.cx, cy: state.view.cy};
  state.exporting = true;
  try {
    canvas.width = w; canvas.height = h;
    if (vp && state.world0) {
      // saved viewport: exact reproducible framing (absolute world coords)
      state.view.cx = vp.cx - state.world0[0];
      state.view.cy = vp.cy - state.world0[1];
      state.view.ppm = w / vp.worldW;
    } else {
      state.view.ppm = save.ppm * (w / save.w);
    }
    await fn();
  } finally {
    canvas.width = save.w; canvas.height = save.h;
    state.view.ppm = save.ppm;
    state.view.cx = save.cx;
    state.view.cy = save.cy;
    state.exporting = false;
    state.dirty = true;
    drawScene();
  }
}

/* ── saved export viewports: named, fixed framings for reproducible exports ── */
const VP_KEY = 'stgui_viewports';
function loadViewports() {
  try { return JSON.parse(localStorage.getItem(VP_KEY)) || []; } catch (e) { return []; }
}
function saveViewports(v) { localStorage.setItem(VP_KEY, JSON.stringify(v)); }
function selectedViewport() {
  return loadViewports().find(v => v.name === $('exp-vp').value) || null;
}
function refreshVpSelect(selName) {
  const sel = $('exp-vp');
  sel.innerHTML = '<option value="">current map view</option>';
  for (const vp of loadViewports()) {
    const o = document.createElement('option');
    o.value = vp.name;
    o.textContent = '⌗ ' + vp.name;
    sel.appendChild(o);
  }
  if (selName) sel.value = selName;
  if (sel.value !== (selName || '')) sel.value = '';
  $('exp-vp-del').style.display = sel.value ? '' : 'none';
}
$('exp-vp').onchange = () => { $('exp-vp-del').style.display = $('exp-vp').value ? '' : 'none'; };
$('exp-vp-del').onclick = () => {
  const n = $('exp-vp').value;
  if (!n || !confirm(`Delete viewport "${n}"?`)) return;
  saveViewports(loadViewports().filter(v => v.name !== n));
  refreshVpSelect();
};

/* define a new viewport: hide the modal, show an aspect-locked frame over the
   map (aspect = the chosen export resolution); pan/zoom underneath, confirm */
$('exp-vp-new').onclick = () => {
  if (!state.world0) { toast('Load a simulation or forcing file first', true); return; }
  const [w, h] = expSize();
  $('modal-export').style.display = 'none';
  // hide every panel while framing — only the map and the frame remain
  const hidden = [];
  for (const id of ['sidebar', 'sideresize', 'sidebtn', 'floatpanels',
                    'colorbars', 'coords', 'drawhint']) {
    const e = $(id);
    if (!e) continue;
    hidden.push([e, e.style.display]);
    e.style.display = 'none';
  }
  const ov = $('vpselect');
  const rect = $('vpselect-rect');
  ov.style.display = 'block';
  const layout = () => {
    const aspect = h / w;
    const availW = window.innerWidth * 0.72;
    const availH = (window.innerHeight - 40 - 58) * 0.78;
    let rw = availW, rh = rw * aspect;
    if (rh > availH) { rh = availH; rw = rh / aspect; }
    rect.style.width = rw + 'px';
    rect.style.height = rh + 'px';
    rect.style.left = (window.innerWidth - rw) / 2 + 'px';
    rect.style.top = 40 + ((window.innerHeight - 40 - 58) - rh) / 2 + 'px';
  };
  layout();
  window.addEventListener('resize', layout);
  const close = () => {
    ov.style.display = 'none';
    window.removeEventListener('resize', layout);
    for (const [e, d] of hidden) e.style.display = d;
    $('modal-export').style.display = 'flex';
  };
  $('vpselect-cancel').onclick = close;
  $('vpselect-ok').onclick = () => {
    const r = rect.getBoundingClientRect();
    const c = screenToRd(r.left + r.width / 2, r.top + r.height / 2);
    const a = screenToRd(r.left, r.top + r.height / 2);
    const b = screenToRd(r.right, r.top + r.height / 2);
    const name = (prompt('Name this viewport:',
                         'viewport ' + (loadViewports().length + 1)) || '').trim();
    if (!name) return;                       // keep framing mode open
    const vps = loadViewports().filter(v => v.name !== name);
    vps.push({name, cx: c[0], cy: c[1], worldW: Math.abs(b[0] - a[0])});
    saveViewports(vps);
    close();
    refreshVpSelect(name);
  };
};

async function doExport() {
  if (!state.runs.length) { toast('Load a simulation first', true); return; }
  const fmt = document.querySelector('input[name=expfmt]:checked').value;
  const [w, h] = expSize();
  const vp = selectedViewport();
  // remember the export settings for next time (incl. the chosen viewport)
  localStorage.setItem('stgui_export_last', JSON.stringify({
    fmt, res: $('exp-res').value, fps: $('exp-fps').value,
    dpf: $('exp-dpf').value, vp: $('exp-vp').value}));
  const msg = $('exp-msg');
  const prog = $('exp-prog');
  const name = state.runs[0].meta.run_name.replace(/\W+/g, '_');
  try {
    if (fmt === 'png') {
      const {path} = await apiJson('/api/picksave?ext=.png&name=' + encodeURIComponent(name + '.png'));
      if (!path) return;
      msg.textContent = 'rendering…';
      await withExportCanvas(w, h, async () => {
        await tilesSettled(6000);
        const blob = await canvasBlob();
        msg.textContent = 'saving…';
        const r = await fetch('/api/savefile?path=' + encodeURIComponent(path),
                              {method: 'POST', body: blob});
        if (!r.ok) throw new Error((await r.json()).error || 'save failed');
      }, vp);
      msg.textContent = '✓ saved to ' + path;
      toast('PNG saved: ' + path);
    } else {
      const fps = Math.max(1, parseInt($('exp-fps').value) || 12);
      const dpf = Math.max(0.25, parseFloat($('exp-dpf').value) || 1);
      const d0 = Date.parse($('exp-t0').value + 'T00:00:00Z') / 86400000;
      const d1 = Date.parse($('exp-t1').value + 'T00:00:00Z') / 86400000;
      if (!isFinite(d0) || !isFinite(d1) || d1 <= d0) throw new Error('invalid period');
      const nFrames = Math.max(2, Math.floor((d1 - d0) / dpf) + 1);
      const {path} = await apiJson('/api/picksave?ext=.mp4&name=' + encodeURIComponent(name + '.mp4'));
      if (!path) return;
      const {job} = await apiJson('/api/video/start', {path, fps});
      prog.style.display = '';
      const clockSave = state.clock.t;
      const playingSave = state.clock.playing;
      state.clock.playing = false;
      await withExportCanvas(w, h, async () => {
        await tilesSettled(6000);
        for (let i = 0; i < nFrames; i++) {
          state.clock.t = Math.min(d0 + i * dpf, state.clock.t1);
          if (typeof ForcingTab !== 'undefined')      // forcing layer must not be stale
            await ForcingTab.setTime(state.clock.t);
          drawScene();
          const blob = await canvasBlob();
          const r = await fetch(`/api/video/frame?job=${job}&i=${i}`,
                                {method: 'POST', body: blob});
          if (!r.ok) throw new Error('frame upload failed');
          prog.firstElementChild.style.width = Math.round((i + 1) / nFrames * 90) + '%';
          msg.textContent = `frame ${i + 1}/${nFrames}`;
        }
      }, vp);
      state.clock.t = clockSave;
      state.clock.playing = playingSave;
      msg.textContent = 'encoding mp4…';
      await apiJson('/api/video/finish', {job});
      prog.firstElementChild.style.width = '100%';
      msg.textContent = '✓ saved to ' + path;
      toast('MP4 saved: ' + path);
    }
  } catch (e) {
    msg.textContent = '✗ ' + e.message;
    toast('Export failed: ' + e.message, true);
  }
}

function openExport() {
  if (!state.runs.length) { toast('Load a simulation first', true); return; }
  $('modal-export').style.display = 'flex';
  $('exp-prog').style.display = 'none';
  $('exp-msg').textContent = '';
  $('exp-t0').value = fmtDate(state.clock.t0);
  $('exp-t1').value = fmtDate(state.clock.t1);
  // restore the last-used export settings (resolution, format, fps, viewport)
  let last = {};
  try { last = JSON.parse(localStorage.getItem('stgui_export_last')) || {}; } catch (e) {}
  if (last.res) $('exp-res').value = last.res;
  if (last.fmt) {
    const r = document.querySelector(`input[name=expfmt][value="${last.fmt}"]`);
    if (r) r.checked = true;
    $('exp-mp4opts').style.display = last.fmt === 'mp4' ? '' : 'none';
  }
  if (last.fps) $('exp-fps').value = last.fps;
  if (last.dpf) $('exp-dpf').value = last.dpf;
  refreshVpSelect(last.vp || '');
  refreshExpDuration();
}
function refreshExpDuration() {
  const fps = Math.max(1, parseInt($('exp-fps').value) || 12);
  const dpf = Math.max(0.25, parseFloat($('exp-dpf').value) || 1);
  const d0 = Date.parse($('exp-t0').value + 'T00:00:00Z') / 86400000;
  const d1 = Date.parse($('exp-t1').value + 'T00:00:00Z') / 86400000;
  const n = isFinite(d0) && isFinite(d1) && d1 > d0 ? Math.floor((d1 - d0) / dpf) + 1 : 0;
  $('exp-duration').textContent = n ? `${n} frames → ${(n / fps).toFixed(1)} s of video` : '';
}
document.querySelectorAll('input[name=expfmt]').forEach(r => r.onchange = () => {
  $('exp-mp4opts').style.display =
    document.querySelector('input[name=expfmt]:checked').value === 'mp4' ? '' : 'none';
});
for (const id of ['exp-fps', 'exp-dpf', 'exp-t0', 'exp-t1'])
  $(id).oninput = refreshExpDuration;
$('btn-export').onclick = openExport;
$('btn-do-export').onclick = doExport;

