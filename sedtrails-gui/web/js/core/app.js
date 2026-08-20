"use strict";
/* ── Data locations popup (📁 top right): every folder the GUI writes to ── */
async function showAppData() {
  const body = $('appdata-body');
  body.innerHTML = '<div class="dim small">loading…</div>';
  $('modal-appdata').style.display = 'flex';
  let locs;
  try {
    locs = (await apiJson('/api/appinfo')).locations;
  } catch (e) {
    body.textContent = 'failed: ' + e.message;
    return;
  }
  body.innerHTML = '';
  for (const l of locs) {
    const d = document.createElement('div');
    d.className = 'field';
    const lbl = document.createElement('div');
    lbl.className = 'lbl';
    lbl.textContent = l.label;
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:6px;align-items:center';
    const p = document.createElement('code');
    p.style.cssText = 'flex:1;font-size:11px;overflow:hidden;text-overflow:ellipsis;' +
                      'white-space:nowrap' + (l.exists ? '' : ';opacity:.5');
    p.textContent = l.path + (l.exists ? '' : '  (not created yet)');
    p.title = l.path;
    row.append(p);
    if (l.exists) {
      const b = document.createElement('button');
      b.className = 'mini';
      b.textContent = '📂';
      b.title = 'open in file explorer';
      b.onclick = () => api('/api/openfolder', {path: l.path})
        .catch(e => toast('could not open folder: ' + e.message, true));
      row.append(b);
    }
    d.append(lbl, row);
    if (l.note) {
      const n = document.createElement('div');
      n.className = 'dim small';
      n.textContent = l.note;
      d.append(n);
    }
    body.append(d);
  }
  const note = document.createElement('div');
  note.className = 'dim small';
  note.style.marginTop = '8px';
  note.textContent = 'View settings (colors, collapsed sections, last opened files) live in ' +
                     'the browser’s localStorage for this address, not in a file.';
  body.append(note);
}

/* ═════════════════════════ init ═════════════════════════ */
async function init() {
  // entering the Viewer re-checks the config's result file: loads it if new,
  // replaces the loaded run if the file on disk changed since the import.
  // The forcing layers live here too (merged tab) — re-sync them as well.
  Tabs.register('viewer', {enter() {
    if (typeof autoImportResult === 'function') autoImportResult();
    if (typeof ForcingTab !== 'undefined') ForcingTab.enter();
  }});
  $('tb-appdata').onclick = showAppData;
  refreshToolButtons();
  refreshRules();
  refreshModelLayers();
  rebuildPaletteTex();
  if (softwareGL) {
    $('sel-decim').value = '16';
    state.decim = 16;
    toast('Software GPU detected — particle decimation set to 1/16');
  }
  try {
    const pol = await apiJson('/api/polygons');
    state.polygons = pol.polygons || [];
    ensurePolyColors();
    refreshPolyList(); refreshRules();
    LayersPanel.refresh();           // panels were built before the store arrived
  } catch (e) {}
  syncVis();
  // load the last config right away (any tab): it brings the forcing file and
  // an existing result file along — no "Open simulation" popup anymore
  if (typeof SettingsTab !== 'undefined') SettingsTab.init().catch(() => {});
  updateTimelineUI();
  requestAnimationFrame(render);
}
init();
