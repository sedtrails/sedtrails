"use strict";
/* ═════════════════════════ UI: simulations ═════════════════════════ */
function fmtStamp(epochSec) {                    // 'YYYY-MM-DD HH:MM' local time
  const d = new Date(epochSec * 1000);
  const p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
         `${p(d.getHours())}:${p(d.getMinutes())}`;
}

function refreshSimList() {
  const el = $('simlist');
  el.innerHTML = '';
  for (const run of state.runs) {
    const card = document.createElement('div');
    card.className = 'simcard';
    const pct = Math.round(run.loaded / run.T * 100);
    const sig = run.meta.source_signature || [];
    const cached = sig[1] ? `<br>cached copy of the results file of ${fmtStamp(sig[1])}` : '';
    card.innerHTML = `
      <div class="name">
        <input type="checkbox" ${run.visible ? 'checked' : ''} title="show/hide">
        <span class="tintslot"></span>
        <span class="grow" style="overflow:hidden;text-overflow:ellipsis">${run.meta.run_name}</span>
        <button class="mini danger" title="remove">✕</button>
      </div>
      <div class="meta">${fmtInt(run.N)} particles${run.stride > 1 ? ` (1/${run.stride})` : ''} · ${run.T} steps${run.meta.has_burial ? ' · burial' : ''}${cached}</div>
      <div class="loadbar"><div style="width:${pct}%"></div></div>`;
    card.querySelector('input').onchange = ev => {
      run.visible = ev.target.checked; updateCounts(); state.dirty = true;
    };
    card.querySelector('.tintslot').appendChild(
      mkColorBtn(() => run.tint, c => { run.tint = c; state.dirty = true; }, 'simulation color'));
    card.querySelector('button.danger').onclick = () => removeRun(run);
    el.appendChild(card);
  }
  if (typeof LayersPanel !== 'undefined') LayersPanel.refresh();
}

function removeRun(run) {
  state.runs = state.runs.filter(r => r !== run);
  run.frames = null; run.burFrames = null;
  if (state.mesh && state.mesh.runId === run.id) {
    state.mesh = null;
    refreshModelLayers();
    for (const r of state.runs) loadMesh(r);
  }
  updateClockRange(); updateTimelineUI();
  refreshSimList(); refreshRules(); updateCounts();
  state.dirty = true;
}

/* ═════════════════════════ UI: polygons (pills) ═════════════════════════ */
function refreshPolyList() {
  const el = $('polylist');
  el.innerHTML = '';
  const wrap = document.createElement('div');
  wrap.className = 'pillwrap';
  for (const p of state.polygons) {
    const pill = document.createElement('span');
    pill.className = 'pill';
    const isSel = state.editSel && state.editSel.name === p.name;
    if (isSel) pill.classList.add('selected');
    pill.title = isSel ? 'editing — drag vertices on the map; click to stop'
                       : 'click to highlight & edit';
    const dot = document.createElement('span');
    dot.className = 'dot';
    dot.style.background = p.color || '#888';
    dot.title = 'change polygon color';
    dot.onclick = ev => {
      ev.stopPropagation();
      const pop = $('swatchpop');
      pop.style.display = 'block';
      const r = dot.getBoundingClientRect();
      pop.style.left = Math.min(r.left, window.innerWidth - 190) + 'px';
      pop.style.top = Math.min(r.bottom + 4, window.innerHeight - 140) + 'px';
      swatchCb = c => {
        p.color = c; dot.style.background = c;
        savePolygons(); rebuildPaletteTex(); state.dirty = true;
      };
    };
    const name = document.createElement('b');
    name.textContent = p.name;
    const del = document.createElement('button');
    del.textContent = '✕'; del.title = 'delete polygon';
    del.onclick = ev => {
      ev.stopPropagation();
      state.polygons = state.polygons.filter(x => x !== p);
      if (state.editSel && state.editSel.name === p.name) state.editSel = null;
      onPolygonsChanged();
    };
    pill.append(dot, name, del);
    pill.onclick = () => {
      state.editSel = isSel ? null : {name: p.name};
      state.tool = state.editSel ? 'edit' : 'pan';
      refreshPolyList(); state.dirty = true;
    };
    wrap.appendChild(pill);
  }
  el.appendChild(wrap);
  if (!state.polygons.length)
    el.innerHTML = '<div class="dim small" style="margin:4px 0">No polygons yet — import .txt polygons or draw one on the map.</div>';
}

function onPolygonsChanged() {
  ensurePolyColors();
  savePolygons(); refreshPolyList(); refreshRules(); requestClassifyAll(); state.dirty = true;
  if (typeof LayersPanel !== 'undefined') LayersPanel.refresh();
  // the Populations tab lists objects in its seed-from dropdowns — keep current
  if (typeof SeedingTab !== 'undefined') SeedingTab.renderPanel();
}
const savePolygons = debounce(() => {
  api('/api/polygons', {polygons: state.polygons}).catch(e => toast('Saving polygons failed: ' + e.message, true));
}, 800);

/* ═════════════════════════ UI: rules ═════════════════════════ */
function mkSelect(options, value, onchange) {
  const s = document.createElement('select');
  for (const [val, label] of options) {
    const o = document.createElement('option');
    o.value = val; o.textContent = label;
    if (val === value) o.selected = true;
    s.appendChild(o);
  }
  s.onchange = () => onchange(s.value);
  return s;
}
function simOptions() {
  return [['*', state.runs.length > 1 ? 'in all simulations' : 'Particles']]
    .concat(state.runs.map(r => [r.id, 'in ' + r.meta.run_name]));
}
function polyOptions() {
  return allPolys().map(p => [p.key, p.name])
    .concat([['*any*', 'any polygon'], ['*none*', 'no polygon']]);
}

function condWidgets(div, rule, which, refresh) {
  const cKey = which === 1 ? 'cond' : 'cond2';
  const pKey = which === 1 ? 'poly' : 'poly2';
  const tKey = which === 1 ? 'thresh' : 'thresh2';
  div.appendChild(mkSelect(which === 1 ? CONDS : CONDS2, rule[cKey],
    v => { rule[cKey] = v; refresh(); onRulesChanged(); }));
  if (['start', 'end', 'ever'].includes(rule[cKey])) {
    div.appendChild(mkSelect(polyOptions(), rule[pKey],
      v => {
        rule[pKey] = v;
        // convenience: a plain color rule follows the selected polygon's color
        if (which === 1 && rule.action === 'color' && !rule.colorTouched
            && !['*any*', '*none*'].includes(v))
          rule.color = polyColorOf(v);
        refresh(); onRulesChanged();
      }));
  } else if (['bur_ge', 'bur_lt'].includes(rule[cKey])) {
    const inp = document.createElement('input');
    inp.type = 'number'; inp.step = '0.01'; inp.min = '0'; inp.value = rule[tKey];
    inp.oninput = () => { rule[tKey] = parseFloat(inp.value) || 0; onRulesChanged(); };
    div.appendChild(inp);
  }
}

function refreshRules() {
  const el = $('rulelist');
  el.innerHTML = '';
  const word = t => { const s = document.createElement('span'); s.className = 'word'; s.textContent = t; return s; };

  state.rules.forEach((rule, idx) => {
    const div = document.createElement('div');
    div.className = 'rule';
    if (state.runs.length > 1 || rule.sim !== '*') {
      div.appendChild(word('Particles'));
      div.appendChild(mkSelect(simOptions(), rule.sim, v => { rule.sim = v; onRulesChanged(); }));
      div.appendChild(word('that'));
    } else div.appendChild(word('Particles that'));

    condWidgets(div, rule, 1, refreshRules);

    if (rule.cond2) {
      div.appendChild(word('and'));
      condWidgets(div, rule, 2, refreshRules);
      const rm = document.createElement('button');
      rm.className = 'mini'; rm.textContent = '✕and'; rm.title = 'remove second condition';
      rm.onclick = () => { rule.cond2 = null; refreshRules(); onRulesChanged(); };
      div.appendChild(rm);
    } else {
      const add = document.createElement('button');
      add.className = 'mini'; add.textContent = '＋and';
      add.title = 'add a second condition (e.g. start in A AND end in B)';
      add.onclick = () => {
        const polys = allPolys();
        rule.cond2 = 'end';
        rule.poly2 = polys.length ? polys[0].key : '*any*';
        rule.thresh2 = 0.05;
        refreshRules(); onRulesChanged();
      };
      div.appendChild(add);
    }

    div.appendChild(word('are'));
    div.appendChild(mkSelect(ACTIONS, rule.action, v => { rule.action = v; refreshRules(); onRulesChanged(); }));
    if (rule.action === 'color') {
      div.appendChild(mkColorBtn(() => rule.color,
        c => { rule.color = c; rule.colorTouched = true; onRulesChanged(); }));
    } else if (rule.action === 'colorby') {
      div.appendChild(mkSelect(COLORBYS, rule.colorby, v => { rule.colorby = v; onRulesChanged(); }));
    }

    const tools = document.createElement('span');
    tools.className = 'tools';
    const up = document.createElement('button'); up.className = 'mini'; up.textContent = '↑'; up.title = 'evaluate earlier';
    up.disabled = idx === 0;
    up.onclick = () => { state.rules.splice(idx-1, 0, state.rules.splice(idx, 1)[0]); refreshRules(); onRulesChanged(); };
    const del = document.createElement('button'); del.className = 'mini danger'; del.textContent = '✕';
    del.onclick = () => { state.rules.splice(idx, 1); refreshRules(); onRulesChanged(); };
    tools.appendChild(up); tools.appendChild(del);
    div.appendChild(tools);
    el.appendChild(div);
  });

  const div = document.createElement('div');
  div.className = 'rule default';
  div.appendChild(word(state.rules.length ? 'All other particles are' : 'All particles are'));
  div.appendChild(mkSelect(DEFAULT_APPS, state.defaultApp.action, v => {
    state.defaultApp.action = v; state.defaultAutoSet = true; refreshRules(); onRulesChanged();
  }));
  if (state.defaultApp.action === 'color') {
    div.appendChild(mkColorBtn(() => state.defaultApp.color,
      c => { state.defaultApp.color = c; onRulesChanged(); }));
  } else if (state.defaultApp.action === 'colorby') {
    div.appendChild(mkSelect(COLORBYS, state.defaultApp.colorby, v => { state.defaultApp.colorby = v; onRulesChanged(); }));
  }
  el.appendChild(div);

  const keys = new Set(allPolys().map(p => p.key));
  const refsMissing = r => ['start','end','ever'].includes(r.cond) && !['*any*','*none*'].includes(r.poly) && !keys.has(r.poly)
    || (r.cond2 && ['start','end','ever'].includes(r.cond2) && !['*any*','*none*'].includes(r.poly2) && !keys.has(r.poly2));
  if (state.rules.some(refsMissing)) {
    const w = document.createElement('div');
    w.className = 'dim small';
    w.textContent = '⚠ some rules reference deleted polygons and are ignored';
    el.appendChild(w);
  }
  refreshScaleRows();
}

$('btn-add-rule').onclick = () => {
  const polys = allPolys();
  const firstKey = polys.length ? polys[0].key : '*any*';
  state.rules.push({sim: '*', cond: 'start', poly: firstKey,
                    thresh: 0.05, cond2: null, poly2: '*any*', thresh2: 0.05,
                    action: 'color',
                    color: polys.length ? polyColorOf(firstKey) : '#d14b4b',
                    colorby: 'burial'});
  refreshRules(); onRulesChanged();
};

/* ═════════════════════════ UI: modals ═════════════════════════ */
document.querySelectorAll('[data-close]').forEach(b => {
  b.onclick = () => { $(b.dataset.close).style.display = 'none'; };
});

let browseTarget = null;
document.querySelectorAll('[data-browse]').forEach(b => {
  b.onclick = async () => {
    const r = await pickNative(b.dataset.browse, 'nc');
    if (r === null) openBrowser(b.dataset.browse, b.dataset.ext, false);
  };
});

async function pickNative(targetInput, kind) {
  let res;
  try {
    const dir = localStorage.getItem('lastDir') || '';
    res = await apiJson('/api/pickfile?kind=' + kind + '&dir=' + encodeURIComponent(dir));
    if (res.paths === null) throw new Error(res.error || 'picker unavailable');
  } catch (e) {
    return null;
  }
  if (!res.paths.length) return [];
  const p = res.paths[0];
  localStorage.setItem('lastDir', p.replace(/[\\\/][^\\\/]*$/, ''));
  if (targetInput) {
    $(targetInput).value = p;
    if (targetInput === 'inp-nc') autofillName();
  }
  return res.paths;
}

async function openBrowser(targetInput, ext, dirMode, title) {
  browseTarget = {input: targetInput, ext, dirMode};
  $('browse-title').textContent = title || (dirMode ? 'Select folder' : `Select ${ext} file`);
  $('browse-usedir').style.display = dirMode ? '' : 'none';
  $('modal-browse').style.display = 'flex';
  const start = $(targetInput) && $(targetInput).value ?
    $(targetInput).value.replace(/[\\\/][^\\\/]*$/, '') : (localStorage.getItem('lastDir') || '');
  await browseTo(start);
}

async function browseTo(dir) {
  try {
    const res = await apiJson('/api/browse?dir=' + encodeURIComponent(dir || ''));
    $('browse-path').value = res.dir;
    if (res.dir) localStorage.setItem('lastDir', res.dir);
    const list = $('browse-list');
    list.innerHTML = '';
    if (res.parent !== null && res.dir) {
      const up = document.createElement('div');
      up.className = 'fe'; up.innerHTML = '📁 ..';
      up.onclick = () => browseTo(res.parent);
      list.appendChild(up);
    }
    for (const d of res.dirs) {
      const el = document.createElement('div');
      el.className = 'fe'; el.innerHTML = `📁 ${d.name}`;
      el.onclick = () => browseTo(d.path);
      list.appendChild(el);
    }
    for (const f of res.files) {
      if (browseTarget.ext && !f.name.toLowerCase().endsWith(browseTarget.ext)) continue;
      const el = document.createElement('div');
      el.className = 'fe';
      el.innerHTML = `📄 ${f.name}<span class="sz">${(f.size / 1e6).toFixed(1)} MB</span>`;
      el.onclick = () => {
        if (browseTarget.input) $(browseTarget.input).value = f.path;
        if (browseTarget.onPick) browseTarget.onPick(f.path);
        $('modal-browse').style.display = 'none';
        if (browseTarget.input === 'inp-nc') autofillName();
      };
      list.appendChild(el);
    }
    $('browse-hint').textContent = res.dir || 'Select a drive or folder';
  } catch (e) { toast('Browse failed: ' + e.message, true); }
}
$('browse-go').onclick = () => browseTo($('browse-path').value);
$('browse-path').onkeydown = e => { if (e.key === 'Enter') browseTo($('browse-path').value); };
$('browse-usedir').onclick = () => {
  const dir = $('browse-path').value;
  if (browseTarget.onPick) browseTarget.onPick(dir);
  $('modal-browse').style.display = 'none';
};

function autofillName() {
  const p = $('inp-nc').value;
  if (!$('inp-name').value && p) {
    const parts = p.replace(/\\/g, '/').split('/');
    const stem = parts[parts.length - 1].replace(/\.nc$/i, '');
    $('inp-name').value = stem === 'sedtrails_results' ? (parts[parts.length - 2] || stem) : stem;
  }
}

async function openAddSim() {
  $('modal-open').style.display = 'flex';
  $('import-prog').style.display = 'none';
  $('import-msg').textContent = '';
  try {
    const reg = await apiJson('/api/registry');
    const sel = $('sel-recent');
    sel.innerHTML = '<option value="">— pick a recent simulation to reopen —</option>';
    reg.simulations.forEach((s, i) => {
      const o = document.createElement('option');
      o.value = i;
      o.textContent = `${s.name}  ·  ${fmtInt(s.n_particles)} × ${s.n_timesteps}`;
      sel.appendChild(o);
    });
    sel.onchange = () => {
      const s = reg.simulations[sel.value];
      if (!s) return;
      sel.value = '';
      $('inp-nc').value = s.nc_path;
      $('inp-forcing').value = s.forcing_path || '';
      $('inp-name').value = s.name;
      doImport();
    };
  } catch (e) {}
}
/* ＋ Add simulation = pick a results netCDF, import with sensible defaults
   (name from the file, forcing from the open forcing file). The old
   "Open simulation" modal stays in the DOM only as a fallback when the
   native file dialog is unavailable. */
$('btn-add-sim').onclick = async () => {
  const picked = await pickNative(null, 'nc');
  if (picked === null) { openAddSim(); return; }
  if (picked.length) importPath(picked[0]).catch(() => {});
};
$('btn-import').onclick = () => doImport();

function resultDisplayName(ncPath) {
  const parts = ncPath.replace(/\\/g, '/').split('/');
  const stem = parts[parts.length - 1].replace(/\.nc$/i, '');
  return stem === 'sedtrails_results' ? (parts[parts.length - 2] || stem) : stem;
}

/* import a results netCDF without the modal; progress shows under the sim list */
async function importPath(ncPath, name, opts) {
  const dispName = name || resultDisplayName(ncPath);
  const frc = (typeof ForcingTab !== 'undefined' && ForcingTab.state.meta)
    ? ForcingTab.state.meta.path : null;
  const st = $('sim-import-status');
  st.style.display = ''; st.textContent = `importing ${dispName}…`;
  try {
    const {job_id} = await apiJson('/api/import',
      {nc_path: ncPath, forcing_path: frc, name: dispName});
    for (;;) {
      await new Promise(r => setTimeout(r, 500));
      const j = await apiJson('/api/job/' + job_id);
      if (j.status === 'error') throw new Error(j.message);
      if (j.status === 'unknown') throw new Error('server restarted during import — retry');
      st.textContent = `importing ${dispName}… ${Math.round((j.progress || 0) * 100)}%`;
      if (j.status === 'done') { await loadRun(j.bundle, 1, dispName); break; }
    }
    st.style.display = 'none';
  } catch (e) {
    st.style.display = 'none';
    if (!(opts && opts.silent)) toast('Import failed: ' + e.message, true);
    throw e;
  }
}

/* auto-load the loaded config's result file (outputs/directory/
   sedtrails_results.nc) into the viewer — silent no-op when there is none.
   Re-imports when the file on disk changed (rerun / external update). */
let autoImportBusy = false;
async function autoImportResult() {
  if (autoImportBusy) return;
  autoImportBusy = true;
  try {
    const cfgPath = typeof SettingsTab !== 'undefined' ? SettingsTab.getPath() : null;
    if (!cfgPath) return;
    const r = await apiJson('/api/config/result', {config_path: cfgPath});
    if (!r.exists) return;
    const run = await apiJson('/api/run/status');
    if (run.state === 'running') return;           // being written right now
    // same naming as the after-run auto-import (run.js) so reruns dedupe
    const dispName = cfgPath.replace(/\\/g, '/').split('/').pop()
      .replace(/\.ya?ml$/i, '') || 'run';
    const cur = state.runs.find(x => x.meta.run_name === dispName);
    if (cur) {
      const sig = cur.meta.source_signature || [];
      if (sig[0] === r.size && sig[1] === r.mtime) return;   // already up to date
      removeRun(cur);                        // results file changed → reload it
    }
    await importPath(r.path, dispName, {silent: true});
  } catch (e) { /* best-effort */ }
  finally { autoImportBusy = false; }
}

const SERVER_LOST =
  'lost the connection to the local server — the python process probably crashed or was ' +
  'stopped. Check the terminal running sedtrails_viewer.py for the error, restart it, and ' +
  'retry: already-imported data is cached, so the retry continues where it left off.';

async function doImport() {
  const nc = $('inp-nc').value.trim();
  if (!nc) { toast('Choose a results netCDF first', true); return; }
  const stride = parseInt($('inp-stride').value);
  const prog = $('import-prog');
  prog.style.display = ''; prog.firstElementChild.style.width = '2%';
  $('import-msg').textContent = 'starting…';
  try {
    let job_id;
    try {
      ({job_id} = await apiJson('/api/import', {
        nc_path: nc, forcing_path: $('inp-forcing').value.trim() || null,
        name: $('inp-name').value.trim() || null}));
    } catch (e) {
      throw new Error(e.message === 'Failed to fetch' ? SERVER_LOST : e.message);
    }
    let netFails = 0;
    while (true) {
      await new Promise(r => setTimeout(r, 500));
      let j;
      try {
        j = await apiJson('/api/job/' + job_id);
        netFails = 0;
      } catch (e) {
        if (++netFails >= 6) throw new Error(SERVER_LOST);
        continue;
      }
      if (j.status === 'unknown')
        throw new Error('the server was restarted during the import — retry '
          + '(already-imported data is cached and will be reused).');
      prog.firstElementChild.style.width = Math.round((j.progress || 0) * 100) + '%';
      $('import-msg').textContent = j.message || '';
      if (j.status === 'done') {
        $('inp-nc').value = ''; $('inp-forcing').value = '';
        $('inp-name').value = '';
        await loadRun(j.bundle, stride);
        break;
      }
      if (j.status === 'error') throw new Error(j.message);
    }
  } catch (e) {
    $('import-msg').textContent = '✗ ' + e.message;
    toast('Import failed: ' + e.message, true);
  }
}

/* polygon import + draw dialogs */
async function importPolygonPaths(paths) {
  let total = 0;
  for (const path of paths) {
    try {
      const res = await apiJson('/api/polygons/import', {path});
      for (const p of res.polygons) {
        p.type = p.type || 'polygon';
        if (!state.polygons.some(x => x.name === p.name)) state.polygons.push(p);
        total++;
      }
    } catch (e) { toast('Polygon import failed: ' + e.message, true); return; }
  }
  if (!total) { toast('No polygons found there', true); return; }
  toast(`Imported ${total} polygon(s)`);
  onPolygonsChanged();
}

$('btn-poly-import').onclick = async () => {
  const picked = await pickNative(null, 'txt');
  if (picked !== null) {
    if (picked.length) importPolygonPaths(picked);
    return;
  }
  browseTarget = {ext: '.txt', dirMode: true, onPick: path => importPolygonPaths([path])};
  $('browse-title').textContent = 'Select polygon folder or .txt file';
  $('browse-usedir').style.display = '';
  $('modal-browse').style.display = 'flex';
  browseTo(localStorage.getItem('lastDir') || '');
};

/* one draw flow for every object type — armed from the Objects panel and
   usable on any map tab (clicks are handled in interact.js) */
const DRAW_HINTS = {
  polygon: 'Click to add vertices · Enter/double-click to finish · Esc to cancel',
  point: 'Click one or more points · Enter/double-click to finish · Esc to cancel',
  transect: 'Click start and end of the transect · Esc to cancel',
  bbox: 'Click two opposite corners of the box · Esc to cancel',
};
const OBJ_LABEL = {polygon: 'polygon', point: 'point set', transect: 'transect', bbox: 'box'};
const EDIT_HINT = 'Drag a point to move · click an edge to insert · right-click a point to ' +
  'delete · Alt-click adds (point sets)';

function startDrawObject(mode) {
  if (!state.world0) { toast('Open a forcing file or simulation first', true); return; }
  if (state.tool === 'draw' && state.drawMode === mode) {   // toggle the tool off
    state.draft = []; state.tool = 'pan';
  } else {
    state.tool = 'draw'; state.drawMode = mode;
    state.draft = []; state.editSel = null;
  }
  refreshToolButtons(); state.dirty = true;
  if (typeof ObjectsPanel !== 'undefined') ObjectsPanel.refresh();
}
$('btn-poly-draw').onclick = () => startDrawObject('polygon');   // legacy entry point

function refreshToolButtons() {
  $('btn-poly-draw').classList.toggle('active', state.tool === 'draw');
  canvas.classList.toggle('drawing', state.tool === 'draw');
  if (state.tool !== 'edit') canvas.classList.remove('vtx');
  const editing = state.tool === 'edit' && state.editSel;
  $('drawhint-text').textContent = editing ? EDIT_HINT : DRAW_HINTS[state.drawMode || 'polygon'];
  $('btn-edit-done').style.display = editing ? '' : 'none';
  $('drawhint').style.display = state.tool === 'draw' || editing ? 'flex' : 'none';
}

/* the green ✓ Done button in the hint bar closes edit mode (same as Esc) */
$('btn-edit-done').onclick = () => {
  state.editSel = null;
  state.tool = 'pan';
  refreshToolButtons();
  state.dirty = true;
  if (typeof ObjectsPanel !== 'undefined') ObjectsPanel.refresh();
};

function finishDraft() {
  const mode = state.drawMode || 'polygon';
  const need = {polygon: 3, point: 1, transect: 2, bbox: 2}[mode];
  if (state.draft.length < need) {
    state.draft = []; state.tool = 'pan'; refreshToolButtons(); state.dirty = true;
    if (typeof ObjectsPanel !== 'undefined') ObjectsPanel.refresh();
    return;
  }
  $('modal-poly-title').textContent = 'New ' + OBJ_LABEL[mode];
  $('poly-name').value = '';
  $('modal-poly').style.display = 'flex';
  $('poly-name').focus();
}
$('poly-save').onclick = () => {
  const mode = state.drawMode || 'polygon';
  let name = $('poly-name').value.trim() || OBJ_LABEL[mode] + ' ' + (Date.now() % 10000);
  while (state.polygons.some(p => p.name === name)) name += '*';
  let verts = state.draft.map(v => [Math.round(v[0] * 100) / 100, Math.round(v[1] * 100) / 100]);
  if (mode === 'bbox') {                       // store the full axis-aligned outline
    const xs = verts.map(v => v[0]), ys = verts.map(v => v[1]);
    verts = [[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.min(...ys)],
             [Math.max(...xs), Math.max(...ys)], [Math.min(...xs), Math.max(...ys)]];
  }
  state.polygons.push({name, type: mode, verts});
  state.draft = [];
  state.tool = 'pan';
  $('modal-poly').style.display = 'none';
  refreshToolButtons();
  onPolygonsChanged();
};

