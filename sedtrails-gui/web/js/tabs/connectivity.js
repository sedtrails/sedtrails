"use strict";
/* ═════════════════════════ Connectivity tab (v0) ═════════════════════════
   Polygon-to-polygon connectivity matrix over an imported simulation: which
   particles start in polygon i and end in (or ever visit) polygon j. Uses the
   shared polygon library (draw/import polygons in the Viewer tab) and the
   cached per-polygon classification. Future metrics (time-lagged, flux-
   weighted, network measures) become new `mode` values on the same endpoint. */

const ConnectivityTab = (() => {
  let lastResult = null;

  async function refreshSims() {
    const sel = $('con-sim');
    const cur = sel.value;
    sel.innerHTML = '';
    try {
      const reg = await apiJson('/api/registry');
      for (const s of reg.simulations) {
        const o = document.createElement('option');
        o.value = s.id;
        o.textContent = `${s.name} · ${fmtInt(s.n_particles)} × ${s.n_timesteps}`;
        sel.appendChild(o);
      }
      // prefer the run loaded in the Particle viewer: a remembered selection
      // from an older session silently computes against the wrong simulation
      const loaded = (state.runs || []).map(r => r.id);
      const has = v => [...sel.options].some(o => o.value === v);
      if (loaded.includes(cur) && has(cur)) sel.value = cur;
      else if (loaded.length && has(loaded[loaded.length - 1])) sel.value = loaded[loaded.length - 1];
      else if (has(cur)) sel.value = cur;
    } catch (e) {}
    refreshInfo();
  }

  function refreshInfo() {
    const n = allPolys().length;      // closed shapes only (polygons & boxes)
    $('con-info').textContent = n >= 2 ?
      `${n} polygon(s) from the Objects panel` :
      'Draw or import at least 2 polygons (Objects panel, top right) to compute connectivity.';
    $('con-compute').disabled = n < 2 || !$('con-sim').value;
  }

  /* auto-generated geomorphic cells (Pearson et al. 2021): k-means on the
     forcing bathymetry → auto_cell_* polygons in the shared Objects panel */
  const AUTO_RE = /^(auto_)?cell_\d+$/;      // also matches pre-rename cell_NN

  async function genCells() {
    const meta = (typeof ForcingTab !== 'undefined') && ForcingTab.state.meta;
    if (!meta) return toast('open a forcing file first (Forcing tab)', true);
    const scalars = (meta.variables || []).filter(v => !v.is_vector_comp);
    const bed = scalars.find(v => /bed/i.test(v.name)) || scalars[0];
    const n = Math.max(2, Math.min(200, parseInt($('con-ncells').value, 10) || 12));
    $('con-gencells').disabled = true;
    $('con-status').textContent = 'clustering bathymetry…';
    try {
      const r = await apiJson('/api/polygons/auto',
                              {forcing_id: meta.id, n, var: bed ? bed.name : null});
      state.polygons = state.polygons.filter(p => !AUTO_RE.test(p.name));
      for (const p of r.polygons)
        state.polygons.push({name: p.name, type: 'polygon', verts: p.verts});
      onPolygonsChanged();
      $('con-status').textContent =
        `${r.polygons.length} cells generated` +
        (bed ? ` (k-means on ${bed.name})` : '') + ' — see the Objects panel';
    } catch (e) {
      toast('auto cells: ' + e.message, true);
      $('con-status').textContent = '';
    }
    $('con-gencells').disabled = false;
    refreshInfo();
  }

  /* drop every auto-generated cell in one click (also wired from the Objects
     panel toolbar via ConnectivityTab.removeAutoCells) */
  function removeAutoCells() {
    const n = state.polygons.filter(p => AUTO_RE.test(p.name)).length;
    if (!n) return toast('no auto-generated cells to remove', true);
    if (!confirm(`Remove all ${n} auto-generated cell(s)?`)) return;
    state.polygons = state.polygons.filter(p => !AUTO_RE.test(p.name));
    onPolygonsChanged();
    if (typeof ObjectsPanel !== 'undefined') ObjectsPanel.refresh();
    $('con-status').textContent = `${n} auto-generated cell(s) removed`;
    refreshInfo();
  }

  async function compute() {
    const bundle = $('con-sim').value;
    if (!bundle) return toast('import a simulation first (Viewer tab)', true);
    const polys = allPolys().map(p => ({key: p.key, verts: p.verts}));
    $('con-compute').disabled = true;
    $('con-status').textContent = 'computing… (first time classifies all particles)';
    try {
      lastResult = await apiJson('/api/connectivity/compute', {
        bundle_id: bundle, polygons: polys,
        mode: $('con-mode').value, normalize: $('con-norm').value,
      });
      state.connectivity = lastResult;   // the map layer (render.js) draws this
      state.dirty = true;
      renderMatrix(lastResult);
      const simLbl = $('con-sim').selectedOptions.length ?
        $('con-sim').selectedOptions[0].textContent : bundle;
      let status = `${simLbl} — ${fmtInt(lastResult.n_particles)} particles · ` +
        `${fmtInt(lastResult.n_unassigned)} start outside all polygons`;
      if (lastResult.n_unassigned === lastResult.n_particles) {
        status += ' ⚠ no particle starts inside any polygon — check that the ' +
          'simulation selected above matches the polygons\' area';
      }
      $('con-status').textContent = status;
    } catch (e) {
      toast('connectivity: ' + e.message, true);
      $('con-status').textContent = '';
    }
    $('con-compute').disabled = false;
  }

  function cellColor(v, vmax) {
    if (!vmax || v <= 0) return 'transparent';
    const f = Math.min(1, v / vmax);
    return `color-mix(in srgb, var(--accent) ${Math.round(f * 82)}%, var(--panel))`;
  }

  function renderMatrix(r) {
    const frac = r.normalize === 'fraction';
    const vmax = frac ? 1 : Math.max(1, ...r.matrix.flat());
    const fmt = v => frac ? (v * 100).toFixed(1) + '%' : fmtInt(v);
    let html = '<table class="conmat"><tr><th class="corner">from \\ to</th>';
    for (const l of r.labels) html += `<th title="${l}">${l}</th>`;
    html += '<th class="dim">n start</th></tr>';
    r.matrix.forEach((row, i) => {
      html += `<tr><th title="${r.labels[i]}">` +
        `<span class="swatch" style="background:${polyColorOf(r.labels[i])}"></span> ${r.labels[i]}</th>`;
      row.forEach((v, j) => {
        const fg = frac ? v > vmax * 0.55 : v > vmax * 0.55;
        html += `<td style="background:${cellColor(v, vmax)}${fg ? ';color:#fff' : ''}">${fmt(v)}</td>`;
      });
      html += `<td class="dim">${fmtInt(r.row_counts[i])}</td></tr>`;
    });
    html += '</table>';
    $('con-result').innerHTML = html;
    $('con-resultsec').open = true;      // sections collapse by default now
    $('con-csv').style.display = '';
    if (typeof LayersPanel !== 'undefined') LayersPanel.refresh();
  }

  function exportCsv() {
    if (!lastResult) return;
    const r = lastResult;
    const rows = [['from\\to', ...r.labels, 'n_start']];
    r.matrix.forEach((row, i) => rows.push([r.labels[i], ...row, r.row_counts[i]]));
    const csv = rows.map(x => x.join(',')).join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv'}));
    a.download = `connectivity_${r.mode}_${r.normalize}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  let wired = false;
  function enter() {
    if (!wired) {
      $('con-compute').onclick = compute;
      $('con-csv').onclick = exportCsv;
      $('con-sim').onchange = refreshInfo;
      $('con-gencells').onclick = genCells;
      $('con-delcells').onclick = removeAutoCells;
      wired = true;
    }
    if (typeof ForcingTab !== 'undefined') ForcingTab.enter();  // map background
    refreshSims();
  }

  Tabs.register('connectivity', {enter});
  return {refreshSims, removeAutoCells, AUTO_RE};
})();
