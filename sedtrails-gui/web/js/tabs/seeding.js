"use strict";
/* ═════════════════════════ Populations tab ═════════════════════════
   One card per population covering the whole population schema. Everything
   that determines the release POSITIONS (source object, distribution,
   coordinates, seed, counts) is edited ONLY in the "Modify seeding…" modal —
   the card shows a read-only summary of it; all other settings stay editable
   in the card. Settings are bound to the same config object as the Settings
   tab; previews come from the backend, which runs sedtrails' own seeding
   strategies (exact WYSIWYG). The old draw-tool/confirm-card machinery below
   is kept but unused. */

const SeedingTab = (() => {
  const S = {
    tool: null,            // {mode: 'point'|'transect'|'bbox'|'poly', pop: index}
    draft: [],             // clicked world coords for the armed tool
    pending: null,         // finished-but-unconfirmed shape {mode, pts, pop}
    previews: {},          // popIndex -> {buf, n, visible:'auto'|true|false, key, label}
    downAt: null,
    applied: {},           // popName -> {obj, verts} — the source object last
                           // applied to that population (drives the ⟳ Update button)
  };

  const popSchema = () => SchemaForm.resolve({$ref: SchemaForm.URN('population')}, null)[0];
  const cfg = () => SettingsTab.getConfig();
  const pops = () => {
    const c = cfg();
    return (c && c.particles && c.particles.populations) || [];
  };
  const cfgBase = () => {
    const p = SettingsTab.getPath();
    return p ? p.replace(/[\\/][^\\/]+$/, '') : null;
  };

  /* preview visibility: ONE group toggle (Layers panel), same on every map
     tab — the old per-population tri-state ('auto' = seeding tab only) is
     gone; previews are hollow rings so they never read as simulated particles */
  let showPreviews = true;
  function previewShown(pv) { return !!pv && showPreviews; }
  function setShowPreviews(v) { showPreviews = !!v; state.dirty = true; }
  function setPreviewVisible(i, v) {               // legacy per-pop hook (unused)
    const pv = S.previews[i] = S.previews[i] || {};
    pv.visible = v;
    state.dirty = true;
  }

  /* ── strategy mutation helpers (used on Apply only) ─────────────────────── */
  function strategyOf(pop) {
    const seeding = pop.seeding = pop.seeding || {};
    return seeding.strategy = seeding.strategy || {};
  }
  function ensureBranch(pop, name, fresh) {
    const st = strategyOf(pop);
    if (!st[name]) {
      for (const k of Object.keys(st)) delete st[k];
      st[name] = fresh;
    }
    return st[name];
  }
  const rd = v => Math.round(v * 10) / 10;

  /* ── draw tools: clicks accumulate a draft; finishing → pending confirm ─── */
  function applyClick(mode, pt) {
    S.draft.push(pt);
    if (mode === 'transect' && S.draft.length === 2) finishShape();
    else if (mode === 'bbox' && S.draft.length === 2) finishShape();
    state.dirty = true;
  }

  function finishShape() {
    if (!S.tool || !S.draft.length) return cancelTool();
    if (S.tool.mode === 'poly' && S.draft.length < 3) return cancelTool();
    S.pending = {mode: S.tool.mode, pts: S.draft.slice(), pop: S.tool.pop};
    S.tool = null;
    S.draft = [];
    canvas.classList.remove('drawing');
    $('drawhint').style.display = 'none';
    state.dirty = true;
    renderPanel();
  }

  function armTool(mode, popIndex) {
    if (S.tool && S.tool.mode === mode && S.tool.pop === popIndex) {
      // clicking the active tool again finishes an open-ended shape
      if (mode === 'point' || mode === 'poly') return finishShape();
      return cancelTool();
    }
    S.pending = null;
    S.tool = {mode, pop: popIndex};
    S.draft = [];
    canvas.classList.add('drawing');
    const hints = {
      point: 'Click release points · Enter or the tool button to finish · Esc to cancel',
      transect: 'Click start and end of the segment · Esc to cancel',
      bbox: 'Click two opposite corners of the box · Esc to cancel',
      poly: 'Click to add vertices · Enter/double-click to finish · Esc to cancel',
    };
    $('drawhint-text').textContent = hints[mode];
    $('btn-edit-done').style.display = 'none';
    $('drawhint').style.display = 'flex';
    renderPanel();
  }

  function cancelTool() {
    S.tool = null;
    S.draft = [];
    canvas.classList.remove('drawing');
    $('drawhint').style.display = 'none';
    state.dirty = true;
    renderPanel();
  }

  function afterEdit() {
    SettingsTab.touch();
    state.dirty = true;
    renderPanel();
    refreshPreviews();
  }

  /* ── seed from an object in the central store (Objects panel) ───────────
     Turns the picked object into a pending shape; applyFromObject() consumes
     it right away (the confirm-card flow is gone). */
  function usePendingFromObject(o, popIndex) {
    const type = o.type || 'polygon';
    let pts = o.verts.map(v => v.slice());
    if (type === 'bbox') {                       // stored as 4 corners → 2 corners
      const xs = pts.map(v => v[0]), ys = pts.map(v => v[1]);
      pts = [[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]];
    }
    const mode = {polygon: 'poly', bbox: 'bbox', transect: 'transect', point: 'point'}[type];
    S.pending = {mode, pts, pop: popIndex, from: o.name};
    state.dirty = true;
    renderPanel();
  }

  /* ── apply the pending shape to the chosen population. REPLACE semantics:
     "seed from object X" always mirrors X exactly (so ⟳ Update re-applies
     changed coordinates without duplicating locations/segments) ────────────── */
  function applyPending(popIndex, asWhat) {
    const p = S.pending;
    const pop = pops()[popIndex];
    if (!p || !pop) return;
    if (p.mode === 'point') {
      const br = ensureBranch(pop, 'point', {locations: []});
      br.locations = p.pts.map(pt => `${rd(pt[0])},${rd(pt[1])}`);
    } else if (p.mode === 'transect') {
      const br = ensureBranch(pop, 'transect', {segments: [], k: 100});
      br.segments = [
        `${rd(p.pts[0][0])},${rd(p.pts[0][1])} ${rd(p.pts[1][0])},${rd(p.pts[1][1])}`];
    } else {                                      // bbox | poly → random / grid area
      const name = asWhat === 'grid' ? 'grid' : 'random';
      const st = strategyOf(pop);
      if (st[name] === undefined) {               // switching branch: start fresh
        for (const k of Object.keys(st)) delete st[k];
        st[name] = name === 'grid' ? {separation: {dx: 100, dy: 100}}
                                   : {nlocations: 100, seed: 42};
      }
      const br = st[name];
      if (p.mode === 'bbox') {
        const [a, b] = p.pts;
        br.bbox = `${rd(Math.min(a[0], b[0]))},${rd(Math.min(a[1], b[1]))} ` +
                  `${rd(Math.max(a[0], b[0]))},${rd(Math.max(a[1], b[1]))}`;
        delete br.poly;
      } else {
        br.poly = p.pts.map(q => `${rd(q[0])},${rd(q[1])}`);
        delete br.bbox;
      }
    }
    // remember which object seeded this population: if that object is edited
    // later, the population card offers a one-click ⟳ Update
    if (p.from) {
      const src = state.polygons.find(o => o.name === p.from);
      if (src) S.applied[pop.name || String(popIndex)] =
        {obj: p.from, verts: JSON.stringify(src.verts)};
    }
    S.pending = null;
    afterEdit();
  }

  /* apply an object's coordinates to a population IMMEDIATELY (no confirm
     card): the strategy is derived from the object type — point set → point,
     transect → transect, polygon/box → area. asWhat picks the area
     distribution ('random'|'grid'); default keeps the population's current. */
  function applyFromObject(o, popIndex, asWhat) {
    const pop = pops()[popIndex];
    if (!pop) return;
    if (!asWhat) {
      const st = pop.seeding && pop.seeding.strategy;
      asWhat = st && st.grid ? 'grid' : 'random';
    }
    usePendingFromObject(o, popIndex);
    applyPending(popIndex, asWhat);
  }

  /* switch random ↔ grid keeping the current area when the source object is gone */
  function switchAreaBranch(pop, name) {
    const st = strategyOf(pop);
    const old = st.random || st.grid || {};
    const bbox = old.bbox, poly = old.poly;
    for (const k of Object.keys(st)) delete st[k];
    st[name] = name === 'grid' ? {separation: {dx: 100, dy: 100}}
                               : {nlocations: 100, seed: 42};
    if (bbox !== undefined) st[name].bbox = bbox;
    if (poly !== undefined) st[name].poly = poly;
    afterEdit();
  }

  function buildConfirmCard() {
    const p = S.pending;
    const card = document.createElement('div');
    card.className = 'rule default';
    const summary = {
      point: `${p.pts.length} release point${p.pts.length > 1 ? 's' : ''}`,
      transect: 'transect segment',
      bbox: 'bounding box',
      poly: `polygon (${p.pts.length} vertices)`,
    }[p.mode];
    const head = document.createElement('div');
    head.style.cssText = 'width:100%;font-weight:600';
    head.textContent = p.from ? `"${p.from}" as ${summary}` : `New ${summary}`;
    card.appendChild(head);

    const row = document.createElement('div');
    row.className = 'row small';
    row.style.width = '100%';
    row.appendChild(document.createTextNode('apply to'));
    const selPop = document.createElement('select');
    pops().forEach((pop, i) => {
      const op = document.createElement('option');
      op.value = i;
      op.textContent = pop.name || 'population ' + (i + 1);
      selPop.appendChild(op);
    });
    selPop.value = Math.min(p.pop, pops().length - 1);
    row.appendChild(selPop);

    let selAs = null;
    if (p.mode === 'bbox' || p.mode === 'poly') {
      row.appendChild(document.createTextNode('as'));
      selAs = document.createElement('select');
      for (const [v, txt] of [['random', 'random area'], ['grid', 'regular grid area']]) {
        const op = document.createElement('option');
        op.value = v; op.textContent = txt;
        selAs.appendChild(op);
      }
      const st = (pops()[p.pop] || {}).seeding && pops()[p.pop].seeding.strategy;
      selAs.value = st && st.grid ? 'grid' : 'random';
      row.appendChild(selAs);
    } else {
      const what = document.createElement('span');
      what.className = 'dim';
      what.textContent = p.mode === 'point' ? 'as point release locations (added)'
                                            : 'as a transect segment (added)';
      row.appendChild(what);
    }
    card.appendChild(row);

    const note = document.createElement('div');
    note.className = 'hint';
    note.style.width = '100%';
    note.textContent = (p.mode === 'bbox' || p.mode === 'poly')
      ? 'Replaces the release area of the chosen population. Nothing changes until Apply.'
      : 'Added to the chosen population. Nothing changes until Apply.';
    card.appendChild(note);

    const btns = document.createElement('div');
    btns.className = 'row';
    btns.style.width = '100%';
    const ok = document.createElement('button');
    ok.className = 'primary';
    ok.textContent = 'Apply';
    ok.onclick = () => applyPending(parseInt(selPop.value, 10), selAs ? selAs.value : null);
    const no = document.createElement('button');
    no.textContent = 'Cancel';
    no.onclick = () => { S.pending = null; state.dirty = true; renderPanel(); };
    btns.appendChild(ok);
    btns.appendChild(no);
    card.appendChild(btns);
    return card;
  }

  /* ── canvas events (active only on this tab while a tool is armed) ──────── */
  canvas.addEventListener('mousedown', e => { S.downAt = [e.clientX, e.clientY]; });
  canvas.addEventListener('click', e => {
    if (Tabs.current !== 'seeding' || !S.tool || !state.world0) return;
    if (S.downAt && Math.hypot(e.clientX - S.downAt[0], e.clientY - S.downAt[1]) > 4) return;
    applyClick(S.tool.mode, screenToRd(e.clientX, e.clientY));
  });
  canvas.addEventListener('dblclick', e => {
    if (Tabs.current === 'seeding' && S.tool &&
        (S.tool.mode === 'poly' || S.tool.mode === 'point')) {
      e.preventDefault();
      S.draft.pop();                       // drop the extra click from the dblclick
      finishShape();
    }
  });
  window.addEventListener('keydown', e => {
    if (Tabs.current !== 'seeding') return;
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (S.tool) {
      if (e.key === 'Enter' && (S.tool.mode === 'poly' || S.tool.mode === 'point')) finishShape();
      else if (e.key === 'Escape') cancelTool();
    } else if (S.pending && e.key === 'Escape') {
      S.pending = null; state.dirty = true; renderPanel();
    }
  });

  /* ── preview (exact seed locations from the backend) ────────────────────── */
  const refreshPreviews = debounce(async () => {
    const c = cfg();
    if (!c) return;
    for (let i = 0; i < pops().length; i++) {
      const pop = pops()[i];
      const key = JSON.stringify([pop.seeding && pop.seeding.strategy,
                                  pop.seeding && pop.seeding.quantity, state.world0]);
      const pv = S.previews[i] = S.previews[i] || {visible: 'auto'};
      if (pv.key === key) continue;
      pv.key = key;
      try {
        const r = await apiJson('/api/seeding/preview',
                                {population: SettingsTab.prune(pop), base: cfgBase()});
        const arr = new Float32Array(r.points.length * 2);
        const w0 = state.world0 || [0, 0];
        r.points.forEach((p, k) => { arr[k*2] = p[0] - w0[0]; arr[k*2+1] = p[1] - w0[1]; });
        if (!pv.buf) pv.buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, pv.buf);
        gl.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
        pv.n = r.points.length;
        pv.label = `${fmtInt(r.n_locations)} locations × ${fmtInt(r.quantity)} = ` +
                   `${fmtInt(r.n_particles)} particles` + (r.capped ? ' (preview capped)' : '');
        pv.error = null;
      } catch (e) {
        pv.n = 0;
        pv.label = null;
        pv.error = String(e.message || e);
      }
      const lbl = $(`seed-count-${i}`);
      if (lbl) {
        lbl.textContent = pv.label || pv.error || '';
        lbl.classList.toggle('err', !!pv.error);
      }
      state.dirty = true;
    }
    if (typeof LayersPanel !== 'undefined') LayersPanel.refresh();
  }, 350);

  /* ── map layer: previews + draft/pending shape ──────────────────────────── */
  function drawShapePts(mode, pts, w0, buf) {
    const closed = mode === 'poly' && pts.length > 2;
    const isBox = mode === 'bbox' && pts.length === 2;
    let disp = pts;
    if (isBox) {
      const [a, b] = pts;
      disp = [[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]], [a[0], a[1]]];
    } else if (closed) {
      disp = [...pts, pts[0]];
    }
    const arr = new Float32Array(disp.length * 2);
    disp.forEach((p, k) => { arr[k*2] = p[0] - w0[0]; arr[k*2+1] = p[1] - w0[1]; });
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
    gl.uniform1f(progFlat.u.uRound, 0);
    if (disp.length > 1 && mode !== 'point')
      gl.drawArrays(gl.LINE_STRIP, 0, disp.length);
    gl.uniform1f(progFlat.u.uRound, 1);
    gl.drawArrays(gl.POINTS, 0, mode === 'point' ? pts.length : disp.length);
    gl.uniform1f(progFlat.u.uRound, 0);
  }

  function drawLayer() {
    if (!state.world0) return;
    gl.useProgram(progFlat.prog);
    gl.uniform4fv(progFlat.u.uView, viewUniform());
    gl.uniform1f(progFlat.u.uPtSize, 7);
    if (!drawLayer.buf) drawLayer.buf = gl.createBuffer();

    for (const [i, pv] of Object.entries(S.previews)) {
      if (!previewShown(pv) || !pv.n || !pv.buf) continue;
      gl.bindBuffer(gl.ARRAY_BUFFER, pv.buf);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
      const rgb = hex2rgb(PALETTE[i % PALETTE.length]);
      gl.uniform4f(progFlat.u.uColor, rgb[0]/255, rgb[1]/255, rgb[2]/255, 0.9);
      gl.uniform1f(progFlat.u.uRound, 2);      // hollow ring ≠ simulated particles
      gl.drawArrays(gl.POINTS, 0, pv.n);
      gl.uniform1f(progFlat.u.uRound, 0);
    }
    if (Tabs.current === 'seeding') {
      const w0 = state.world0;
      gl.uniform4f(progFlat.u.uColor, 0.85, 0.55, 0.1, 1.0);
      if (S.draft.length && S.tool)
        drawShapePts(S.tool.mode, S.draft, w0, drawLayer.buf);
      else if (S.pending)
        drawShapePts(S.pending.mode, S.pending.pts, w0, drawLayer.buf);
    }
  }

  /* ── panel: one card per population covering the WHOLE population schema
     (particle settings + seeding) — the Settings tab links here ──────────── */
  function renderPanel() {
    const box = $('seeding-pops');
    const c = cfg();
    if (!c) {
      box.innerHTML = '<div class="placeholder">Load a configuration in the Settings tab first.</div>';
      return;
    }
    const openState = {};
    box.querySelectorAll('details[data-pointer]').forEach(d => openState[d.dataset.pointer] = d.open);
    box.innerHTML = '';
    const ps = popSchema();
    const ctx = {
      cfg: c,
      onChange: () => { SettingsTab.touch(); refreshPreviews(); },
      rerender: renderPanel,
      openState,
    };
    // full population form; name/type/seeding render explicitly at the top of
    // each card, barriers is not exposed, strategy params render chooser-less
    ctx.override = (path, schema, root) => {
      const p = path.map(x => typeof x === 'number' ? '*' : x).join('/');
      if (p.endsWith('populations/*/name') || p.endsWith('populations/*/particle_type') ||
          p.endsWith('populations/*/seeding') || p.endsWith('populations/*/barriers'))
        return null;
      if (p.endsWith('populations/*/seeding/strategy'))
        return null;                       // rendered explicitly inside renderSeeding
      return SettingsTab.override(ctx, path, schema, root);
    };

    pops().forEach((pop, i) => {
      const card = document.createElement('details');
      card.className = 'sf-group sf-pop';
      card.dataset.pointer = `seedpop/${i}`;
      card.open = openState[`seedpop/${i}`] === true;   // collapsed by default
      card.innerHTML =
        `<summary><span class="swatch" style="background:${PALETTE[i % PALETTE.length]}"></span>` +
        ` ${pop.name || 'population ' + (i + 1)} <span class="dim small">(${pop.particle_type || '?'})</span>` +
        `<span class="poptools"><button class="mini" data-act="dup" title="duplicate">⧉</button>` +
        `<button class="mini danger" data-act="del" title="remove">✕</button></span></summary>`;
      card.querySelector('[data-act=dup]').onclick = e => {
        e.preventDefault();
        const copy = JSON.parse(JSON.stringify(pop));
        copy.name = (copy.name || 'population') + '_copy';
        pops().splice(i + 1, 0, copy);
        afterEdit();
      };
      card.querySelector('[data-act=del]').onclick = e => {
        e.preventDefault();
        if (!confirm(`Remove population "${pop.name || i + 1}"?`)) return;
        pops().splice(i, 1);
        afterEdit();
      };
      const body = document.createElement('div');
      body.className = 'sec';
      // explicit order: name/type on top, then seeding (the important part,
      // always open), then the remaining particle settings
      const base = ['particles', 'populations', i];
      body.appendChild(SchemaForm.fieldRow(ctx, ps.properties.name, 'name',
                                           [...base, 'name'], true));
      body.appendChild(SchemaForm.fieldRow(ctx, ps.properties.particle_type,
                                           'particle_type', [...base, 'particle_type'], true));
      body.appendChild(renderSeeding(ctx, [...base, 'seeding']));
      SchemaForm.renderObject(ctx, ps, ps, base, body);
      card.appendChild(body);
      box.appendChild(card);
    });

    const addRow = document.createElement('div');
    addRow.className = 'row';
    addRow.style.margin = '8px 10px';
    const add = document.createElement('button');
    add.textContent = '＋ Add population';
    add.onclick = () => {
      if (!c.particles) c.particles = {};
      if (!c.particles.populations) c.particles.populations = [];
      c.particles.populations.push(
        SettingsTab.newPopulation(c.particles.populations.length + 1));
      afterEdit();
    };
    addRow.appendChild(add);
    box.appendChild(addRow);
  }

  /* one-line description of the current release positions (panel summary) */
  function strategySummary(st, srcName) {
    if (!st || !Object.keys(st).length) return 'No release locations yet.';
    const inObj = srcName ? ` in "${srcName}"` : '';
    if (st.point) return `point release — ${(st.point.locations || []).length} location(s)`;
    if (st.transect) return `transect${srcName ? ` "${srcName}"` : ''} — ` +
      `${(st.transect.segments || []).length} segment(s), k=${st.transect.k}`;
    if (st.random) return `random points${inObj} — ${st.random.nlocations} locations, ` +
      `seed ${st.random.seed}`;
    if (st.grid) {
      const s = st.grid.separation || {};
      return `regular grid${inObj} — dx ${s.dx} m × dy ${s.dy} m`;
    }
    if (st.file_points) return 'release locations from a points file';
    return '';
  }

  /* the seeding block inside a population card — always visible (no collapse):
     a read-only summary of the release positions + the "Modify seeding…"
     button (the modal is the ONLY place positions change), ⟳ Update when the
     source object moved, the live count, then the schema-driven form for the
     remaining (non-position) seeding fields like release_type */
  function renderSeeding(ctx, path) {
    const i = path[2];                 // ['particles', 'populations', i, 'seeding']
    const pop = pops()[i];
    const ps = popSchema();
    const g = document.createElement('div');
    g.className = 'sf-group sf-pinned';
    g.dataset.pointer = path.join('/');
    g.innerHTML = '<div class="sf-pinnedhead">seeding</div>';
    const body = document.createElement('div');
    body.className = 'sec';

    const lk = S.applied[pop.name || String(i)];
    const src = lk && state.polygons.find(q => q.name === lk.obj);
    const st = pop.seeding && pop.seeding.strategy;

    const srow = document.createElement('div');
    srow.className = 'row small';
    const sum = document.createElement('span');
    sum.className = 'grow';
    sum.textContent = strategySummary(st, src && lk.obj);
    if (!st || !Object.keys(st).length) sum.className = 'grow dim';
    srow.appendChild(sum);
    body.appendChild(srow);

    const brow = document.createElement('div');
    brow.className = 'row';
    const mod = document.createElement('button');
    mod.className = 'grow';
    mod.textContent = '✎ Modify seeding…';
    mod.title = 'select an object and set distribution, coordinates, seed, counts';
    mod.onclick = () => openSeedingModal(i);
    brow.appendChild(mod);
    // ⟳ Update: re-sync coordinates from a changed source object (settings kept)
    if (src && JSON.stringify(src.verts) !== lk.verts) {
      const upd = document.createElement('button');
      upd.className = 'mini okbtn';
      upd.textContent = '⟳ Update';
      upd.title = `"${lk.obj}" changed — re-apply its coordinates to this population`;
      upd.onclick = () => applyFromObject(src, i);
      brow.appendChild(upd);
    }
    body.appendChild(brow);

    const pv = S.previews[i];
    const count = document.createElement('div');
    count.className = 'hint';
    count.id = `seed-count-${i}`;
    count.textContent = pv && (pv.label || pv.error) || '';
    body.appendChild(count);

    SchemaForm.renderObject(ctx, ps.properties.seeding, ps, path, body);
    g.appendChild(body);
    return g;
  }

  /* strategy parameters only — kept for reference, no longer rendered in the
     panel (position settings moved into the Modify-seeding modal) */
  function renderStrategyParams(ctx, path) {
    const ps = popSchema();
    const stratSchema = ps.properties.seeding.properties.strategy;
    const cur = SchemaForm.get(ctx.cfg, path) || {};
    const active = Object.keys(stratSchema.properties).find(o => cur[o] !== undefined);
    const wrap = document.createElement('div');
    wrap.dataset.pointer = path.join('/');
    if (!active) {
      wrap.innerHTML =
        '<div class="hint">No release locations yet — pick an object above.</div>';
      return wrap;
    }
    if (active === 'file_points') {
      const note = document.createElement('div');
      note.className = 'row small dim';
      note.textContent = 'release locations come from a points file';
      wrap.appendChild(note);
    }
    const body = document.createElement('div');
    body.className = 'sf-branch';
    SchemaForm.renderObject(ctx, stratSchema.properties[active], ps, [...path, active], body);
    wrap.appendChild(body);
    return wrap;
  }

  /* ── "Modify seeding…" modal: the ONLY place release positions change.
     All edits happen on a deep-cloned draft strategy; Apply commits it to the
     population, Cancel/✕ just hides the modal and the draft is dropped. */
  const M = {pop: -1, draft: null, from: null};

  function openSeedingModal(i) {
    const pop = pops()[i];
    if (!pop) return;
    M.pop = i;
    M.draft = JSON.parse(JSON.stringify((pop.seeding && pop.seeding.strategy) || {}));
    const lk = S.applied[pop.name || String(i)];
    M.from = (lk && state.polygons.some(q => q.name === lk.obj)) ? lk.obj : null;
    $('modal-seeding-title').textContent =
      `Modify seeding — ${pop.name || 'population ' + (i + 1)}`;
    $('modal-seeding-count').textContent = '';
    renderModal();
    modalCount();
    $('modal-seeding').style.display = 'flex';
  }

  /* derive a fresh draft strategy from an object; area objects keep the draft's
     current distribution + its parameters (same rules as applyPending) */
  function draftFromObject(o, draft) {
    const type = o.type || 'polygon';
    let pts = o.verts.map(v => v.slice());
    if (type === 'point')
      return {point: {locations: pts.map(pt => `${rd(pt[0])},${rd(pt[1])}`)}};
    if (type === 'transect')
      return {transect: {segments: [
        `${rd(pts[0][0])},${rd(pts[0][1])} ${rd(pts[1][0])},${rd(pts[1][1])}`],
        k: (draft.transect && draft.transect.k) || 100}};
    const grid = !!draft.grid;
    const br = grid
      ? {separation: (draft.grid.separation && {...draft.grid.separation}) || {dx: 100, dy: 100}}
      : {nlocations: (draft.random && draft.random.nlocations) || 100,
         seed: draft.random && draft.random.seed !== undefined ? draft.random.seed : 42};
    if (type === 'bbox') {
      const xs = pts.map(v => v[0]), ys = pts.map(v => v[1]);
      br.bbox = `${rd(Math.min(...xs))},${rd(Math.min(...ys))} ` +
                `${rd(Math.max(...xs))},${rd(Math.max(...ys))}`;
    } else {
      br.poly = pts.map(q => `${rd(q[0])},${rd(q[1])}`);
    }
    return grid ? {grid: br} : {random: br};
  }

  /* switch the draft's area distribution keeping the area itself */
  function draftSwitchArea(draft, name) {
    const old = draft.random || draft.grid || {};
    const br = name === 'grid' ? {separation: {dx: 100, dy: 100}}
                               : {nlocations: 100, seed: 42};
    if (old.bbox !== undefined) br.bbox = old.bbox;
    if (old.poly !== undefined) br.poly = old.poly;
    return {[name]: br};
  }

  /* modal field helpers — every edit mutates M.draft and refreshes the count */
  function mRow(labelTxt, el) {
    const row = document.createElement('div');
    row.className = 'row small';
    const l = document.createElement('span');
    l.className = 'sf-label dim';
    l.textContent = labelTxt;
    row.appendChild(l);
    row.appendChild(el);
    return row;
  }
  function mNum(labelTxt, get, set) {
    const inp = document.createElement('input');
    inp.type = 'number';
    inp.className = 'grow';
    inp.value = get();
    inp.oninput = () => {
      const v = parseFloat(inp.value);
      if (isFinite(v)) { set(v); modalCount(); }
    };
    return mRow(labelTxt, inp);
  }
  function mLines(labelTxt, lines, set, ph) {
    const wrap = document.createElement('div');
    wrap.className = 'field';
    wrap.innerHTML = `<div class="lbl">${labelTxt}</div>`;
    const ta = document.createElement('textarea');
    ta.rows = 4;
    ta.style.width = '100%';
    ta.placeholder = ph;
    ta.value = (lines || []).join('\n');
    ta.oninput = () => {
      set(ta.value.split('\n').map(s => s.trim()).filter(Boolean));
      modalCount();
    };
    wrap.appendChild(ta);
    return wrap;
  }

  function renderModal() {
    const body = $('modal-seeding-body');
    body.innerHTML = '';
    const draft = M.draft;
    const glyph = {polygon: '⬠', point: '•', transect: '╱', bbox: '▭'};

    // 1. the source object — the strategy type follows it
    const sel = document.createElement('select');
    sel.className = 'grow';
    const ph = document.createElement('option');
    ph.value = '';
    ph.textContent = state.polygons.length
      ? '— manual coordinates —'
      : '— no objects yet (Objects panel, top right) —';
    sel.appendChild(ph);
    state.polygons.forEach((o, k) => {
      const op = document.createElement('option');
      op.value = k;
      op.textContent = `${glyph[o.type || 'polygon']} ${o.name}`;
      if (o.name === M.from) op.selected = true;
      sel.appendChild(op);
    });
    sel.onchange = () => {
      const o = state.polygons[parseInt(sel.value, 10)];
      if (o) { M.from = o.name; M.draft = draftFromObject(o, M.draft); }
      else M.from = null;
      renderModal();
      modalCount();
    };
    body.appendChild(mRow('seed from', sel));

    // 2. derived strategy + type-specific position settings
    const active = ['point', 'transect', 'random', 'grid', 'file_points']
      .find(k => draft[k] !== undefined);
    const kind = {point: 'point release', transect: 'transect release',
                  random: 'area release', grid: 'area release',
                  file_points: 'points from file'}[active];
    const krow = document.createElement('span');
    krow.className = 'grow dim';
    krow.textContent = kind || 'pick an object above';
    body.appendChild(mRow('strategy', krow));

    if (active === 'random' || active === 'grid') {
      const dsel = document.createElement('select');
      dsel.className = 'grow';
      for (const [v, txt] of [['random', 'random points'], ['grid', 'regular grid']]) {
        const op = document.createElement('option');
        op.value = v; op.textContent = txt;
        dsel.appendChild(op);
      }
      dsel.value = active;
      dsel.onchange = () => {
        M.draft = draftSwitchArea(M.draft, dsel.value);
        renderModal();
        modalCount();
      };
      body.appendChild(mRow('distribution', dsel));

      const br = draft[active];
      const area = document.createElement('span');
      area.className = 'grow dim';
      area.textContent = br.poly ? `polygon, ${br.poly.length} vertices`
                       : br.bbox ? `box ${br.bbox}` : 'no area yet';
      body.appendChild(mRow('area', area));
      if (active === 'random') {
        body.appendChild(mNum('n locations', () => br.nlocations,
                              v => { br.nlocations = Math.max(1, Math.round(v)); }));
        body.appendChild(mNum('seed', () => br.seed,
                              v => { br.seed = Math.round(v); }));
      } else {
        const sep = br.separation = br.separation || {dx: 100, dy: 100};
        body.appendChild(mNum('dx (m)', () => sep.dx, v => { sep.dx = v; }));
        body.appendChild(mNum('dy (m)', () => sep.dy, v => { sep.dy = v; }));
      }
    } else if (active === 'point') {
      body.appendChild(mLines('release locations (x,y per line)',
        draft.point.locations, v => { draft.point.locations = v; }, '75000,455000'));
    } else if (active === 'transect') {
      body.appendChild(mLines('segments (x1,y1 x2,y2 per line)',
        draft.transect.segments, v => { draft.transect.segments = v; },
        '75000,455000 76000,456000'));
      body.appendChild(mNum('k (locations per segment)', () => draft.transect.k,
                            v => { draft.transect.k = Math.max(2, Math.round(v)); }));
    } else if (active === 'file_points') {
      const inp = document.createElement('input');
      inp.type = 'text';
      inp.className = 'grow';
      inp.value = draft.file_points.path || '';
      inp.oninput = () => { draft.file_points.path = inp.value; modalCount(); };
      body.appendChild(mRow('points file', inp));
    } else {
      const hint = document.createElement('div');
      hint.className = 'hint';
      hint.textContent = 'Pick an object to derive the release strategy from its ' +
        'type: point set → point release, transect → transect, polygon/box → area.';
      body.appendChild(hint);
    }
  }

  /* live particle count in the modal footer, computed on the draft */
  const modalCount = debounce(async () => {
    const pop = pops()[M.pop];
    const el = $('modal-seeding-count');
    if (!pop || !el) return;
    if (!M.draft || !Object.keys(M.draft).length) { el.textContent = ''; return; }
    try {
      const p2 = JSON.parse(JSON.stringify(SettingsTab.prune(pop)));
      p2.seeding = p2.seeding || {};
      p2.seeding.strategy = M.draft;
      const r = await apiJson('/api/seeding/preview', {population: p2, base: cfgBase()});
      el.textContent = `${fmtInt(r.n_locations)} locations × ${fmtInt(r.quantity)} = ` +
                       `${fmtInt(r.n_particles)} particles`;
      el.classList.remove('err');
    } catch (e) {
      el.textContent = String(e.message || e);
      el.classList.add('err');
    }
  }, 350);

  function applySeedingModal() {
    const pop = pops()[M.pop];
    if (!pop) return;
    pop.seeding = pop.seeding || {};
    pop.seeding.strategy = M.draft;
    const key = pop.name || String(M.pop);
    if (M.from) {
      const src = state.polygons.find(q => q.name === M.from);
      if (src) S.applied[key] = {obj: M.from, verts: JSON.stringify(src.verts)};
    } else {
      delete S.applied[key];           // manual coordinates: no source link
    }
    $('modal-seeding').style.display = 'none';
    afterEdit();
  }

  function wire() {
    $('seed-goto-settings').onclick = () => Tabs.activate('settings');
    $('modal-seeding-apply').onclick = applySeedingModal;
  }

  let wired = false;
  async function enter() {
    if (!wired) { wire(); wired = true; }
    await SettingsTab.init();
    if (typeof ForcingTab !== 'undefined') ForcingTab.enter();  // map background
    renderPanel();
    refreshPreviews();
  }

  Tabs.register('seeding', {enter, leave() { if (S.tool) cancelTool(); }});
  return {drawLayer, refreshPreviews, renderPanel,
          previews: () => S.previews, previewShown, setPreviewVisible,
          showPreviews: () => showPreviews, setShowPreviews,
          popNames: () => pops().map((p, i) => p.name || 'population ' + (i + 1))};
})();
