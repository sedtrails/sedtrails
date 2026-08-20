"use strict";
/* ═════════════════════════ floating Objects panel ═════════════════════════
   THE central manager for all map shapes (polygons, boxes, transects, point
   sets) — one shared store (state.polygons, persisted server-side). Objects
   are created here (draw / import), renamed, re-colored, re-ordered, edited
   and deleted here — other tabs only PICK from this store (seeding areas,
   filter rules, connectivity). Visibility of the whole group is toggled in
   the Layers panel; per-object eyes live here. */

const ObjectsPanel = (() => {
  const el = () => $('op-body');
  const GLYPH = {polygon: '⬠', point: '•', transect: '╱', bbox: '▭'};
  const fmtC = v => String(Math.round(v * 100) / 100);

  /* editable coordinate table over a live [[x,y],…] array (object verts or
     the current draft) — commit on change, so typing keeps focus */
  function coordTable(verts, {onChange, minVerts, addLabel}) {
    const {mini} = LayersPanel.ui;
    const t = document.createElement('div');
    t.className = 'op-coords';
    const head = document.createElement('div');
    head.className = 'op-crow dim';
    head.innerHTML = '<span></span><i>x [m]</i><i>y [m]</i>';
    t.appendChild(head);
    verts.forEach((v, i) => {
      const r = document.createElement('div');
      r.className = 'op-crow';
      const idx = document.createElement('span');
      idx.textContent = i + 1;
      const num = (k) => {
        const inp = document.createElement('input');
        inp.value = fmtC(v[k]);
        inp.onchange = () => {
          const f = parseFloat(inp.value);
          if (isFinite(f)) { v[k] = f; onChange(); }
          else inp.value = fmtC(v[k]);
        };
        return inp;
      };
      const del = mini('✕', 'remove this point', () => {
        if (verts.length <= minVerts) {
          toast(`needs at least ${minVerts} point(s)`, true);
          return;
        }
        verts.splice(i, 1);
        onChange();
      });
      r.append(idx, num(0), num(1), del);
      t.appendChild(r);
    });
    const foot = document.createElement('div');
    foot.className = 'op-crow';
    foot.appendChild(mini('＋ point', addLabel, () => {
      const last = verts[verts.length - 1];
      const off = 25 / state.view.ppm * (window.devicePixelRatio || 1);
      verts.push(last ? [last[0] + off, last[1] + off] : [0, 0]);
      onChange();
    }));
    t.appendChild(foot);
    return t;
  }

  function refresh() {
    const body = el();
    if (!body || typeof LayersPanel === 'undefined') return;
    const {eyeBtn, row, label, mini} = LayersPanel.ui;
    body.innerHTML = '';

    /* ── toolbar: draw a new object / import from file ── */
    const bar = document.createElement('div');
    bar.className = 'op-toolbar';
    for (const [mode, txt, title] of [
      ['point', '• points', 'draw a set of points (click, Enter to finish)'],
      ['transect', '╱ transect', 'draw a transect (2 clicks)'],
      ['bbox', '▭ box', 'draw a box (2 corner clicks)'],
      ['polygon', '⬠ polygon', 'draw a polygon (Enter/double-click to finish)'],
    ]) {
      const b = document.createElement('button');
      b.className = 'mini' +
        (state.tool === 'draw' && (state.drawMode || 'polygon') === mode ? ' active' : '');
      b.textContent = txt;
      b.title = title;
      b.onclick = () => startDrawObject(mode);
      bar.appendChild(b);
    }
    const imp = document.createElement('button');
    imp.className = 'mini';
    imp.textContent = '⇪ import';
    imp.title = 'import polygons (.txt file or folder)';
    imp.onclick = () => $('btn-poly-import').onclick();
    bar.appendChild(imp);
    // bulk-remove the auto-generated connectivity cells (only shown if any)
    if (typeof ConnectivityTab !== 'undefined' &&
        state.polygons.some(p => ConnectivityTab.AUTO_RE.test(p.name))) {
      const del = document.createElement('button');
      del.className = 'mini danger';
      del.textContent = '✕ auto cells';
      del.title = 'remove all auto-generated cells (auto_cell_*)';
      del.onclick = () => ConnectivityTab.removeAutoCells();
      bar.appendChild(del);
    }
    body.appendChild(bar);

    /* live coordinates of the shape being drawn */
    if (state.tool === 'draw' && state.draft.length) {
      body.appendChild(row(label('new ' +
        (OBJ_LABEL[state.drawMode || 'polygon'] || 'object') + ' — coordinates', true)));
      body.appendChild(coordTable(state.draft, {
        minVerts: 1,
        addLabel: 'add a point (or click the map)',
        onChange: () => { state.dirty = true; refresh(); },
      }));
    }

    if (!state.polygons.length) {
      body.appendChild(row(label('no objects yet — draw or import above', true)));
      return;
    }

    state.polygons.forEach((p, i) => {
      const type = p.type || 'polygon';
      const editing = state.editSel && state.editSel.name === p.name;

      const glyph = document.createElement('span');
      glyph.className = 'dim';
      glyph.style.cssText = 'width:14px;text-align:center;flex:none';
      glyph.textContent = GLYPH[type];
      glyph.title = type;

      const nameInp = document.createElement('input');
      nameInp.className = 'lp-name';
      nameInp.value = p.name;
      nameInp.title = 'click to rename';
      nameInp.onchange = () => {
        let nn = nameInp.value.trim() || p.name;
        while (state.polygons.some(q => q !== p && q.name === nn)) nn += '*';
        if (state.editSel && state.editSel.name === p.name) state.editSel = {name: nn};
        p.name = nn;
        onPolygonsChanged();
        refresh();
      };

      const colorBtn = typeof mkColorBtn === 'function'
        ? mkColorBtn(() => p.color || '#445566',
                     c => {
                       p.color = c;
                       savePolygons();
                       if (typeof rebuildPaletteTex === 'function') rebuildPaletteTex();
                       if (typeof refreshRules === 'function') refreshRules();
                       state.dirty = true;
                       LayersPanel.refresh();
                     },
                     'object color')
        : LayersPanel.ui.swatch(p.color || '#445566');

      const edit = mini('✎', editing ? 'stop editing' : 'edit points (map + table)', () => {
        state.editSel = editing ? null : {name: p.name};
        state.tool = state.editSel ? 'edit' : 'pan';
        if (typeof refreshToolButtons === 'function') refreshToolButtons();
        state.dirty = true;
        refresh();
      });
      if (editing) edit.classList.add('active');

      body.appendChild(row(
        eyeBtn(!p.hidden, () => {
          p.hidden = !p.hidden;
          savePolygons();
          state.dirty = true;
          LayersPanel.refresh();
        }),
        colorBtn, glyph, nameInp, edit,
        mini('▲', 'move up', () => {
          if (i === 0) return;
          [state.polygons[i - 1], state.polygons[i]] =
            [state.polygons[i], state.polygons[i - 1]];
          onPolygonsChanged();
          refresh();
        }),
        mini('✕', 'delete object', () => {
          if (!confirm(`Delete "${p.name}"?`)) return;
          state.polygons.splice(i, 1);
          if (state.editSel && state.editSel.name === p.name) state.editSel = null;
          onPolygonsChanged();
          refresh();
        })));

      if (editing) {
        body.appendChild(coordTable(p.verts, {
          minVerts: MIN_VERTS[type] || 3,
          addLabel: 'add a point (or click an edge on the map)',
          onChange: () => { onPolygonsChanged(); refresh(); },
        }));
      }
    });
  }

  /* collapse persistence + header chevron (shared with the Layers panel) */
  LayersPanel.wirePanel('objectspanel', 'stgui_objectspanel_open');

  return {refresh};
})();
ObjectsPanel.refresh();
