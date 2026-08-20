"use strict";
/* ═════════════════════════ Overlays section (ex Layers panel) ═════════════
   The floating Layers panel was merged into the Viewer tab's sidebar: this
   module now renders the Overlays section (#ov-body) — seeding previews,
   objects master eye and the connectivity network, the drawn things that have
   no other native controls in the sidebar. Particles/forcing/background got
   native sidebar controls in the merge. Still a VIEW over the existing
   globals (state.*, SeedingTab.previews) — it owns no state of its own.
   The module keeps its name because refresh() is called from everywhere and
   the Objects panel reuses its ui helpers + wirePanel. */

const LayersPanel = (() => {
  const el = () => $('ov-body');

  function eyeBtn(shown, onClick) {
    const b = document.createElement('button');
    b.className = 'eye' + (shown ? '' : ' off');
    b.textContent = '👁';
    b.onclick = ev => { ev.preventDefault(); ev.stopPropagation(); onClick(); };
    return b;
  }

  function row(...els) {
    const r = document.createElement('div');
    r.className = 'lp-row';
    for (const e of els) r.appendChild(e);
    return r;
  }

  function label(txt, dimmed) {
    const s = document.createElement('span');
    s.className = 'grow' + (dimmed ? ' dim' : '');
    s.style.cssText = 'overflow:hidden;text-overflow:ellipsis;white-space:nowrap';
    s.textContent = txt;
    return s;
  }

  function groupHead(txt, ...extras) {
    const s = document.createElement('div');
    s.className = 'lp-group';
    s.appendChild(document.createTextNode(txt));
    const tools = document.createElement('span');
    tools.className = 'lp-tools';
    for (const e of extras) e && tools.appendChild(e);
    s.appendChild(tools);
    return s;
  }

  /* collapsible group: <details> whose summary is the group header.
     Extras (↗ link, group eye) sit right-aligned in a .lp-tools span so every
     group header reads the same: ▸ NAME … [link][eye] */
  function groupBox(key, txt, ...extras) {
    const d = document.createElement('details');
    d.open = localStorage.getItem('stgui_obj_' + key) !== '0';
    d.addEventListener('toggle', () =>
      localStorage.setItem('stgui_obj_' + key, d.open ? '1' : '0'));
    const s = document.createElement('summary');
    s.className = 'lp-group';
    s.appendChild(document.createTextNode(txt));
    const tools = document.createElement('span');
    tools.className = 'lp-tools';
    for (const e of extras) e && tools.appendChild(e);
    s.appendChild(tools);
    d.appendChild(s);
    return d;
  }

  function swatch(color, hollow) {
    const s = document.createElement('span');
    s.className = 'swatch' + (hollow ? ' hollow' : '');
    if (hollow) s.style.borderColor = color;
    else s.style.background = color;
    return s;
  }

  function mini(txt, title, fn) {
    const b = document.createElement('button');
    b.className = 'mini';
    b.textContent = txt;
    b.title = title;
    b.onclick = ev => { ev.preventDefault(); ev.stopPropagation(); fn(); };
    return b;
  }

  function link(tab, title) {
    return mini('↗', title, () => Tabs.activate(tab));
  }

  const OBJ_GLYPH = {polygon: '⬠', point: '•', transect: '╱', bbox: '▭'};
  const sync = () => { if (typeof syncVis === 'function') syncVis(); };

  function refresh() {
    const body = el();
    if (body) {
      body.innerHTML = '';

      /* ── seeding previews: ONE eye + per-population legend ── */
      if (typeof SeedingTab !== 'undefined') {
        const names = SeedingTab.popNames();
        if (names.length) {
          const shown = SeedingTab.showPreviews();
          body.appendChild(groupHead('Seeding previews',
            link('seeding', 'open the Populations tab'),
            eyeBtn(shown, () => {
              SeedingTab.setShowPreviews(!shown);
              refresh();
            })));
          const previews = SeedingTab.previews();
          names.forEach((nm, i) => {
            const pv = previews[i];
            body.appendChild(row(
              swatch(PALETTE[i % PALETTE.length], true),
              label(nm + (pv && pv.n ? '' : ' (no preview)'), !shown || !(pv && pv.n))));
          });
        }
      }

      /* ── objects: master eye + a one-line summary (per-object visibility
         and management live in the floating Objects panel, top right) ── */
      body.appendChild(groupHead('Objects',
        eyeBtn(state.showOutlines, () => {
          state.showOutlines = !state.showOutlines;
          sync(); state.dirty = true; refresh();
        })));
      body.appendChild(row(label(state.polygons.length
        ? `${state.polygons.length} object(s) — manage in the Objects panel (top right)`
        : 'none drawn yet — see the Objects panel (top right)', true)));

      /* ── connectivity network: nodes at cell centroids + directed edges from
         the last computed matrix (Connectivity tab); drawn by drawConnectivity */
      if (state.connectivity) {
        body.appendChild(groupHead('Connectivity',
          link('connectivity', 'open the Connectivity tab'),
          eyeBtn(state.showConnectivity, () => {
            state.showConnectivity = !state.showConnectivity;
            state.dirty = true; refresh();
          })));
        const con = state.connectivity;
        body.appendChild(row(label(
          `${con.labels.length} cells · ` +
          (con.mode === 'start_end' ? 'start → end' : 'start → ever visited') +
          (con.normalize === 'fraction' ? ' · fractions' : ' · counts'),
          !state.showConnectivity)));
        body.appendChild(row(label('edge width ∝ value · node size ∝ departures', true)));
      }
    }

    if (typeof ObjectsPanel !== 'undefined') ObjectsPanel.refresh();
  }

  /* collapse persistence + header chevron button for a floating panel —
     the chevron matches the sidebar's hide button so panel-level collapse
     reads differently from the small group triangles inside */
  function wirePanel(id, lsKey) {
    const panel = $(id);
    if (!panel) return;
    const tgl = panel.querySelector('summary .panel-toggle');
    const syncTgl = () => { if (tgl) tgl.textContent = panel.open ? '▾' : '▸'; };
    if (tgl) tgl.onclick = ev => {     // buttons swallow the summary click
      ev.preventDefault(); ev.stopPropagation();
      panel.open = !panel.open;
    };
    panel.open = localStorage.getItem(lsKey) === '1';   // collapsed by default
    panel.addEventListener('toggle', () => {
      localStorage.setItem(lsKey, panel.open ? '1' : '0');
      syncTgl();
    });
    syncTgl();
  }
  // (the floating #layerspanel is hidden — no wirePanel for it anymore)

  /* small shared UI helpers, reused by the Objects panel */
  return {refresh, wirePanel,
          ui: {eyeBtn, row, label, groupHead, groupBox, swatch, mini, link}};
})();
LayersPanel.refresh();
