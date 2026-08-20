"use strict";
/* ═════════════════════════ Settings tab ═════════════════════════
   Schema-driven editor for the SedTRAILS simulation yaml. Loads/saves through
   the backend (comment-preserving via ruamel), validates against the packaged
   JSON schemas, and shows a live output-size estimate.                       */

const SettingsTab = (() => {
  let cfg = null;            // the config object bound to the form
  let cfgPath = null;        // absolute path of the loaded yaml (null = unsaved)
  let snapshot = '';         // JSON snapshot for dirty detection
  let roundtrip = null;      // 'ruamel' | 'pyyaml'
  let schemasReady = false;
  let formCtx = null;

  const el = id => $(id);
  const popSchema = () => SchemaForm.resolve({$ref: SchemaForm.URN('population')},
                                             null)[0];

  /* ── defaults for chooser branches ─────────────────────────────────────── */
  const FRESH_TRACER = {
    vanwesten: () => ({flow_field_name: ['bed_load_velocity', 'suspended_velocity']}),
    soulsby: () => ({flow_field_name: ['grain_velocity']}),
    passive_tracer: () => ({flow_field_name: ['depth_avg_flow_velocity']}),
  };
  const FRESH_STRATEGY = {
    point: () => ({locations: []}),
    transect: () => ({segments: [], k: 100}),
    random: () => ({bbox: '', nlocations: 100, seed: 42}),
    grid: () => ({bbox: '', separation: {dx: 100, dy: 100}}),
    file_points: () => ({path: ''}),
  };
  const CHARACTERISTICS = {
    passive: () => ({diffusion_coefficient: 0.0}),
    sand: () => ({density: 2650.0, grain_size: 0.0005}),
    mud: () => ({density: 2000.0, size: 0.00005}),
  };
  const CHAR_DEF = {passive: 'passive_characteristics', sand: 'sand_characteristics',
                    mud: 'mud_characteristics'};

  function newPopulation(n) {
    const prev = cfg.particles && cfg.particles.populations &&
                 cfg.particles.populations[cfg.particles.populations.length - 1];
    if (prev) {
      const copy = JSON.parse(JSON.stringify(prev));
      copy.name = `population_${n}`;
      return copy;
    }
    return {
      name: `population_${n}`, particle_type: 'sand',
      characteristics: CHARACTERISTICS.sand(),
      tracer_methods: {vanwesten: FRESH_TRACER.vanwesten()},
      transport_probability: 'no_probability',
      seeding: {quantity: 1, burial_depth: {constant: 0},
                strategy: {random: FRESH_STRATEGY.random()}},
    };
  }

  /* ── override table for the generic walker ─────────────────────────────── */
  function override(ctx, path, schema, root) {
    const p = path.map(x => typeof x === 'number' ? '*' : x).join('/');

    // particles (populations incl. seeding) live entirely in the Populations
    // tab — hidden here. Existing yaml values are preserved untouched.
    if (p === 'particles') return null;

    // populations (particle settings + seeding) live in the Populations tab —
    // Settings shows a link instead. renderPopulations below stays for reuse.
    if (p === 'particles/populations') {
      const row = document.createElement('div');
      row.className = 'sf-field';
      row.dataset.pointer = path.join('/');
      const n = (SchemaForm.get(cfg, path) || []).length;
      row.innerHTML = '<span class="sf-label dim">populations</span>' +
        `<span class="dim small grow">${n} population(s) — configured in the Populations tab</span>`;
      const btn = document.createElement('button');
      btn.className = 'mini';
      btn.textContent = '↗ Populations';
      btn.title = 'open the Populations tab';
      btn.onclick = () => Tabs.activate('seeding');
      row.appendChild(btn);
      return row;
    }
    if (p === 'visualization') return null;                    // dashboard: not GUI-relevant
    // 'general' and 'domain' contain only unimplemented/legacy options
    // (preprocess, n_runs, display_input_metadata, subset/pol cropping) — the
    // few live ones (input model, numerical scheme) are re-homed under Inputs.
    // Hidden values in existing yamls are preserved untouched.
    if (p === 'general' || p === 'domain') return null;
    if (p === 'inputs') return renderInputsGroup(ctx);

    // physics stays ONE collapsible group: its constants / bed_shear_stress /
    // bed_slope sub-objects render flat inside (dim sub-label, no nested groups)
    if (p === 'physics/constants' || p === 'physics/bed_shear_stress' ||
        p === 'physics/bed_slope') {
      const wrap = document.createElement('div');
      wrap.dataset.pointer = path.join('/');
      const head = document.createElement('div');
      head.className = 'row small dim';
      head.textContent = SchemaForm.label(path[path.length - 1]);
      wrap.appendChild(head);
      const [sub, subRoot] = SchemaForm.resolve(schema, root);
      SchemaForm.renderObject(ctx, sub, subRoot, path, wrap);
      return wrap;
    }

    if (p.endsWith('populations/*/characteristics')) return renderCharacteristics(ctx, path);
    if (p.endsWith('seeding/class')) return null;              // unused legacy option

    // release_type: free text in the schema, but only two values are meaningful
    if (p.endsWith('seeding/release_type'))
      return SchemaForm.fieldRow(ctx, {...schema, enum: ['instantaneous', 'continuous']},
                                 'release_type', path, false);

    // seeding lives in its own tab — Settings shows a link instead (the Seeding
    // tab renders the subtree directly, so this only affects the Settings form)
    if (p.endsWith('populations/*/seeding')) {
      const row = document.createElement('div');
      row.className = 'sf-field';
      row.dataset.pointer = path.join('/');
      row.innerHTML = '<span class="sf-label dim">seeding</span>' +
        '<span class="dim small grow">configured in the Seeding tab</span>';
      const btn = document.createElement('button');
      btn.className = 'mini';
      btn.textContent = '↗ Seeding';
      btn.title = 'open the Seeding tab';
      btn.onclick = () => Tabs.activate('seeding');
      row.appendChild(btn);
      return row;
    }

    if (p === 'outputs/directory') {
      const req = (SchemaForm.getSchemas().main.properties.outputs.required || [])
                    .includes('directory');
      return addBrowse(SchemaForm.fieldRow(ctx, mergedSchema(schema, root),
                                           'directory', path, req), 'dir');
    }

    if (p.endsWith('populations/*/tracer_methods')) {
      const ps = popSchema();
      return SchemaForm.chooser(ctx, path, ['vanwesten', 'soulsby', 'passive_tracer'],
        (active, bpath) => {
          const body = document.createElement('div');
          body.className = 'sf-branch';
          const [sub, subRoot] = SchemaForm.resolve(
            ps.properties.tracer_methods.properties[active], ps);
          SchemaForm.renderObject(ctx, sub, subRoot, bpath, body);
          return body;
        },
        o => FRESH_TRACER[o](), 'tracer method');
    }

    if (p.endsWith('seeding/strategy')) {
      const ps = popSchema();
      const stratSchema = ps.properties.seeding.properties.strategy;
      return SchemaForm.chooser(ctx, path,
        ['point', 'transect', 'random', 'grid', 'file_points'],
        (active, bpath) => {
          const body = document.createElement('div');
          body.className = 'sf-branch';
          SchemaForm.renderObject(ctx, stratSchema.properties[active], ps, bpath, body);
          return body;
        },
        o => FRESH_STRATEGY[o](), 'seeding strategy');
    }

    if (p.endsWith('seeding/burial_depth')) {
      const ps = popSchema();
      const bdSchema = ps.properties.seeding.properties.burial_depth;
      const BD_LBL = {constant: 'depth below bed [m]',
                      random: 'max depth [m] (uniform 0–max)'};
      const wrap = document.createElement('div');
      wrap.dataset.pointer = path.join('/');
      const head = document.createElement('div');
      head.className = 'row small dim';
      head.innerHTML = 'initial burial depth ' + SchemaForm.infoBtn({description:
        'Burial depth of the particles at release, below the local bed level. ' +
        '"constant" buries every particle at the same depth; "random" draws each ' +
        'particle uniformly between 0 and the given maximum.'});
      wrap.appendChild(head);
      wrap.appendChild(SchemaForm.chooser(ctx, path, ['constant', 'random'],
        (active, bpath) => {
          const body = document.createElement('div');
          body.className = 'sf-branch';
          body.appendChild(SchemaForm.fieldRow(ctx, bdSchema.properties[active],
                                               BD_LBL[active], bpath, false));
          return body;
        },
        o => 0, ''));                       // the header above already labels it
      return wrap;
    }

    if (p === 'outputs/netcdf/compression') {
      const row = document.createElement('div');
      row.className = 'sf-field';
      row.dataset.pointer = path.join('/');
      row.innerHTML = '<span class="sf-label dim">compression</span>';
      const sel = document.createElement('select');
      sel.className = 'sf-input';
      for (const [v, txt] of [['', '(default: auto)'], ['auto', 'auto'],
                              ['true', 'on'], ['false', 'off']]) {
        const op = document.createElement('option');
        op.value = v; op.textContent = txt;
        sel.appendChild(op);
      }
      const cur = SchemaForm.get(cfg, path);
      sel.value = cur === undefined ? '' : String(cur);
      sel.onchange = () => {
        const obj = SchemaForm.containerAt(cfg, path.slice(0, -1));
        if (sel.value === '') delete obj.compression;
        else obj.compression = sel.value === 'auto' ? 'auto' : sel.value === 'true';
        ctx.onChange(path.join('/'));
      };
      row.appendChild(sel);
      row.insertAdjacentHTML('beforeend', SchemaForm.infoBtn(schema));
      return row;
    }

    return undefined;                                    // generic rendering
  }

  /* ── custom Inputs group (re-homed general options, forcing link, morpho) ── */
  function addBrowse(row, kind) {
    const btn = document.createElement('button');
    btn.className = 'mini';
    btn.textContent = '…';
    btn.title = 'Browse';
    btn.onclick = async () => {
      try {
        const r = await apiJson('/api/pickfile?kind=' + kind);
        if (r.paths && r.paths[0]) {
          const inp = row.querySelector('.sf-input');
          inp.value = r.paths[0];
          inp.dispatchEvent(new Event('change'));
        }
      } catch (e) { toast(String(e), true); }
    };
    row.appendChild(btn);
    return row;
  }

  function mergedSchema(propSchema, root) {
    const [sub] = SchemaForm.resolve(propSchema, root);
    const m = {...sub, ...propSchema};
    delete m.$ref;
    return m;
  }

  function selectRow(labelTxt, options, value, onSet, info) {
    const row = document.createElement('div');
    row.className = 'sf-field';
    row.innerHTML = `<span class="sf-label dim">${labelTxt}</span>`;
    const sel = document.createElement('select');
    sel.className = 'sf-input';
    for (const [v, txt] of options) {
      const o = document.createElement('option');
      o.value = v;
      o.textContent = txt;
      sel.appendChild(o);
    }
    sel.value = value;
    sel.onchange = () => onSet(sel.value);
    row.appendChild(sel);
    if (info) row.insertAdjacentHTML('beforeend',
      `<button class="mini info" data-info="${info.replace(/"/g, '&quot;')}" tabindex="-1">ⓘ</button>`);
    return row;
  }

  let morphoInclude = null;   // UI-only: 'yes' | 'no' (no dedicated yaml key exists)

  function renderInputsGroup(ctx) {
    const main = SchemaForm.getSchemas().main;
    const P = main.properties;
    const inputsP = P.inputs.properties;
    const imSchema = SchemaForm.resolve(P.general.properties.input_model, main)[0];

    const g = document.createElement('details');
    g.className = 'sf-group';
    g.dataset.pointer = 'inputs';
    g.open = ctx.openState['inputs'] === true;   // collapsed by default
    g.innerHTML = `<summary>inputs ${SchemaForm.infoBtn(P.inputs)}</summary>`;
    const body = document.createElement('div');
    body.className = 'sec';

    // forcing file + browse + quick-link to the Viewer tab (forcing layers)
    const frcRow = addBrowse(
      SchemaForm.fieldRow(ctx, inputsP.data, 'forcing', ['inputs', 'data'], true), 'nc');
    const view = document.createElement('button');
    view.className = 'mini';
    view.textContent = '↗ view';
    view.title = 'open in the Viewer tab';
    view.onclick = () => Tabs.activate('viewer');
    frcRow.appendChild(view);
    body.appendChild(frcRow);

    // live options re-homed from the hidden "general" section
    body.appendChild(SchemaForm.fieldRow(ctx, imSchema.properties.format,
      'format', ['general', 'input_model', 'format'], false));
    body.appendChild(SchemaForm.fieldRow(ctx, imSchema.properties.reference_date,
      'reference date', ['general', 'input_model', 'reference_date'], false));
    body.appendChild(SchemaForm.fieldRow(ctx, imSchema.properties.morfac,
      'morfac', ['general', 'input_model', 'morfac'], false));
    body.appendChild(SchemaForm.fieldRow(ctx,
      mergedSchema(P.general.properties.numerical_scheme, main),
      'numerical scheme', ['general', 'numerical_scheme'], false));

    body.appendChild(SchemaForm.fieldRow(ctx, inputsP.read_interval,
      'read interval', ['inputs', 'read_interval'], false));
    body.appendChild(SchemaForm.fieldRow(ctx, inputsP.repeat_eulerian_fields,
      'repeat eulerian fields', ['inputs', 'repeat_eulerian_fields'], false));

    // morphodynamics: bed level source (bed_level_data override)
    const hasSep = SchemaForm.get(cfg, ['inputs', 'bed_level_data']) !== undefined;
    if (morphoInclude === null || hasSep) morphoInclude = 'yes';
    body.appendChild(selectRow('include morphodynamics?',
      [['yes', 'Yes (default)'], ['no', 'No']], morphoInclude,
      v => {
        morphoInclude = v;
        if (v === 'no' && cfg.inputs) {
          delete cfg.inputs.bed_level_data;
          ctx.onChange('inputs/bed_level_data');
        }
        ctx.rerender();
      },
      'Whether the bed level evolves during the simulation. The bed comes from ' +
      'the forcing file by default; a separate bed-level file can override it ' +
      '(measured surveys, tiled morphodynamic predictions).'));
    if (morphoInclude === 'yes') {
      body.appendChild(selectRow('bed level source',
        [['forcing', 'from forcing file (default)'], ['separate', 'from separate file [BETA]']],
        hasSep ? 'separate' : 'forcing',
        v => {
          if (v === 'forcing') { if (cfg.inputs) delete cfg.inputs.bed_level_data; }
          else SchemaForm.containerAt(cfg, ['inputs']).bed_level_data = '';
          ctx.onChange('inputs/bed_level_data');
          ctx.rerender();
        },
        inputsP.bed_level_data.description));
      if (hasSep) {
        body.appendChild(addBrowse(SchemaForm.fieldRow(ctx, inputsP.bed_level_data,
          'bed level file', ['inputs', 'bed_level_data'], false), 'nc'));
      }
    }
    // inputs.comp_dir is hidden: unused legacy option

    g.appendChild(body);
    return g;
  }

  /* ── populations UI ────────────────────────────────────────────────────── */
  function renderPopulations(ctx, path) {
    const wrap = document.createElement('div');
    wrap.dataset.pointer = path.join('/');
    const pops = SchemaForm.get(cfg, path) || [];
    const ps = popSchema();

    pops.forEach((pop, i) => {
      const card = document.createElement('details');
      card.className = 'sf-group sf-pop';
      card.dataset.pointer = [...path, i].join('/');
      card.open = ctx.openState[card.dataset.pointer] === true;
      const name = pop && pop.name ? pop.name : `population ${i + 1}`;
      const type = pop && pop.particle_type ? pop.particle_type : '?';
      card.innerHTML =
        `<summary><span class="swatch" style="background:${PALETTE[i % PALETTE.length]}"></span>` +
        ` ${name} <span class="dim small">(${type})</span>` +
        `<span class="poptools"><button class="mini" data-act="dup" title="duplicate">⧉</button>` +
        `<button class="mini danger" data-act="del" title="remove">✕</button></span></summary>`;
      card.querySelector('[data-act=dup]').onclick = e => {
        e.preventDefault();
        const copy = JSON.parse(JSON.stringify(pop));
        copy.name = (copy.name || 'population') + '_copy';
        pops.splice(i + 1, 0, copy);
        ctx.onChange(path.join('/'));
        ctx.rerender();
      };
      card.querySelector('[data-act=del]').onclick = e => {
        e.preventDefault();
        pops.splice(i, 1);
        ctx.onChange(path.join('/'));
        ctx.rerender();
      };
      const body = document.createElement('div');
      body.className = 'sec';
      SchemaForm.renderObject(ctx, ps, ps, [...path, i], body);
      card.appendChild(body);
      wrap.appendChild(card);
    });

    const addRow = document.createElement('div');
    addRow.className = 'row';
    const add = document.createElement('button');
    add.textContent = '＋ Add population';
    add.onclick = () => {
      SchemaForm.containerAt(cfg, path.slice(0, -1));
      if (!cfg.particles.populations) cfg.particles.populations = [];
      cfg.particles.populations.push(newPopulation(cfg.particles.populations.length + 1));
      ctx.onChange(path.join('/'));
      ctx.rerender();
    };
    addRow.appendChild(add);
    wrap.appendChild(addRow);
    return wrap;
  }

  function renderCharacteristics(ctx, path) {
    const wrap = document.createElement('div');
    wrap.className = 'sf-chooser';
    wrap.dataset.pointer = path.join('/');
    const popPath = path.slice(0, -1);
    const pop = SchemaForm.get(cfg, popPath) || {};
    const type = pop.particle_type || 'passive';
    const ps = popSchema();
    const [sub] = SchemaForm.resolve({$ref: '#/$defs/' + CHAR_DEF[type]}, ps);
    const head = document.createElement('div');
    head.className = 'row small dim';
    head.textContent = `characteristics (${type})`;
    wrap.appendChild(head);
    const body = document.createElement('div');
    body.className = 'sf-branch';
    SchemaForm.renderObject(ctx, sub, ps, path, body);
    wrap.appendChild(body);
    return wrap;
  }

  /* ── dirty / snapshot ──────────────────────────────────────────────────── */
  function markClean() {
    snapshot = JSON.stringify(cfg); updateChip();
    // the Run tab renders its "⚠ unsaved changes" hint from isDirty() — clear it now
    if (typeof RunTab !== 'undefined') RunTab.refreshSummary();
  }
  function isDirty() { return cfg !== null && JSON.stringify(cfg) !== snapshot; }
  function updateChip() {
    const chip = el('cfgchip');
    if (chip) {
      if (!cfg) chip.textContent = '';
      else {
        const base = cfgPath ? cfgPath.split(/[\\/]/).pop() : '(unsaved config)';
        chip.textContent = base + (isDirty() ? ' ●' : '');
        chip.title = cfgPath || '';
      }
    }
    // topbar save icon mirrors the dirty state: grey when nothing to save
    const sv = el('tb-save');
    if (sv) {
      const needsSave = !!cfg && (isDirty() || !cfgPath);
      sv.disabled = !needsSave;
      sv.title = !cfg ? 'no config loaded' : needsSave ? 'Save config' : 'Saved ✓';
      sv.classList.toggle('attn', needsSave);     // small dot badge, not a blue button
    }
    const sa = el('tb-saveas');
    if (sa) sa.disabled = !cfg;
  }
  window.addEventListener('beforeunload', e => {
    if (isDirty()) { e.preventDefault(); e.returnValue = ''; }
  });

  /* ── prune empty containers before save/validate ───────────────────────── */
  function pruned(o) {
    if (Array.isArray(o)) return o.map(pruned);
    if (o && typeof o === 'object') {
      const out = {};
      for (const [k, v] of Object.entries(o)) {
        const pv = pruned(v);
        const emptyObj = pv && typeof pv === 'object' && !Array.isArray(pv)
                         && Object.keys(pv).length === 0;
        if (pv !== undefined && !emptyObj) out[k] = pv;
      }
      return out;
    }
    return o;
  }

  /* ── validation display ────────────────────────────────────────────────── */
  function showErrors(errors) {
    document.querySelectorAll('.sf-err').forEach(x => x.classList.remove('sf-err'));
    const box = el('set-errors');
    if (!errors || !errors.length) {
      box.innerHTML = cfg ? '<span class="ok">✓ configuration is valid</span>' : '';
      return;
    }
    box.innerHTML = errors.map(e =>
      `<div class="errline" data-goto="${e.pointer}">▸ <b>${e.pointer || '(root)'}</b>: ${e.message}</div>`).join('');
    for (const e of errors) {
      let ptr = e.pointer;
      while (ptr) {
        const hit = document.querySelector(`#settings-form [data-pointer="${ptr}"]`);
        if (hit) { hit.classList.add('sf-err'); break; }
        ptr = ptr.split('/').slice(0, -1).join('/');
      }
    }
    box.querySelectorAll('.errline').forEach(d => d.onclick = () => {
      let ptr = d.dataset.goto;
      while (ptr) {
        const hit = document.querySelector(`#settings-form [data-pointer="${ptr}"]`);
        if (hit) {
          let g = hit.closest('details');
          while (g) { g.open = true; g = g.parentElement.closest('details'); }
          hit.scrollIntoView({block: 'center'});
          break;
        }
        ptr = ptr.split('/').slice(0, -1).join('/');
      }
    });
  }

  /* ── size estimate ─────────────────────────────────────────────────────── */
  const refreshEstimate = debounce(async () => {
    if (!cfg) return;
    try {
      const est = await apiJson('/api/config/estimate', {config: pruned(cfg)});
      const box = el('set-estimate');
      if (!est.particles || !est.slots) {
        box.textContent = 'output size: unknown (set duration, save_interval and seeding)';
        return;
      }
      const mb = est.bytes / 1024 / 1024;
      const size = mb > 1024 ? (mb / 1024).toFixed(1) + ' GB' : mb.toFixed(1) + ' MB';
      box.textContent =
        `${est.particles_exact ? '' : '≈ '}${fmtInt(est.particles)} particles × ` +
        `${fmtInt(est.slots)} output slots × ${est.bytes_per_particle_slot} B ` +
        `= ${size} uncompressed` +
        (est.compression ? ' (compression will be enabled)' : '');
    } catch (e) { /* estimate is best-effort */ }
  }, 400);

  /* ── form rendering ────────────────────────────────────────────────────── */
  function renderForm() {
    formCtx = SchemaForm.render(el('settings-form'), cfg, {
      onChange: () => { updateChip(); refreshEstimate(); },
      override,
    });
    updateChip();
    refreshEstimate();
  }

  /* ── file operations ───────────────────────────────────────────────────── */
  async function openPath(p) {
    const r = await apiJson('/api/config/load', {path: p});
    cfg = r.config || {};
    cfgPath = r.path;
    roundtrip = r.roundtrip;
    localStorage.setItem('stgui_last_config', cfgPath);
    el('set-path').value = cfgPath;
    if (roundtrip === 'pyyaml') {
      toast('ruamel.yaml not installed — comments in this file will not be preserved on save', true);
    }
    markClean();
    renderForm();
    showErrors(null);
    // the referenced forcing file loads (and shows) immediately; an existing
    // result file of this config loads into the viewer as well
    if (typeof ForcingTab !== 'undefined')
      ForcingTab.autoOpenFromConfig().catch(() => {});
    if (typeof autoImportResult === 'function') autoImportResult();
    const v = await apiJson('/api/config/validate', {config: pruned(cfg)});
    showErrors(v.errors);
  }

  async function save(saveAs) {
    if (!cfg) return;
    let p = cfgPath;
    if (saveAs || !p) {
      const sug = p ? p.split(/[\\/]/).pop() : 'sedtrails-config.yaml';
      const r = await apiJson(`/api/picksave?ext=.yaml&name=${encodeURIComponent(sug)}`);
      if (!r.path) return;
      p = r.path;
    }
    try {
      const r = await apiJson('/api/config/save', {path: p, config: pruned(cfg)});
      cfgPath = r.path;
      el('set-path').value = cfgPath;
      localStorage.setItem('stgui_last_config', cfgPath);
      markClean();
      showErrors(r.validation.errors);
      toast(r.validation.valid ? 'Saved ✓' : 'Saved — but the config has validation errors',
            !r.validation.valid);
    } catch (e) { toast('save failed: ' + e, true); }
  }

  /* ── wiring ────────────────────────────────────────────────────────────── */
  // init runs from app.js (startup) AND from tab hooks — memoize the promise
  // so concurrent callers share one run instead of double-loading the config
  let initP = null;
  function initOnce() {
    return initP || (initP = init().catch(e => { initP = null; throw e; }));
  }
  async function init() {
    if (schemasReady) return;
    const s = await apiJson('/api/schema');
    SchemaForm.setSchemas(s);
    schemasReady = true;

    el('set-open').onclick = async () => {
      try {
        const r = await apiJson('/api/pickfile?kind=yaml');
        if (r.paths && r.paths[0]) await openPath(r.paths[0]);
        else if (r.error) toast('native dialog unavailable: ' + r.error, true);
      } catch (e) { toast(String(e), true); }
    };
    el('set-path').addEventListener('keydown', async e => {
      if (e.key !== 'Enter') return;
      const p = el('set-path').value.trim();
      if (!p) return;
      try { await openPath(p); } catch (err) { toast(String(err), true); }
    });
    el('tb-save').onclick = () => save(false);
    el('tb-saveas').onclick = () => save(true);

    async function newFromTemplate() {
      const r = await apiJson('/api/config/template');
      cfg = r.config;
      cfgPath = null;
      el('set-path').value = '';
      markClean();
      renderForm();
      showErrors(null);
    }

    const last = localStorage.getItem('stgui_last_config');
    if (last) {
      try { await openPath(last); } catch (e) { /* file moved — fine */ }
    }
    if (!cfg) {
      el('settings-form').innerHTML =
        '<div class="placeholder">Open an existing SedTRAILS yaml (Open…, or type a path and ' +
        'press Enter) — or <a href="#" id="set-template-link">start from a template</a>.<br>' +
        'All options below come straight from the packaged JSON schemas — hover the ⓘ ' +
        'symbols for documentation.</div>';
      const link = el('set-template-link');
      if (link) link.onclick = e => {
        e.preventDefault();
        newFromTemplate().catch(err => toast(String(err), true));
      };
    }
    updateChip();
  }

  Tabs.register('settings', {enter() { initOnce().catch(e => toast(String(e), true)); }});
  if (Tabs.current === 'settings') initOnce().catch(e => toast(String(e), true));

  return {
    getConfig: () => cfg,
    getPath: () => cfgPath,
    getPruned: () => cfg ? pruned(cfg) : null,
    isDirty,
    openPath,
    override,                                   // reused by the Seeding tab
    prune: pruned,
    touch: () => { updateChip(); refreshEstimate(); },
    newPopulation,                              // reused by the Populations tab
    save: () => save(false),
    schemasReady: () => schemasReady,
    init: initOnce,
  };
})();
