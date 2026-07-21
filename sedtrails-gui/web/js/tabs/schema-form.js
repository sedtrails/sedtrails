"use strict";
/* ═════════════════ schema-driven form generator (Settings tab) ═════════════════
   Renders a JSON-Schema (draft 2020-12) as an editable form bound directly to a
   plain config object. Generic walker + a small override table for the parts
   with structure the schema alone can't express nicely (populations, exactly-
   one-of choosers). Values the user leaves empty stay ABSENT from the config,
   so schema defaults keep applying at run time (shown as placeholders).      */

const SchemaForm = (() => {
  let SCHEMAS = null;                       // {main, population, visualization}
  const URN = f => `urn:sedtrails:config:${f}.schema.json`;

  function setSchemas(s) { SCHEMAS = s; }

  /* ── $ref resolution; returns [schema, root-for-#-refs] ─────────────────── */
  function resolve(schema, root) {
    let guard = 0;
    while (schema && schema.$ref && guard++ < 8) {
      const ref = schema.$ref;
      if (ref.startsWith('#/')) {
        let t = root;
        for (const part of ref.slice(2).split('/')) t = t && t[part];
        if (!t) break;
        schema = t;
      } else {
        const name = ['main', 'population', 'visualization'].find(n => ref === URN(n));
        if (!name) break;
        root = SCHEMAS[name];
        schema = root;
      }
    }
    return [schema || {}, root];
  }

  /* ── small helpers ──────────────────────────────────────────────────────── */
  const label = k => k.replace(/_/g, ' ');
  const fmtVal = v => typeof v === 'boolean' ? (v ? 'True' : 'False') : String(v);
  const fmtDefault = d => typeof d === 'object' ? JSON.stringify(d) : fmtVal(d);

  function infoBtn(schema) {
    if (!schema.description && schema.default === undefined) return '';
    const parts = [];
    if (schema.description) parts.push(schema.description);
    if (schema.default !== undefined) parts.push(`Default: ${fmtDefault(schema.default)}`);
    const txt = parts.join('\n\n').replace(/"/g, '&quot;');
    return `<button class="mini info" data-info="${txt}" tabindex="-1">ⓘ</button>`;
  }

  function ensure(obj, key, init) {
    if (obj[key] === undefined || obj[key] === null) obj[key] = init;
    return obj[key];
  }

  /* getter for the container at `path` inside cfg, creating objects on demand */
  function containerAt(cfg, path) {
    let o = cfg;
    for (const p of path) o = ensure(o, p, typeof p === 'number' ? [] : {});
    return o;
  }

  /* ── leaf widgets ───────────────────────────────────────────────────────── */
  function coerce(schema, raw) {
    if (raw === '') return undefined;
    const t = Array.isArray(schema.type) ? schema.type[0] : schema.type;
    if (t === 'integer') { const v = parseInt(raw, 10); return isNaN(v) ? undefined : v; }
    if (t === 'number') { const v = parseFloat(raw); return isNaN(v) ? undefined : v; }
    if (t === 'boolean') return raw === 'true';
    return raw;
  }

  function leafWidget(schema, value, onSet) {
    const t = Array.isArray(schema.type) ? schema.type.find(x => x !== 'null') : schema.type;
    let el;
    if (schema.enum) {
      el = document.createElement('select');
      const opts = ['', ...schema.enum.filter(o => o !== schema.default)];
      for (const o of opts) {
        const op = document.createElement('option');
        op.value = o;
        op.textContent = o === '' ?
          (schema.default !== undefined ? `${fmtVal(schema.default)} (default)` : '(unset)') : o;
        el.appendChild(op);
      }
      if (value !== undefined && value !== null && String(value) === String(schema.default)) value = '';
      el.value = value === undefined || value === null ? '' : String(value);
      el.onchange = () => onSet(coerce(schema, el.value));
    } else if (t === 'boolean') {
      el = document.createElement('select');
      const options = schema.default !== undefined
        ? [['', `${fmtVal(schema.default)} (default)`],
           [String(!schema.default), fmtVal(!schema.default)]]
        : [['', '(unset)'], ['true', 'True'], ['false', 'False']];
      for (const [v, txt] of options) {
        const op = document.createElement('option');
        op.value = v;
        op.textContent = txt;
        el.appendChild(op);
      }
      if (value !== undefined && value !== null && value === schema.default) value = '';
      el.value = value === undefined || value === null ? '' : String(value);
      el.onchange = () => onSet(coerce({type: 'boolean'}, el.value));
    } else if (t === 'number' || t === 'integer') {
      el = document.createElement('input');
      el.type = 'number';
      el.step = t === 'integer' ? '1' : 'any';
      if (schema.minimum !== undefined) el.min = schema.minimum;
      if (schema.maximum !== undefined) el.max = schema.maximum;
      if (schema.default !== undefined) el.placeholder = `${schema.default} (default)`;
      el.value = value === undefined || value === null ? '' : value;
      el.onchange = () => onSet(coerce(schema, el.value));
    } else if (schema.type === 'array') {
      el = document.createElement('textarea');
      el.rows = Math.min(4, Math.max(2, (value || []).length));
      el.placeholder = 'one entry per line';
      el.value = (value || []).join('\n');
      el.onchange = () => {
        const arr = el.value.split('\n').map(s => s.trim()).filter(Boolean);
        onSet(arr.length ? arr : undefined);
      };
    } else {                                         // string / mixed
      el = document.createElement('input');
      el.type = 'text';
      if (schema.default !== undefined) el.placeholder = `${schema.default} (default)`;
      el.value = value === undefined || value === null ? '' : String(value);
      el.onchange = () => onSet(el.value === '' ? undefined : el.value);
    }
    return el;
  }

  /* ── field row ──────────────────────────────────────────────────────────── */
  function fieldRow(ctx, schema, key, path, required) {
    const pointer = path.join('/');
    const row = document.createElement('div');
    row.className = 'sf-field';
    row.dataset.pointer = pointer;
    const lbl = document.createElement('span');
    lbl.className = 'sf-label dim';
    lbl.innerHTML = label(key) + (required ? ' <b class="req">*</b>' : '');
    row.appendChild(lbl);

    const parent = () => containerAt(ctx.cfg, path.slice(0, -1));
    const widget = leafWidget(schema, get(ctx.cfg, path), v => {
      const obj = parent();
      if (v === undefined) delete obj[path[path.length - 1]];
      else obj[path[path.length - 1]] = v;
      ctx.onChange(pointer);
    });
    widget.classList.add('sf-input');
    row.appendChild(widget);
    row.insertAdjacentHTML('beforeend', infoBtn(schema));
    return row;
  }

  function get(cfg, path) {
    let o = cfg;
    for (const p of path) { if (o === undefined || o === null) return undefined; o = o[p]; }
    return o;
  }

  /* ── exactly-one-of chooser (tracer_methods, strategy, burial_depth) —
     a dropdown selects the active branch; its fields render below ─────────── */
  function chooser(ctx, path, options, renderBranch, freshBranch, title) {
    const wrap = document.createElement('div');
    wrap.className = 'sf-chooser';
    wrap.dataset.pointer = path.join('/');
    const current = get(ctx.cfg, path) || {};
    const active = options.find(o => current[o] !== undefined) || null;

    const bar = document.createElement('div');
    bar.className = 'row small';
    if (title) {
      const lbl = document.createElement('span');
      lbl.className = 'sf-label dim';
      lbl.textContent = title;
      bar.appendChild(lbl);
    }
    const sel = document.createElement('select');
    sel.className = 'grow';
    if (!active) {
      const ph = document.createElement('option');
      ph.value = '';
      ph.textContent = `— choose ${title || 'an option'} —`;
      sel.appendChild(ph);
    }
    for (const o of options) {
      const op = document.createElement('option');
      op.value = o;
      op.textContent = label(o);
      sel.appendChild(op);
    }
    if (active) sel.value = active;
    sel.onchange = () => {
      if (!sel.value) return;
      const obj = containerAt(ctx.cfg, path.slice(0, -1));
      obj[path[path.length - 1]] = {[sel.value]: freshBranch(sel.value)};
      ctx.onChange(path.join('/'));
      ctx.rerender();
    };
    bar.appendChild(sel);
    wrap.appendChild(bar);
    if (active) wrap.appendChild(renderBranch(active, [...path, active]));
    return wrap;
  }

  /* ── object walker ──────────────────────────────────────────────────────── */
  function renderObject(ctx, schema, root, path, container) {
    const props = schema.properties || {};
    const required = new Set(schema.required || []);
    for (const key of Object.keys(props)) {
      const p = [...path, key];
      const ov = ctx.override(p, props[key], root);
      if (ov === null) continue;                       // suppressed
      if (ov) { container.appendChild(ov); continue }  // custom element
      const [sub, subRoot] = resolve(props[key], root);
      const merged = {...sub, ...props[key]};          // ref-site desc/default win
      delete merged.$ref;
      if (sub.type === 'object' && (sub.properties || sub.$defs)) {
        const g = document.createElement('details');
        g.className = 'sf-group';
        g.dataset.pointer = p.join('/');
        // ALL groups collapsed by default; only a user click (kept in openState
        // across re-renders) opens one
        g.open = ctx.openState[p.join('/')] === true;
        g.innerHTML = `<summary>${label(key)} ${infoBtn(merged)}</summary>`;
        const body = document.createElement('div');
        body.className = 'sec';
        renderObject(ctx, sub, subRoot, p, body);
        g.appendChild(body);
        container.appendChild(g);
      } else {
        container.appendChild(fieldRow(ctx, merged, key, p, required.has(key)));
      }
    }
  }

  /* ── public: render the whole form ──────────────────────────────────────── */
  function render(container, cfg, opts) {
    const openState = {};
    container.querySelectorAll('details[data-pointer]').forEach(d => {
      openState[d.dataset.pointer] = d.open;
    });
    container.innerHTML = '';
    const ctx = {
      cfg,
      onChange: opts.onChange,
      rerender: () => render(container, cfg, opts),
      override: (path, schema, root) => opts.override(ctx, path, schema, root),
      openState,
    };
    renderObject(ctx, SCHEMAS.main, SCHEMAS.main, [], container);
    return ctx;
  }

  return {setSchemas, getSchemas: () => SCHEMAS, resolve, render, fieldRow, chooser,
          renderObject, leafWidget, containerAt, get, label, infoBtn, URN};
})();

/* ── info popover (single, shared) — shows on hover, hides on leave ────────── */
(() => {
  let showTimer = null;
  const show = btn => {
    const pop = $('infopop');
    if (!pop) return;
    pop.textContent = btn.dataset.info;
    pop.style.display = 'block';
    const r = btn.getBoundingClientRect();
    pop.style.left = Math.min(r.left, window.innerWidth - 340) + 'px';
    pop.style.top = (r.bottom + 6) + 'px';
  };
  const hide = () => {
    clearTimeout(showTimer);
    const pop = $('infopop');
    if (pop) pop.style.display = 'none';
  };
  document.addEventListener('mouseover', e => {
    const btn = e.target.closest && e.target.closest('button.info');
    if (btn) {
      clearTimeout(showTimer);
      showTimer = setTimeout(() => show(btn), 120);
    }
  });
  document.addEventListener('mouseout', e => {
    if (e.target.closest && e.target.closest('button.info')) hide();
  });
  document.addEventListener('click', e => {         // touch / keyboard fallback
    const btn = e.target.closest && e.target.closest('button.info');
    if (btn) { show(btn); e.preventDefault(); } else hide();
  });
})();
