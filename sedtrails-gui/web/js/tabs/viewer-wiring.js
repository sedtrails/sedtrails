"use strict";
/* ═════════════════════════ playback UI ═════════════════════════ */
function togglePlay() {
  if (!clockHasSources()) return;
  const c = state.clock;
  if (!c.playing && c.t >= c.t1 - 1e-9) c.t = c.t0;
  c.playing = !c.playing;
  refreshPlayBtn();
  state.dirty = true;
}
function refreshPlayBtn() { $('btn-play').textContent = state.clock.playing ? '⏸' : '▶'; }
function stepDays(d) {
  const c = state.clock;
  c.playing = false; refreshPlayBtn();
  c.t = Math.min(Math.max(c.t + d, c.t0), Math.min(c.t1, loadedFrontier()));
  updateTimelineUI(); state.dirty = true;
}
function updateTimelineUI() {
  const c = state.clock;
  const span = Math.max(c.t1 - c.t0, 1e-9);
  $('timeline').value = Math.round((c.t - c.t0) / span * 1000);
  // track: solid accent up to the playhead, muted accent to the loaded
  // frontier (data still streaming in), plain track beyond
  const pos = Math.max(0, Math.min(100, (c.t - c.t0) / span * 100));
  const front = Math.max(pos, (loadedFrontier() - c.t0) / span * 100);
  const loadedCol = 'color-mix(in srgb, var(--accent) 30%, var(--border))';
  $('timeline').style.setProperty('--tl-bg',
    `linear-gradient(90deg, var(--accent) 0%, var(--accent) ${pos}%, ` +
    `${loadedCol} ${pos}%, ${loadedCol} ${front}%, var(--border) ${front}%)`);
  updateCoverageBands(span);
  const day = Math.floor(c.t - c.t0) + 1;
  const total = Math.ceil(span);
  const hm = new Date(c.t * 86400000).toISOString().slice(11, 16);
  $('datelabel').textContent = `${fmtDate(c.t)} ${hm} · day ${day}/${total}`;
}

/* labeled bands above the slider: which part of the clock range each time
   source covers (forcing may repeat — shown as a faint full-width band around
   the solid native span). Hover a band for the exact date range. */
function updateCoverageBands(span) {
  const c = state.clock;
  const pct = t => Math.max(0, Math.min(100, (t - c.t0) / span * 100));
  const band = (el, a, b, color, repeats, title) => {
    const has = a != null;
    el.classList.toggle('on', has);
    if (!has) { el.style.background = 'transparent'; el.title = ''; return; }
    const faint = repeats ? `color-mix(in srgb, ${color} 25%, transparent)`
                          : 'transparent';
    const solid = `color-mix(in srgb, ${color} 55%, transparent)`;
    el.style.background = `linear-gradient(90deg, ${faint} 0%, ${faint} ${a}%, ` +
      `${solid} ${a}%, ${solid} ${b}%, ${faint} ${b}%, ${faint} 100%)`;
    el.title = title;
  };
  const ftd = typeof forcingTimeDays === 'function' ? forcingTimeDays() : null;
  if (ftd) {
    const cfg = typeof SettingsTab !== 'undefined' ? SettingsTab.getConfig() : null;
    const repeats = !!(cfg && cfg.inputs && cfg.inputs.repeat_eulerian_fields !== false);
    band($('tl-cov-frc'), pct(ftd[0]), pct(ftd[ftd.length - 1]), 'var(--ok)', repeats,
         `Forcing data: ${fmtDate(ftd[0])} → ${fmtDate(ftd[ftd.length - 1])}` +
         (repeats ? '\nRepeats (is cycled) outside this span — faint part of the band.'
                  : '\nNo forcing outside this span.'));
  } else {
    band($('tl-cov-frc'), null);
  }
  if (state.runs.length) {
    let p0 = Infinity, p1 = -Infinity;
    for (const r of state.runs) {
      const td = r.meta.time_days;
      p0 = Math.min(p0, r.epoch + td[0]);
      p1 = Math.max(p1, r.epoch + td[td.length - 1]);
    }
    band($('tl-cov-par'), pct(p0), pct(p1), 'var(--warn)', false,
         `Particle results: ${fmtDate(p0)} → ${fmtDate(p1)}`);
  } else {
    band($('tl-cov-par'), null);
  }
}
$('btn-play').onclick = togglePlay;
$('btn-step-f').onclick = () => stepDays(1);
$('btn-step-b').onclick = () => stepDays(-1);
$('timeline').oninput = () => {
  const c = state.clock;
  c.playing = false; refreshPlayBtn();
  c.t = c.t0 + (c.t1 - c.t0) * $('timeline').value / 1000;
  c.t = Math.min(c.t, loadedFrontier());
  updateTimelineUI(); state.dirty = true;
};
$('rng-speed').oninput = () => {
  // log-mapped: slider 0..1 → 1 h/s .. 30 days/s (default 3 h/s)
  const f = parseFloat($('rng-speed').value);
  const lo = 1 / 24, hi = 30;
  const speed = Math.pow(10, Math.log10(lo) + (Math.log10(hi) - Math.log10(lo)) * f);
  state.clock.speed = speed;
  $('lbl-speed').textContent = speed >= 1 ?
    speed.toFixed(speed < 10 ? 1 : 0) + ' d/s' : (speed * 24).toFixed(1) + ' h/s';
};

/* ═════════════════════════ control wiring ═════════════════════════ */
$('rng-size').oninput = () => { state.pointSize = parseFloat($('rng-size').value); state.dirty = true; };
$('rng-alpha').oninput = () => { state.alpha = parseFloat($('rng-alpha').value); state.dirty = true; };
$('sel-decim').onchange = () => {
  state.decim = parseInt($('sel-decim').value);
  schedulePathRebuild(); state.dirty = true;
};
$('rng-pwidth').oninput = () => { state.paths.width = parseFloat($('rng-pwidth').value); state.dirty = true; };
$('rng-palpha').oninput = () => { state.paths.alpha = parseFloat($('rng-palpha').value); state.dirty = true; };
$('sel-pathcolor').onchange = () => {
  state.paths.mode = $('sel-pathcolor').value;
  $('pathcolor-swatch').style.display = state.paths.mode === 'solid' ? '' : 'none';
  if (state.paths.mode === 'burial') state.runs.forEach(streamBurial);
  schedulePathRebuild(); refreshScaleRows(); updateColorbars(); state.dirty = true;
};
$('pathcolor-swatch').appendChild(mkColorBtn(() => state.paths.color,
  c => { state.paths.color = c; state.dirty = true; }));
$('pathcolor-swatch').style.display = 'none';
$('sel-basemap').onchange = () => { state.basemap = $('sel-basemap').value; state.dirty = true; };
$('sel-model').onchange = () => { state.modelLayer = $('sel-model').value; state.dirty = true; };
$('rng-mesh-alpha').oninput = () => { state.meshAlpha = parseFloat($('rng-mesh-alpha').value); state.dirty = true; };

/* eye show/hide toggles — hidden = greyed-out related controls, stable layout */
function syncVis() {
  const eyes = [['eye-basemap', state.showBasemap], ['eye-model', state.showModel],
                ['eye-poly', state.showOutlines], ['eye-points', state.showPoints],
                ['eye-paths', state.paths.show]];
  for (const [id, on] of eyes) $(id).classList.toggle('off', !on);
  $('sel-basemap').classList.toggle('grey', !state.showBasemap);
  $('sel-model').classList.toggle('grey', !state.showModel || !state.mesh);
  $('row-mesh-alpha').classList.toggle('grey', !state.showModel || !state.mesh);
  $('poly-opts').classList.toggle('grey', !state.showOutlines);
  $('points-opts').classList.toggle('grey', !state.showPoints);
  $('paths-opts').classList.toggle('grey', !state.paths.show);
  $('row-fade-alpha').classList.toggle('grey', !(state.fade.bur || state.fade.imm));
}
function wireEye(id, toggle) {
  $(id).onclick = ev => {
    ev.preventDefault(); ev.stopPropagation();   // don't collapse the section
    toggle(); syncVis(); state.dirty = true;
  };
}
wireEye('eye-basemap', () => { state.showBasemap = !state.showBasemap; });
wireEye('eye-model', () => { state.showModel = !state.showModel; state.modelAutoSet = true; });
wireEye('eye-poly', () => { state.showOutlines = !state.showOutlines; });
wireEye('eye-points', () => { state.showPoints = !state.showPoints; });
wireEye('eye-paths', () => {
  state.paths.show = !state.paths.show;
  if (state.paths.show) { schedulePathRebuild(); if (rulesNeedBurial()) state.runs.forEach(streamBurial); }
  refreshScaleRows(); updateColorbars(); updatePathsNote();
});

/* fading of buried / immobile particles */
$('chk-fade-bur').onchange = () => {
  state.fade.bur = $('chk-fade-bur').checked;
  if (state.fade.bur) state.runs.forEach(streamBurial);
  syncVis(); state.dirty = true;
};
$('inp-fade-depth').onchange = () => {
  state.fade.depth = Math.max(parseFloat($('inp-fade-depth').value) || 0.05, 0.001);
  state.dirty = true;
};
$('chk-fade-imm').onchange = () => {
  state.fade.imm = $('chk-fade-imm').checked;
  if (state.fade.imm) state.runs.forEach(streamFlags);
  syncVis(); state.dirty = true;
};
$('rng-fade-alpha').oninput = () => {
  state.fade.alpha = parseFloat($('rng-fade-alpha').value); state.dirty = true;
};
$('btn-hide-side').onclick = () => {
  $('sidebar').classList.add('hidden');
  $('sidebtn').style.display = 'block';
  $('sideresize').style.display = 'none';
};
$('sidebtn').onclick = () => {
  $('sidebar').classList.remove('hidden');
  $('sidebtn').style.display = 'none';
  $('sideresize').style.display = '';
};

/* resizable sidebar: drag the divider; the width persists and applies on every
   tab (the sidebar width is a CSS variable, --sidebar-w) */
(() => {
  const rs = $('sideresize');
  if (!rs) return;
  const clamp = w => Math.max(320, Math.min(w, window.innerWidth - 380));
  const apply = w => document.documentElement.style.setProperty('--sidebar-w', w + 'px');
  const saved = parseInt(localStorage.getItem('stgui_sidebar_w'), 10);
  if (isFinite(saved)) apply(clamp(saved));
  rs.onmousedown = e => {
    e.preventDefault();
    rs.classList.add('dragging');
    const move = ev => apply(clamp(ev.clientX));
    const up = ev => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
      rs.classList.remove('dragging');
      localStorage.setItem('stgui_sidebar_w', String(clamp(ev.clientX)));
    };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  };
})();

