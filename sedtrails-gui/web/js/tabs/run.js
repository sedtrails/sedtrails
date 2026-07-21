"use strict";
/* ═════════════════════════ Run tab ═════════════════════════
   Launch `sedtrails run` on the loaded config, stream the log, show progress
   (output-slot count from the streaming netCDF) with an ETA, and hand the
   finished result straight to the Viewer tab. */

const RunTab = (() => {
  let pollTimer = null;
  let logOffset = 0;
  let lastState = null;
  let autoImported = false;   // auto-import fires once per run

  const el = id => $(id);
  const fmtBytes = b => b == null ? '?' :
    b > 1024**3 ? (b / 1024**3).toFixed(1) + ' GB' :
    b > 1024**2 ? (b / 1024**2).toFixed(1) + ' MB' : Math.round(b / 1024) + ' kB';
  const fmtDur = s => {
    if (s == null) return '–';
    s = Math.round(s);
    const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), ss = s % 60;
    return (h ? h + 'h ' : '') + (h || m ? m + 'm ' : '') + ss + 's';
  };

  /* ── summary card ───────────────────────────────────────────────────────── */
  async function refreshSummary() {
    const cfg = SettingsTab.getConfig();
    const path = SettingsTab.getPath();
    const box = el('run-summary');
    if (!cfg || !path) {
      box.innerHTML = '<div class="placeholder">Load (and save) a configuration in the ' +
        'Settings tab first — the run uses the yaml file on disk.</div>';
      el('run-start').disabled = true;
      return;
    }
    el('run-start').disabled = false;
    let est = null;
    try { est = await apiJson('/api/config/estimate', {config: SettingsTab.getPruned()}); }
    catch (e) {}
    const pops = (cfg.particles && cfg.particles.populations || [])
      .map(p => p.name || '?').join(', ');
    box.innerHTML =
      `<div class="row small"><span class="dim" style="width:78px">Config</span><span class="grow" style="word-break:break-all">${path}</span></div>` +
      `<div class="row small"><span class="dim" style="width:78px">Populations</span><span>${pops || '–'}</span></div>` +
      `<div class="row small"><span class="dim" style="width:78px">Duration</span><span>${(cfg.time && cfg.time.duration) || '(forcing length)'}</span></div>` +
      `<div class="row small"><span class="dim" style="width:78px">Est. output</span><span>` +
      (est && est.bytes ? `${fmtInt(est.particles)} particles × ${fmtInt(est.slots)} slots ≈ ${fmtBytes(est.bytes)}` : 'unknown') +
      `</span></div>` +
      (SettingsTab.isDirty() ? '<div class="hint" style="color:var(--warn)">⚠ unsaved changes — save first, the run reads the file on disk</div>' : '');
  }

  /* ── run control ────────────────────────────────────────────────────────── */
  async function start() {
    const path = SettingsTab.getPath();
    if (!path) return toast('save the configuration first', true);
    if (SettingsTab.isDirty() && !confirm('The config has unsaved changes; the run uses the file on disk. Start anyway?')) return;
    try {
      await apiJson('/api/run/start', {config_path: path});
      el('run-log').textContent = '';
      logOffset = 0;
      autoImported = false;
      startPolling();
    } catch (e) { toast('start failed: ' + e.message, true); }
  }

  async function stop() {
    if (!confirm('Stop the running simulation? The output file may be left incomplete.')) return;
    try { await apiJson('/api/run/stop', {}); } catch (e) { toast(String(e), true); }
  }

  function startPolling() {
    clearInterval(pollTimer);
    pollTimer = setInterval(poll, 1000);
    poll();
  }

  async function poll() {
    let st;
    try { st = await apiJson('/api/run/status'); }
    catch (e) { return; }
    const running = st.state === 'running';
    el('run-start').style.display = running ? 'none' : '';
    el('run-stop').style.display = running ? '' : 'none';
    el('run-state').textContent = st.state === 'idle' ? '' :
      `${st.state}${st.returncode !== null && st.state !== 'running' ? ` (exit ${st.returncode})` : ''}` +
      (st.elapsed ? ` · elapsed ${fmtDur(st.elapsed)}` : '');
    el('run-state').style.color =
      st.state === 'error' ? 'var(--err)' : st.state === 'done' ? 'var(--ok)' : 'var(--dim)';

    const p = st.progress;
    const bar = el('run-prog');
    if (p && st.state !== 'idle') {
      bar.style.display = '';
      bar.firstElementChild.style.width = (st.state === 'done' ? 100 : (p.pct || 0)) + '%';
      // always show whatever progress source is available (slots > log % > bytes)
      const parts = [];
      if (p.pct != null) parts.push(`${(st.state === 'done' ? 100 : p.pct).toFixed(0)}%`);
      if (p.written_slots != null && p.total_slots)
        parts.push(`output slot ${p.written_slots}/${p.total_slots}`);
      else if (p.bytes != null)
        parts.push(`output ${fmtBytes(p.bytes)}` + (p.est_bytes ? ` of ≈${fmtBytes(p.est_bytes)}` : ''));
      if (running && p.eta_s != null) parts.push(`~${fmtDur(p.eta_s)} remaining`);
      el('run-progtxt').textContent = parts.join(' · ');
    } else {
      bar.style.display = 'none';
      el('run-progtxt').textContent = '';
    }

    try {
      const lg = await apiJson('/api/run/log?offset=' + logOffset);
      if (lg.lines.length) {
        logOffset = lg.offset;
        const pre = el('run-log');
        const stick = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 20;
        pre.textContent += lg.lines.join('\n') + '\n';
        if (pre.textContent.length > 400000) pre.textContent = pre.textContent.slice(-300000);
        if (stick) pre.scrollTop = pre.scrollHeight;
      }
    } catch (e) {}

    el('run-import').style.display = st.state === 'done' && st.result_path ? '' : 'none';
    el('run-import').onclick = () => importResult(st);

    if (!running && lastState === 'running') {
      toast(st.state === 'done' ? 'Simulation finished ✓' :
            st.state === 'stopped' ? 'Simulation stopped' : 'Simulation failed — see log', st.state === 'error');
      refreshSummary();
      // finished cleanly → results go straight into the viewer
      if (st.state === 'done' && st.result_path && !autoImported) {
        autoImported = true;
        importResult(st);
      }
    }
    lastState = st.state;
    if (!running && st.state !== 'idle') { clearInterval(pollTimer); pollTimer = null; }
  }

  /* silent import: no modal — the GUI already knows result path, forcing and name.
     A rerun replaces the previously imported run of the same name (update). */
  async function importResult(st) {
    Tabs.activate('viewer');
    const frc = (typeof ForcingTab !== 'undefined' && ForcingTab.state.meta) ?
      ForcingTab.state.meta.path : null;
    const cfgName = (st.config_path || '').split(/[\\/]/).pop().replace(/\.ya?ml$/i, '') || 'run';
    toast('Importing result…');
    try {
      const {job_id} = await apiJson('/api/import',
        {nc_path: st.result_path, forcing_path: frc, name: cfgName});
      for (;;) {
        await new Promise(r => setTimeout(r, 500));
        const j = await apiJson('/api/job/' + job_id);
        if (j.status === 'error') throw new Error(j.message);
        if (j.status === 'unknown') throw new Error('server restarted during import — retry');
        if (j.status === 'done') {
          state.runs.filter(r => r.meta.run_name === cfgName).forEach(removeRun);
          await loadRun(j.bundle, 1);
          toast('Result loaded ✓');
          break;
        }
      }
    } catch (e) { toast('Import failed: ' + e.message, true); }
  }

  let wired = false;
  async function enter() {
    if (!wired) {
      el('run-start').onclick = start;
      el('run-stop').onclick = stop;
      wired = true;
    }
    await SettingsTab.init();
    refreshSummary();
    startPolling();
  }

  Tabs.register('run', {enter, leave() {
    // keep polling while a run is active so the toast still fires elsewhere
    if (lastState !== 'running') { clearInterval(pollTimer); pollTimer = null; }
  }});
  return {refreshSummary};
})();
