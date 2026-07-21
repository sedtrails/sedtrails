"use strict";
/* ═════════════════════════ init ═════════════════════════ */
async function init() {
  // entering the Viewer re-checks the config's result file: loads it if new,
  // replaces the loaded run if the file on disk changed since the import.
  // The forcing layers live here too (merged tab) — re-sync them as well.
  Tabs.register('viewer', {enter() {
    if (typeof autoImportResult === 'function') autoImportResult();
    if (typeof ForcingTab !== 'undefined') ForcingTab.enter();
  }});
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
