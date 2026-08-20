"use strict";
/* ═════════════════════════ tab switching (GUI) ═════════════════════════ */
const TAB_TITLES = {
  settings: 'SedTRAILS settings', seeding: 'Populations',
  run: 'Run', viewer: 'Viewer', connectivity: 'Connectivity',
};

const Tabs = {
  current: null,
  hooks: {},                       // name -> {enter(), leave()}
  register(name, h) { this.hooks[name] = h || {}; },
  activate(name) {
    if (name === this.current) return;
    const old = this.current;
    if (old && this.hooks[old] && this.hooks[old].leave) this.hooks[old].leave();
    this.current = name;
    document.body.className = 'tab-' + name;
    document.querySelectorAll('.tabbtn').forEach(b =>
      b.classList.toggle('active', b.dataset.tab === name));
    document.querySelectorAll('.tabpanel').forEach(p =>
      p.classList.toggle('active', p.dataset.tab === name));
    const t = $('paneltitle');
    if (t) t.textContent = TAB_TITLES[name] || name;
    if (this.hooks[name] && this.hooks[name].enter) this.hooks[name].enter();
    if (typeof LayersPanel !== 'undefined') LayersPanel.refresh();
    if (typeof state === 'object') state.dirty = true;
  },
};

document.querySelectorAll('.tabbtn').forEach(b =>
  b.onclick = () => Tabs.activate(b.dataset.tab));
Tabs.activate('settings');
