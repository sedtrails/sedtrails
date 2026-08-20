"use strict";
/* ═════════════════════════ global state ═════════════════════════ */
const state = {
  runs: [],
  world0: null,
  polygons: [],                                  // flat list: {name, verts, color}
  rules: [],
  defaultApp: {action: 'tint', color: '#8b939e', colorby: 'burial'},
  defaultAutoSet: false,
  view: {cx: 0, cy: 0, ppm: 0.01},
  clock: {playing: false, t: 0, t0: 0, t1: 1, speed: 0.125},   // 3 h/s default
  pointSize: 3, alpha: 0.85, decim: 1,
  pointCmap: 'viridis',
  ranges: {burial: [null, null], age: [null, null], first: [null, null],
           dist: [null, null]},                  // user overrides (null = auto)
  fade: {bur: false, depth: 0.05, imm: false, alpha: 0.1},
  paths: {show: false, width: 1.5, alpha: 0.6, mode: 'age',
          color: '#1f6fd6', cmap: 'viridis'},
  showBasemap: true,
  basemap: 'sat',
  showModel: false,
  modelLayer: 'bed',
  meshAlpha: 1.0,
  mesh: null,
  showOutlines: true,
  showPoints: true,
  showParticles: true,                           // master eye (Layers panel group)
  showConnectivity: true,
  connectivity: null,                            // last matrix (Connectivity tab)
  tool: 'pan',
  draft: [],
  editSel: null,
  dirty: true,
  selCount: 0,
  exporting: false,
};

const CONDS = [
  ['start', 'start in'], ['end', 'end in'], ['ever', 'ever visit'],
  ['bur_ge', 'are buried ≥ (m)'], ['bur_lt', 'are buried < (m)'], ['always', '— (all)'],
];
const CONDS2 = CONDS.filter(c => c[0] !== 'always');
const ACTIONS = [['hide', 'hidden'], ['color', 'colored'], ['colorby', 'colored by']];
const COLORBYS = [
  ['burial', 'burial depth'], ['age', 'age'], ['first', 'time of first polygon visit'],
];
const DEFAULT_APPS = [
  ['tint', 'in their simulation color'],
  ['color', 'colored'],
  ['colorby', 'colored by'],
  ['hide', 'hidden'],
];

