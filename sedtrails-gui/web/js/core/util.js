"use strict";
/* ═════════════════════════ helpers & constants ═════════════════════════ */
const $ = id => document.getElementById(id);
const SENT = 65535;
const PALETTE = ['#3d7fd1','#f0842c','#3aa35c','#d14b4b','#9166cf','#d668b5',
                 '#b0a028','#2ba7ae','#8a5f4a','#6e86d0','#54b389','#d98ba8',
                 '#a8935e','#79b23f','#e0937c','#7f8a96'];
const SWATCHES = ['#d14b4b','#f0842c','#e8c531','#3aa35c','#2ba7ae','#3d7fd1','#9166cf','#d668b5',
                  '#8c2f2f','#a85a1c','#9a8420','#256e3d','#1d7076','#28558c','#61458a','#93477c',
                  '#f2a3a3','#f7c797','#f2e394','#a3d9b5','#9fd6da','#a9c7ec','#c8b5e8','#e8b5d8',
                  '#111111','#555555','#999999','#cccccc','#ffffff','#7a5c45','#c9b380','#8fd14f'];
const SIM_TINTS = ['#1f6fd6','#f0842c','#3aa35c','#d668b5','#b0a028','#9166cf'];
const NEUTRAL = '#8b939e';
const CMAPS = {
  viridis: [[68,1,84],[72,36,117],[65,68,135],[53,95,141],[42,120,142],
    [33,145,140],[34,168,132],[68,190,112],[122,209,81],[189,223,38],[253,231,37]],
  turbo: [[48,18,59],[70,107,227],[40,187,235],[62,240,161],[162,252,60],
    [241,204,32],[249,112,21],[196,31,6],[122,4,3]],
  plasma: [[13,8,135],[84,2,163],[139,10,165],[185,50,137],[219,92,104],
    [244,136,73],[254,188,43],[240,249,33]],
  RdBu: [[103,0,31],[178,24,43],[214,96,77],[244,165,130],[253,219,199],[247,247,247],
    [209,229,240],[146,197,222],[67,147,195],[33,102,172],[5,48,97]].reverse(),
  gray: [[45,45,45],[235,235,235]],
  // terrain-like for bed level: deep blue → shallow cyan → sand → vegetated land
  topo: [[8,40,88],[28,84,146],[60,140,190],[125,195,215],[190,230,225],
    [240,232,180],[210,200,120],[150,160,90],[120,120,70]],
  // cyclic (first stop == last stop) — for direction/phase maps on [-180, 180]
  phase: [[201,58,180],[212,68,70],[190,133,38],[131,178,56],[47,187,145],
    [58,153,214],[121,95,222],[201,58,180]],
};
const MAX_RULES = 16;

function hex2rgb(h) {
  return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
}
function fmtInt(n) { return n.toLocaleString('en-US'); }
function debounce(fn, ms) {
  let t = null;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
function toast(msg, isErr) {
  const el = document.createElement('div');
  el.className = 'toast' + (isErr ? ' err' : '');
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), isErr ? 6000 : 3500);
  if (isErr) console.error(msg);
}
async function api(route, body) {
  const opt = body === undefined ? {} :
    {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)};
  const r = await fetch(route, opt);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).error || msg; } catch (e) {}
    throw new Error(msg);
  }
  return r;
}
const apiJson = async (route, body) => (await api(route, body)).json();

/* ── RD New (EPSG:28992) ⇄ WGS84 ⇄ WebMercator (±0.3 m) ── */
function rdToWgs(x, y) {
  const dX = (x - 155000) * 1e-5, dY = (y - 463000) * 1e-5;
  const somN = 3235.65389*dY - 32.58297*dX*dX - 0.24750*dY*dY - 0.84978*dX*dX*dY
    - 0.06550*dY*dY*dY - 0.01709*dX*dX*dY*dY - 0.00738*dX + 0.00530*dX*dX*dX*dX
    - 0.00039*dX*dX*dY*dY*dY + 0.00033*dX*dX*dX*dX*dY - 0.00012*dX*dY;
  const somE = 5260.52916*dX + 105.94684*dX*dY + 2.45656*dX*dY*dY - 0.81885*dX*dX*dX
    + 0.05594*dX*dY*dY*dY - 0.05607*dX*dX*dX*dY + 0.01199*dY - 0.00256*dX*dX*dX*dY*dY
    + 0.00128*dX*dY*dY*dY*dY + 0.00022*dY*dY - 0.00022*dX*dX + 0.00026*dX*dX*dX*dX*dX;
  return [5.38720621 + somE/3600, 52.15517440 + somN/3600];
}
function wgsToRd(lon, lat) {
  const dp = 0.36*(lat - 52.15517440), dl = 0.36*(lon - 5.38720621);
  const x = 155000 + 190094.945*dl - 11832.228*dp*dl - 114.221*dp*dp*dl - 32.391*dl*dl*dl
    - 0.705*dp - 2.340*dp*dp*dp*dl - 0.608*dp*dl*dl*dl - 0.008*dl*dl + 0.148*dp*dp*dl*dl*dl;
  const y = 463000 + 309056.544*dp + 3638.893*dl*dl + 73.077*dp*dp - 157.984*dp*dl*dl
    + 59.788*dp*dp*dp + 0.433*dl - 6.439*dp*dp*dl*dl - 0.032*dp*dl + 0.092*dl*dl*dl*dl
    - 0.054*dp*dl*dl*dl*dl;
  return [x, y];
}
const EARTH_R = 6378137, MERC_HALF = Math.PI * EARTH_R;
function wgsToMerc(lon, lat) {
  return [EARTH_R * lon * Math.PI / 180,
          EARTH_R * Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360))];
}
function mercToWgs(mx, my) {
  return [mx / EARTH_R * 180 / Math.PI,
          (2 * Math.atan(Math.exp(my / EARTH_R)) - Math.PI / 2) * 180 / Math.PI];
}

