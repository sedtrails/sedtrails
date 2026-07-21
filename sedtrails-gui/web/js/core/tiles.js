"use strict";
/* ═════════════════════════ live tile layer ═════════════════════════ */
const TILE_PROVIDERS = {
  sat: {url: (z,x,y) => `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${y}/${x}`,
        maxZoom: 19, attribution: 'Imagery © Esri, Maxar, Earthstar Geographics'},
  gray: {url: (z,x,y) => `https://basemaps.cartocdn.com/light_all/${z}/${x}/${y}.png`,
         maxZoom: 19, attribution: '© CARTO · © OpenStreetMap contributors'},
};
const tileCache = new Map();
let tileFetches = 0;

function tileKey(p, z, x, y) { return `${p}/${z}/${x}/${y}`; }

function requestTile(provider, z, x, y) {
  const key = tileKey(provider, z, x, y);
  if (tileCache.has(key)) return tileCache.get(key);
  if (tileFetches >= 14) return null;
  const entry = {loading: true};
  tileCache.set(key, entry);
  tileFetches++;
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => {
    tileFetches--;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    entry.loading = false;
    entry.tex = tex;
    trimTileCache();
    state.dirty = true;
  };
  img.onerror = () => { tileFetches--; entry.loading = false; entry.failed = true; };
  img.src = TILE_PROVIDERS[provider].url(z, x, y);
  return entry;
}

function trimTileCache() {
  if (tileCache.size <= 400) return;
  const keys = [...tileCache.keys()];
  for (let i = 0; i < keys.length - 320; i++) {
    const e = tileCache.get(keys[i]);
    if (e && e.tex) gl.deleteTexture(e.tex);
    tileCache.delete(keys[i]);
  }
}

function localToWgs(lx, ly) {
  return rdToWgs(lx + state.world0[0], ly + state.world0[1]);
}

function drawTiles() {
  if (!state.showBasemap || !state.world0) { $('attribution').textContent = ''; return; }
  const prov = TILE_PROVIDERS[state.basemap];
  $('attribution').textContent = prov.attribution;

  const [ , latC] = localToWgs(state.view.cx, state.view.cy);
  const mercPerPx = (1 / state.view.ppm) / Math.cos(latC * Math.PI / 180);
  let z = Math.max(3, Math.min(prov.maxZoom, Math.round(Math.log2(156543.03 / mercPerPx))));

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width / dpr, h = canvas.height / dpr;
  let mx0 = Infinity, my0 = Infinity, mx1 = -Infinity, my1 = -Infinity;
  for (const [px, py] of [[0,0],[w,0],[0,h],[w,h]]) {
    const [lx, ly] = screenToWorld(px, py);
    const m = wgsToMerc(...localToWgs(lx, ly));
    mx0 = Math.min(mx0, m[0]); mx1 = Math.max(mx1, m[0]);
    my0 = Math.min(my0, m[1]); my1 = Math.max(my1, m[1]);
  }
  let tx0, tx1, ty0, ty1;
  while (true) {
    const n = 2 ** z, f = n / (2 * MERC_HALF);
    tx0 = Math.floor((mx0 + MERC_HALF) * f); tx1 = Math.floor((mx1 + MERC_HALF) * f);
    ty0 = Math.floor((MERC_HALF - my1) * f); ty1 = Math.floor((MERC_HALF - my0) * f);
    if ((tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= 80 || z <= 3) break;
    z--;
  }

  const draws = new Map();
  for (let x = tx0; x <= tx1; x++) for (let y = ty0; y <= ty1; y++) {
    const n = 2 ** z;
    if (x < 0 || y < 0 || x >= n || y >= n) continue;
    const e = requestTile(state.basemap, z, x, y);
    if (e && e.tex) { draws.set(tileKey(state.basemap, z, x, y), {z, x, y, tex: e.tex}); continue; }
    let az = z, ax = x, ay = y;
    for (let k = 0; k < 5 && az > 3; k++) {
      az--; ax >>= 1; ay >>= 1;
      const a = tileCache.get(tileKey(state.basemap, az, ax, ay));
      if (a && a.tex) { draws.set(tileKey(state.basemap, az, ax, ay), {z: az, x: ax, y: ay, tex: a.tex}); break; }
    }
  }

  const list = [...draws.values()].sort((a, b) => a.z - b.z);
  if (!drawTiles.buf) drawTiles.buf = gl.createBuffer();
  gl.useProgram(progTex.prog);
  gl.uniform4fv(progTex.u.uView, viewUniform());
  gl.activeTexture(gl.TEXTURE0);
  gl.uniform1i(progTex.u.uTex, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, drawTiles.buf);
  gl.enableVertexAttribArray(0); gl.enableVertexAttribArray(1);
  gl.disable(gl.BLEND);
  const w0 = state.world0;
  for (const t of list) {
    const n = 2 ** t.z;
    const tmx0 = t.x / n * 2 * MERC_HALF - MERC_HALF;
    const tmx1 = (t.x + 1) / n * 2 * MERC_HALF - MERC_HALF;
    const tmy1 = MERC_HALF - t.y / n * 2 * MERC_HALF;
    const tmy0 = MERC_HALF - (t.y + 1) / n * 2 * MERC_HALF;
    const corner = (mx, my) => {
      const [x, y] = wgsToRd(...mercToWgs(mx, my));
      return [x - w0[0], y - w0[1]];
    };
    const [sw, se, nw, ne] = [corner(tmx0, tmy0), corner(tmx1, tmy0),
                              corner(tmx0, tmy1), corner(tmx1, tmy1)];
    const quad = new Float32Array([
      sw[0], sw[1], 0, 1,   se[0], se[1], 1, 1,
      nw[0], nw[1], 0, 0,   ne[0], ne[1], 1, 0,
    ]);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.DYNAMIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
    gl.bindTexture(gl.TEXTURE_2D, t.tex);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
  gl.disableVertexAttribArray(1);
  gl.enable(gl.BLEND);
}

