"use strict";
/* ═════════════════════════ WebGL setup ═════════════════════════ */
const canvas = $('glcanvas');
const gl = canvas.getContext('webgl2', {antialias: false, alpha: false, depth: false,
                                        stencil: true, preserveDrawingBuffer: true});
if (!gl) document.body.innerHTML = '<p style="padding:40px">WebGL2 is not available in this browser.</p>';

let softwareGL = false;
try {
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : '';
  softwareGL = /swiftshader|llvmpipe|software/i.test(renderer);
} catch (e) {}

function compile(vsSrc, fsSrc) {
  const mk = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
      throw new Error(gl.getShaderInfoLog(s) + '\n' + src);
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const info = gl.getActiveUniform(p, i);
    u[info.name.replace('[0]', '')] = gl.getUniformLocation(p, info.name);
  }
  return {prog: p, u};
}

// Rule encoding:
// cond: 0 always/none, 1 start-in, 2 end-in, 3 ever-in, 4 burial>=, 5 burial<
// arg:  polygon index, -1 = any polygon, -2 = no polygon
// action: 0 hide, 1 solid, 2 by-burial, 3 by-age, 4 by-first (+defaults 5/6/7)
const PT_VS = `#version 300 es
precision highp float; precision highp int;
layout(location=0) in uvec2 aPos;
layout(location=1) in uvec2 aPosNext;
layout(location=2) in uvec2 aIds;
layout(location=3) in uint aFirst;
layout(location=4) in uint aBur0;
layout(location=5) in uint aBur1;
layout(location=6) in uint aMaskV;   // ever visited
layout(location=7) in uint aMask0;   // inside at start
layout(location=8) in uint aMaskE;   // inside at end
layout(location=9) in uint aFlg;     // 1 mobile, 0 immobile, 255 unknown
uniform vec4 uQuant, uView;
uniform float uFrac, uPointSize, uAlpha, uNowDays, uBurMax;
uniform vec3 uFade;                  // x: fade-depth (quantized, <=0 off), y: fade-immobile, z: faded alpha
uniform vec2 uVRbur, uVRage, uVRfirst;
uniform int uNRules;
uniform ivec4 uRuleI[${MAX_RULES}];
uniform vec4 uRuleC[${MAX_RULES}];
uniform vec2 uRuleT[${MAX_RULES}];
uniform int uDefAction;
uniform vec3 uDefColor, uTint;
uniform sampler2D uCmap, uPalette;
out vec4 vColor;
void cull() { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; vColor = vec4(0.0); }
bool maskHit(uint mask, int arg) {
  if (arg == -1) return mask != 0u;
  if (arg == -2) return mask == 0u;
  return (mask & (1u << uint(arg))) != 0u;
}
bool condHit(int cond, int arg, float th, float bur) {
  if (cond == 0) return true;
  if (cond == 4) return bur >= th;
  if (cond == 5) return bur < th;
  return maskHit(cond == 1 ? aMask0 : cond == 2 ? aMaskE : aMaskV, arg);
}
vec3 valueColor(int action, float bur, float first) {
  float v, lo, hi;
  if (action == 2) { v = bur / 65534.0 * uBurMax; lo = uVRbur.x; hi = uVRbur.y; }
  else if (action == 3) { v = uNowDays; lo = uVRage.x; hi = uVRage.y; }
  else {
    if (first >= 65535.0) return vec3(0.545, 0.576, 0.62);
    v = first; lo = uVRfirst.x; hi = uVRfirst.y;
  }
  return texture(uCmap, vec2(clamp((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0), 0.5)).rgb;
}
vec3 paletteColor(int id) {
  if (id == 255) return vec3(0.545, 0.576, 0.62);
  return texelFetch(uPalette, ivec2(id & 31, 0), 0).rgb;
}
void main() {
  if (aPos.x == 65535u) { cull(); return; }
  vec2 p = vec2(aPos) * uQuant.xy + uQuant.zw;
  if (aPosNext.x != 65535u)
    p = mix(p, vec2(aPosNext) * uQuant.xy + uQuant.zw, uFrac);
  float bur = (aBur0 == 65535u) ? 0.0
            : (aBur1 == 65535u) ? float(aBur0)
            : mix(float(aBur0), float(aBur1), uFrac);
  int act = -1;
  vec3 col = vec3(0.0);
  for (int i = 0; i < ${MAX_RULES}; i++) {
    if (i >= uNRules) break;
    if (condHit(uRuleI[i].x, uRuleI[i].y, uRuleT[i].x, bur)
        && condHit(uRuleI[i].z, uRuleI[i].w, uRuleT[i].y, bur)) {
      act = int(uRuleC[i].w); col = uRuleC[i].rgb; break;
    }
  }
  if (act < 0) {
    act = uDefAction;
    col = (act == 5) ? paletteColor(int(aIds.x))
        : (act == 6) ? paletteColor(int(aIds.y))
        : (act == 7) ? uTint : uDefColor;
    if (act >= 5) act = 1;
  }
  if (act == 0) { cull(); return; }
  if (act >= 2) col = valueColor(act, bur, float(aFirst));
  gl_Position = vec4(p * uView.xy + uView.zw, 0.0, 1.0);
  gl_PointSize = uPointSize;
  float a = uAlpha;
  if (uFade.x > 0.0) a *= mix(1.0, uFade.z, clamp(bur / uFade.x, 0.0, 1.0));
  if (uFade.y > 0.5 && aFlg == 0u) a *= uFade.z;
  vColor = vec4(col, a);
}`;
const PT_FS = `#version 300 es
precision mediump float;
in vec4 vColor;
uniform float uRound;
out vec4 frag;
void main() {
  vec3 col = vColor.rgb;
  if (uRound > 0.5) {
    vec2 c = gl_PointCoord - 0.5;
    float r2 = dot(c, c);
    if (r2 > 0.25) discard;
    col *= 1.0 - 0.3 * smoothstep(0.15, 0.25, r2);   // darker rim → particle look
  }
  frag = vec4(col * vColor.a, vColor.a);
}`;

const SEG_VS = `#version 300 es
precision highp float; precision highp int;
layout(location=1) in uvec4 aSeg;
layout(location=2) in uvec2 aTV;
uniform vec4 uQuant, uView;
uniform vec2 uCanvas, uVR;
uniform float uWidthPx, uNow, uPmax;
uniform int uSolid;
uniform vec3 uSolidColor;
uniform sampler2D uCmap;
out vec4 vColor;
void main() {
  if (float(aTV.x) > uNow || aSeg.x == 65535u || aSeg.z == 65535u) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vColor = vec4(0.0); return;
  }
  vec2 p0 = vec2(aSeg.xy) * uQuant.xy + uQuant.zw;
  vec2 p1 = vec2(aSeg.zw) * uQuant.xy + uQuant.zw;
  vec2 c0 = p0 * uView.xy + uView.zw;
  vec2 c1 = p1 * uView.xy + uView.zw;
  vec2 dpx = (c1 - c0) * uCanvas * 0.5;
  float len = max(length(dpx), 1e-4);
  vec2 dir = dpx / len;
  vec2 nrm = vec2(-dir.y, dir.x);
  float along = float(gl_VertexID >> 1);
  float side = float((gl_VertexID & 1) * 2 - 1);
  vec2 px = mix(vec2(0.0), dpx, along)
          + dir * (along * 2.0 - 1.0) * uWidthPx * 0.5
          + nrm * side * uWidthPx * 0.5;
  gl_Position = vec4(c0 + px / (uCanvas * 0.5), 0.0, 1.0);
  vec3 col;
  if (uSolid == 1) col = uSolidColor;
  else {
    float v = float(aTV.y) / 65534.0 * uPmax;
    col = texture(uCmap, vec2(clamp((v - uVR.x) / max(uVR.y - uVR.x, 1e-9), 0.0, 1.0), 0.5)).rgb;
  }
  vColor = vec4(col, 1.0);
}`;
const SEG_FS = `#version 300 es
precision mediump float;
in vec4 vColor; out vec4 frag;
void main() { frag = vec4(vColor.rgb * vColor.a, vColor.a); }`;

const MESH_VS = `#version 300 es
precision highp float;
layout(location=0) in vec2 aPos;
layout(location=1) in float aVal;
uniform vec4 uView;
uniform vec2 uOff;
out float vVal;
void main() {
  vVal = aVal;
  gl_Position = vec4((aPos + uOff) * uView.xy + uView.zw, 0.0, 1.0);
}`;
const MESH_FS = `#version 300 es
precision mediump float;
in float vVal;
uniform vec2 uClim;
uniform float uAlpha;
uniform sampler2D uRamp;
out vec4 frag;
void main() {
  if (vVal < -9000.0) discard;
  float f = clamp((vVal - uClim.x) / (uClim.y - uClim.x), 0.0, 1.0);
  vec4 c = texture(uRamp, vec2(f, 0.5));
  frag = vec4(c.rgb * uAlpha, uAlpha);
}`;

const FLAT_VS = `#version 300 es
layout(location=0) in vec2 aPos;
uniform vec4 uView;
uniform float uPtSize;   // px; only POINTS draws care — set it before every POINTS draw
void main() { gl_Position = vec4(aPos * uView.xy + uView.zw, 0.0, 1.0); gl_PointSize = uPtSize; }`;
const FLAT_FS = `#version 300 es
precision mediump float;
uniform vec4 uColor;
uniform float uRound;    // 1 = round point with darker rim, 2 = hollow ring (seeding previews)
out vec4 frag;
void main() {
  vec3 col = uColor.rgb;
  float a = uColor.a;
  if (uRound > 0.5) {
    vec2 c = gl_PointCoord - 0.5;
    float r2 = dot(c, c);
    if (r2 > 0.25) discard;
    if (uRound > 1.5) a *= smoothstep(0.07, 0.11, r2);   // hollow center
    col *= 1.0 - 0.35 * smoothstep(0.14, 0.25, r2);
  }
  frag = vec4(col * a, a);
}`;

const TEX_VS = `#version 300 es
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
uniform vec4 uView;
out vec2 vUV;
void main() { vUV = aUV; gl_Position = vec4(aPos * uView.xy + uView.zw, 0.0, 1.0); }`;
const TEX_FS = `#version 300 es
precision mediump float;
in vec2 vUV; uniform sampler2D uTex; out vec4 frag;
void main() { vec4 c = texture(uTex, vUV); frag = vec4(c.rgb, 1.0); }`;

const SCREEN_VS = `#version 300 es
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main() { vUV = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;
const BLIT_FS = `#version 300 es
precision mediump float;
in vec2 vUV; uniform sampler2D uTex; uniform float uAlpha; out vec4 frag;
void main() { frag = texture(uTex, vUV) * uAlpha; }`;

const progPt = compile(PT_VS, PT_FS);
const progSeg = compile(SEG_VS, SEG_FS);
const progMesh = compile(MESH_VS, MESH_FS);
const progFlat = compile(FLAT_VS, FLAT_FS);
const progTex = compile(TEX_VS, TEX_FS);
const progBlit = compile(SCREEN_VS, BLIT_FS);

function makeRampTexture(stops) {
  const tex = gl.createTexture();
  const px = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const f = i / 255 * (stops.length - 1);
    const j = Math.min(Math.floor(f), stops.length - 2);
    const w = f - j;
    for (let k = 0; k < 3; k++)
      px[i*4+k] = Math.round(stops[j][k] * (1-w) + stops[j+1][k] * w);
    px[i*4+3] = 255;
  }
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, px);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  return tex;
}
const cmapTextures = {};
function cmapTex(name) {
  if (!cmapTextures[name]) cmapTextures[name] = makeRampTexture(CMAPS[name] || CMAPS.viridis);
  return cmapTextures[name];
}

const paletteTex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, paletteTex);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
function rebuildPaletteTex() {
  // palette slots = polygon colors, so palette-by-polygon coloring matches
  // each polygon's own color
  const polys = allPolys();
  const px = new Uint8Array(32 * 4);
  for (let i = 0; i < 32; i++) {
    const rgb = hex2rgb(polys[i] && polys[i].color ? polys[i].color : PALETTE[i % PALETTE.length]);
    px[i*4] = rgb[0]; px[i*4+1] = rgb[1]; px[i*4+2] = rgb[2]; px[i*4+3] = 255;
  }
  gl.bindTexture(gl.TEXTURE_2D, paletteTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 32, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, px);
}

const screenQuad = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, screenQuad);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);

const off = {fbo: gl.createFramebuffer(), tex: gl.createTexture(), w: 0, h: 0};
function ensureOffFBO() {
  if (off.w === canvas.width && off.h === canvas.height) return;
  off.w = canvas.width; off.h = canvas.height;
  gl.bindTexture(gl.TEXTURE_2D, off.tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, off.w, off.h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.bindFramebuffer(gl.FRAMEBUFFER, off.fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, off.tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

/* ═════════════════════════ swatch color picker ═════════════════════════ */
let swatchCb = null;
{
  const grid = document.querySelector('#swatchpop .grid');
  for (const c of SWATCHES) {
    const d = document.createElement('div');
    d.className = 'sw';
    d.style.background = c;
    d.onclick = () => { if (swatchCb) swatchCb(c); closeSwatchPop(); };
    grid.appendChild(d);
  }
  document.querySelector('#swatchpop .more').onclick = () => {
    const inp = $('swatch-custom');
    inp.oninput = () => { if (swatchCb) swatchCb(inp.value); };
    inp.click();
    closeSwatchPop();
  };
  window.addEventListener('mousedown', e => {
    if (!$('swatchpop').contains(e.target) && !e.target.classList.contains('colorbtn'))
      closeSwatchPop();
  });
}
function closeSwatchPop() { $('swatchpop').style.display = 'none'; swatchCb = null; }
function mkColorBtn(getColor, setColor, title) {
  const b = document.createElement('span');
  b.className = 'colorbtn';
  b.style.background = getColor();
  b.title = title || 'change color';
  b.onclick = ev => {
    ev.stopPropagation();
    const pop = $('swatchpop');
    pop.style.display = 'block';
    const r = b.getBoundingClientRect();
    pop.style.left = Math.min(r.left, window.innerWidth - 190) + 'px';
    pop.style.top = Math.min(r.bottom + 4, window.innerHeight - 140) + 'px';
    swatchCb = c => { setColor(c); b.style.background = c; };
  };
  return b;
}

