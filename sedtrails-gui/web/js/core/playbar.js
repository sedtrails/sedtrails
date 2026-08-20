"use strict";
/* ═════════════════════════ playback / clock ═════════════════════════ */
function forcingTimeDays() {
  return (typeof ForcingTab !== 'undefined' && ForcingTab.state.timeDays &&
          ForcingTab.state.timeDays.length) ? ForcingTab.state.timeDays : null;
}

function clockHasSources() {
  return state.runs.length > 0 || !!forcingTimeDays();
}

function updateClockRange() {
  // union of every time source: all particle runs + the open forcing file
  let t0 = Infinity, t1 = -Infinity;
  for (const r of state.runs) {
    const td = r.meta.time_days;
    t0 = Math.min(t0, r.epoch + td[0]);
    t1 = Math.max(t1, r.epoch + td[td.length - 1]);
  }
  const ftd = forcingTimeDays();
  if (ftd) {
    t0 = Math.min(t0, ftd[0]);
    t1 = Math.max(t1, ftd[ftd.length - 1]);
  }
  if (!isFinite(t0)) return;                 // no sources — keep defaults
  state.clock.t0 = t0; state.clock.t1 = t1;
  state.clock.t = Math.min(Math.max(state.clock.t, t0), t1);
  if (!isFinite(state.clock.t)) state.clock.t = t0;
}

function frameAt(run, absDay) {
  const td = run.meta.time_days;
  const local = absDay - run.epoch;
  let lo = 0, hi = td.length - 1;
  if (local <= td[0]) return {k: 0, frac: 0};
  if (local >= td[hi]) return {k: hi, frac: 0};
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (td[mid] <= local) lo = mid; else hi = mid;
  }
  const span = td[hi] - td[lo];
  return {k: lo, frac: span > 0 ? (local - td[lo]) / span : 0};
}

function loadedFrontier() {
  let front = state.clock.t1;
  for (const r of state.runs) {
    if (!r.visible) continue;
    const td = r.meta.time_days;
    const k = Math.max(0, Math.min(r.loaded - 1, td.length - 1));
    if (r.loaded < td.length) front = Math.min(front, r.epoch + td[Math.max(0, k - 1)]);
  }
  return front;
}

function fmtDate(absDay) {
  return new Date(absDay * 86400000).toISOString().slice(0, 10);
}

