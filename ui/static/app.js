import * as THREE from "three";
import { ParametricGeometry } from "three/addons/geometries/ParametricGeometry.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);

const els = {
  form: $("form"),
  btnPredict: $("btn-predict"),
  btnLoadExample: $("btn-load-example"),
  driving: $("driving"),
  p: $("p"),
  all: $("all"),
  status: $("status"),
  canvas: $("c"),
  chartHr: $("chart-hr"),
  chartBp: $("chart-bp"),
  chartCo: $("chart-co"),
};

let heartMaterials = [];

let liveSim = null;

// Served from ui/static/
// const GLB_URL = "a_heart_model_with_arteries_clean.glb";
const GLB_URL = "realistic_human_heart.glb";

function setStatus(s) {
  els.status.textContent = s;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function expEase(start, target, tSeconds, tauSeconds) {
  // First-order response: start -> target
  const tau = Math.max(1e-6, tauSeconds);
  const k = 1 - Math.exp(-tSeconds / tau);
  return start + (target - start) * k;
}

// =============================
// HeartbeatPulse (Option A)
// Self-contained so you can disable/remove it easily.
// - To disable: set ENABLE_HEARTBEAT_PULSE = false
// - To delete: remove this block + the 3 call sites marked "HeartbeatPulse call site"
// =============================
const ENABLE_HEARTBEAT_PULSE = true;

class HeartbeatPulse {
  constructor({ bpm = 72, amplitude = 0.045 } = {}) {
    this.bpm = bpm;
    this.amplitude = amplitude;
    this.object = null;
    this.baseScale = null;
    this.startTime = null;
  }

  attach(object3d, nowSeconds = 0) {
    if (!object3d) return;
    this.object = object3d;
    this.baseScale = object3d.scale.clone();
    this.startTime = nowSeconds;
  }

  detach() {
    if (this.object && this.baseScale) {
      this.object.scale.copy(this.baseScale);
    }
    this.object = null;
    this.baseScale = null;
    this.startTime = null;
  }

  setBpm(bpm) {
    const v = Number(bpm);
    if (!Number.isFinite(v)) return;
    this.bpm = clamp(v, 30, 180);
  }

  update(nowSeconds) {
    if (!this.object || !this.baseScale) return;

    const t = Number(nowSeconds);
    if (!Number.isFinite(t)) return;

    const bpm = clamp(this.bpm, 30, 180);
    const omega = (2 * Math.PI * bpm) / 60;
    const phase = omega * (t - (this.startTime ?? 0));

    // 0..1 pulse
    const s = (Math.sin(phase) + 1) / 2;
    // Make systole sharper than diastole
    const pulse = Math.pow(s, 2.4);

    const a = clamp(this.amplitude, 0, 0.15);

    // Non-uniform scale looks more like contraction than uniform "balloon" scaling,
    // but is still pure scale-based (Option A).
    const sx = 1 + a * pulse;
    const sy = 1 - a * 0.55 * pulse;
    const sz = 1 + a * pulse;

    this.object.scale.set(this.baseScale.x * sx, this.baseScale.y * sy, this.baseScale.z * sz);
  }
}

let heartbeatPulse = ENABLE_HEARTBEAT_PULSE ? new HeartbeatPulse() : null;

function makeField({ key, label, help, min, max, step, defaultValue }) {
  const wrap = document.createElement("div");
  wrap.className = "field";

  const l = document.createElement("label");
  l.textContent = label;

  const input = document.createElement("input");
  input.type = "number";
  input.step = String(step ?? 1);
  if (min != null) input.min = String(min);
  if (max != null) input.max = String(max);
  input.value = String(defaultValue ?? "");
  input.inputMode = "decimal";
  input.dataset.key = key;

  wrap.appendChild(l);
  wrap.appendChild(input);

  if (help) {
    const hint = document.createElement("div");
    hint.className = "hint";
    hint.textContent = help;
    wrap.appendChild(hint);
  }

  return wrap;
}

const INPUTS = [
  {
    key: "anesthesia",
    label: "Anesthesia drugs (0–100%)",
    help: "Higher -> HR relaxes/reduces, BP reduces slightly.",
    min: 0,
    max: 100,
    step: 1,
    defaultValue: 30,
  },
  {
    key: "oxygen",
    label: "Oxygen cylinder supply (0–100%)",
    help: "Lower -> oxygenation worsens.",
    min: 0,
    max: 100,
    step: 1,
    defaultValue: 100,
  },
  {
    key: "blood_loss",
    label: "Blood loss (mL)",
    help: "Higher -> BP drops; HR compensates. >2500–3000 mL can decompensate.",
    min: 0,
    max: 4000,
    step: 25,
    defaultValue: 0,
  },
];

const SIM_DURATION_S = 180; // show 1–3 min edge behaviors
const SIM_DT_S = 1;

function logistic01(x) {
  const v = Number(x);
  if (!Number.isFinite(v)) return 0;
  return 1 / (1 + Math.exp(-v));
}

function computeVitalsAtTime({ anesthesia, oxygen, blood_loss }, tSeconds) {
  const t = Math.max(0, Number(tSeconds) || 0);

  const a = clamp((anesthesia ?? 0) / 100, 0, 1);
  const o = clamp((oxygen ?? 100) / 100, 0, 1);
  const bloodMl = clamp(Number(blood_loss ?? 0), 0, 4000);

  const bMild = clamp(bloodMl / 2000, 0, 1);
  const bSevere = clamp((bloodMl - 2600) / 800, 0, 1); // ~2600..3400

  const anesthesiaMax = clamp((a - 0.8) / 0.2, 0, 1); // 0 below 80%, 1 at 100%
  const compAllowed = 1 - anesthesiaMax; // max anesthesia suppresses compensation

  // Baseline steady-state targets (non-extreme).
  // Blood-loss compensation is explicitly damped by anesthesia.
  const hrT = clamp(80 * (1 - 0.15 * a) + 25 * bMild * (0.2 + 0.8 * compAllowed), 30, 180);
  const mapT = clamp(85 * (1 - 0.10 * a) - 35 * bMild, 20, 130);
  const coT = clamp(
    5.2 * (1 - 0.10 * a - 0.35 * bMild) + ((hrT - 80) / 100) * 0.8 * (0.4 + 0.6 * compAllowed),
    0.8,
    10.0
  );

  // Start values.
  const hr0 = 80;
  const map0 = 85;
  const co0 = 5.2;

  // Response times (blood loss effects are faster).
  const tauHr = lerp(18, 10, bMild);
  const tauMap = lerp(22, 12, bMild);
  const tauCo = lerp(24, 14, bMild);

  // Smooth approach to targets.
  let hr = expEase(hr0, hrT, t, tauHr);
  let map = expEase(map0, mapT, t, tauMap);
  let co = expEase(co0, coT, t, tauCo);

  // Oxygen affects SpO2 and risk strongly; mild hypoxia also increases HR and reduces MAP/CO slightly.
  const hypoxia = clamp((0.25 - o) / 0.25, 0, 1); // only below ~25%
  hr = clamp(hr + 20 * hypoxia * compAllowed, 30, 180);
  map = clamp(map - 10 * hypoxia, 15, 130);
  co = clamp(co - 0.6 * hypoxia, 0.8, 10);

  // A) Zero oxygen: delayed collapse (do not drop instantly).
  const isZeroOxygen = o <= 0.01;
  if (isZeroOxygen) {
    const collapseStart = 20;
    const collapseDuration = lerp(150, 85, anesthesiaMax); // max anesthesia -> faster collapse
    const collapseP = clamp((t - collapseStart) / collapseDuration, 0, 1);
    const collapse = Math.pow(collapseP, 1.35);

    // Transient tachycardia then bradycardia.
    const tachyUp = clamp(t / 18, 0, 1);
    const hrTachy = lerp(hr, 135, tachyUp);
    const hrBrady = 42;
    hr = lerp(hrTachy, hrBrady, collapse);

    // MAP/CO drop with a short delay.
    const mapEarly = lerp(map, 55, clamp(t / 20, 0, 1));
    map = lerp(mapEarly, 35, collapse);

    const coEarly = lerp(co, 3.2, clamp(t / 20, 0, 1));
    co = lerp(coEarly, 1.6, collapse);
  }

  // B) Extreme blood loss: compensation breaks then sudden decompensation.
  const isExtremeBleed = bloodMl >= 2800;
  if (isExtremeBleed) {
    const compRamp = clamp(t / 20, 0, 1);
    const hrHigh = lerp(hr, 145, compRamp) * (0.25 + 0.75 * compAllowed);

    // Sudden decomp after short delay.
    const decompStart = lerp(65, 30, bSevere); // more severe -> earlier
    const decomp = logistic01((t - decompStart) / 3.5);

    // Unstable HR after decomp (deterministic oscillation).
    const wobble = 10 * Math.sin(t * 0.85) + 6 * Math.sin(t * 0.37);
    const hrAfter = 120 + wobble;
    hr = lerp(hrHigh, hrAfter, decomp);

    // Very low MAP/CO after decomp.
    const mapAfter = lerp(45, 28, bSevere);
    const coAfter = lerp(2.2, 1.3, bSevere);
    map = lerp(map, mapAfter, decomp);
    co = lerp(co, coAfter, decomp);

    // If anesthesia is maxed, suppress all compensation (override the tachy behavior).
    if (anesthesiaMax > 0) {
      hr = lerp(hr, 40, anesthesiaMax);
      map = lerp(map, 45, anesthesiaMax);
      co = lerp(co, 2.2, anesthesiaMax);
    }
  }

  // C) Maximum anesthesia: suppress all compensatory mechanisms (dominant effect).
  if (anesthesiaMax > 0) {
    // Clamp into the requested ranges.
    hr = lerp(hr, 42, anesthesiaMax);
    map = lerp(map, 48, anesthesiaMax);
    co = lerp(co, 2.4, anesthesiaMax);
  }

  // Keep within plausible bounds.
  hr = clamp(hr, 25, 180);
  map = clamp(map, 15, 130);
  co = clamp(co, 0.6, 10);

  const spo2 = clamp(0.98 - 0.55 * (1 - o) - 0.14 * (bloodMl / 4000), 0.35, 1.0);
  const riskFromSpo2 = clamp((0.94 - spo2) / 0.44, 0, 1);
  const riskFromMap = clamp((65 - map) / 45, 0, 1);
  const risk = clamp(Math.max(riskFromSpo2, riskFromMap) + 0.15 * a, 0, 1);

  return { hr, map, co, spo2, risk };
}

function buildForm() {
  els.form.innerHTML = "";
  for (const cfg of INPUTS) {
    els.form.appendChild(makeField(cfg));
  }
}

function getInputs() {
  const out = {};
  for (const cfg of INPUTS) {
    const input = els.form.querySelector(`input[data-key="${cfg.key}"]`);
    const raw = input.value;
    if (raw === "") throw new Error(`Missing value for: ${cfg.label}`);
    const v = Number(raw);
    if (!Number.isFinite(v)) throw new Error(`Invalid number for: ${cfg.label}`);
    out[cfg.key] = v;
  }
  return out;
}

function resetInputs() {
  stopLiveSimulation();
  for (const cfg of INPUTS) {
    const input = els.form.querySelector(`input[data-key="${cfg.key}"]`);
    if (input) input.value = String(cfg.defaultValue ?? "");
  }
  els.driving.textContent = "—";
  els.p.textContent = "—";
  els.all.textContent = "—";
  setStatus("Ready");
  applyHeartColor(heartColorFromInputs({ oxygen: 100, blood_loss: 0 }));
  clearCharts();
}

function startLiveSimulation(inputs) {
  stopLiveSimulation();
  clearCharts();

  liveSim = {
    inputs,
    startedAt: null,
    lastSampleAt: 0,
    sampleDt: 0.2,
    windowSeconds: 20,
    maxSeconds: SIM_DURATION_S,
    t: [],
    hr: [],
    map: [],
    co: [],
    spo2: [],
    risk: [],
  };

  setStatus("Running (live)...");
}

function stopLiveSimulation() {
  if (!liveSim) return;
  liveSim = null;
  setStatus("Ready");
}

function pushSample(buf, t, v, maxPoints) {
  buf.t.push(t);
  buf.hr.push(v.hr);
  buf.map.push(v.map);
  buf.co.push(v.co);
  buf.spo2.push(v.spo2);
  buf.risk.push(v.risk);

  if (buf.t.length > maxPoints) {
    const drop = buf.t.length - maxPoints;
    buf.t.splice(0, drop);
    buf.hr.splice(0, drop);
    buf.map.splice(0, drop);
    buf.co.splice(0, drop);
    buf.spo2.splice(0, drop);
    buf.risk.splice(0, drop);
  }
}

function updateLiveSimulation(nowSeconds) {
  if (!liveSim) return;
  const sim = liveSim;

  if (sim.startedAt == null) {
    sim.startedAt = nowSeconds;
    sim.lastSampleAt = nowSeconds;
  }

  const tSim = Math.max(0, nowSeconds - sim.startedAt);

  // Sample at a steady cadence regardless of frame rate.
  const maxPoints = Math.ceil((sim.windowSeconds / sim.sampleDt) * 2);
  while (nowSeconds - sim.lastSampleAt >= sim.sampleDt) {
    const tS = Math.max(0, sim.lastSampleAt - sim.startedAt);
    const v = computeVitalsAtTime(sim.inputs, tS);
    pushSample(sim, tS, v, maxPoints);
    sim.lastSampleAt += sim.sampleDt;
  }

  const n = sim.t.length;
  if (n > 0) {
    const hr = sim.hr[n - 1];
    const map = sim.map[n - 1];
    const co = sim.co[n - 1];
    const spo2 = sim.spo2[n - 1];
    const risk = sim.risk[n - 1];

    heartbeatPulse?.setBpm(hr);
    els.driving.textContent = risk.toFixed(3);
    els.p.textContent = `HR ${hr.toFixed(0)} bpm, MAP ${map.toFixed(0)} mmHg, CO ${co.toFixed(2)} L/min`;
    els.all.textContent = JSON.stringify(
      {
        inputs: {
          anesthesia_percent: sim.inputs.anesthesia,
          oxygen_percent: sim.inputs.oxygen,
          blood_loss_ml: sim.inputs.blood_loss,
        },
        derived_vitals: {
          at_time_seconds: sim.t[n - 1],
          heart_rate_bpm: hr,
          mean_arterial_pressure_mmhg: map,
          cardiac_output_l_min: co,
          spo2_percent: spo2 * 100,
        },
        risk_score_0to1: risk,
        notes: "Heuristic prototype (live).",
      },
      null,
      2
    );
  }

  // Render scrolling window
  const xMax = tSim;
  const xMin = Math.max(0, xMax - sim.windowSeconds);

  drawScrollingChart(els.chartHr, sim.t, sim.hr, { xMin, xMax, units: "bpm", line: "#ff7a7a" });
  drawScrollingChart(els.chartBp, sim.t, sim.map, { xMin, xMax, units: "mmHg", line: "#ff7a7a" });
  drawScrollingChart(els.chartCo, sim.t, sim.co, { xMin, xMax, units: "L/min", line: "#ff7a7a" });

  if (tSim >= sim.maxSeconds) {
    setStatus("Done");
    stopLiveSimulation();
  }
}

function heartColorFromInputs({ oxygen, blood_loss }) {
  // User rules:
  // - blood loss: light red -> pale red
  // - low oxygen: slight blue tint
  const o = clamp((oxygen ?? 100) / 100, 0, 1);
  const b = clamp((blood_loss ?? 0) / 2000, 0, 1);

  const baseRed = new THREE.Color("#ff2d2d");
  const paleRed = new THREE.Color("#ffd6d6");
  const blueTint = new THREE.Color("#4da3ff");

  const c = baseRed.clone();
  // Blood loss makes it paler.
  c.lerp(paleRed, b * 0.85);
  // Oxygen reduction adds a slight blue tint.
  c.lerp(blueTint, (1 - o) * 0.35);
  return c;
}

function applyHeartColor(color) {
  if (!heartMaterials.length) return;
  const c = color;
  for (const m of heartMaterials) {
    if (m && m.color) m.color.copy(c);
    if (m && m.emissive) m.emissive.copy(c).multiplyScalar(0.25);
    if (m) m.needsUpdate = true;
  }
}

function simulateOnce() {
  const { anesthesia, oxygen, blood_loss } = getInputs();

  // Build time-series graphs (0..SIM_DURATION_S) and show the end-state summary.
  const series = simulateSeries({ anesthesia, oxygen, blood_loss });
  const iEnd = Math.max(0, series.t.length - 1);
  const hr = series.hr[iEnd];
  const map = series.map[iEnd];
  const co = series.co[iEnd];
  const spo2 = series.spo2[iEnd];
  const risk = series.risk[iEnd];

  // HeartbeatPulse call site: drive beat rate from simulated HR (end-state).
  heartbeatPulse?.setBpm(hr);

  els.driving.textContent = risk.toFixed(3);
  els.p.textContent = `HR ${hr.toFixed(0)} bpm, MAP ${map.toFixed(0)} mmHg, CO ${co.toFixed(2)} L/min`;
  els.all.textContent = JSON.stringify(
    {
      inputs: { anesthesia_percent: anesthesia, oxygen_percent: oxygen, blood_loss_ml: blood_loss },
      derived_vitals: {
        at_time_seconds: series.t[iEnd] ?? SIM_DURATION_S,
        heart_rate_bpm: hr,
        mean_arterial_pressure_mmhg: map,
        cardiac_output_l_min: co,
        spo2_percent: spo2 * 100,
      },
      risk_score_0to1: risk,
      notes: "Heuristic prototype (not ML model).",
    },
    null,
    2
  );

  // Heart tint driven directly by oxygen + blood loss rules.
  applyHeartColor(heartColorFromInputs({ oxygen, blood_loss }));

  drawChart(els.chartHr, series.t, series.hr, { units: "bpm", line: "#ff7a7a" });
  drawChart(els.chartBp, series.t, series.map, { units: "mmHg", line: "#ff7a7a" });
  drawChart(els.chartCo, series.t, series.co, { units: "L/min", line: "#ff7a7a" });
}

function clearCharts() {
  for (const c of [els.chartHr, els.chartBp, els.chartCo]) {
    if (!c) continue;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
    ctx.fillText("—", 6, 18);
  }
}

function simulateSeries({ anesthesia, oxygen, blood_loss }) {
  const t = [];
  const hr = [];
  const map = [];
  const co = [];
  const spo2 = [];
  const risk = [];

  for (let s = 0; s <= SIM_DURATION_S; s += SIM_DT_S) {
    const v = computeVitalsAtTime({ anesthesia, oxygen, blood_loss }, s);
    t.push(s);
    hr.push(v.hr);
    map.push(v.map);
    co.push(v.co);
    spo2.push(v.spo2);
    risk.push(v.risk);
  }

  return { t, hr, map, co, spo2, risk };
}

function drawChart(canvas, xs, ys, { units, line }) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background
  ctx.fillStyle = "rgba(0,0,0,0.20)";
  ctx.fillRect(0, 0, w, h);

  const padL = 34;
  const padR = 10;
  const padT = 10;
  const padB = 18;

  const xMin = xs[0] ?? 0;
  const xMax = xs[xs.length - 1] ?? 1;
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
    yMin = 0;
    yMax = 1;
  }
  if (Math.abs(yMax - yMin) < 1e-6) {
    yMax = yMin + 1;
  }

  // Nice padding
  const yPad = (yMax - yMin) * 0.12;
  yMin -= yPad;
  yMax += yPad;

  const xTo = (x) => padL + ((x - xMin) / (xMax - xMin)) * (w - padL - padR);
  const yTo = (y) => padT + (1 - (y - yMin) / (yMax - yMin)) * (h - padT - padB);

  // Grid
  ctx.strokeStyle = "rgba(255,255,255,0.10)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const yy = padT + (i / 4) * (h - padT - padB);
    ctx.moveTo(padL, yy);
    ctx.lineTo(w - padR, yy);
  }
  ctx.stroke();

  // Axes labels
  ctx.fillStyle = "rgba(231,238,247,0.65)";
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  ctx.fillText(`${yMax.toFixed(0)} ${units}`, 6, padT + 10);
  ctx.fillText(`${yMin.toFixed(0)} ${units}`, 6, h - padB);
  ctx.fillText(`0s`, padL, h - 6);
  ctx.fillText(`${xMax}s`, w - padR - 26, h - 6);

  // Line
  ctx.strokeStyle = line || "#ff7a7a";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < xs.length; i++) {
    const xx = xTo(xs[i]);
    const yy = yTo(ys[i]);
    if (i === 0) ctx.moveTo(xx, yy);
    else ctx.lineTo(xx, yy);
  }
  ctx.stroke();
}

function drawScrollingChart(canvas, xs, ys, { xMin, xMax, units, line }) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(0,0,0,0.20)";
  ctx.fillRect(0, 0, w, h);

  const padL = 34;
  const padR = 10;
  const padT = 10;
  const padB = 18;

  const x0 = Number.isFinite(xMin) ? xMin : 0;
  const x1 = Number.isFinite(xMax) ? xMax : 1;
  const span = Math.max(1e-6, x1 - x0);

  // Y range from only visible points.
  let yMin = Infinity;
  let yMax = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i];
    if (x < x0 || x > x1) continue;
    const y = ys[i];
    if (!Number.isFinite(y)) continue;
    yMin = Math.min(yMin, y);
    yMax = Math.max(yMax, y);
  }
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
    yMin = 0;
    yMax = 1;
  }
  if (Math.abs(yMax - yMin) < 1e-6) yMax = yMin + 1;
  const yPad = (yMax - yMin) * 0.12;
  yMin -= yPad;
  yMax += yPad;

  const xTo = (x) => padL + ((x - x0) / span) * (w - padL - padR);
  const yTo = (y) => padT + (1 - (y - yMin) / (yMax - yMin)) * (h - padT - padB);

  // Grid
  ctx.strokeStyle = "rgba(255,255,255,0.10)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const yy = padT + (i / 4) * (h - padT - padB);
    ctx.moveTo(padL, yy);
    ctx.lineTo(w - padR, yy);
  }
  ctx.stroke();

  // Labels
  ctx.fillStyle = "rgba(231,238,247,0.65)";
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  ctx.fillText(`${yMax.toFixed(0)} ${units}`, 6, padT + 10);
  ctx.fillText(`${yMin.toFixed(0)} ${units}`, 6, h - padB);
  ctx.fillText(`${x0.toFixed(0)}s`, padL, h - 6);
  ctx.fillText(`${x1.toFixed(0)}s`, w - padR - 32, h - 6);

  // Trace
  ctx.strokeStyle = line || "#ff7a7a";
  ctx.lineWidth = 2;
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i];
    if (x < x0 || x > x1) continue;
    const y = ys[i];
    if (!Number.isFinite(y)) continue;
    const xx = xTo(x);
    const yy = yTo(y);
    if (!started) {
      ctx.moveTo(xx, yy);
      started = true;
    } else {
      ctx.lineTo(xx, yy);
    }
  }
  if (started) ctx.stroke();
}

function initThree() {
  const renderer = new THREE.WebGLRenderer({ canvas: els.canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const clock = new THREE.Clock();

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05070b);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  camera.position.set(0.2, 0.3, 7);

  const ambient = new THREE.AmbientLight(0xffffff, 0.55);
  scene.add(ambient);

  const dir = new THREE.DirectionalLight(0xffffff, 0.9);
  dir.position.set(2, 3, 4);
  scene.add(dir);

  const rim = new THREE.DirectionalLight(0xffffff, 0.35);
  rim.position.set(-3, 1, -2);
  scene.add(rim);

  // Load GLB as the main heart model.
  const heartGroup = new THREE.Group();
  scene.add(heartGroup);
  loadGlbInto(heartGroup, camera)
    .then((root) => {
      // HeartbeatPulse call site: attach to GLB root.
      heartbeatPulse?.attach(root, clock.getElapsedTime());
    })
    .catch((e) => {
      setStatus(e instanceof Error ? e.message : String(e));
      // Fallback: show a simple parametric heart if GLB load fails.
      const fallbackGeometry = makeHeartGeometry();
      const fallbackMaterial = new THREE.MeshStandardMaterial({
        color: new THREE.Color("red"),
        roughness: 0.35,
        metalness: 0.05,
      });
      const fallback = new THREE.Mesh(fallbackGeometry, fallbackMaterial);
      fallback.rotation.x = -Math.PI * 0.06;
      fallback.rotation.y = Math.PI * 0.18;
      heartGroup.add(fallback);
      heartMaterials = [fallbackMaterial];
      applyHeartColor(new THREE.Color("red"));

      // HeartbeatPulse call site: attach to fallback mesh.
      heartbeatPulse?.attach(fallback, clock.getElapsedTime());
    });

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(50, 50),
    new THREE.MeshStandardMaterial({ color: 0x0a101a, roughness: 1.0, metalness: 0.0 })
  );
  floor.position.y = -2.0;
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.minDistance = 2.5;
  controls.maxDistance = 16;
  controls.target.set(0, 0, 0);
  controls.update();

  function resize() {
    const w = els.canvas.clientWidth;
    const h = els.canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  window.addEventListener("resize", resize);
  resize();

  // No animations beyond rendering; keep camera fixed.
  renderer.setAnimationLoop(() => {
    const t = clock.getElapsedTime();
    controls.update();

    // Live charts update (ECG monitor-like sweep)
    try {
      updateLiveSimulation(t);
    } catch (e) {
      console.warn("Live simulation disabled due to error:", e);
      stopLiveSimulation();
    }

    // HeartbeatPulse call site: update per-frame with auto-disable on failure.
    if (heartbeatPulse) {
      try {
        heartbeatPulse.update(t);
      } catch (err) {
        console.warn("HeartbeatPulse disabled due to error:", err);
        try {
          heartbeatPulse.detach();
        } catch {
          // ignore
        }
        heartbeatPulse = null;
      }
    }

    renderer.render(scene, camera);
  });
}

async function loadGlbInto(targetGroup, camera) {
  setStatus("Loading .glb model...");
  const loader = new GLTFLoader();

  const gltf = await new Promise((resolve, reject) => {
    loader.load(
      GLB_URL,
      (g) => resolve(g),
      undefined,
      (err) => reject(err)
    );
  });

  // Clear previous
  while (targetGroup.children.length) targetGroup.remove(targetGroup.children[0]);
  heartMaterials = [];

  const root = gltf.scene || gltf.scenes?.[0];
  if (!root) throw new Error("GLB loaded but scene is empty");

  // Collect materials that can be recolored.
  root.traverse((obj) => {
    if (!obj.isMesh) return;
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    for (const m of mats) {
      if (m && m.color) heartMaterials.push(m);
    }
  });

  // Center + scale to a consistent size.
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  box.getSize(size);
  const center = new THREE.Vector3();
  box.getCenter(center);

  root.position.sub(center);

  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const desired = 3.2;
  const scale = desired / maxDim;
  root.scale.setScalar(scale);

  root.rotation.y = Math.PI * 0.2;
  root.rotation.x = -Math.PI * 0.03;

  targetGroup.add(root);

  // Ensure initial color matches requested default (red).
  applyHeartColor(new THREE.Color("#ff2d2d"));

  // Frame camera a bit nicer around the model.
  camera.position.set(0.2, 0.3, 7);
  setStatus("Ready");

  return root;
}

function makeHeartGeometry() {
  // True 3D parametric heart surface (smooth volume, not a 2D extrude).
  // Parameterization:
  //   u controls the classic heart curve
  //   v rotates that curve around the vertical axis to form a closed surface
  // This yields a symmetric but clearly 3D heart-like solid.
  function heartSurface(u, v, target) {
    const uu = u * Math.PI; // 0..π
    const vv = v * Math.PI * 2; // 0..2π

    // Classic heart curve radius
    const r = 16 * Math.pow(Math.sin(uu), 3);
    const y =
      13 * Math.cos(uu) -
      5 * Math.cos(2 * uu) -
      2 * Math.cos(3 * uu) -
      Math.cos(4 * uu);

    // Rotate radius around y-axis
    const x = r * Math.sin(vv);
    const z = r * Math.cos(vv);

    target.set(x, y, z);
  }

  const geom = new ParametricGeometry(heartSurface, 90, 90);
  geom.center();
  // Scale down and slightly squash depth to look more heart-like.
  geom.scale(0.07, 0.07, 0.07);
  geom.scale(1.15, 1.2, 0.9);
  geom.computeVertexNormals();
  return geom;
}

async function main() {
  try {
    buildForm();
    initThree();

    els.btnPredict.addEventListener("click", () => {
      try {
        const inputs = getInputs();
        // Start a live scrolling trace (like a monitor sweep).
        startLiveSimulation(inputs);
      } catch (e) {
        setStatus(e instanceof Error ? e.message : String(e));
      }
    });

    els.btnLoadExample.addEventListener("click", () => {
      resetInputs();
    });

    resetInputs();
  } catch (e) {
    setStatus(e instanceof Error ? e.message : String(e));
  }
}

main();
