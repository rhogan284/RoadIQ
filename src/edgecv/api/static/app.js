// RoadIQ dashboard (component 8). Talks only to the read API (component 7).
"use strict";

const CLASS_NAME = { D00: "Longitudinal", D10: "Transverse", D20: "Alligator",
                     D40: "Pothole", other: "Other" };
const BAND_COLOUR = { good: "#3f7f62", fair: "#c08a2e", poor: "#a83e2c" };
const UNSURVEYED = "#c6cdd4";
const ACCEPTED = new Set(["auto-accepted", "confirmed", "reclassified"]);

const S = { run: null, summary: null, geo: null, worklist: [], instances: [], queue: [],
            current: null, segment: null, order: new Map() };
const $ = (id) => document.getElementById(id);
const api = async (path, opts) => {
  const r = await fetch(`/api${path}`, opts);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
};
const fmtBytes = (n) => n == null ? "–" : n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB`
  : n >= 1e6 ? `${(n / 1e6).toFixed(0)} MB` : `${(n / 1e3).toFixed(0)} KB`;
const fmtDate = (s, opts) => s ? new Date(s).toLocaleString("en-AU", opts) : "—";
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
// "osm:411059290:0" → "W411059290 §1" — way id and 100 m section, as in the mock-up.
const segRef = (ref) => {
  const m = /^osm:(\d+):(\d+)$/.exec(ref || "");
  return m ? `W${m[1]} §${Number(m[2]) + 1}` : (ref || "—");
};
const label = (inst) => {
  const cls = inst.state === "reclassified" && inst.new_class ? inst.new_class : inst.defect_class;
  return `${cls} ${CLASS_NAME[cls] ?? cls}`;
};

// ------------------------------------------------------------------ map
const map = L.map("map", { preferCanvas: true, zoomSnap: 0.25, zoomControl: true, attributionControl: true })
  .setView([-33.8843, 151.1945], 16);
// Greyed OSM tiles for street context only (CSS filter on .basemap). If they cannot load
// (no internet at the demo) the segments still draw on the plain background.
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, className: "basemap",
  attribution: "© OpenStreetMap contributors",
}).addTo(map);
let segLayer = null, dotLayer = L.layerGroup().addTo(map), selLayer = L.layerGroup().addTo(map);

const legend = L.control({ position: "bottomleft" });
legend.onAdd = () => {
  const d = L.DomUtil.create("div", "legend");
  d.id = "legend";
  return d;
};
legend.addTo(map);

// State arterials: driven, never scored. Drawn once, under everything else.
const ARTERIAL = "#b9c2cb";
api("/network").then((fc) => L.geoJSON(fc, {
  style: { color: ARTERIAL, weight: 7, opacity: 0.8 }, interactive: false,
}).addTo(map).bringToBack()).catch(() => {});

// The vehicle: an arrow at the newest published frame, rotated to its heading, and the
// trail of frames that have landed. The gap between them is the pipeline's lag.
const trailLine = L.polyline([], { color: "#0d6d92", weight: 4, opacity: 0.55 }).addTo(map);
const carIcon = (heading) => L.divIcon({ className: "car", iconSize: [30, 30], iconAnchor: [15, 15],
  html: `<svg width="30" height="30" viewBox="-15 -15 30 30" style="transform:rotate(${heading || 0}deg)">
    <circle r="13" fill="#0d6d92" stroke="#fff" stroke-width="2.5"/>
    <path d="M0,-8 L6,6 L0,2.5 L-6,6 Z" fill="#fff"/></svg>` });
let carMarker = null;

async function renderPosition() {
  if (!S.run) return;
  const p = await api(`/runs/${S.run}/position`).catch(() => null);
  if (!p || !p.car) { if (carMarker) { carMarker.remove(); carMarker = null; } trailLine.setLatLngs([]); return; }
  const c = p.car;
  const kmh = c.speed_mps == null ? "" : ` · ${Math.round(c.speed_mps * 3.6)} km/h`;
  const tip = `Vehicle · frame ${c.seq}${kmh}` +
    (p.landed_seq != null && c.source === "bus" ? ` · pipeline ${c.seq - p.landed_seq} frames behind` : "");
  if (!carMarker) {
    carMarker = L.marker([c.lat, c.lon], { icon: carIcon(c.heading_deg), zIndexOffset: 1000 })
      .bindTooltip(tip, { direction: "top", offset: [0, -14] }).addTo(map);
  } else {
    carMarker.setLatLng([c.lat, c.lon]).setIcon(carIcon(c.heading_deg)).setTooltipContent(tip);
  }
  trailLine.setLatLngs(p.trail);
}

function renderLegend() {
  const km = { good: 0, fair: 0, poor: 0, none: 0 };
  for (const f of S.geo.features) {
    const b = f.properties.condition_band;
    km[BAND_COLOUR[b] ? b : "none"] += (f.properties.length_m || 0) / 1000;
  }
  const row = (c, t) => `<div><i style="background:${c}"></i>${t}</div>`;
  $("legend").innerHTML = `<h4>CONDITION</h4>` +
    row(BAND_COLOUR.good, `Good&nbsp; ${km.good.toFixed(1)} km`) +
    row(BAND_COLOUR.fair, `Fair&nbsp; ${km.fair.toFixed(1)} km`) +
    row(BAND_COLOUR.poor, `Poor&nbsp; ${km.poor.toFixed(1)} km`) +
    row(UNSURVEYED, `Not surveyed&nbsp; ${km.none.toFixed(1)} km`) +
    row(ARTERIAL, "Not ours (state road)");
}

function renderMap(fit) {
  if (segLayer) segLayer.remove();
  segLayer = L.geoJSON(S.geo, {
    style: (f) => {
      const c = BAND_COLOUR[f.properties.condition_band];
      return c ? { color: c, weight: 6, opacity: 0.95 }
               : { color: UNSURVEYED, weight: 5, opacity: 0.9, dashArray: "6 6" };
    },
    onEachFeature: (f, layer) => layer.on("click", () => selectSegment(f.properties.segment_id)),
  }).addTo(map);
  if (fit) {
    map.invalidateSize();
    const b = segLayer.getBounds();
    if (b.isValid()) map.fitBounds(b, { padding: [8, 8], animate: false });
  }

  dotLayer.clearLayers();
  for (const inst of S.instances) {
    if (inst.state === "rejected" || inst.defect_class === "other") continue;
    const pending = inst.state === "pending";
    L.circleMarker([inst.lat, inst.lon], {
      radius: 2.6, weight: 0.8, color: "#fff",
      fillColor: pending ? "#d9a03a" : "#8e2f20", fillOpacity: 0.9,
    }).on("click", () => showInstance(inst.cluster_key)).addTo(dotLayer);
  }
  renderLegend();
}

function highlightSegment(segmentId) {
  selLayer.clearLayers();
  const f = S.geo.features.find((x) => x.properties.segment_id === segmentId);
  if (!f) return;
  const p = f.properties;
  const halo = L.geoJSON(f, { style: { color: "#0d6d92", weight: 16, opacity: 0.25 } });
  const line = L.geoJSON(f, { style: { color: "#0d6d92", weight: 6 } });
  selLayer.addLayer(halo).addLayer(line);
  const idx = p.condition_index == null ? "not scored" : `${Math.round(p.condition_index)}/100`;
  line.bindTooltip(`${esc(segRef(p.road_ref))} · ${idx}`, { permanent: true, direction: "top",
    className: "seg-label", offset: [0, -8] }).openTooltip();
}

// ------------------------------------------------------------------ stats + work list
function renderStats() {
  const s = S.summary;
  $("s-km").textContent = `${s.assessed_km.toFixed(1)} km`;
  $("s-km-l").textContent = s.gap_m > 0 ? `assessed · ${Math.round(s.gap_m)} m not assessed`
                                        : "assessed this survey";
  $("s-def").textContent = s.defects.toLocaleString();
  $("s-def-l").textContent = `defects on the register · ${s.pending_review} to review`;
  $("s-acc").textContent = `${s.frames_accounted_pct}%`;
  $("s-acc-l").textContent = `frames accounted for · ${s.frames_dropped} dropped`;
  $("s-mb").textContent = fmtBytes(s.bytes_stored);
  $("s-mb-l").textContent = `uploaded, of ${fmtBytes(s.raw_bytes)} captured`;
  const r = s.run;
  $("chrome-meta").textContent =
    `Survey ${fmtDate(r.started_at, { day: "numeric", month: "short", year: "numeric" })}` +
    `  ·  ${r.authority_id}  ·  ${r.source_kind}  ·  self-hosted`;
}

function renderWorklist() {
  const prev = S.worklist.find((w) => w.previous_at);
  $("map-note").textContent = prev
    ? `worst-first · compared with ${fmtDate(prev.previous_at, { month: "long", year: "numeric" })}`
    : "worst-first · first survey of this network";
  $("worklist").innerHTML = S.worklist.map((w, i) => {
    const c = w.change;
    const change = c == null ? `<span class="same">first survey</span>`
      : c < 0 ? `<span class="worse">${-c} pts worse</span>`
      : c > 0 ? `<span class="better">${c} pts better</span>` : `<span class="same">unchanged</span>`;
    return `<tr data-seg="${w.segment_id}" class="${w.segment_id === S.segment ? "sel" : ""}">
      <td>${i + 1}</td>
      <td>${esc(segRef(w.road_ref))} · ${esc(w.road_name)}</td>
      <td><span class="sw" style="background:${BAND_COLOUR[w.condition_band]}"></span>${Math.round(w.condition_index)}/100</td>
      <td>${w.defects}</td><td>${change}</td>
      <td class="muted">${w.previous_at ? fmtDate(w.previous_at, { month: "short", year: "numeric" }) : "—"}</td></tr>`;
  }).join("");
  for (const tr of $("worklist").querySelectorAll("tr")) {
    tr.addEventListener("click", () => selectSegment(Number(tr.dataset.seg)));
  }
}

function selectSegment(segmentId) {
  S.segment = segmentId;
  highlightSegment(segmentId);
  renderWorklist();
  const on = S.instances.filter((i) => i.segment_id === segmentId && i.state !== "rejected"
                                       && i.defect_class !== "other");
  on.sort((a, b) => (ACCEPTED.has(b.state) - ACCEPTED.has(a.state)) || b.confidence - a.confidence);
  if (on.length) showInstance(on[0].cluster_key);
}

// ------------------------------------------------------------------ evidence
const canvas = $("frame"), ctx = canvas.getContext("2d");

function drawBox(b, sx, sy, { strong, pending, rejected, other }) {
  const x = b.bbox_x * sx, y = b.bbox_y * sy, w = b.bbox_w * sx, h = b.bbox_h * sy;
  const col = pending ? "#d9a03a" : rejected ? "#a83e2c" : other ? "#c6cdd4" : "#4e9cbe";
  ctx.save();
  ctx.strokeStyle = col;
  ctx.lineWidth = strong ? 4 : 2.5;
  if (pending || rejected || other) ctx.setLineDash([12, 8]);
  ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]);
  if (!pending && !rejected && !other) {           // the mock-up's corner ticks
    ctx.lineWidth = strong ? 9 : 6;
    const t = Math.min(26, w / 3, h / 3);
    for (const [cx, cy, dx, dy] of [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]) {
      ctx.beginPath(); ctx.moveTo(cx + t * dx, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + t * dy); ctx.stroke();
    }
  }
  const cls = b.defect_class;
  const text = `${cls} ${(CLASS_NAME[cls] ?? cls).toUpperCase()} · ${b.confidence.toFixed(2)}` +
               (pending ? " · REVIEW" : rejected ? " · REJECTED" : "");
  ctx.font = "bold 17px Helvetica, Arial, sans-serif";
  const tw = ctx.measureText(text).width + 16;
  const ty = Math.max(0, y - 28);
  ctx.fillStyle = pending ? "#b4801f" : rejected ? "#a83e2c" : other ? "#8b96a3" : "#0d6d92";
  ctx.fillRect(x, ty, tw, 26);
  ctx.fillStyle = "#fff";
  ctx.fillText(text, x + 8, ty + 19);
  ctx.restore();
}

function drawPlaceholder(msg) {
  ctx.fillStyle = "#2a2d31"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#c5d2da"; ctx.font = "18px Helvetica, Arial"; ctx.fillText(msg, 24, 40);
}

async function showInstance(key) {
  const inst = await api(`/instances/${S.run}/${key}`);
  S.current = inst;
  if (inst.segment_id && inst.segment_id !== S.segment) {
    S.segment = inst.segment_id; highlightSegment(inst.segment_id); renderWorklist();
  }
  const idx = S.queue.findIndex((q) => q.cluster_key === key);
  $("queue-label").textContent = idx >= 0 ? `Review queue · ${idx + 1} of ${S.queue.length}`
                                          : `Review queue · ${S.queue.length} pending`;
  $("ev-title").textContent = `Defect ${key.slice(0, 6)}  ·  ${label(inst)}`;
  const chip = $("ev-state");
  chip.textContent = inst.state === "pending" ? "Awaiting sign-off" : inst.state.replace("-", " ");
  chip.className = "chip " + (inst.state === "pending" ? "pending"
    : inst.state === "rejected" ? "rejected" : "accepted");

  const img = new Image();
  img.onload = () => {
    const scale = Math.min(960 / img.width, 720 / img.height);
    canvas.width = Math.round(img.width * scale);
    canvas.height = Math.round(img.height * scale);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const sx = canvas.width / inst.width, sy = canvas.height / inst.height;
    const boxes = [...inst.frame_boxes].sort((a, b) =>
      (a.detection_id === inst.detection_id) - (b.detection_id === inst.detection_id));
    for (const b of boxes) {
      drawBox(b, sx, sy, { strong: b.detection_id === inst.detection_id,
        pending: b.state === "pending", rejected: b.state === "rejected",
        other: b.defect_class === "other" });
    }
  };
  img.onerror = () => {
    canvas.width = 960; canvas.height = 540;
    drawPlaceholder("Full frame not available — only the crop left the vehicle.");
    if (inst.crop_sha256) {
      const crop = new Image();
      crop.onload = () => ctx.drawImage(crop, 24, 64, Math.min(crop.width, 900), Math.min(crop.height, 440));
      crop.src = `/api/blobs/crop/${inst.crop_sha256}?fmt=${inst.crop_format || "webp"}`;
    }
  };
  img.src = `/api/frames/${S.run}/${inst.seq}/image`;

  const kmh = inst.speed_mps == null ? "—" : `${Math.round(inst.speed_mps * 3.6)} km/h`;
  $("hud-l").textContent = `frame ${String(inst.seq).padStart(5, "0")}  ·  ` +
    `${S.summary.run.target_fps} fps  ·  ${Math.round(inst.latency_ms)} ms inference`;
  $("hud-r").textContent = `${inst.lat.toFixed(4)}, ${inst.lon.toFixed(4)}  ·  ${kmh}`;
  const others = inst.frame_boxes.length - 1;
  $("ev-caption").textContent = `The frame the vehicle saw, source ${inst.source_ref.split("/").pop()}. ` +
    `${inst.frame_boxes.length} box${inst.frame_boxes.length === 1 ? "" : "es"} on this frame` +
    (others > 0 ? "; only the padded crops left the vehicle." : "; only the padded crop left the vehicle.");
  const area = (100 * inst.area_px / (inst.width * inst.height)).toFixed(1);
  const meta = [
    ["Confidence", `${inst.confidence.toFixed(2)}  ·  ${inst.confidence >= 0.5 ? "above threshold" : "below threshold — review"}`],
    ["Segment", inst.road_name ? `${segRef(inst.road_ref)} · ${inst.road_name}` : "unlocated"],
    ["Severity", `${inst.severity}  ·  ${area}% of frame`],
    ["Sightings", `${inst.observation_count} frame${inst.observation_count > 1 ? "s" : ""} · 1 asset record`],
    ["Surface", inst.surface_type ? `${inst.surface_type[0].toUpperCase()}${inst.surface_type.slice(1)}` : "—"],
    ["First seen", fmtDate(inst.first_seen_at, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" })],
    ["Detector", `${inst.detector} · ${inst.detector_version}`],
    ["Ground truth", `${inst.ground_truth_boxes} labelled box${inst.ground_truth_boxes === 1 ? "" : "es"} on this image`],
  ];
  $("ev-meta").innerHTML = meta.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
  for (const id of ["btn-confirm", "btn-reject", "btn-order"]) $(id).disabled = false;
  $("btn-confirm").textContent = inst.state === "confirmed" ? "Confirmed ✓" : "Confirm defect";
  $("btn-reject").textContent = inst.state === "rejected" ? "Rejected ✓" : "Reject";
  $("btn-order").textContent = S.order.has(key) ? "On work order ✓" : "Add to work order";
  $("btn-order").classList.toggle("done", S.order.has(key));
}

async function review(state) {
  const key = S.current.cluster_key;
  await api(`/instances/${S.run}/${key}/review`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state, reviewed_by: "officer" }),
  });
  const pos = S.queue.findIndex((q) => q.cluster_key === key);
  await refresh(false);
  // Keep the officer moving through the queue after a decision.
  const next = pos >= 0 && S.queue.length ? S.queue[Math.min(pos, S.queue.length - 1)] : null;
  await showInstance(next && state !== "pending" ? next.cluster_key : key);
}

function step(delta) {
  if (!S.queue.length) return;
  const at = S.current ? S.queue.findIndex((q) => q.cluster_key === S.current.cluster_key) : -1;
  const i = at < 0 ? 0 : (at + delta + S.queue.length) % S.queue.length;
  showInstance(S.queue[i].cluster_key);
}

// ------------------------------------------------------------------ bench
async function renderBench() {
  const b = await api(`/runs/${S.run}/bench`).catch(() => null);
  S.benched = !!b;
  if (!b) { $("bench").innerHTML = `<p class="muted">No benchmark row yet — run <code>make evaluate</code>.</p>`; return; }
  const g = b.grid_point || {};
  const det = (g.detectors || []).map((d) => `${d.name} ${d.version}`).join(", ");
  const pc = Object.entries(g.per_class || {}).map(([c, m]) =>
    `<tr><td>${c} ${CLASS_NAME[c]}</td><td>${m.precision.toFixed(3)}</td><td>${m.recall.toFixed(3)}</td><td>${m.f1.toFixed(3)}</td><td class="muted">${m.gt}</td></tr>`).join("") +
    Object.entries(g.unscored || {}).map(([c, m]) =>
    `<tr class="muted"><td>${c} ${CLASS_NAME[c]}</td><td colspan="4">unscored — ${esc(m.reason)} (${m.pred} predictions)</td></tr>`).join("");
  $("bench").innerHTML = `<p class="muted">${esc(det)} · IoU ≥ ${g.iou_min} · ` +
    `${b.frames_processed} frames · p50 ${Math.round(b.p50_ms)} ms · p95 ${Math.round(b.p95_ms)} ms · ` +
    `${g.throughput_fps ?? "–"} fps</p><table><tr><th>Class</th><th>P</th><th>R</th><th>F1</th><th>GT boxes</th></tr>${pc}` +
    `<tr><td><b>All</b></td><td><b>${Number(b.precision).toFixed(3)}</b></td><td><b>${Number(b.recall).toFixed(3)}</b></td><td><b>${Number(b.f1).toFixed(3)}</b></td><td></td></tr></table>`;
}

// ------------------------------------------------------------------ export
function exportCsv(ev) {
  ev.preventDefault();
  const q = (s) => `"${String(s ?? "").replace(/"/g, '""')}"`;
  const lines = [["rank", "segment", "road", "condition_index", "band", "defects", "change"].join(",")];
  S.worklist.forEach((w, i) => lines.push([i + 1, q(w.road_ref), q(w.road_name),
    w.condition_index, w.condition_band, w.defects, w.change ?? ""].join(",")));
  if (S.order.size) {
    lines.push("", ["cluster_key", "class", "confidence", "road", "lat", "lon", "state"].join(","));
    for (const i of S.order.values()) lines.push([i.cluster_key, q(label(i)), i.confidence,
      q(i.road_name), i.lat, i.lon, i.state].join(","));
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
  a.download = `work-order-${S.run.slice(0, 8)}.csv`;
  a.click();
}

// ------------------------------------------------------------------ load
async function refresh(fit) {
  const [summary, geo, worklist, instances] = await Promise.all([
    api(`/runs/${S.run}/summary`), api(`/runs/${S.run}/segments.geojson`),
    api(`/runs/${S.run}/worklist?limit=25`), api(`/runs/${S.run}/instances?limit=50000`)]);
  Object.assign(S, { summary, geo, worklist, instances });
  S.queue = instances.filter((i) => i.state === "pending" && i.defect_class !== "other");
  renderStats(); renderMap(fit); renderWorklist();
  if (S.segment) highlightSegment(S.segment);
}

async function loadRun(runId) {
  Object.assign(S, { run: runId, current: null, segment: null, benched: false });
  S.order.clear();
  await refresh(true);
  renderBench();
  renderPosition();
  logReset();
  if (S.worklist.length) selectSegment(S.worklist[0].segment_id);
}

// ------------------------------------------------------------------ live mode
// Poll every LIVE_MS. The current run's panels refresh in place (map view, selected
// segment and the open defect are kept), and a run that starts later is followed
// automatically unless the viewer has picked an older run from the menu.
const LIVE_MS = 5000;
const CAR_MS = 1000;
S.follow = true;
S.busy = false;

function renderRunOptions(runs) {
  const sel = $("run-select");
  const keep = sel.value;
  sel.innerHTML = runs.map((r) => `<option value="${r.run_id}">${fmtDate(r.started_at,
    { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })} · ${esc(r.source_kind)} · ${r.run_id.slice(0, 8)}${r.ended_at ? "" : " · live"}</option>`).join("");
  sel.value = keep || (runs[0] && runs[0].run_id);
}

function renderLive(run) {
  const live = run && !run.ended_at;
  S.live = !!live;
  const badge = $("live");
  badge.hidden = !live;
  if (live) badge.textContent = `● LIVE · ${S.summary.frames_ingested.toLocaleString()} frames landed`;
}

async function tick() {
  if (S.busy || document.hidden) return;
  S.busy = true;
  try {
    const runs = await api("/runs");
    renderRunOptions(runs);
    const newest = runs[0];
    if (newest && S.follow && newest.run_id !== S.run) {
      $("run-select").value = newest.run_id;
      await loadRun(newest.run_id);
    } else if (S.run) {
      await refresh(false);
      if (!S.current && S.worklist.length) selectSegment(S.worklist[0].segment_id);
      if (!S.benched) renderBench();
    }
    renderLive(runs.find((r) => r.run_id === S.run));
  } catch (e) {
    console.warn("live refresh failed", e);
  } finally {
    S.busy = false;
  }
}

// ------------------------------------------------------------------ frame log
// One line per frame, in the order frames LANDED in Postgres (writer insert order).
const LOG = { cursor: null, paused: false, n: 0, busy: false, times: [], html: [] };
const LOG_MAX_LINES = 3000;
const pad = (v, n) => String(v).padEnd(n);
const fmtTime = (s) => { const d = new Date(s);
  return d.toLocaleTimeString("en-AU", { hour12: false }) + "." + String(d.getMilliseconds()).padStart(3, "0"); };

function logReset() {
  Object.assign(LOG, { cursor: null, n: 0, times: [], html: [] });
  $("log-body").textContent = "";
}

function logLine(l) {
  const dets = l.detections || [];
  const [lvl, cls] = l.status !== "ok" ? ["ERROR", "err"] : dets.length ? ["DEFECT", "det"] : ["CLEAN", "ok"];
  const level = `<span class="${cls}">${pad(lvl, 6)}</span>`;
  const pos = l.lat == null ? "no-fix".padEnd(21) : `${l.lat.toFixed(5)},${l.lon.toFixed(5)}`;
  const kmh = l.speed_mps == null ? "  –  " : `${Math.round(l.speed_mps * 3.6)}km/h`;
  const found = l.status !== "ok" ? `<span class="err">${esc(l.error || l.status)}</span>`
    : dets.length ? `<span class="det">${dets.length} det</span>  ` +
      dets.map((d) => `<span class="cls">${d.c}</span>:${Number(d.p).toFixed(2)}`).join(" ")
    : `<span class="dim">no damage</span>`;
  return `<span class="t">${fmtTime(l.captured_at)}</span> ${level} ` +
    `seq=${String(l.seq).padStart(5, "0")}  ${pad(l.worker_id.replace(/^worker-/, "w-").slice(0, 8), 8)}  ` +
    `${String(Math.round(l.latency_ms)).padStart(4)}ms  ${pos}  ${pad(kmh, 7)}  ` +
    `<span class="dim">${pad(esc(l.source_ref), 26)}</span>  ${found}`;
}

async function logPoll() {
  if (LOG.busy || !S.run) return;
  LOG.busy = true;
  try {
    const q = LOG.cursor == null ? "limit=200" : `after_id=${LOG.cursor}&limit=500`;
    const r = await api(`/runs/${S.run}/log?${q}`);
    const body = $("log-body");
    const atBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 40;
    if (r.lines.length) {
      const fresh = r.lines.map(logLine);
      LOG.html.push(...fresh);
      LOG.n += r.lines.length;
      if (LOG.html.length > LOG_MAX_LINES) {
        // Drop the oldest so a 5,758-frame run doesn't grow the DOM without bound.
        LOG.html = LOG.html.slice(-LOG_MAX_LINES);
        body.innerHTML = LOG.html.join("\n") + "\n";
      } else {
        body.insertAdjacentHTML("beforeend", fresh.join("\n") + "\n");
      }
      // Rate from the frames' own clock, not from poll timing: frames captured in the
      // newest 10 s of what has landed.
      LOG.times.push(...r.lines.map((l) => Date.parse(l.captured_at)));
      const newest = LOG.times[LOG.times.length - 1];
      LOG.times = LOG.times.filter((t) => newest - t <= 10000);
      if (atBottom) body.scrollTop = body.scrollHeight;
    }
    LOG.cursor = r.cursor;
    const span = LOG.times.length > 1 ? (LOG.times[LOG.times.length - 1] - Math.min(...LOG.times)) / 1000 : 0;
    const rate = span > 0 ? (LOG.times.length - 1) / span : 0;
    $("log-meta").textContent = `run ${S.run.slice(0, 8)} · ${LOG.n.toLocaleString()} lines` +
      (S.live ? ` · ~${rate.toFixed(1)} frames/s landing` : " · run finished") +
      (LOG.paused ? " · paused" : "");
  } catch (e) {
    $("log-meta").textContent = `log unavailable: ${e.message}`;
  } finally {
    LOG.busy = false;
  }
}

async function boot() {
  $("btn-confirm").onclick = () => review("confirmed");
  $("btn-reject").onclick = () => review("rejected");
  $("btn-order").onclick = () => {
    const i = S.current; if (!i) return;
    S.order.has(i.cluster_key) ? S.order.delete(i.cluster_key) : S.order.set(i.cluster_key, i);
    showInstance(i.cluster_key);
  };
  $("q-prev").onclick = () => step(-1);
  $("q-next").onclick = () => step(1);
  $("export").onclick = exportCsv;
  const runs = await api("/runs");
  const sel = $("run-select");
  renderRunOptions(runs);
  sel.onchange = () => {
    // Picking anything but the newest run stops auto-follow; picking the newest resumes it.
    S.follow = sel.value === sel.options[0].value;
    loadRun(sel.value);
  };
  setInterval(tick, LIVE_MS);
  setInterval(() => { if (S.live && !document.hidden) renderPosition(); }, CAR_MS);
  setInterval(() => { if (!$("log").hidden && !LOG.paused) logPoll(); }, 1000);
  $("live").onclick = () => { $("log").hidden = !$("log").hidden; if (!$("log").hidden) logPoll(); };
  $("log-close").onclick = () => { $("log").hidden = true; };
  $("log-clear").onclick = () => { $("log-body").textContent = ""; LOG.n = 0; LOG.html = []; };
  $("log-pause").onclick = () => {
    LOG.paused = !LOG.paused;
    $("log-pause").textContent = LOG.paused ? "Resume" : "Pause";
    if (!LOG.paused) logPoll();
  };
  const first = runs[0];
  if (!first) { $("ev-caption").textContent = "No survey runs yet — run make e2e. Waiting…"; return; }
  sel.value = first.run_id;
  await loadRun(first.run_id);
  renderLive(first);
  if (location.hash === "#log") { $("log").hidden = false; logPoll(); }   // deep link for demos
}

boot().catch((e) => { $("ev-caption").textContent = `Could not load: ${e.message}`; });
