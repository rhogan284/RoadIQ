// RoadIQ dashboard (component 8). Talks only to the read API (component 7).
"use strict";

const CLASS_NAME = { D00: "Longitudinal", D10: "Transverse", D20: "Alligator",
                     D40: "Pothole", other: "Other" };
const BAND_COLOUR = { good: "#3f7f62", fair: "#c08a2e", poor: "#a83e2c" };
const UNSURVEYED = "#9aa5b1";          // darker than before so unsurveyed roads are visible
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

// "Zoom to council roads" button under the +/− controls, and the same fit on load.
const fitCtl = L.control({ position: "topleft" });
fitCtl.onAdd = () => {
  const b = L.DomUtil.create("button", "fit-btn");
  b.type = "button"; b.textContent = "⤢";
  b.title = "Zoom to council roads"; b.setAttribute("aria-label", b.title);
  L.DomEvent.disableClickPropagation(b);
  b.onclick = fitNetwork;
  return b;
};
fitCtl.addTo(map);
function fitNetwork() {
  if (!segLayer) return;
  map.invalidateSize();                         // the map box may have changed size
  const b = segLayer.getBounds();
  if (b.isValid()) map.fitBounds(b, { padding: [20, 20], maxZoom: 17, animate: false });
}
window.addEventListener("resize", () => map.invalidateSize());

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

// The planned drive: dashed ahead of the car, faint behind it. Split by distance along
// the route (frame seq × metres per frame), which is exactly how feed-sim places the car.
const planDone = L.polyline([], { color: "#0d6d92", weight: 3, opacity: 0.15, interactive: false }).addTo(map);
const planAhead = L.polyline([], { color: "#0b5f80", weight: 4.5, opacity: 0.95, dashArray: "1 9",
  lineCap: "round", interactive: false }).addTo(map);
const hav = (a, b) => {
  const r = Math.PI / 180, dla = (b[0] - a[0]) * r, dlo = (b[1] - a[1]) * r;
  const h = Math.sin(dla / 2) ** 2 + Math.cos(a[0] * r) * Math.cos(b[0] * r) * Math.sin(dlo / 2) ** 2;
  return 2 * 6371008.8 * Math.asin(Math.sqrt(h));
};
async function loadPlan() {
  S.plan = null;
  planDone.setLatLngs([]); planAhead.setLatLngs([]);
  const p = await api(`/runs/${S.run}/route`).catch(() => null);
  if (!p || !p.points || p.points.length < 2) return;
  const cum = [0];
  for (let i = 1; i < p.points.length; i++) cum.push(cum[i - 1] + hav(p.points[i - 1], p.points[i]));
  S.plan = { ...p, cum, step: p.speed_mps / p.fps };
  renderPlan(null);
}
function renderPlan(seq) {
  const P = S.plan;
  if (!P) return;
  const d = seq == null ? 0 : (seq * P.step) % P.cum[P.cum.length - 1];
  let i = 0;
  while (i < P.cum.length - 2 && P.cum[i + 1] < d) i++;
  const t = (d - P.cum[i]) / ((P.cum[i + 1] - P.cum[i]) || 1);
  const a = P.points[i], b = P.points[i + 1];
  const at = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  planDone.setLatLngs([...P.points.slice(0, i + 1), at]);
  planAhead.setLatLngs(S.live ? [at, ...P.points.slice(i + 1)] : []);
}

// Where the vehicle drove but nothing was scored: state roads ("not ours") and turns,
// using the segmenter's own snap, so the dashed line is exactly what the score ignored.
const trackLayer = L.layerGroup().addTo(map);
const OFF_COUNCIL = "#6f7d89";
async function renderTrack() {
  if (!S.run) return;
  const t = await api(`/runs/${S.run}/track`).catch(() => null);
  if (!t) return;
  trackLayer.clearLayers();
  for (const s of t.stretches) {
    if (s.council || s.points.length < 2) continue;
    L.polyline(s.points, { color: OFF_COUNCIL, weight: 4, opacity: 0.9, dashArray: "5 7" })
      .bindTooltip("Driven, not surveyed — state road or turn (not the council's)", { sticky: true })
      .addTo(trackLayer);
  }
}

async function renderPosition() {
  if (!S.run) return;
  const p = await api(`/runs/${S.run}/position`).catch(() => null);
  if (!p || !p.car) { if (carMarker) { carMarker.remove(); carMarker = null; } trailLine.setLatLngs([]); return; }
  S.feedState = p.feed_state;
  renderControls();
  const c = p.car;
  renderPlan(c.seq);
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
    row(ARTERIAL, "Not ours (state road)") +
    `<div><i style="background:repeating-linear-gradient(90deg,#0d6d92 0 2px,transparent 2px 7px)"></i>Planned route ahead</div>` +
    `<div><i style="background:repeating-linear-gradient(90deg,${OFF_COUNCIL} 0 5px,transparent 5px 9px)"></i>Driven, not surveyed</div>` +
    `<div><i class="dot" style="background:#d9a03a"></i>Defect awaiting review</div>` +
    `<div><i class="dot" style="background:#8e2f20"></i>Defect accepted</div>`;
}

function renderMap(fit) {
  if (segLayer) segLayer.remove();
  segLayer = L.geoJSON(S.geo, {
    style: (f) => {
      const c = BAND_COLOUR[f.properties.condition_band];
      return c ? { color: c, weight: 6, opacity: 0.95 }
               : { color: UNSURVEYED, weight: 4, opacity: 0.8, dashArray: "6 6" };
    },
    onEachFeature: (f, layer) => layer.on("click", () => selectSegment(f.properties.segment_id)),
  }).addTo(map);
  if (fit) fitNetwork();

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
  // No frames yet: "–" and a hint, not a "0%" that reads like a failure.
  if (!s.frames_offered) {
    $("s-acc").textContent = "–";
    $("s-acc-l").textContent = "frames accounted for · waiting for the first frame";
  } else {
    $("s-acc").textContent = `${s.frames_accounted_pct}%`;
    $("s-acc-l").textContent = `frames accounted for · ` +
      (s.frames_in_flight ? `${s.frames_in_flight.toLocaleString()} in flight · ` : "") +
      `${s.frames_dropped.toLocaleString()} dropped`;
  }
  $("s-mb").textContent = fmtBytes(s.bytes_stored);
  $("s-mb-l").textContent = s.raw_bytes ? `crops uploaded, of ${fmtBytes(s.raw_bytes)} captured`
                                        : "crops uploaded";
  $("chrome-meta").textContent = s.run.authority_id;
}

function renderWorklist() {
  const prev = S.worklist.find((w) => w.previous_at);
  $("map-note").textContent = prev
    ? `worst-first · compared with ${fmtDate(prev.previous_at, { month: "long", year: "numeric" })}`
    : "worst-first · first survey of this network";
  if (!S.worklist.length) {
    $("worklist").innerHTML = `<tr class="empty"><td colspan="6">No roads scored yet. ` +
      `Roads appear here, worst first, once the run has been scored.</td></tr>`;
    return;
  }
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
  for (const tr of $("worklist").querySelectorAll("tr[data-seg]")) {
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
  else showSegment(segmentId);                     // no defects: show the road itself
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

// Light placeholder instead of a big dark box when there's no photo to show.
function drawEmpty(msg) {
  $("frame-wrap").classList.add("empty");
  canvas.width = 960; canvas.height = 400;
  ctx.fillStyle = "#e9eef4"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#5b6674"; ctx.font = "20px Helvetica, Arial"; ctx.textAlign = "center";
  ctx.fillText(msg, canvas.width / 2, canvas.height / 2);
  ctx.textAlign = "start";
}

function setReviewEnabled(on) {
  for (const id of ["btn-confirm", "btn-reject", "btn-order", "btn-reclass", "reclass-class"]) $(id).disabled = !on;
  if (!on) {
    $("btn-confirm").textContent = "Confirm defect"; $("btn-reject").textContent = "Reject";
    $("btn-reclass").textContent = "Reclassify"; $("btn-order").textContent = "Add to work order";
    $("btn-order").classList.remove("done");
  }
}

// Back to the starting state: nothing selected.
function resetEvidence() {
  S.current = null;
  $("ev-title").textContent = "Select a defect";
  $("ev-state").textContent = "—"; $("ev-state").className = "chip muted-chip";
  $("ev-meta").innerHTML = "";
  $("ev-caption").textContent = "Click a road, a defect dot, or a work-list row.";
  drawEmpty("Nothing selected yet");
  setReviewEnabled(false);
}

// A road with no open defects: show what we know about the road instead.
function showSegment(segmentId) {
  const f = S.geo.features.find((x) => x.properties.segment_id === segmentId);
  if (!f) return;
  showTab("evidence");
  S.current = null;
  const p = f.properties, band = p.condition_band;
  $("ev-title").textContent = `${segRef(p.road_ref)} · ${p.road_name || "Unnamed road"}`;
  $("ev-state").textContent = band || "not surveyed";
  $("ev-state").className = "chip " + (BAND_COLOUR[band] ? band : "muted-chip");
  drawEmpty(band ? "No open defects on this road" : "Not surveyed in this run");
  $("ev-caption").textContent = band ? "Road summary. Defect dots on the map open their photos."
                                     : "Drive this road in a run to score it.";
  const counts = p.counts ? Object.entries(p.counts).filter(([k, v]) => k !== "pending" && v)
    .map(([k, v]) => `${v} × ${k} ${CLASS_NAME[k] ?? ""}`.trim()).join(", ") : "";
  const meta = [
    ["Condition", p.condition_index == null ? "not scored" : `${Math.round(p.condition_index)}/100 · ${band}`],
    ["Length", `${p.length_m} m`],
    ["Covered", p.coverage_m == null ? "—" : `${Math.round(p.coverage_m)} m`],
    ["Frames", p.frames_assessed == null ? "—" : `${p.frames_assessed} assessed`],
    ["Defects", counts || "none recorded"],
  ];
  $("ev-meta").innerHTML = meta.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
  setReviewEnabled(false);
  renderQueue();
}

async function showInstance(key) {
  showTab("evidence");                              // any defect click lands on Evidence
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
  $("frame-wrap").classList.remove("empty");
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
  for (const id of ["btn-confirm", "btn-reject", "btn-order", "btn-reclass", "reclass-class"]) $(id).disabled = false;
  $("btn-confirm").textContent = inst.state === "confirmed" ? "Confirmed ✓" : "Confirm defect";
  $("btn-reject").textContent = inst.state === "rejected" ? "Rejected ✓" : "Reject";
  // Reclassify: preselect the type the defect currently has (the corrected one, if any).
  const currentClass = inst.state === "reclassified" && inst.new_class ? inst.new_class : inst.defect_class;
  $("reclass-class").value = CLASS_NAME[currentClass] && currentClass !== "other" ? currentClass : "D00";
  $("btn-reclass").textContent = inst.state === "reclassified" ? "Reclassified ✓" : "Reclassify";
  $("btn-order").textContent = S.order.has(key) ? "On work order ✓" : "Add to work order";
  $("btn-order").classList.toggle("done", S.order.has(key));
  renderQueue();                                    // highlight this defect in the queue list
}

// Send a review decision. The API re-scores the whole run before replying, so this can
// take a moment: lock the buttons, say what's happening, and report a failure plainly.
async function review(state, newClass = null) {
  if (!S.current) return;
  const key = S.current.cluster_key;
  const btns = ["btn-confirm", "btn-reject", "btn-reclass", "reclass-class"];
  for (const id of btns) $(id).disabled = true;            // stop double-clicks
  $("ev-caption").textContent = "Saving decision and re-scoring the road…";
  try {
    await api(`/instances/${S.run}/${key}/review`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state, new_class: newClass, reviewed_by: "officer" }),
    });
  } catch (e) {
    $("ev-caption").textContent = `Couldn't save the decision (${e.message}). Try again.`;
    notify("error", `A review didn't save: ${e.message}`);
    for (const id of btns) $(id).disabled = false;
    return;
  }
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

// ------------------------------------------------------------------ tabs
function showTab(name) {
  for (const b of document.querySelectorAll(".tab")) b.classList.toggle("active", b.dataset.tab === name);
  for (const p of document.querySelectorAll(".tab-page")) p.hidden = p.dataset.page !== name;
}

function renderQueue() {
  $("queue-count").textContent = S.queue.length;
  const cur = S.current && S.current.cluster_key;
  $("queue-list").innerHTML = S.queue.length
    ? S.queue.map((q) => `
      <li data-key="${esc(q.cluster_key)}" class="${q.cluster_key === cur ? "sel" : ""}">
        <span><b>${esc(label(q))}</b><br><span class="muted">${esc(q.road_name || "unlocated road")}</span></span>
        <span class="muted">${q.confidence.toFixed(2)}</span>
      </li>`).join("")
    : `<li class="muted">Nothing waiting for review.</li>`;
  for (const li of $("queue-list").querySelectorAll("li[data-key]")) {
    li.onclick = () => showInstance(li.dataset.key);
  }
}

function renderHealth() {
  const s = S.summary;
  const n = (v) => (v == null ? "–" : v.toLocaleString());
  const rows = [
    ["Frames offered", n(s.frames_offered)],   ["Landed", n(s.frames_ingested)],
    ["Processed", n(s.frames_processed)],      ["In flight", n(s.frames_in_flight)],
    ["Dropped", n(s.frames_dropped)],
    ["Accounted for", s.frames_offered ? `${s.frames_accounted_pct}%` : "–"],
    ["Road assessed", `${s.assessed_km} km`],  ["Not assessed", `${s.gap_m} m`],
    ["Segments scored", n(s.segments_scored)], ["Crops uploaded", fmtBytes(s.bytes_stored)],
  ];
  $("health").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

// ------------------------------------------------------------------ bench
async function renderBench() {
  const b = await api(`/runs/${S.run}/bench`).catch(() => null);
  S.benched = !!b;
  if (!b) { $("bench").innerHTML = `<p class="muted">Benchmark results appear here after the run finishes and is scored.</p>`; return; }
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
  const sameRun = S.dataRun === S.run, prevGeo = S.geo, prevInst = S.instances;
  Object.assign(S, { summary, geo, worklist, instances });
  if (sameRun && prevGeo) checkChanges(prevGeo, prevInst);
  S.dataRun = S.run;
  S.queue = instances.filter((i) => i.state === "pending" && i.defect_class !== "other");
  renderStats(); renderMap(fit); renderWorklist(); renderQueue(); renderHealth();
  if (S.segment) highlightSegment(S.segment);
}

async function loadRun(runId) {
  Object.assign(S, { run: runId, current: null, segment: null, benched: false });
  S.order.clear();
  resetEvidence();
  await refresh(true);
  renderBench();
  await loadPlan();
  renderPosition();
  renderTrack();
  logReset();
  if (S.worklist.length) selectSegment(S.worklist[0].segment_id);
}

// ------------------------------------------------------------------ notifications
// The dashboard compares each refresh with the previous one and announces what changed.
// They live only while the page is open (the API has no event history yet).
const NOTES = { list: [], unread: 0, apiDown: false, lastExit: null };
const BAND_RANK = { poor: 1, fair: 2, good: 3 };

function notify(level, text, onClick) {
  NOTES.list.unshift({ level, text, onClick, at: new Date() });
  NOTES.list = NOTES.list.slice(0, 50);                     // keep the newest 50
  if ($("notes").hidden) NOTES.unread++;
  renderNotes();
}

function renderNotes() {
  $("bell-count").hidden = NOTES.unread === 0;
  $("bell-count").textContent = NOTES.unread;
  const ul = $("notes-list");
  if (!NOTES.list.length) { ul.innerHTML = `<li class="muted">Nothing yet.</li>`; return; }
  ul.innerHTML = NOTES.list.map((n, i) =>
    `<li data-i="${i}" class="${n.level}${n.onClick ? " clickable" : ""}">${esc(n.text)}
       <time>${n.at.toLocaleTimeString("en-AU")}</time></li>`).join("");
  for (const li of ul.querySelectorAll("li.clickable")) {
    li.onclick = () => { $("notes").hidden = true; NOTES.list[li.dataset.i].onClick(); };
  }
}

function checkChanges(prevGeo, prevInst) {
  // New defects: cluster keys we haven't seen before (summarised if many arrive at once).
  const seen = new Set(prevInst.map((i) => i.cluster_key));
  const fresh = S.instances.filter((i) => !seen.has(i.cluster_key) && i.defect_class !== "other");
  if (fresh.length > 3) {
    notify("info", `${fresh.length} new defects detected`);
  } else {
    for (const i of fresh) {
      notify(i.state === "pending" ? "warn" : "info",
        `New defect: ${label(i)} on ${i.road_name || "an unlocated road"} (${i.confidence.toFixed(2)})` +
        (i.state === "pending" ? " — needs review" : ""),
        () => showInstance(i.cluster_key));
    }
  }
  // Condition changes: a road moved between good / fair / poor.
  const before = new Map(prevGeo.features.map((f) => [f.properties.segment_id, f.properties.condition_band]));
  let firstScored = 0;
  for (const f of S.geo.features) {
    const p = f.properties, was = before.get(p.segment_id), now = p.condition_band;
    if (!BAND_RANK[now] || was === now) continue;
    if (!BAND_RANK[was]) { firstScored++; continue; }
    notify(BAND_RANK[now] < BAND_RANK[was] ? "warn" : "info",
      `${segRef(p.road_ref)} ${p.road_name || ""}: ${was} → ${now}`,
      () => selectSegment(p.segment_id));
  }
  if (firstScored) notify("info", `${firstScored} road${firstScored > 1 ? "s" : ""} scored for the first time`);
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

// A "live" run that hasn't landed a new frame for STALL_MS is stalled: the feed died or was
// never started. Flag it, and let the officer cancel it and start a new run.
const STALL_MS = 3 * 60 * 1000;
function checkStall(run) {
  const was = S.stalled;
  if (!run || run.ended_at) { S.stalled = false; return; }
  const n = S.summary.frames_ingested;
  if (S.stallRun !== run.run_id || n !== S.lastFrames) {
    S.stallRun = run.run_id; S.lastFrames = n; S.lastProgressAt = Date.now();
  }
  const quiet = Date.now() - S.lastProgressAt;
  const age = Date.now() - Date.parse(run.started_at);
  S.stalled = S.feedState !== "paused" && ((n === 0 && age > STALL_MS) || quiet > STALL_MS);
  if (S.stalled && !was) notify("warn", "This run has stalled: no new frames for over 3 minutes. Cancel it to start a new run.");
}

function renderLive(run) {
  const live = run && !run.ended_at;
  S.live = !!live;
  checkStall(run);
  const badge = $("live");
  badge.hidden = !live;
  badge.classList.toggle("stalled", !!S.stalled);
  if (live && S.stalled) {
    badge.textContent = `⚠ Stalled · ${S.summary.frames_ingested.toLocaleString()} frames`;
  } else if (live) {
    const paused = S.feedState === "paused";
    badge.textContent = `${paused ? "❚❚ PAUSED" : "● LIVE"} · ${S.summary.frames_ingested.toLocaleString()} frames landed`;
    badge.classList.toggle("paused", paused);
  }
  renderControls();
}

function renderControls() {
  const live = S.live && S.feedState !== "cancelled" && S.feedState !== "done";
  $("run-ctl").hidden = !live;
  const paused = S.feedState === "paused";
  $("ctl-pause").textContent = paused ? "▶ Resume" : "❚❚ Pause";
  const badge = $("live");
  if (S.live && badge.textContent) {
    badge.textContent = badge.textContent.replace(/^(● LIVE|❚❚ PAUSED)/, paused ? "❚❚ PAUSED" : "● LIVE");
    badge.classList.toggle("paused", paused);
  }
}

async function runControl(action) {
  if (!S.run) return;
  if (action === "cancel" && !confirm("Cancel this run? Frames already sent are kept, " +
      "segmented and scored; the rest of the dataset is not replayed.")) return;
  for (const b of ["ctl-pause", "ctl-cancel"]) $(b).disabled = true;
  try {
    await api(`/runs/${S.run}/control`, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action }) });
  } catch (e) {
    alert(`Could not ${action}: ${e.message}`);
  } finally {
    setTimeout(() => { for (const b of ["ctl-pause", "ctl-cancel"]) $(b).disabled = false; }, 800);
  }
}

// ------------------------------------------------------------------ new run
// "Starting…" until feed-sim registers the run (a few seconds of route planning), and
// "Scoring…" after the feed ends while the runner drains, segments and benchmarks.
async function renderRunner() {
  const r = await api("/runner").catch(() => null);
  if (r && r.exit_code && r.exit_code !== "0" && r.exit_code !== NOTES.lastExit) {
    notify("error", `The runner stopped with an error (exit ${r.exit_code}). Open the frame log to see why.`);
    NOTES.lastExit = r.exit_code;
  }
  const note = $("runner-note");
  const feeding = S.live && !S.stalled && !["cancelled", "done"].includes(S.feedState);
  let text = "";
  if (r && r.alive) {
    if (r.run_id !== S.run) text = "Starting run…";
    else if (!feeding) text = "Scoring run…";
  }
  note.hidden = !text;
  note.textContent = text;
  $("new-run").hidden = feeding || !!(r && r.alive);
}

async function startRun(form) {
  const f = new FormData(form);
  const body = { fps: Number(f.get("fps")), route_mode: f.get("route_mode"),
                 max_frames: f.get("max_frames") ? Number(f.get("max_frames")) : null,
                 route_seed: f.get("route_seed") === "" ? null : Number(f.get("route_seed")) };
  const r = await fetch("/api/runs/start", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  S.follow = true;                        // jump to the new run as soon as it registers
  renderRunner();
  return r.json();
}

async function tick() {
  if (S.busy || document.hidden) return;
  S.busy = true;
  try {
    const runs = await api("/runs");
    if (NOTES.apiDown) { NOTES.apiDown = false; notify("info", "Connection to the system restored"); }
    renderRunOptions(runs);
    const newest = runs[0];
    if (newest && S.follow && newest.run_id !== S.run) {
      $("run-select").value = newest.run_id;
      await loadRun(newest.run_id);
    } else if (S.run) {
      await refresh(false);
      if (!S.current && S.segment == null && S.worklist.length) selectSegment(S.worklist[0].segment_id);
      if (!S.benched) renderBench();
      if (S.live) renderTrack();
    }
    renderLive(runs.find((r) => r.run_id === S.run));
    renderRunner();
  } catch (e) {
    console.warn("live refresh failed", e);
    if (!NOTES.apiDown) notify("error", `Lost connection to the system (${e.message})`);
    NOTES.apiDown = true;
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
  $("btn-reclass").onclick = () => review("reclassified", $("reclass-class").value);
  $("btn-order").onclick = () => {
    const i = S.current; if (!i) return;
    S.order.has(i.cluster_key) ? S.order.delete(i.cluster_key) : S.order.set(i.cluster_key, i);
    showInstance(i.cluster_key);
  };
  $("q-prev").onclick = () => step(-1);
  $("q-next").onclick = () => step(1);
  $("export").onclick = exportCsv;
  for (const b of document.querySelectorAll(".tab")) b.onclick = () => showTab(b.dataset.tab);
  $("bell").onclick = () => { $("notes").hidden = !$("notes").hidden; NOTES.unread = 0; renderNotes(); };
  $("notes-clear").onclick = () => { NOTES.list = []; NOTES.unread = 0; renderNotes(); };
  resetEvidence();
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
  $("new-run").onclick = () => { $("new-run-err").textContent = ""; $("new-run-dlg").showModal(); };
  $("new-run-go").onclick = async (ev) => {
    ev.preventDefault();
    $("new-run-go").disabled = true;
    try {
      await startRun($("new-run-form"));
      $("new-run-dlg").close();
    } catch (e) {
      $("new-run-err").textContent = `Could not start: ${e.message}`;
    } finally {
      $("new-run-go").disabled = false;
    }
  };
  $("ctl-pause").onclick = () => runControl(S.feedState === "paused" ? "resume" : "pause");
  $("ctl-cancel").onclick = () => runControl("cancel");
  $("live").onclick = () => { $("log").hidden = !$("log").hidden; if (!$("log").hidden) logPoll(); };
  $("log-close").onclick = () => { $("log").hidden = true; };
  $("log-clear").onclick = () => { $("log-body").textContent = ""; LOG.n = 0; LOG.html = []; };
  $("log-pause").onclick = () => {
    LOG.paused = !LOG.paused;
    $("log-pause").textContent = LOG.paused ? "Resume" : "Pause";
    if (!LOG.paused) logPoll();
  };
  const first = runs[0];
  if (!first) { $("ev-caption").textContent = "No survey runs yet — press ＋ New run."; return; }
  sel.value = first.run_id;
  await loadRun(first.run_id);
  renderLive(first);
  renderRunner();
  if (location.hash === "#new") $("new-run").click();                      // deep link
  if (location.hash === "#log") { $("log").hidden = false; logPoll(); }   // deep link for demos
}

boot().catch((e) => { $("ev-caption").textContent = `Could not load: ${e.message}`; });