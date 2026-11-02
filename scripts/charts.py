"""Charts for the C4 report's Validation and Testing section, from a sweep's files.

    make charts BENCH_OUT=bench-out/2026-11-02
    uv run --with matplotlib python -m scripts.charts bench-out/2026-11-02

Reads only the files scripts/sweep.py and edgecv.bench.export wrote — no database —
and writes <out>/charts/*.png (200 dpi, sized for a Word page) plus <out>/results.md
(the same numbers as tables, for the report and for readers who cannot see colour).

A chart whose data is missing is skipped with a message, never drawn empty.

Colour: the validated light categorical palette (slots 1-5 pass CVD and normal-vision
separation on adjacent pairs; slots 3-5 are under 3:1 contrast on the surface, so
every series is also direct-labelled or named in a legend, and results.md is the
table view). One hue per entity everywhere: "1 worker" is the same blue in every
chart, "inference" the same orange.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --- tokens ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
             "#184f95", "#0d366b"]
CRITICAL = "#d03b3b"

STAGE_COLOR = {"queue_ms": SERIES[0], "infer_ms": SERIES[1], "write_ms": SERIES[2]}
STAGE_NAME = {"queue_ms": "Queue wait", "infer_ms": "Inference", "write_ms": "DB write",
              "e2e_ms": "End to end"}
WORKER_COLOR = {1: SERIES[0], 2: SERIES[1], 3: SERIES[2], 4: SERIES[3]}
SERVICE_ORDER = ["worker", "writer", "postgres", "redis", "other"]
SERVICE_COLOR = dict(zip(SERVICE_ORDER, SERIES))
CLASS_COLOR = {"D00": SERIES[0], "D10": SERIES[1], "D20": SERIES[2], "D40": SERIES[3]}
CLASS_NAME = {"D00": "D00 longitudinal crack", "D10": "D10 transverse crack",
              "D20": "D20 alligator crack", "D40": "D40 pothole"}
WIDTH_IN = 7.2

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.facecolor": SURFACE,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6,
    "lines.markersize": 5.5, "text.color": INK,
})


# --- loading --------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def num(v):
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


class Sweep:
    def __init__(self, root: Path):
        self.root = root
        self.runs = read_csv(root / "runs.csv")
        hw = root / "hardware.json"
        self.hw = json.loads(hw.read_text()) if hw.exists() else {}

    def grid(self, name: str) -> list[dict]:
        return [r for r in self.runs if r["grid"] == name]

    def first(self, name: str) -> dict | None:
        rows = self.grid(name)
        return rows[0] if rows else None

    def file(self, label: str, name: str) -> Path:
        return self.root / label / name

    def summary(self, label: str) -> dict:
        return json.loads(self.file(label, "summary.json").read_text())

    def point(self, label: str) -> dict:
        return json.loads(self.file(label, "point.json").read_text())

    def footnote(self) -> str:
        h = self.hw
        if not h:
            return ""
        return (f"{h.get('host_cpu')} · Docker VM {h.get('docker_vm_cpus')} vCPU / "
                f"{h.get('docker_vm_mem_gb')} GB · YOLOv12s on CPU "
                f"({h.get('worker_threads')} threads per worker), no GPU · "
                f"weights {Path(h.get('weights', '')).name} · commit {h.get('commit')}")


# --- figure helpers -------------------------------------------------------------------

def figure(sweep: Sweep, title: str, subtitle: str, *, ncols: int = 1, nrows: int = 1,
           height: float = 3.6, sharex: bool = False, sharey: bool = False):
    """Title, subtitle and footnote sit at fixed distances in INCHES from the edges,
    so they never collide with the plot whatever the figure's height."""
    import textwrap
    lines = textwrap.wrap(subtitle, 118)
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH_IN, height), sharex=sharex,
                             sharey=sharey, squeeze=False)
    fig.text(0.012, 1 - 0.10 / height, title, ha="left", va="top", fontsize=11.5,
             weight="bold", color=INK)
    fig.text(0.012, 1 - 0.36 / height, "\n".join(lines), ha="left", va="top",
             fontsize=8.5, color=INK_2, linespacing=1.35)
    fig._top_in = 0.36 + 0.165 * len(lines) + 0.3
    note = sweep.footnote()
    if note:
        fig.text(0.012, 0.04 / height, note, ha="left", va="bottom", fontsize=6.3,
                 color=MUTED)
    return fig, axes


def fig_legend(fig, handles, labels, ncols: int = 4):
    """A figure-level legend in the band above the footnote."""
    h = fig.get_figheight()
    fig.legend(handles, labels, loc="lower center", ncols=ncols,
               bbox_to_anchor=(0.5, 0.22 / h))


def save(fig, out: Path, name: str, bottom_in: float = 0.6) -> Path:
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - fig._top_in / h, bottom=bottom_in / h, left=0.1,
                        right=0.97, wspace=0.34, hspace=0.5)
    path = out / name
    # tight: long tick labels on the left or value labels on the right are kept,
    # never clipped.
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def service_group(svc: str) -> str:
    return svc if svc in ("worker", "writer", "postgres", "redis") else "other"


# --- A1: latency distribution ---------------------------------------------------------

def chart_latency_cdf(s: Sweep, out: Path):
    base = s.first("baseline")
    if not base:
        return print("  skip latency CDF: no baseline run")
    frames = read_csv(s.file(base["label"], "frames.csv"))
    e2e_all = [num(f["e2e_ms"]) for f in frames if num(f["e2e_ms"]) is not None]
    p50, p95 = np.percentile(e2e_all, 50), np.percentile(e2e_all, 95)
    fig, axes = figure(s, f"Half of all frames are in the database within {p50:,.0f} ms; "
                          f"95% within {p95 / 1000:.1f} s",
                       f"Cumulative share of frames by latency · baseline run, "
                       f"{len(frames):,} frames, {base['workers']} workers, "
                       f"{num(base['offered_fps']):g} fps offered", height=3.7)
    ax = axes[0][0]
    for i, key in enumerate(["infer_ms", "e2e_ms"]):
        v = np.sort([x for x in (num(f[key]) for f in frames) if x is not None])
        if not v.size:
            continue
        y = np.arange(1, v.size + 1) / v.size * 100
        color = SERIES[1] if key == "infer_ms" else SERIES[0]
        ax.plot(v, y, color=color, label=STAGE_NAME[key])
        for p in (50, 95):
            xv = float(np.percentile(v, p))
            ax.plot([xv], [p], "o", color=color, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, zorder=5)
            ax.annotate(f"p{p} {xv:,.0f} ms", (xv, p), xytext=(6, -10 if i else 6),
                        textcoords="offset points", fontsize=7.5, color=INK_2)
    hi = max(float(np.percentile([num(f["e2e_ms"]) or 0 for f in frames], 99.5)), 1)
    ax.set_xlim(0, hi * 1.08)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Latency (ms) — axis cut at the 99.5th percentile")
    ax.set_ylabel("Frames at or below (%)")
    ax.legend(loc="lower right")
    return save(fig, out, "01_latency_cdf.png", bottom_in=0.65)


# --- B1: where the time goes ----------------------------------------------------------

def chart_latency_stages(s: Sweep, out: Path):
    base = s.first("baseline")
    if not base:
        return print("  skip stage breakdown: no baseline run")
    frames = [f for f in read_csv(s.file(base["label"], "frames.csv"))
              if num(f["e2e_ms"]) is not None]
    if not frames:
        return print("  skip stage breakdown: no written_at (run predates migration 003)")
    e2e = np.array([num(f["e2e_ms"]) for f in frames])
    bands = [("Typical frame\n(p45–p55)", 45, 55), ("Slow frame\n(p92.5–p97.5)", 92.5, 97.5),
             ("Slowest 1%\n(p99+)", 99, 100)]
    rows = []
    for name, lo, hi in bands:
        a, b = np.percentile(e2e, lo), np.percentile(e2e, hi)
        sel = [f for f, v in zip(frames, e2e) if a <= v <= b]
        rows.append((name, {k: float(np.median([num(f[k]) for f in sel]))
                            for k in ("queue_ms", "infer_ms", "write_ms")}, len(sel)))
    slow = rows[1][1]
    fig, axes = figure(s, "Slow frames are slow because they wait in the queue"
                       if slow["queue_ms"] > slow["infer_ms"] else "Where a frame's time goes",
                       "Median time in each stage, for frames grouped by end-to-end "
                       "latency · baseline run", height=3.3)
    ax = axes[0][0]
    ax.grid(axis="y", visible=False)
    for yi, (name, st, n) in enumerate(rows[::-1]):
        left = 0.0
        for k in ("queue_ms", "infer_ms", "write_ms"):
            w = st[k]
            ax.barh(yi, w, left=left, height=0.55, color=STAGE_COLOR[k],
                    edgecolor=SURFACE, linewidth=1.5,
                    label=STAGE_NAME[k] if yi == 0 else None)
            if w > 0.09 * max(sum(r[1].values()) for r in rows):
                ax.text(left + w / 2, yi, f"{w:,.0f}", ha="center", va="center",
                        fontsize=7.5, color="#ffffff" if k != "write_ms" else INK)
            left += w
        ax.text(left, yi, f"  {left:,.0f} ms", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.set_xlabel("Milliseconds")
    ax.set_xlim(0, max(sum(r[1].values()) for r in rows) * 1.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=3)
    return save(fig, out, "02_latency_stages.png", bottom_in=1.0)


# --- B2: throughput ceiling -----------------------------------------------------------

def _by_workers(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[int(num(r["workers"]))].append(r)
    for w in groups:
        groups[w].sort(key=lambda r: num(r["offered_fps"]))
    return dict(sorted(groups.items()))


def chart_throughput(s: Sweep, out: Path):
    rows = s.grid("throughput")
    if not rows:
        return print("  skip throughput: no throughput grid")
    groups = _by_workers(rows)
    fps_max = max(num(r["offered_fps"]) for r in rows)
    buffer = rows[0]["buffer"]
    ceil = {w: max(num(r["achieved_fps"]) for r in g) for w, g in groups.items()}
    title = "Throughput ceiling: " + " · ".join(
        f"{c:.1f} fps with {w} worker{'s' if w > 1 else ''}" for w, c in ceil.items())
    fig, axes = figure(s, title,
                       f"Offered vs achieved rate, frames dropped, and p95 end-to-end "
                       f"latency · {rows[0]['frames_offered']} frames per point, "
                       f"buffer {buffer} frames", ncols=3, height=3.5)
    a1, a2, a3 = axes[0]
    a1.plot([0, fps_max], [0, fps_max], color=AXIS, linewidth=1, zorder=1)
    a1.text(fps_max * 0.97, fps_max * 0.92, "keeps up", rotation=0, ha="right",
            fontsize=7, color=MUTED)
    for w, g in groups.items():
        x = [num(r["offered_fps"]) for r in g]
        c = WORKER_COLOR.get(w, SERIES[w % len(SERIES)])
        lab = f"{w} worker{'s' if w > 1 else ''}"
        a1.plot(x, [num(r["achieved_fps"]) for r in g], "-o", color=c, label=lab,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        a2.plot(x, [num(r["drop_pct"]) for r in g], "-o", color=c,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        a3.plot(x, [num(r["e2e_p95_ms"]) / 1000 for r in g], "-o", color=c,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    a1.set_title("Achieved fps", fontsize=9)
    a2.set_title("Frames dropped (%)", fontsize=9)
    a3.set_title("p95 end to end (s, log)", fontsize=9)
    a3.set_yscale("log")
    for a in (a1, a2, a3):
        a.set_xlabel("Offered fps")
        a.set_xticks(sorted({num(r["offered_fps"]) for r in rows}))
    a1.set_xlim(0, fps_max * 1.1)
    a1.set_ylim(0, fps_max * 1.05)
    a2.set_ylim(bottom=0)
    fig_legend(fig, *a1.get_legend_handles_labels())
    return save(fig, out, "03_throughput_ceiling.png", bottom_in=1.0)


# --- A2: coverage ---------------------------------------------------------------------

def chart_coverage(s: Sweep, out: Path):
    rows = [r for r in s.runs if num(r["frames_offered"])]
    if not rows:
        return print("  skip coverage: no runs")
    order = {"baseline": 0, "chaos": 1, "prevalence": 2, "throughput": 3}
    rows.sort(key=lambda r: (order.get(r["grid"], 9), num(r["workers"]) or 0,
                             num(r["prevalence"]) or 0, num(r["offered_fps"]) or 0))
    fig, axes = figure(s, "Every frame is accounted for: landed or counted as dropped",
                       "Share of offered frames, per run · a dropped frame is refused at "
                       "the buffer and reported as metres of road not assessed",
                       height=1.9 + 0.26 * len(rows))
    ax = axes[0][0]
    ax.grid(axis="y", visible=False)
    for yi, r in enumerate(rows[::-1]):
        off = num(r["frames_offered"])
        landed = num(r["frames_landed"]) / off * 100
        dropped = num(r["frames_dropped"]) / off * 100
        lost = max(0.0, 100 - landed - dropped)
        ax.barh(yi, landed, color=SERIES[0], height=0.62, edgecolor=SURFACE,
                linewidth=1.2, label="Landed in Postgres" if yi == 0 else None)
        ax.barh(yi, dropped, left=landed, color=SERIES[1], height=0.62, edgecolor=SURFACE,
                linewidth=1.2, label="Refused at buffer (counted)" if yi == 0 else None)
        if lost > 0.05:
            ax.barh(yi, lost, left=landed + dropped, color=CRITICAL, height=0.62,
                    label="Unaccounted" if yi == 0 else None)
        ax.text(101, yi, f"{int(num(r['frames_landed'])):,} / {int(off):,}",
                va="center", fontsize=7, color=INK_2)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows[::-1]], fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of offered frames")
    fig_legend(fig, *ax.get_legend_handles_labels(), ncols=3)
    return save(fig, out, "04_coverage_by_run.png", bottom_in=0.95)


# --- A3: chaos ------------------------------------------------------------------------

def chart_chaos(s: Sweep, out: Path):
    r = s.first("chaos")
    if not r:
        return print("  skip chaos: no chaos run")
    bus = read_csv(s.file(r["label"], "bus.csv"))
    pt = s.point(r["label"])
    if not bus or not pt.get("kill"):
        return print("  skip chaos: no bus samples or no kill recorded")
    t0 = pt["t_feed_start"]
    t = np.array([num(b["t"]) - t0 for b in bus])
    kill_t = pt["kill"]["t"] - t0
    pending = np.array([num(b["pending"]) or 0 for b in bus])
    backlog = np.array([num(b["stream_len"]) or 0 for b in bus])
    lag_unknown = np.array([num(b["lag"]) is None for b in bus])
    landed = np.array([num(b["landed"]) for b in bus])
    offered = int(num(r["frames_offered"]))
    fps = num(r["offered_fps"])
    published_n = offered - int(num(r["frames_dropped"]))
    fig, axes = figure(s, f"One of two workers killed mid-run: {int(landed[-1]):,} of "
                          f"{published_n:,} frames still landed",
                       f"worker-1 SIGKILLed {kill_t:.0f} s into a {offered:,}-frame run "
                       f"(2 workers, {fps:g} fps). The survivor carries the load, the "
                       f"backlog grows, and the dead worker's claimed frames stay pending "
                       f"until the survivor reclaims them with XAUTOCLAIM.",
                       nrows=2, height=4.6, sharex=True)
    a1, a2 = axes[0][0], axes[1][0]
    a1.plot(t, backlog, color=SERIES[0], label="Backlog on the stream (XLEN)")
    a1.plot(t, pending, color=SERIES[1], linewidth=1.0,
            label="Pending (claimed, not acked)")
    if lag_unknown.any():
        a1.fill_between(t, 0, 1, where=lag_unknown, transform=a1.get_xaxis_transform(),
                        color=GRID, alpha=0.45, linewidth=0, zorder=0,
                        label="Redis reports lag as unknown")
    a1.set_ylabel("Frames on the bus")
    a1.legend(loc="upper left", fontsize=7.5)
    published = np.minimum(np.clip(t, 0, None) * fps, offered)
    a2.plot(t, published, color=AXIS, linewidth=1.2, label="Published by feed-sim")
    a2.plot(t, landed, color=SERIES[0], label="Landed in Postgres")
    a2.set_ylabel("Frames (cumulative)")
    a2.set_xlabel("Seconds since feed start")
    a2.legend(loc="lower right")
    for a in (a1, a2):
        a.axvline(kill_t, color=CRITICAL, linewidth=1)
    a2.annotate("worker-1 killed", (kill_t, a2.get_ylim()[1] * 0.85), xytext=(4, 0),
                textcoords="offset points", fontsize=7.5, color=CRITICAL)
    # The reclaim: the first moment after the kill that pending drops back to zero.
    after = np.where((t > kill_t) & (pending == 0))[0]
    peak_i = int(np.argmax(np.where(t > kill_t, pending, -1)))
    if after.size and pending[peak_i] > 0:
        rec_i = after[after > peak_i][0] if (after > peak_i).any() else after[-1]
        a1.annotate(f"drained · pending back to 0 at {t[rec_i]:.0f} s",
                    (t[rec_i], 0), xytext=(6, 30), textcoords="offset points",
                    fontsize=7.5, color=INK_2,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6})
    a2.plot([t[-1]], [landed[-1]], "o", color=SERIES[0], markeredgecolor=SURFACE,
            markeredgewidth=1.2, zorder=5)
    a2.annotate(f"{int(landed[-1]):,} landed of {offered - int(num(r['frames_dropped'])):,} "
                f"published", (t[-1], landed[-1]), xytext=(0, 9), textcoords="offset points",
                ha="right", fontsize=7.5, color=INK_2)
    return save(fig, out, "05_chaos_timeline.png", bottom_in=0.65)


# --- A4/A5: storage vs prevalence -----------------------------------------------------

def chart_storage(s: Sweep, out: Path):
    rows = sorted(s.grid("prevalence"), key=lambda r: num(r["prevalence"]))
    if not rows:
        return print("  skip storage: no prevalence grid")
    sums = {r["label"]: s.summary(r["label"]) for r in rows}
    x = [num(r["prevalence"]) * 100 for r in rows]
    raw = [sums[r["label"]]["bytes"]["raw_per_km"] / 1e6 for r in rows]
    stored = [num(r["bytes_stored"]) / 1e6 / num(r["assessed_km"]) for r in rows]
    red = [num(r["reduction_x"]) for r in rows]
    fig, axes = figure(s, f"Storage reduction falls from {red[0]:.1f}× to {red[-1]:.1f}× "
                          f"as the share of damaged frames rises",
                       "Bytes kept per km of road, by share of frames with damage · "
                       "raw = every JPEG offered; stored = crops + thumbnails only",
                       ncols=2, height=3.5)
    a1, a2 = axes[0]
    idx = np.arange(len(rows))
    a1.bar(idx - 0.2, raw, 0.38, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5,
           label="Raw frames")
    a1.bar(idx + 0.2, stored, 0.38, color=SERIES[1], edgecolor=SURFACE, linewidth=1.5,
           label="Stored by RoadIQ")
    for i in idx:
        a1.text(i - 0.2, raw[i] * 1.12, f"{raw[i]:,.0f}", ha="center", fontsize=7,
                color=INK_2)
        a1.text(i + 0.2, stored[i] * 1.12, f"{stored[i]:,.1f}", ha="center", fontsize=7,
                color=INK_2)
    a1.set_yscale("log")
    a1.set_xticks(idx, [f"{v:.0f}% damaged" for v in x])
    a1.set_ylabel("MB per km (log)")
    a1.grid(axis="x", visible=False)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2)
    a2.plot(x, red, "-o", color=SERIES[0], markeredgecolor=SURFACE, markeredgewidth=1.2)
    for xi, ri in zip(x, red):
        a2.annotate(f"{ri:.1f}×", (xi, ri), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color=INK_2)
    a2.set_xlabel("Frames with damage (%)")
    a2.set_ylabel("Reduction (raw ÷ stored)")
    a2.set_ylim(0, max(red) * 1.25)
    a2.set_xlim(0, 100)
    return save(fig, out, "06_storage_vs_prevalence.png", bottom_in=0.95)


# --- A6: accuracy ---------------------------------------------------------------------

def chart_accuracy(s: Sweep, out: Path):
    base = s.first("baseline")
    if not base:
        return print("  skip accuracy: no baseline run")
    summ = s.summary(base["label"])
    per = (summ.get("accuracy") or {}).get("per_class") or {}
    if not per:
        return print("  skip accuracy: no per-class scores")
    classes = [c for c in ("D00", "D10", "D20", "D40") if c in per]
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    fig, axes = figure(s, "Detector accuracy by defect class",
                       f"Box level, IoU ≥ 0.5, confidence ≥ 0.25 · {summ['frames_landed']:,} "
                       f"RDD2022 test images · D40 has no ground truth in this split, "
                       f"so it is not scored", height=3.5)
    ax = axes[0][0]
    ax.grid(axis="x", visible=False)
    idx = np.arange(len(classes))
    w = 0.26
    for j, (k, name) in enumerate(metrics):
        vals = [per[c][k] for c in classes]
        ax.bar(idx + (j - 1) * w, vals, w * 0.95, color=SERIES[j], edgecolor=SURFACE,
               linewidth=1.5, label=name)
        for i, v in enumerate(vals):
            ax.text(idx[i] + (j - 1) * w, v + 0.015, f"{v:.2f}", ha="center",
                    fontsize=6.8, color=INK_2)
    ax.set_xticks(idx, [f"{CLASS_NAME[c]}\n(n = {per[c]['gt']:,} boxes)" for c in classes])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(loc="upper right", ncols=3)
    return save(fig, out, "07_accuracy_per_class.png", bottom_in=0.8)


def chart_pr(s: Sweep, out: Path):
    base = s.first("baseline")
    if not base:
        return print("  skip PR curve: no baseline run")
    curve = read_csv(s.file(base["label"], "pr_curve.csv"))
    summ = s.summary(base["label"])
    if not curve:
        return print("  skip PR curve: empty")
    fig, axes = figure(s, "Precision–recall by defect class",
                       "Each line sweeps the confidence threshold from high (left) down "
                       "to the pipeline's 0.25 floor (line end). AP50 is the area under "
                       "the line, so it is lower than a full-range mAP50.", height=3.9)
    ax = axes[0][0]
    by = defaultdict(list)
    for row in curve:
        by[row["class"]].append((num(row["recall"]), num(row["precision"]),
                                 num(row["confidence"])))
    for c, pts in by.items():
        r, p, conf = zip(*pts)
        ax.plot(r, p, color=CLASS_COLOR[c],
                label=f"{CLASS_NAME[c]} · AP50 {summ['ap50'].get(c, 0):.2f}")
        ax.plot([r[-1]], [p[-1]], "o", color=CLASS_COLOR[c], markeredgecolor=SURFACE,
                markeredgewidth=1.2)
        # The operating point the deployed pipeline uses is the line end (conf 0.25).
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    if summ.get("map50_scored") is not None:
        ax.text(0.98, 0.97, f"mean AP50 (scored classes) {summ['map50_scored']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK)
    return save(fig, out, "08_pr_curves.png", bottom_in=0.6)


def chart_confusion(s: Sweep, out: Path):
    base = s.first("baseline")
    if not base:
        return print("  skip confusion: no baseline run")
    rows = read_csv(s.file(base["label"], "confusion.csv"))
    if not rows:
        return print("  skip confusion: empty")
    counts = {(r["gt_class"], r["pred_class"]): int(num(r["count"])) for r in rows}
    gts = sorted({g for g, _ in counts if g != "background"}) + ["background"]
    preds = sorted({p for _, p in counts if p != "missed"}) + ["missed"]
    m = np.array([[counts.get((g, p), 0) for p in preds] for g in gts], dtype=float)
    share = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    fig, axes = figure(s, "What the detector calls each kind of damage",
                       "Rows: ground truth. Columns: prediction. Shade = share of the row; "
                       "the number is the box count. IoU ≥ 0.5, class-agnostic matching.",
                       height=4.3)
    ax = axes[0][0]
    ax.grid(False)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(gts)):
        for j in range(len(preds)):
            if m[i, j] == 0:
                continue
            ax.text(j, i, f"{int(m[i, j]):,}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if share[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(preds)), [p if p != "missed" else "missed\n(no box)"
                                      for p in preds])
    ax.set_yticks(range(len(gts)), [g if g != "background" else "background\n(false alarm)"
                                    for g in gts])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for sp in ax.spines.values():
        sp.set_visible(False)
    return save(fig, out, "09_confusion_matrix.png", bottom_in=0.75)


# --- B3/B4: resources ------------------------------------------------------------------

def _resource_series(samples: list[dict], t0: float):
    """Per service group, per second: each container's readings in that second are
    averaged first (docker stats refreshes ~twice a second), then summed across
    containers — the same rule as edgecv.bench.resources.summarise."""
    per = defaultdict(list)
    for sm in samples:
        per[(service_group(sm["service"]), sm["container"],
             int(num(sm["t"]) - t0))].append(sm)
    cpu = defaultdict(lambda: defaultdict(float))
    mem = defaultdict(lambda: defaultdict(float))
    for (g, _c, sec), group in per.items():
        cpu[g][sec] += sum(num(x["cpu_pct"]) for x in group) / len(group)
        mem[g][sec] += sum(num(x["mem_mb"]) for x in group) / len(group)
    return cpu, mem


def chart_resources_timeline(s: Sweep, out: Path):
    base = s.first("baseline")
    if not base:
        return print("  skip resource timeline: no baseline run")
    samples = read_csv(s.file(base["label"], "resources.csv"))
    pt = s.point(base["label"])
    if not samples:
        return print("  skip resource timeline: no samples")
    t0 = pt["t_feed_start"]
    cpu, mem = _resource_series(samples, t0)
    fig, axes = figure(s, "The workers do almost all the work",
                       f"CPU and memory per service over the baseline run · CPU is % of "
                       f"one core (a 6-thread worker flat out ≈ 600%) · memory is the "
                       f"container cgroup, summed over replicas",
                       nrows=2, height=4.8, sharex=True)
    a1, a2 = axes[0][0], axes[1][0]
    for g in SERVICE_ORDER:
        if g not in cpu:
            continue
        xs = sorted(cpu[g])
        a1.plot(xs, [cpu[g][x] for x in xs], color=SERVICE_COLOR[g], linewidth=1.3)
        a2.plot(xs, [mem[g][x] / 1024 for x in xs], color=SERVICE_COLOR[g], linewidth=1.3,
                label=g)
    feed_end = pt["t_feed_end"] - t0
    for a in (a1, a2):
        a.axvspan(0, feed_end, color=GRID, alpha=0.35, linewidth=0, zorder=0)
    a1.text(2, a1.get_ylim()[1] * 0.97, "feed running", fontsize=7, color=MUTED, va="top")
    a1.set_ylabel("CPU (% of one core)")
    a2.set_ylabel("Memory (GB)")
    a2.set_xlabel("Seconds since feed start")
    fig_legend(fig, *a2.get_legend_handles_labels(), ncols=5)
    return save(fig, out, "10_resources_timeline.png", bottom_in=0.95)


def chart_resources_per_service(s: Sweep, out: Path):
    base = s.first("baseline")
    if not base:
        return print("  skip resources per service: no baseline run")
    res = s.point(base["label"]).get("resources") or {}
    groups = defaultdict(lambda: {"cpu_mean_pct": 0.0, "cpu_peak_pct": 0.0,
                                  "mem_total_peak_mb": 0.0})
    for svc, v in res.items():
        if svc == "pipeline":
            continue
        g = groups[service_group(svc)]
        g["cpu_mean_pct"] += v["cpu_mean_pct"]
        g["cpu_peak_pct"] += v["cpu_peak_pct"]
        g["mem_total_peak_mb"] += v["mem_total_peak_mb"]
    order = [g for g in SERVICE_ORDER if g in groups]
    if not order:
        return print("  skip resources per service: no summary")
    fig, axes = figure(s, "Cost of each service during the baseline run",
                       "Mean CPU while the run was live (tick = peak second), and peak "
                       "memory · summed over replicas", ncols=2, height=3.2)
    a1, a2 = axes[0]
    y = np.arange(len(order))[::-1]
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
        a.set_yticks(y, order)
    means = [groups[g]["cpu_mean_pct"] for g in order]
    peaks = [groups[g]["cpu_peak_pct"] for g in order]
    mems = [groups[g]["mem_total_peak_mb"] / 1024 for g in order]
    a1.barh(y, means, height=0.55, color=SERIES[0])
    a1.scatter(peaks, y, marker="|", s=120, color=INK_2, linewidths=1.2, zorder=4)
    for yi, m_, p_ in zip(y, means, peaks):
        a1.text(max(m_, p_), yi, f"  {m_:,.0f}%", va="center", fontsize=7.5, color=INK_2)
    a1.set_xlabel("CPU (% of one core)")
    a1.set_xlim(0, max(peaks) * 1.3)
    a2.barh(y, mems, height=0.55, color=SERIES[0])
    for yi, v in zip(y, mems):
        a2.text(v, yi, f"  {v:.2f} GB", va="center", fontsize=7.5, color=INK_2)
    a2.set_xlabel("Peak memory (GB)")
    a2.set_xlim(0, max(mems) * 1.35)
    return save(fig, out, "11_resources_per_service.png", bottom_in=0.65)


def chart_cpu_vs_rate(s: Sweep, out: Path):
    rows = s.grid("throughput")
    if not rows:
        return print("  skip CPU vs rate: no throughput grid")
    groups = _by_workers(rows)
    cost = {w: np.mean([(num(r["worker_cpu_mean_pct"]) / 100) / num(r["achieved_fps"])
                        for r in g if num(r["offered_fps"]) >= 8]) for w, g in groups.items()}
    ws = sorted(cost)
    title = (f"More workers cost more CPU per frame: {cost[ws[0]]:.2f} core-s with "
             f"{ws[0]}, {cost[ws[-1]]:.2f} with {ws[-1]}" if len(ws) > 1
             else "Worker CPU by offered rate")
    fig, axes = figure(s, title,
                       "Mean worker CPU while each run was live, and CPU-seconds spent per "
                       "frame processed · throughput grid · title averages the points at "
                       "8 fps and above", ncols=2, height=3.4)
    a1, a2 = axes[0]
    for w, g in groups.items():
        c = WORKER_COLOR.get(w, SERIES[w % len(SERIES)])
        x = [num(r["offered_fps"]) for r in g]
        cpu = [num(r["worker_cpu_mean_pct"]) for r in g]
        per_frame = [(cp / 100) / num(r["achieved_fps"]) if cp and num(r["achieved_fps"])
                     else None for cp, r in zip(cpu, g)]
        lab = f"{w} worker{'s' if w > 1 else ''}"
        a1.plot(x, cpu, "-o", color=c, label=lab, markeredgecolor=SURFACE,
                markeredgewidth=1.2)
        a2.plot(x, per_frame, "-o", color=c, markeredgecolor=SURFACE, markeredgewidth=1.2)
    for a in (a1, a2):
        a.set_xlabel("Offered fps")
        a.set_xticks(sorted({num(r["offered_fps"]) for r in rows}))
    a1.set_ylabel("Worker CPU (% of one core)")
    a2.set_ylabel("Core-seconds per frame")
    a1.set_ylim(bottom=0)
    a2.set_ylim(bottom=0)
    fig_legend(fig, *a1.get_legend_handles_labels())
    return save(fig, out, "12_cpu_vs_offered_rate.png", bottom_in=0.95)


# --- results.md -----------------------------------------------------------------------

def _f(v, fmt="{:,.0f}"):
    v = num(v) if not isinstance(v, (int, float)) else v
    return "–" if v is None else fmt.format(v)


def results_md(s: Sweep, out_root: Path) -> Path:
    L = ["# RoadIQ benchmark results", "",
         f"Generated by `scripts/charts.py` from `{out_root}`. Every number below is "
         "measured; nothing is a target.", ""]
    if s.hw:
        h = s.hw
        L += ["## Test environment", "",
              "| | |", "|---|---|",
              f"| Host | {h.get('host_cpu')}, {h.get('host_cores')} cores, "
              f"{h.get('host_mem_gb')} GB |",
              f"| Docker VM | {h.get('docker_vm_cpus')} vCPU, {h.get('docker_vm_mem_gb')} GB, "
              f"Docker {h.get('docker_version')} |",
              f"| Detector | {h.get('detector')} · `{Path(h.get('weights', '')).name}` · "
              f"{h.get('gpu')} · {h.get('worker_threads')} threads per worker |",
              f"| Code | commit `{h.get('commit')}` |", ""]
    base = s.first("baseline")
    if base:
        sm = s.summary(base["label"])
        L += ["## Baseline run — latency by stage (ms)", "",
              f"{sm['frames_landed']:,} of {sm['frames_offered']:,} frames landed, "
              f"{sm['frames_dropped']} dropped · {base['workers']} workers · "
              f"{num(base['offered_fps']):g} fps offered, {sm['achieved_fps']} achieved · "
              f"coverage {sm['coverage']['coverage_pct']}% of "
              f"{sm['coverage']['assessed_m'] / 1000:.2f} km", "",
              "| Stage | p50 | p90 | p95 | p99 | max |", "|---|---|---|---|---|---|"]
        for k in ("queue_ms", "infer_ms", "write_ms", "e2e_ms"):
            st = sm["stages"][k]
            L.append(f"| {STAGE_NAME[k]} | {_f(st['p50'])} | {_f(st['p90'])} | "
                     f"{_f(st['p95'])} | {_f(st['p99'])} | {_f(st.get('max'))} |")
        acc = sm.get("accuracy") or {}
        if acc.get("per_class"):
            L += ["", "## Baseline run — accuracy (IoU ≥ 0.5, confidence ≥ 0.25)", "",
                  "| Class | GT boxes | Predictions | Precision | Recall | F1 | AP50 |",
                  "|---|---|---|---|---|---|---|"]
            for c, m in acc["per_class"].items():
                L.append(f"| {CLASS_NAME.get(c, c)} | {m['gt']:,} | {m['pred']:,} | "
                         f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
                         f"{sm['ap50'].get(c, 0):.3f} |")
            L.append(f"| **All scored** | | | **{acc['precision']:.3f}** | "
                     f"**{acc['recall']:.3f}** | **{acc['f1']:.3f}** | "
                     f"**{sm['map50_scored']:.3f}** |")
        res = s.point(base["label"]).get("resources") or {}
        if res:
            L += ["", "## Baseline run — resources while live", "",
                  "| Service | Replicas | Mean CPU (% of a core) | Peak CPU | "
                  "Peak memory, one replica (MB) | Peak memory, all replicas (MB) |",
                  "|---|---|---|---|---|---|"]
            for svc, v in sorted(res.items(), key=lambda kv: -kv[1].get("cpu_mean_pct", 0)):
                if svc == "pipeline":
                    continue
                L.append(f"| {svc} | {v['replicas']} | {_f(v['cpu_mean_pct'], '{:,.1f}')} | "
                         f"{_f(v['cpu_peak_pct'], '{:,.1f}')} | {_f(v['mem_peak_mb'])} | "
                         f"{_f(v['mem_total_peak_mb'])} |")
            p = res.get("pipeline", {})
            L.append(f"| **whole pipeline** | | **{_f(p.get('cpu_mean_pct'), '{:,.1f}')}** | "
                     f"**{_f(p.get('cpu_peak_pct'), '{:,.1f}')}** | | "
                     f"**{_f(p.get('mem_total_peak_mb'))}** |")
    cols = [("label", "Run"), ("workers", "Workers"), ("offered_fps", "Offered fps"),
            ("achieved_fps", "Achieved fps"), ("buffer", "Buffer"),
            ("frames_offered", "Offered"), ("frames_dropped", "Dropped"),
            ("frames_landed", "Landed"), ("e2e_p50_ms", "e2e p50 ms"),
            ("e2e_p95_ms", "e2e p95 ms"), ("infer_p50_ms", "infer p50 ms"),
            ("worker_cpu_mean_pct", "Worker CPU %"), ("worker_mem_peak_mb", "Worker mem MB"),
            ("reduction_x", "Bytes reduction ×")]
    if s.runs:
        L += ["", "## Every run", "", "| " + " | ".join(c[1] for c in cols) + " |",
              "|" + "---|" * len(cols)]
        for r in s.runs:
            L.append("| " + " | ".join(r.get(k, "") or "–" for k, _ in cols) + " |")
    chaos = s.first("chaos")
    if chaos:
        L += ["", "## Chaos run", "",
              f"One of two workers SIGKILLed {chaos['chaos_kill_at_s']} s into the feed. "
              f"{chaos['frames_landed']} of {chaos['frames_offered']} offered frames landed, "
              f"{chaos['frames_dropped']} refused at the buffer, "
              f"{int(num(chaos['frames_offered'])) - int(num(chaos['frames_landed'])) - int(num(chaos['frames_dropped']))} "
              f"unaccounted."]
    junit = out_root / "junit.xml"
    if junit.exists():
        import xml.etree.ElementTree as ET
        tree = ET.parse(junit)
        cases = tree.getroot().iter("testcase")
        L += ["", "## Test suite", "", "| Test | Result | Seconds |", "|---|---|---|"]
        n = ok = 0
        for c in cases:
            n += 1
            failed = c.find("failure") is not None or c.find("error") is not None
            skipped = c.find("skipped") is not None
            ok += not failed and not skipped
            name = f"{c.get('classname', '').split('.')[-1]}::{c.get('name')}"
            L.append(f"| `{name}` | {'FAIL' if failed else 'skip' if skipped else 'pass'} | "
                     f"{float(c.get('time', 0)):.1f} |")
        L.insert(L.index("## Test suite") + 1, f"\n**{ok} of {n} pass.**")
    path = out_root / "results.md"
    path.write_text("\n".join(L) + "\n")
    print(f"  wrote {path}")
    return path


CHARTS = [chart_latency_cdf, chart_latency_stages, chart_throughput, chart_coverage,
          chart_chaos, chart_storage, chart_accuracy, chart_pr, chart_confusion,
          chart_resources_timeline, chart_resources_per_service, chart_cpu_vs_rate]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m scripts.charts <sweep output dir>")
    root = Path(sys.argv[1])
    s = Sweep(root)
    out = root / "charts"
    out.mkdir(parents=True, exist_ok=True)
    for chart in CHARTS:
        try:
            chart(s, out)
        except (KeyError, ValueError, IndexError, FileNotFoundError, TypeError) as exc:
            print(f"  FAILED {chart.__name__}: {exc!r}")
    results_md(s, root)


if __name__ == "__main__":
    main()
