"""Per-container CPU, memory, network and disk during a benchmark run, from `docker stats`.

The bench_runs columns `peak_rss_mb` and `cpu_pct` were in the schema from the start
and nothing filled them. This samples every container of the Compose project once a
second for the length of a run, then reduces the samples to those two columns plus a
per-service breakdown.

What the numbers are, precisely, so the report does not overclaim:

  cpu_pct   docker's CPUPerc: percent of ONE core, so a worker using six threads
            flat out reads ~600%. Summed across containers for "pipeline CPU".
  mem       docker's MemUsage: the container cgroup's memory, which is resident
            memory plus some page cache — close to, but not exactly, process RSS.
            Reported as "memory (cgroup)"; the column name peak_rss_mb is the
            schema's, kept for compatibility.
  net/block cumulative counters since the container started; a run's traffic is
            the last sample minus the first.

`docker stats` with no container list streams every running container and picks
up new ones as they start, which is what catches the short-lived feed-sim container.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

_UNITS = {"b": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12,
          "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3, "tib": 1024 ** 4}
_SIZE = re.compile(r"^\s*([0-9.]+)\s*([a-zA-Z]+)\s*$")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# <project>-<service>-<n>  or  <project>-<service>-run-<hex>  (docker compose run)
_NAME = re.compile(r"^(?P<service>.+?)-(?:run-[0-9a-f]+|\d+)$")


def parse_size(text: str) -> float:
    """'512.3MiB' → bytes. Docker mixes SI (network, disk) and IEC (memory) units."""
    m = _SIZE.match(text)
    if not m:
        raise ValueError(f"unparseable size {text!r}")
    return float(m.group(1)) * _UNITS[m.group(2).lower()]


def parse_pair(text: str) -> tuple[float, float]:
    """'1.2MB / 3.4kB' → (1.2e6, 3.4e3)."""
    a, b = text.split("/")
    return parse_size(a), parse_size(b)


def service_of(container: str, project: str) -> tuple[str, str] | None:
    """('roadiq-bench-worker-2', 'roadiq-bench') → ('worker', '2'). None if the
    container is not in the project."""
    prefix = f"{project}-"
    if not container.startswith(prefix):
        return None
    rest = container[len(prefix):]
    m = _NAME.match(rest)
    if not m:
        return rest, ""
    return m.group("service"), rest[len(m.group("service")) + 1:]


def parse_stats_line(line: str, project: str, t: float) -> dict | None:
    line = _ANSI.sub("", line).strip()
    if not line.startswith("{"):
        return None
    rec = json.loads(line)
    svc = service_of(rec.get("Name", ""), project)
    if svc is None:
        return None
    mem_used, _mem_limit = parse_pair(rec["MemUsage"])
    net_rx, net_tx = parse_pair(rec["NetIO"])
    blk_r, blk_w = parse_pair(rec["BlockIO"])
    return {"t": round(t, 3), "container": rec["Name"], "service": svc[0],
            "replica": svc[1], "cpu_pct": float(rec["CPUPerc"].rstrip("%") or 0),
            "mem_mb": round(mem_used / 1024 ** 2, 1),
            "net_rx_b": int(net_rx), "net_tx_b": int(net_tx),
            "blk_r_b": int(blk_r), "blk_w_b": int(blk_w)}


def summarise(samples: list[dict], *, t0: float | None = None,
              t1: float | None = None) -> dict:
    """Reduce samples (optionally inside [t0, t1]) to per-service figures.

    Per service: peak memory of any one replica, mean and peak CPU summed over its
    replicas at each sample time, and network/disk traffic over the window. Plus
    `pipeline` totals across every service.
    """
    window = [s for s in samples
              if (t0 is None or s["t"] >= t0) and (t1 is None or s["t"] <= t1)]
    if not window:
        return {}
    # Bucket to the second so the replicas of a service are summed at a common time.
    # docker stats refreshes about twice a second, so a container can have several
    # readings in one bucket: AVERAGE those first, then sum across containers.
    # Summing them directly doubled every total (2026-11-02 sweep, fixed here).
    per_container: dict[tuple, list] = defaultdict(list)
    peak_mem: dict[str, float] = defaultdict(float)
    first: dict[str, dict] = {}
    last: dict[str, dict] = {}
    for s in window:
        per_container[(s["service"], s["container"], int(s["t"]))].append(s)
        peak_mem[s["service"]] = max(peak_mem[s["service"]], s["mem_mb"])
        first.setdefault(s["container"], s)
        last[s["container"]] = s
    cpu_at: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    mem_at: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for (svc, _c, sec), group in per_container.items():
        cpu_at[svc][sec] += sum(g["cpu_pct"] for g in group) / len(group)
        mem_at[svc][sec] += sum(g["mem_mb"] for g in group) / len(group)

    out: dict = {}
    for svc in cpu_at:
        cpu = list(cpu_at[svc].values())
        traffic = defaultdict(int)
        for c, s1 in last.items():
            if s1["service"] != svc:
                continue
            s0 = first[c]
            for k in ("net_rx_b", "net_tx_b", "blk_r_b", "blk_w_b"):
                traffic[k] += max(0, s1[k] - s0[k])
        out[svc] = {"cpu_mean_pct": round(sum(cpu) / len(cpu), 1),
                    "cpu_peak_pct": round(max(cpu), 1),
                    "mem_peak_mb": round(peak_mem[svc], 1),
                    "mem_total_peak_mb": round(max(mem_at[svc].values()), 1),
                    "replicas": len({c for c, s in last.items() if s["service"] == svc}),
                    **traffic}
    total_cpu: dict[int, float] = defaultdict(float)
    total_mem: dict[int, float] = defaultdict(float)
    for svc in cpu_at:
        for sec, v in cpu_at[svc].items():
            total_cpu[sec] += v
        for sec, v in mem_at[svc].items():
            total_mem[sec] += v
    out["pipeline"] = {"cpu_mean_pct": round(sum(total_cpu.values()) / len(total_cpu), 1),
                       "cpu_peak_pct": round(max(total_cpu.values()), 1),
                       "mem_total_peak_mb": round(max(total_mem.values()), 1)}
    return out


class DockerStatsSampler:
    """Streams `docker stats` in a background thread into a list of sample dicts."""

    def __init__(self, project: str):
        self.project = project
        self.samples: list[dict] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> "DockerStatsSampler":
        self._proc = subprocess.Popen(
            ["docker", "stats", "--format", "{{json .}}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return self

    def _pump(self) -> None:
        assert self._proc and self._proc.stdout
        for raw in self._proc.stdout:
            for line in _ANSI.sub("\n", raw).splitlines():
                try:
                    rec = parse_stats_line(line, self.project, time.time())
                except (ValueError, KeyError, json.JSONDecodeError):
                    continue
                if rec:
                    self.samples.append(rec)

    def stop(self) -> list[dict]:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=5)
        return self.samples


def write_samples(path: Path, samples: list[dict]) -> None:
    import csv
    if not samples:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(samples[0]))
        w.writeheader()
        w.writerows(samples)
