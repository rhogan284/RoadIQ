"""Unit tests for the benchmark export and resource parsing. No services needed."""
import pytest

from edgecv.bench.export import (average_precision, confusion, match_flags,
                                 percentiles, pr_curve)
from edgecv.bench.resources import (parse_pair, parse_size, parse_stats_line,
                                    service_of, summarise)

BOX = [10, 10, 100, 100]
NEAR = [12, 12, 100, 100]       # IoU with BOX ≈ 0.92
FAR = [500, 500, 50, 50]        # IoU with BOX = 0


def p(ref, cls, conf, box, seq=0):
    return (ref, seq, cls, conf, *box)


class TestMatching:
    def test_one_hit_one_false_alarm(self):
        gt = {("a.jpg", "D00"): [BOX]}
        preds = [p("a.jpg", "D00", 0.9, NEAR), p("a.jpg", "D00", 0.4, FAR)]
        confs, flags, n_gt = match_flags(preds, gt, "D00")
        assert confs == [0.9, 0.4]
        assert flags == [True, False]
        assert n_gt == 1

    def test_one_gt_box_is_claimed_once(self):
        gt = {("a.jpg", "D00"): [BOX]}
        preds = [p("a.jpg", "D00", 0.5, NEAR), p("a.jpg", "D00", 0.8, BOX)]
        _confs, flags, _n = match_flags(preds, gt, "D00")
        assert flags == [True, False]          # the 0.8 box wins, the duplicate is FP

    def test_other_classes_ignored(self):
        gt = {("a.jpg", "D00"): [BOX]}
        preds = [p("a.jpg", "D10", 0.9, BOX)]
        assert match_flags(preds, gt, "D00") == ([], [], 1)


class TestCurve:
    def test_prefix_precision_recall(self):
        curve = pr_curve([0.9, 0.7, 0.5], [True, False, True], n_gt=4)
        assert [round(c["precision"], 3) for c in curve] == [1.0, 0.5, 0.667]
        assert [c["recall"] for c in curve] == [0.25, 0.25, 0.5]

    def test_perfect_detector_ap_is_one(self):
        assert average_precision(pr_curve([0.9, 0.8], [True, True], n_gt=2)) == pytest.approx(1.0)

    def test_half_recall_full_precision(self):
        assert average_precision(pr_curve([0.9], [True], n_gt=2)) == pytest.approx(0.5)

    def test_empty(self):
        assert average_precision([]) == 0.0


class TestConfusion:
    def test_hit_confusion_miss_and_false_alarm(self):
        gt = {("a.jpg", "D00"): [BOX], ("b.jpg", "D10"): [BOX], ("c.jpg", "D20"): [BOX]}
        preds = [p("a.jpg", "D00", 0.9, NEAR),      # correct
                 p("b.jpg", "D20", 0.9, NEAR),      # D10 called D20
                 p("a.jpg", "D00", 0.3, FAR)]       # nothing there
        m = confusion(preds, gt)
        assert m[("D00", "D00")] == 1
        assert m[("D10", "D20")] == 1
        assert m[("background", "D00")] == 1
        assert m[("D20", "missed")] == 1            # c.jpg had no prediction at all

    def test_confidence_floor(self):
        gt = {("a.jpg", "D00"): [BOX]}
        m = confusion([p("a.jpg", "D00", 0.2, BOX)], gt, conf_min=0.5)
        assert m == {("D00", "missed"): 1}


class TestPercentiles:
    def test_skips_missing(self):
        out = percentiles([1.0, None, 3.0, float("nan"), 2.0])
        assert out["n"] == 3 and out["p50"] == 2.0

    def test_all_missing(self):
        assert percentiles([None])["p50"] is None


class TestDockerStats:
    def test_units(self):
        assert parse_size("512MiB") == 512 * 1024 ** 2
        assert parse_size("1.5kB") == 1500
        assert parse_size("0B") == 0
        assert parse_pair("1.2MB / 3.4kB") == (1.2e6, 3.4e3)

    def test_service_names(self):
        assert service_of("roadiq-bench-worker-2", "roadiq-bench") == ("worker", "2")
        assert service_of("roadiq-bench-feedsim-run-3f9a1c", "roadiq-bench") == ("feedsim", "run-3f9a1c")
        assert service_of("roadiq-postgres-1", "roadiq-bench") is None

    def test_line_with_ansi_prefix(self):
        line = ('\x1b[2J\x1b[H{"Name":"roadiq-bench-worker-1","CPUPerc":"412.50%",'
                '"MemUsage":"1.5GiB / 7.75GiB","NetIO":"2MB / 1MB","BlockIO":"0B / 4.1kB"}')
        rec = parse_stats_line(line, "roadiq-bench", 10.0)
        assert rec["service"] == "worker" and rec["cpu_pct"] == 412.5
        assert rec["mem_mb"] == 1536.0 and rec["blk_w_b"] == 4100

    def test_summarise_sums_replicas(self):
        def s(t, c, svc, cpu, mem, rx):
            return {"t": t, "container": c, "service": svc, "replica": "",
                    "cpu_pct": cpu, "mem_mb": mem, "net_rx_b": rx, "net_tx_b": 0,
                    "blk_r_b": 0, "blk_w_b": 0}
        samples = [s(1.0, "w1", "worker", 100, 500, 0), s(1.2, "w2", "worker", 300, 700, 0),
                   s(2.0, "w1", "worker", 200, 600, 1000), s(2.1, "w2", "worker", 200, 650, 500),
                   s(1.1, "pg", "postgres", 10, 100, 0), s(2.1, "pg", "postgres", 30, 120, 0)]
        out = summarise(samples)
        assert out["worker"]["cpu_mean_pct"] == 400       # (400 + 400) / 2
        assert out["worker"]["mem_peak_mb"] == 700         # one replica's peak
        assert out["worker"]["mem_total_peak_mb"] == 1250  # both replicas together
        assert out["worker"]["net_rx_b"] == 1500
        assert out["worker"]["replicas"] == 2
        assert out["pipeline"]["cpu_peak_pct"] == 430

    def test_two_samples_per_second_are_averaged_not_summed(self):
        # docker stats refreshes about twice a second; summing both readings of one
        # container in a second doubled every CPU and memory total (found 2026-11-02:
        # 1,825 % on a 15-core VM).
        def s(t, cpu, mem):
            return {"t": t, "container": "w1", "service": "worker", "replica": "1",
                    "cpu_pct": cpu, "mem_mb": mem, "net_rx_b": 0, "net_tx_b": 0,
                    "blk_r_b": 0, "blk_w_b": 0}
        out = summarise([s(1.1, 400, 600), s(1.6, 500, 700),
                         s(2.1, 300, 650), s(2.6, 300, 650)])
        assert out["worker"]["cpu_mean_pct"] == 375        # (450 + 300) / 2
        assert out["worker"]["cpu_peak_pct"] == 450
        assert out["worker"]["mem_total_peak_mb"] == 650
        assert out["worker"]["mem_peak_mb"] == 700         # raw single-sample peak
        assert out["pipeline"]["cpu_peak_pct"] == 450
