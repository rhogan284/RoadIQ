import random
import pytest
from edgecv.feedsim.prevalence import PrevalenceSampler, pace_deadlines

CLEAN = [f"clean_{i}.png" for i in range(50)]
DEFECT = [f"defect_{i}.png" for i in range(20)]


def test_prevalence_is_approximately_honoured():
    sampler = PrevalenceSampler(CLEAN, DEFECT, prevalence=0.02,
                                rng=random.Random(1))
    draws = [sampler.next() for _ in range(10_000)]
    defect_fraction = sum(1 for _p, is_defect in draws if is_defect) / len(draws)
    assert 0.015 < defect_fraction < 0.025


def test_same_seed_reproduces_the_sequence():
    a = PrevalenceSampler(CLEAN, DEFECT, prevalence=0.1, rng=random.Random(7))
    b = PrevalenceSampler(CLEAN, DEFECT, prevalence=0.1, rng=random.Random(7))
    assert [a.next() for _ in range(50)] == [b.next() for _ in range(50)]


def test_defect_draws_come_from_defect_pool():
    sampler = PrevalenceSampler(CLEAN, DEFECT, prevalence=1.0,
                                rng=random.Random(3))
    for _ in range(20):
        path, is_defect = sampler.next()
        assert is_defect and path in DEFECT


def test_zero_prevalence_yields_only_clean():
    sampler = PrevalenceSampler(CLEAN, DEFECT, prevalence=0.0,
                                rng=random.Random(3))
    assert all(not is_defect for _p, is_defect in
               (sampler.next() for _ in range(50)))


def test_rejects_prevalence_out_of_range():
    with pytest.raises(ValueError):
        PrevalenceSampler(CLEAN, DEFECT, prevalence=1.5, rng=random.Random(1))


def test_rejects_empty_pool_it_needs():
    with pytest.raises(ValueError, match="defect"):
        PrevalenceSampler(CLEAN, [], prevalence=0.1, rng=random.Random(1))


def test_pacing_deadlines_do_not_drift():
    deadlines = list(pace_deadlines(start=100.0, fps=10, n=5))
    assert deadlines == [100.0, 100.1, 100.2, 100.3, 100.4]


def test_pacing_rejects_non_positive_fps():
    with pytest.raises(ValueError):
        list(pace_deadlines(start=0.0, fps=0, n=1))
