"""Unit tests for baseline RunningMeanCov and BaselineStore."""
import numpy as np
from app.baseline import RunningMeanCov, BaselineStore


def test_running_mean_cov_converges():
    dim = 4
    r = RunningMeanCov(dim=dim, alpha=0.1, gate_threshold=10.0)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    # feed repeated identical vectors
    for _ in range(50):
        r.update(x)
    # score should be small
    s = r.score(x)
    assert s < 1.0


def test_baseline_store_basic():
    bs = BaselineStore(dim=4, alpha=0.1, gate_threshold=10.0)
    worker = 1
    x = np.zeros(4)
    bs.update(worker, x)
    s = bs.score(worker, x)
    assert s >= 0.0
