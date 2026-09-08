"""Unit tests for TrendScorer"""
import time
from app.trend import TrendScorer


def test_trend_increasing():
    t = TrendScorer(window_size=20, ewma_alpha=0.2)
    wid = 1
    base = time.time()
    # feed increasing deviations
    for i in range(15):
        t.add_sample(wid, base + i, float(i))
    r = t.risk_score(wid)
    # expect some positive risk
    assert r > 0.0
