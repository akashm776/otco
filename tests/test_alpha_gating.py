"""Unit tests for compute_alpha_effective — covers all four gating branches."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.loss import compute_alpha_effective


ALPHA = 0.05
KW = dict(
    entropy_threshold=3.0,
    gap_suppress_easy=0.10,
    gap_downweight_hard=-0.07,
    hard_alpha_scale=0.25,
)


def test_useful_regime():
    a_eff, bucket = compute_alpha_effective(ALPHA, 2.2, -0.03, **KW)
    assert a_eff == ALPHA, f"expected {ALPHA}, got {a_eff}"
    assert bucket == 0, f"expected bucket 0 (useful), got {bucket}"


def test_diffuse_plan():
    a_eff, bucket = compute_alpha_effective(ALPHA, 3.2, -0.03, **KW)
    assert a_eff == 0.0, f"expected 0.0, got {a_eff}"
    assert bucket == 3, f"expected bucket 3 (diffuse), got {bucket}"


def test_too_easy():
    a_eff, bucket = compute_alpha_effective(ALPHA, 2.2, 0.12, **KW)
    assert a_eff == 0.0, f"expected 0.0, got {a_eff}"
    assert bucket == 1, f"expected bucket 1 (too_easy), got {bucket}"


def test_too_hard():
    a_eff, bucket = compute_alpha_effective(ALPHA, 2.2, -0.08, **KW)
    assert abs(a_eff - ALPHA * 0.25) < 1e-9, f"expected {ALPHA * 0.25}, got {a_eff}"
    assert bucket == 2, f"expected bucket 2 (too_hard), got {bucket}"


def test_inactive():
    a_eff, bucket = compute_alpha_effective(0.0, 2.2, -0.03, **KW)
    assert a_eff == 0.0
    assert bucket == 4, f"expected bucket 4 (inactive), got {bucket}"


def test_entropy_takes_priority_over_gap():
    # diffuse plan should win even if gap is in the too-easy range
    a_eff, bucket = compute_alpha_effective(ALPHA, 3.5, 0.15, **KW)
    assert a_eff == 0.0
    assert bucket == 3, f"expected bucket 3 (diffuse), got {bucket}"


def test_boundary_suppress_easy():
    # exactly at threshold — should NOT suppress
    a_eff, bucket = compute_alpha_effective(ALPHA, 2.2, 0.10, **KW)
    assert a_eff == ALPHA, f"gap == threshold should pass, got {a_eff}"
    assert bucket == 0


def test_boundary_downweight_hard():
    # exactly at threshold — should NOT downweight
    a_eff, bucket = compute_alpha_effective(ALPHA, 2.2, -0.07, **KW)
    assert a_eff == ALPHA, f"gap == threshold should pass, got {a_eff}"
    assert bucket == 0


if __name__ == "__main__":
    tests = [
        test_useful_regime,
        test_diffuse_plan,
        test_too_easy,
        test_too_hard,
        test_inactive,
        test_entropy_takes_priority_over_gap,
        test_boundary_suppress_easy,
        test_boundary_downweight_hard,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
