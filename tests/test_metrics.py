import pytest

from lanedet.metrics import LaneEvalAccumulator, bench

YS = list(range(300, 720, 10))


def lane(x0, slope=0.0):
    return [x0 + slope * (y - 300) for y in YS]


def test_perfect_prediction():
    gt = [lane(400, -0.5), lane(800, 0.5)]
    assert bench(gt, gt, YS) == (1.0, 0.0, 0.0)


def test_small_offset_still_correct():
    gt = [lane(400)]
    acc, fp, fn = bench([lane(415)], gt, YS)
    assert acc == 1.0 and fp == 0 and fn == 0


def test_large_offset_is_miss_and_false_positive():
    acc, fp, fn = bench([lane(450)], [lane(400)], YS)
    assert acc == 0.0 and fp == 1.0 and fn == 1.0


def test_missing_lane_is_false_negative():
    acc, fp, fn = bench([lane(400)], [lane(400), lane(800)], YS)
    assert acc == pytest.approx(0.5) and fp == 0.0 and fn == pytest.approx(0.5)


def test_partial_lane_counts_missing_points():
    pred = lane(400)
    pred[:20] = [-2] * 20  # first half of the rows undetected
    acc, _, fn = bench([pred], [lane(400)], YS)
    assert acc == pytest.approx(22 / 42) and fn == 1.0  # below the 85 % match threshold


def test_too_many_predictions_scores_zero():
    assert bench([lane(x) for x in range(100, 600, 100)], [lane(100), lane(200)], YS)[0] == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        bench([[1, 2]], [lane(1)], YS)


def test_accumulator():
    acc = LaneEvalAccumulator()
    acc.add([lane(400)], [lane(400)], YS)
    acc.add([], [lane(400)], YS)
    s = acc.summary()
    assert s["images"] == 2 and s["accuracy"] == pytest.approx(0.5)
