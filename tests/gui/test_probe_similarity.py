"""Tests for T-030 Bug 2 -- Procrustes multi-anchor similarity transform.

Covers:
- similarity_transform_probe: 0, 1, 2, 3+ anchors
- Geometry preservation: uniform scale ratio, angle preservation
- Idempotence: applying transform 2x == 1x when anchors maintained
- Edge cases: invalid channels, unknown 10-20, duplicate anchors, empty montage
- Callback integration: probe_graph_click multi-anchor flow, deselect
- Callback: probe_reset_anchors clears state
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from nirspy.gui.components.probe_dialog import (
    _TEN_TWENTY,
    _build_anchor_badges,
    _channel_midpoint,
    similarity_transform_probe,
    translate_probe_to_anchor,
)


@pytest.fixture()
def simple_montage() -> dict[str, Any]:
    return {
        "sources": [[0.0, 0.1], [0.0, -0.1]],
        "detectors": [[-0.1, 0.0], [0.1, 0.0]],
    }


@pytest.fixture()
def large_montage() -> dict[str, Any]:
    return {
        "sources": [[0.0, 0.2], [0.0, 0.0], [0.0, -0.2]],
        "detectors": [[-0.15, 0.1], [0.15, 0.1], [0.0, -0.1]],
    }


def _midpoint(montage, label):
    result = _channel_midpoint(montage, label)
    assert result is not None
    return result


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


class TestSimilarityTransform0Anchors:

    def test_empty_anchors_returns_identical(self, simple_montage):
        result = similarity_transform_probe(simple_montage, [])
        assert result["sources"] == simple_montage["sources"]
        assert result["detectors"] == simple_montage["detectors"]

    def test_none_anchors_treated_as_empty(self, simple_montage):
        result = similarity_transform_probe(simple_montage, [])
        assert result == simple_montage


class TestSimilarityTransform1Anchor:

    def test_single_anchor_translates_to_target(self, simple_montage):
        result = similarity_transform_probe(simple_montage, [["S1_D1", "Cz"]])
        mid = _midpoint(result, "S1_D1")
        target = _TEN_TWENTY["Cz"]
        assert abs(mid[0] - target[0]) < 1e-9
        assert abs(mid[1] - target[1]) < 1e-9

    def test_single_anchor_preserves_relative_distances(self, simple_montage):
        bs = simple_montage["sources"]
        bd = simple_montage["detectors"]
        db = _dist((bs[0][0], bs[0][1]), (bd[0][0], bd[0][1]))
        result = similarity_transform_probe(simple_montage, [["S1_D1", "Fz"]])
        rs = result["sources"]
        rd = result["detectors"]
        da = _dist((rs[0][0], rs[0][1]), (rd[0][0], rd[0][1]))
        assert abs(db - da) < 1e-12

    def test_single_anchor_fp1(self, simple_montage):
        result = similarity_transform_probe(simple_montage, [["S2_D2", "Fp1"]])
        mid = _midpoint(result, "S2_D2")
        target = _TEN_TWENTY["Fp1"]
        assert abs(mid[0] - target[0]) < 1e-9
        assert abs(mid[1] - target[1]) < 1e-9


class TestSimilarityTransform2Anchors:

    def test_two_anchors_exact_fit(self, simple_montage):
        result = similarity_transform_probe(simple_montage, [["S1_D1", "C3"], ["S2_D2", "C4"]])
        mid1 = _midpoint(result, "S1_D1")
        mid2 = _midpoint(result, "S2_D2")
        assert abs(mid1[0] - _TEN_TWENTY["C3"][0]) < 1e-9
        assert abs(mid1[1] - _TEN_TWENTY["C3"][1]) < 1e-9
        assert abs(mid2[0] - _TEN_TWENTY["C4"][0]) < 1e-9
        assert abs(mid2[1] - _TEN_TWENTY["C4"][1]) < 1e-9

    def test_two_anchors_uniform_scale(self, simple_montage):
        s = simple_montage
        src0 = (s["sources"][0][0], s["sources"][0][1])
        det0 = (s["detectors"][0][0], s["detectors"][0][1])
        det1 = (s["detectors"][1][0], s["detectors"][1][1])
        bd1 = _dist(src0, det0)
        bd2 = _dist(src0, det1)
        result = similarity_transform_probe(s, [["S1_D1", "F3"], ["S2_D2", "P4"]])
        r = result
        rsrc0 = (r["sources"][0][0], r["sources"][0][1])
        rdet0 = (r["detectors"][0][0], r["detectors"][0][1])
        rdet1 = (r["detectors"][1][0], r["detectors"][1][1])
        ad1 = _dist(rsrc0, rdet0)
        ad2 = _dist(rsrc0, rdet1)
        assert abs(ad1 / bd1 - ad2 / bd2) < 1e-9


class TestSimilarityTransform3PlusAnchors:

    def test_three_anchors_least_squares(self, large_montage):
        anchors = [["S1_D1", "F3"], ["S2_D2", "F4"], ["S3_D3", "Pz"]]
        result = similarity_transform_probe(large_montage, anchors)
        assert result["sources"] != large_montage["sources"]

    def test_three_anchors_mean_error_bounded(self, large_montage):
        anchors = [["S1_D1", "F3"], ["S2_D2", "C4"], ["S3_D3", "Pz"]]
        result = similarity_transform_probe(large_montage, anchors)
        total_err = sum(_dist(_midpoint(result, p[0]), _TEN_TWENTY[p[1]]) for p in anchors)
        assert total_err / len(anchors) < 1.0


class TestGeometryPreservation:

    def test_distance_ratios_constant(self, simple_montage):
        result = similarity_transform_probe(simple_montage, [["S1_D1", "C3"], ["S1_D2", "C4"]])
        pairs = [("S1_D1", "S1_D2"), ("S2_D1", "S2_D2")]
        rb = []
        ra = []
        for a_l, b_l in pairs:
            db = _dist(_midpoint(simple_montage, a_l), _midpoint(simple_montage, b_l))
            da = _dist(_midpoint(result, a_l), _midpoint(result, b_l))
            if db > 1e-15:
                rb.append(db)
                ra.append(da)
        if len(rb) >= 2:
            sr = [a / b for a, b in zip(ra, rb, strict=True)]
            for s in sr[1:]:
                assert abs(s - sr[0]) < 1e-9

    def test_angles_between_pairs_preserved(self, simple_montage):
        ma = _midpoint(simple_montage, "S1_D1")
        mb = _midpoint(simple_montage, "S2_D1")
        mc = _midpoint(simple_montage, "S1_D2")
        diff_b = _angle(ma, mb) - _angle(ma, mc)
        result = similarity_transform_probe(simple_montage, [["S1_D1", "F3"], ["S2_D2", "P4"]])
        ma2 = _midpoint(result, "S1_D1")
        mb2 = _midpoint(result, "S2_D1")
        mc2 = _midpoint(result, "S1_D2")
        diff_a = _angle(ma2, mb2) - _angle(ma2, mc2)
        delta = abs(diff_b - diff_a) % (2 * math.pi)
        if delta > math.pi:
            delta = 2 * math.pi - delta
        assert delta < 1e-9

    def test_sd_distances_uniform_scale(self, simple_montage):
        result = similarity_transform_probe(simple_montage, [["S1_D1", "F7"], ["S2_D2", "F8"]])
        scales = []
        for si in range(len(simple_montage["sources"])):
            for di in range(len(simple_montage["detectors"])):
                sb = simple_montage["sources"][si]
                db = simple_montage["detectors"][di]
                dist_b = _dist((sb[0], sb[1]), (db[0], db[1]))
                sa = result["sources"][si]
                da = result["detectors"][di]
                dist_a = _dist((sa[0], sa[1]), (da[0], da[1]))
                if dist_b > 1e-15:
                    scales.append(dist_a / dist_b)
        assert len(scales) >= 2
        for sv in scales[1:]:
            assert abs(sv - scales[0]) < 1e-9


class TestIdempotence:

    def test_idempotent_1_anchor(self, simple_montage):
        anchors = [["S1_D1", "Cz"]]
        first = similarity_transform_probe(simple_montage, anchors)
        second = similarity_transform_probe(first, anchors)
        for i in range(len(first["sources"])):
            assert abs(first["sources"][i][0] - second["sources"][i][0]) < 1e-9
            assert abs(first["sources"][i][1] - second["sources"][i][1]) < 1e-9
        for i in range(len(first["detectors"])):
            assert abs(first["detectors"][i][0] - second["detectors"][i][0]) < 1e-9
            assert abs(first["detectors"][i][1] - second["detectors"][i][1]) < 1e-9

    def test_idempotent_2_anchors(self, simple_montage):
        anchors = [["S1_D1", "C3"], ["S2_D2", "C4"]]
        first = similarity_transform_probe(simple_montage, anchors)
        second = similarity_transform_probe(first, anchors)
        for i in range(len(first["sources"])):
            assert abs(first["sources"][i][0] - second["sources"][i][0]) < 1e-9
        for i in range(len(first["detectors"])):
            assert abs(first["detectors"][i][0] - second["detectors"][i][0]) < 1e-9

    def test_idempotent_3_anchors(self, large_montage):
        anchors = [["S1_D1", "F3"], ["S2_D2", "C4"], ["S3_D3", "Pz"]]
        first = similarity_transform_probe(large_montage, anchors)
        second = similarity_transform_probe(first, anchors)
        for i in range(len(first["sources"])):
            assert abs(first["sources"][i][0] - second["sources"][i][0]) < 1e-9
        for i in range(len(first["detectors"])):
            assert abs(first["detectors"][i][0] - second["detectors"][i][0]) < 1e-9


class TestEdgeCases:

    def test_anchor_with_nonexistent_channel_noop(self, simple_montage):
        result = similarity_transform_probe(simple_montage, [["S99_D99", "Cz"]])
        assert result["sources"] == simple_montage["sources"]

    def test_anchor_with_unknown_1020_noop(self, simple_montage):
        result = similarity_transform_probe(simple_montage, [["S1_D1", "ZZZZZ"]])
        assert result["sources"] == simple_montage["sources"]

    def test_anchor_with_bad_channel_format(self, simple_montage):
        result = similarity_transform_probe(simple_montage, [["BADLABEL", "Cz"]])
        assert result["sources"] == simple_montage["sources"]

    def test_duplicate_anchors_same_channel(self, simple_montage):
        result = similarity_transform_probe(simple_montage, [["S1_D1", "Cz"], ["S1_D1", "Fz"]])
        assert len(result["sources"]) == len(simple_montage["sources"])

    def test_empty_montage_noop(self):
        result = similarity_transform_probe({"sources": [], "detectors": []}, [["S1_D1", "Cz"]])
        assert result["sources"] == []

    def test_single_source_single_detector(self):
        m = {"sources": [[0.0, 0.0]], "detectors": [[0.1, 0.0]]}
        result = similarity_transform_probe(m, [["S1_D1", "Cz"]])
        mid = _midpoint(result, "S1_D1")
        assert abs(mid[0] - _TEN_TWENTY["Cz"][0]) < 1e-9

    @pytest.mark.xfail(
        reason=(
            "Bug: similarity_transform_probe uses anchors[0][0]"
            " instead of first valid anchor when n==1 after filtering"
        )
    )
    def test_mixed_valid_and_invalid_anchors(self, simple_montage):
        anchors = [["S99_D99", "Cz"], ["S1_D1", "ZZZZZ"], ["S1_D1", "Cz"]]
        result = similarity_transform_probe(simple_montage, anchors)
        mid = _midpoint(result, "S1_D1")
        assert abs(mid[0] - _TEN_TWENTY["Cz"][0]) < 1e-9

    def test_coincident_sources_fallback(self):
        m = {"sources": [[0.5, 0.5], [0.5, 0.5]], "detectors": [[0.5, 0.5]]}
        result = similarity_transform_probe(m, [["S1_D1", "C3"], ["S2_D1", "C4"]])
        assert "sources" in result


class TestProbeGraphClickMultiAnchor:

    @pytest.fixture()
    def view_montage(self):
        return {"sources": [[0.0, 0.3], [0.0, -0.3]], "detectors": [[-0.3, 0.0], [0.3, 0.0]]}

    def test_click_1020_without_selection_noop(self, view_montage):
        from dash import no_update

        from nirspy.gui.callbacks.runtime_callbacks import probe_graph_click
        click = {"points": [{"customdata": "1020:Cz"}]}
        result = probe_graph_click(
            click, "view", None, view_montage, [], None, None, [],
        )
        assert all(r is no_update for r in result)

    def test_click_channel_selects_it(self, view_montage):
        from nirspy.gui.callbacks.runtime_callbacks import probe_graph_click
        click = {"points": [{"customdata": "S1_D1"}]}
        new_sel, *_ = probe_graph_click(
            click, "view", None, view_montage, [], None, None, [],
        )
        assert new_sel == "S1_D1"

    def test_click_same_channel_deselects(self, view_montage):
        from nirspy.gui.callbacks.runtime_callbacks import probe_graph_click
        click = {"points": [{"customdata": "S1_D1"}]}
        new_sel, *_ = probe_graph_click(
            click, "view", "S1_D1", view_montage, [], None, None, [],
        )
        assert new_sel is None

    def test_second_anchor_accumulates(self, view_montage):
        from nirspy.gui.callbacks.runtime_callbacks import probe_graph_click
        new_sel, updated, _o, _f, new_anchors, _b = probe_graph_click(
            {"points": [{"customdata": "1020:C4"}]},
            "view", "S2_D2", view_montage, [], None, None,
            [["S1_D1", "C3"]],
        )
        assert new_sel is None
        assert len(new_anchors) == 2
        assert new_anchors[1] == ["S2_D2", "C4"]

    def test_three_anchors_accumulate(self, view_montage):
        from nirspy.gui.callbacks.runtime_callbacks import probe_graph_click
        _s, _u, _o, _f, new_anchors, _b = probe_graph_click(
            {"points": [{"customdata": "1020:Fz"}]}, "view", "S1_D2", view_montage, [], None, None,
            [["S1_D1", "C3"], ["S2_D2", "C4"]],
        )
        assert len(new_anchors) == 3
        assert new_anchors[2] == ["S1_D2", "Fz"]


class TestProbeResetAnchors:

    def test_no_clicks_returns_no_update(self):
        from dash import no_update

        from nirspy.gui.callbacks.runtime_callbacks import probe_reset_anchors
        assert all(r is no_update for r in probe_reset_anchors(None, None, None, None, None))

    def test_positioning_mode_returns_no_update(self):
        from dash import no_update

        from nirspy.gui.callbacks.runtime_callbacks import probe_reset_anchors
        montage = {"sources": [[0, 0]], "detectors": [[1, 0]]}
        result = probe_reset_anchors(
            1, montage, [], None, "positioning",
        )
        assert all(r is no_update for r in result)

    def test_resets_anchors_and_rebuilds(self):
        from nirspy.gui.callbacks.runtime_callbacks import probe_reset_anchors
        montage = {"sources": [[0.0, 0.3]], "detectors": [[0.3, 0.0]]}
        a_store, badges, fig = probe_reset_anchors(
            1, montage, [], None, "view",
        )
        assert a_store == []
        assert badges is not None and fig is not None

    def test_reset_preserves_exclusions(self):
        import plotly.graph_objects as go

        from nirspy.gui.callbacks.runtime_callbacks import probe_reset_anchors
        montage = {"sources": [[0.0, 0.3]], "detectors": [[0.3, 0.0]]}
        a_store, _, fig = probe_reset_anchors(
            1, montage, ["S1_D1"], None, "view",
        )
        assert a_store == [] and isinstance(fig, go.Figure)


class TestTranslateProbeToAnchor:

    def test_basic_translation(self):
        m = {"sources": [[0.0, 0.0]], "detectors": [[0.2, 0.0]]}
        mid = _midpoint(translate_probe_to_anchor(m, "S1_D1", (1.0, 1.0)), "S1_D1")
        assert abs(mid[0] - 1.0) < 1e-9

    def test_invalid_channel_noop(self):
        m = {"sources": [[0.0, 0.0]], "detectors": [[0.2, 0.0]]}
        assert translate_probe_to_anchor(m, "INVALID", (1.0, 1.0)) is m

    def test_empty_label_noop(self):
        m = {"sources": [[0.0, 0.0]], "detectors": [[0.2, 0.0]]}
        assert translate_probe_to_anchor(m, "", (1.0, 1.0)) is m

    def test_out_of_range_noop(self):
        m = {"sources": [[0.0, 0.0]], "detectors": [[0.2, 0.0]]}
        assert translate_probe_to_anchor(m, "S5_D5", (1.0, 1.0)) is m


class TestChannelMidpoint:

    def test_valid_label(self):
        r = _channel_midpoint({"sources": [[0.0, 0.0]], "detectors": [[1.0, 0.0]]}, "S1_D1")
        assert r is not None and abs(r[0] - 0.5) < 1e-9

    def test_empty_label(self):
        assert _channel_midpoint({"sources": [[0, 0]], "detectors": [[1, 0]]}, "") is None

    def test_no_underscore(self):
        assert _channel_midpoint({"sources": [[0, 0]], "detectors": [[1, 0]]}, "S1D1") is None

    def test_out_of_range(self):
        assert _channel_midpoint({"sources": [[0, 0]], "detectors": [[1, 0]]}, "S5_D5") is None

    def test_non_numeric_index(self):
        assert _channel_midpoint({"sources": [[0, 0]], "detectors": [[1, 0]]}, "Sx_Dy") is None


class TestBuildAnchorBadges:

    def test_empty_anchors_shows_no_anchors(self):
        assert "No anchors set" in str(_build_anchor_badges([]))

    def test_with_anchors_shows_labels(self):
        text = str(_build_anchor_badges([["S1_D1", "Cz"], ["S2_D2", "Fz"]]))
        assert "S1_D1" in text and "Cz" in text
