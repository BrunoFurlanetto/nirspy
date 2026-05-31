"""Tests for per-condition epoch extraction editor (T-041).

Coverage
--------
CA-1: Radio toggle "Per-condition windows / groups" appears in ParamEditor
      for epochs_extraction.
CA-2: Defaults of tmin/tmax for epochs_extraction are -0.5/5.0 (not
      BlockAverage -2.0/18.0).
CA-3: Condition windows editor renders with ParamMeta correct for
      epochs_extraction (min/max/step from that block's registry).
CA-4: Condition groups editor renders with ParamMeta correct for
      epochs_extraction.
CA-5: Toggle to "windows" clears groups; toggle to "simple" (windows)
      clears per_condition_windows too.
CA-6: execution_callbacks converts per_condition_groups dict ->
      list[ConditionGroup] when building EpochsExtractionParams.
CA-7: No previously passing tests broken (verified by running full suite --
      this module itself must not import-error or fail).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from nirspy.blocks.epochs import EpochsExtractionParams
from nirspy.gui.callbacks.execution_callbacks import _build_pipeline_from_state
from nirspy.gui.components.condition_groups_editor import (
    render_condition_groups_editor,
)
from nirspy.gui.components.condition_windows_editor import (
    render_condition_windows_editor,
)
from nirspy.gui.components.param_editor import render_param_editor
from nirspy.gui.components.param_metadata import metadata_for

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    block_id: str = "epochs_extraction",
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import uuid

    return {
        "block_id": block_id,
        "instance_id": str(uuid.uuid4()),
        "params": params or {},
        "enabled": True,
    }


def _render_epoch_editor(current_values: dict[str, Any]) -> str:
    """Render the ParamEditor for epochs_extraction and return its HTML repr."""
    result = render_param_editor(
        block_id="epochs_extraction",
        instance_id="step-ee",
        params_class=EpochsExtractionParams,
        current_values=current_values,
    )
    return str(result)


# ---------------------------------------------------------------------------
# CA-1: Radio toggle present in epochs_extraction ParamEditor
# ---------------------------------------------------------------------------


class TestCA1RadioTogglePresent:
    """CA-1: The HRF mode radio toggle appears for epochs_extraction."""

    def test_radio_toggle_in_epochs_extraction(self) -> None:
        html_str = _render_epoch_editor({})
        assert "cg-mode-radio" in html_str

    def test_radio_options_windows_and_groups(self) -> None:
        html_str = _render_epoch_editor({})
        assert "Per-condition windows" in html_str
        assert "Per-condition groups" in html_str

    def test_radio_toggle_not_present_for_unrelated_block(self) -> None:
        from nirspy.blocks.preprocessing import BeerLambertParams

        result = render_param_editor(
            block_id="beer_lambert",
            instance_id="step-bl",
            params_class=BeerLambertParams,
            current_values={},
        )
        assert "cg-mode-radio" not in str(result)

    def test_windows_panel_present(self) -> None:
        html_str = _render_epoch_editor({})
        assert "cg-windows-panel" in html_str

    def test_groups_panel_present(self) -> None:
        html_str = _render_epoch_editor({})
        assert "cg-groups-panel" in html_str


# ---------------------------------------------------------------------------
# CA-2: defaults tmin=-0.5, tmax=5.0 — distinct from BlockAverage
# ---------------------------------------------------------------------------


class TestCA2EpochsExtractionDefaults:
    """CA-2: EpochsExtractionParams defaults differ from BlockAverageParams."""

    def test_default_tmin_minus_half(self) -> None:
        params = EpochsExtractionParams()
        assert params.tmin == -0.5

    def test_default_tmax_five(self) -> None:
        params = EpochsExtractionParams()
        assert params.tmax == 5.0

    def test_defaults_differ_from_block_average(self) -> None:
        from nirspy.blocks.analysis import BlockAverageParams

        ba = BlockAverageParams()
        ee = EpochsExtractionParams()
        assert ba.tmin != ee.tmin
        assert ba.tmax != ee.tmax

    def test_param_meta_tmin_min_range(self) -> None:
        """ParamMeta for epochs_extraction.tmin has min=-30 (not -60 like block_average)."""
        meta_ee = metadata_for("epochs_extraction", "tmin")
        meta_ba = metadata_for("block_average", "tmin")
        assert meta_ee is not None
        assert meta_ba is not None
        assert meta_ee.min != meta_ba.min

    def test_param_meta_tmax_max_range(self) -> None:
        """ParamMeta for epochs_extraction.tmax has max=60 (not 180 like block_average)."""
        meta_ee = metadata_for("epochs_extraction", "tmax")
        meta_ba = metadata_for("block_average", "tmax")
        assert meta_ee is not None
        assert meta_ba is not None
        assert meta_ee.max != meta_ba.max


# ---------------------------------------------------------------------------
# CA-3: condition_windows_editor renders with correct ParamMeta for
#        epochs_extraction (min/max/step differ from block_average)
# ---------------------------------------------------------------------------


class TestCA3WindowsEditorParamMeta:
    """CA-3: Per-condition windows editor uses epochs_extraction ParamMeta."""

    def test_editor_renders_with_epochs_block_id(self) -> None:
        result = render_condition_windows_editor(
            "step-ee",
            {"S1": {"tmin": -0.5, "tmax": 5.0, "baseline_tmin": -0.5, "baseline_tmax": 0.0}},
            available_conditions=["S1"],
            block_id="epochs_extraction",
        )
        html_str = str(result)
        assert "cond-window-row" in html_str
        assert "'condition': 'S1'" in html_str

    def test_epochs_tmax_max_bound_applied(self) -> None:
        """Default fill uses meta.max; epochs max=60, block_average max=180."""
        meta = metadata_for("epochs_extraction", "tmax")
        assert meta is not None
        # _render_row defaults to meta.max when no value given
        result = render_condition_windows_editor(
            "step-ee",
            {},
            available_conditions=["Cond1"],
            block_id="epochs_extraction",
        )
        # The table is hidden (switch off, no current_value), but the row
        # HTML still carries the max attr from ParamMeta.
        html_str = str(result)
        # max for epochs_extraction.tmax is 60
        assert "max=60" in html_str

    def test_block_average_tmax_max_differs(self) -> None:
        """block_average tmax max=180 — different from epochs_extraction."""
        result_ba = render_condition_windows_editor(
            "step-ba",
            {},
            available_conditions=["Cond1"],
            block_id="block_average",
        )
        # max for block_average.tmax is 180
        assert "max=180" in str(result_ba)

    def test_per_condition_windows_field_suppressed_in_fallback(self) -> None:
        """The per_condition_windows field must NOT appear as a generic text input."""
        html_str = _render_epoch_editor({})
        # The field name should not appear as a raw param-input field
        assert "'field': 'per_condition_windows'" not in html_str

    def test_groups_field_suppressed_in_fallback(self) -> None:
        """The groups field must NOT appear as a generic text input."""
        html_str = _render_epoch_editor({})
        assert "'field': 'groups'" not in html_str


# ---------------------------------------------------------------------------
# CA-4: condition_groups_editor renders with correct ParamMeta for
#        epochs_extraction
# ---------------------------------------------------------------------------


class TestCA4GroupsEditorParamMeta:
    """CA-4: Per-condition groups editor uses epochs_extraction ParamMeta."""

    def test_editor_renders_with_epochs_block_id(self) -> None:
        groups: dict[str, Any] = {
            "G1": {
                "label": "G1",
                "condition_names": ["S1"],
                "event_indices": [],
                "tmin": -0.5,
                "tmax": 5.0,
                "baseline_tmin": -0.5,
                "baseline_tmax": 0.0,
            }
        }
        result = render_condition_groups_editor(
            "step-ee",
            groups,
            available_conditions=["S1"],
            block_id="epochs_extraction",
        )
        html_str = str(result)
        assert "cg-label" in html_str
        assert "cg-time" in html_str

    def test_epochs_tmin_min_applied_in_group_card(self) -> None:
        """Group numeric inputs carry epochs_extraction ParamMeta bounds."""
        groups: dict[str, Any] = {
            "G1": {
                "label": "G1",
                "condition_names": ["S1"],
                "event_indices": [],
                "tmin": -0.5,
                "tmax": 5.0,
                "baseline_tmin": -0.5,
                "baseline_tmax": 0.0,
            }
        }
        result = render_condition_groups_editor(
            "step-ee",
            groups,
            available_conditions=["S1"],
            block_id="epochs_extraction",
        )
        html_str = str(result)
        # epochs_extraction tmin has min=-30; block_average has min=-60
        assert "min=-30" in html_str

    def test_empty_groups_shows_add_button(self) -> None:
        result = render_condition_groups_editor(
            "step-ee",
            None,
            block_id="epochs_extraction",
        )
        assert "cg-add" in str(result)

    def test_instance_id_propagated(self) -> None:
        result = render_condition_groups_editor(
            "my-epoch-step",
            None,
            block_id="epochs_extraction",
        )
        assert "'instance_id': 'my-epoch-step'" in str(result)


# ---------------------------------------------------------------------------
# CA-5: toggle_hrf_mode clears correct fields for epochs_extraction
# ---------------------------------------------------------------------------


class TestCA5ToggleClearsCorrectFields:
    """CA-5: toggle_hrf_mode clears groups/per_condition_windows correctly."""

    def _call_toggle(
        self,
        instance_id: str,
        new_mode: str,
        pipeline_state: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        import unittest.mock as mock_lib

        import nirspy.gui.callbacks.param_callbacks as mod
        from nirspy.gui.callbacks.param_callbacks import toggle_hrf_mode

        ctx_mock = MagicMock()
        ctx_mock.triggered_id = {
            "type": "cg-mode-radio",
            "instance_id": instance_id,
        }
        ctx_mock.triggered = [{"value": new_mode}]

        with mock_lib.patch.object(mod, "ctx", ctx_mock):
            result = toggle_hrf_mode([new_mode], pipeline_state)
        return result  # type: ignore[return-value]

    def _make_state(
        self,
        instance_id: str = "step-ee",
        block_id: str = "epochs_extraction",
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "instance_id": instance_id,
                "block_id": block_id,
                "params": params or {},
                "enabled": True,
            }
        ]

    def test_toggle_to_groups_clears_per_condition_windows(self) -> None:
        state = self._make_state(
            params={
                "per_condition_windows": {"S1": {"tmin": -0.5, "tmax": 5.0,
                                                  "baseline_tmin": -0.5,
                                                  "baseline_tmax": 0.0}},
            }
        )
        result = self._call_toggle("step-ee", "groups", state)
        params = result[0]["params"]
        assert params.get("per_condition_windows") == {}

    def test_toggle_to_windows_clears_groups_for_epochs_extraction(self) -> None:
        """For epochs_extraction, toggling to 'windows' clears 'groups' field."""
        state = self._make_state(
            params={
                "per_condition_groups": {
                    "G1": {
                        "label": "G1",
                        "condition_names": ["S1"],
                        "event_indices": [],
                        "tmin": -0.5,
                        "tmax": 5.0,
                        "baseline_tmin": -0.5,
                        "baseline_tmax": 0.0,
                    }
                },
            }
        )
        result = self._call_toggle("step-ee", "windows", state)
        params = result[0]["params"]
        # For epochs_extraction the callback writes params["groups"] = None
        assert params.get("groups") is None

    def test_toggle_to_windows_does_not_clear_per_condition_groups_for_block_average(self) -> None:
        """For block_average, toggling to 'windows' clears per_condition_groups (not groups)."""
        state = self._make_state(
            instance_id="step-ba",
            block_id="block_average",
            params={
                "per_condition_groups": {
                    "G1": {
                        "label": "G1",
                        "condition_names": ["S1"],
                        "event_indices": [],
                        "tmin": -2.0,
                        "tmax": 18.0,
                        "baseline_tmin": -2.0,
                        "baseline_tmax": 0.0,
                    }
                },
            },
        )
        result = self._call_toggle("step-ba", "windows", state)
        params = result[0]["params"]
        assert params.get("per_condition_groups") == {}

    def test_hrf_mode_marker_set_on_toggle(self) -> None:
        state = self._make_state()
        result = self._call_toggle("step-ee", "groups", state)
        assert result[0]["params"].get("_hrf_mode") == "groups"


# ---------------------------------------------------------------------------
# CA-6: execution_callbacks converts per_condition_groups -> list[ConditionGroup]
# ---------------------------------------------------------------------------


class TestCA6ExecutionConversion:
    """CA-6: _build_pipeline_from_state converts groups dict to list[ConditionGroup]."""

    def test_per_condition_groups_dict_converted_to_list(self) -> None:
        """When epochs_extraction has per_condition_groups, they become 'groups'."""
        import uuid

        entry: dict[str, Any] = {
            "block_id": "epochs_extraction",
            "instance_id": str(uuid.uuid4()),
            "enabled": True,
            "params": {
                "per_condition_groups": {
                    "Long": {
                        "label": "Long",
                        "condition_names": ["S1"],
                        "event_indices": [],
                        "tmin": -0.5,
                        "tmax": 5.0,
                        "baseline_tmin": -0.5,
                        "baseline_tmax": 0.0,
                    }
                }
            },
        }
        # _build_pipeline_from_state must instantiate EpochsExtractionParams
        # with groups=[ConditionGroup(...)]; no TypeError expected.
        pipeline = _build_pipeline_from_state([entry])
        assert len(pipeline.steps) == 1
        params = pipeline.steps[0].params  # type: ignore[attr-defined]
        assert params.groups is not None
        assert len(params.groups) == 1
        from nirspy.blocks.analysis import ConditionGroup

        assert isinstance(params.groups[0], ConditionGroup)
        assert params.groups[0].label == "Long"

    def test_empty_per_condition_groups_produces_no_groups(self) -> None:
        import uuid

        entry: dict[str, Any] = {
            "block_id": "epochs_extraction",
            "instance_id": str(uuid.uuid4()),
            "enabled": True,
            "params": {"per_condition_groups": {}},
        }
        pipeline = _build_pipeline_from_state([entry])
        params = pipeline.steps[0].params  # type: ignore[attr-defined]
        # Empty dict means no groups set — params.groups stays None (default)
        assert params.groups is None

    def test_no_per_condition_groups_produces_default_params(self) -> None:
        """Without per_condition_groups, EpochsExtractionParams uses its defaults."""
        import uuid

        entry: dict[str, Any] = {
            "block_id": "epochs_extraction",
            "instance_id": str(uuid.uuid4()),
            "enabled": True,
            "params": {},
        }
        pipeline = _build_pipeline_from_state([entry])
        params = pipeline.steps[0].params  # type: ignore[attr-defined]
        assert params.tmin == -0.5
        assert params.tmax == 5.0
        assert params.groups is None

    def test_multiple_groups_all_converted(self) -> None:
        """All groups in the dict become ConditionGroup instances."""
        import uuid

        entry: dict[str, Any] = {
            "block_id": "epochs_extraction",
            "instance_id": str(uuid.uuid4()),
            "enabled": True,
            "params": {
                "per_condition_groups": {
                    "Short": {
                        "label": "Short",
                        "condition_names": ["S1"],
                        "event_indices": [],
                        "tmin": -0.5,
                        "tmax": 3.0,
                        "baseline_tmin": -0.5,
                        "baseline_tmax": 0.0,
                    },
                    "Long": {
                        "label": "Long",
                        "condition_names": ["S2"],
                        "event_indices": [],
                        "tmin": -0.5,
                        "tmax": 8.0,
                        "baseline_tmin": -0.5,
                        "baseline_tmax": 0.0,
                    },
                }
            },
        }
        pipeline = _build_pipeline_from_state([entry])
        params = pipeline.steps[0].params  # type: ignore[attr-defined]
        assert params.groups is not None
        assert len(params.groups) == 2
        labels = {g.label for g in params.groups}
        assert labels == {"Short", "Long"}

    def test_per_condition_groups_key_stripped_from_final_params(self) -> None:
        """After conversion, 'per_condition_groups' must NOT be passed to params_class."""
        import uuid

        entry: dict[str, Any] = {
            "block_id": "epochs_extraction",
            "instance_id": str(uuid.uuid4()),
            "enabled": True,
            "params": {
                "per_condition_groups": {
                    "G": {
                        "label": "G",
                        "condition_names": ["S1"],
                        "event_indices": [],
                        "tmin": -0.5,
                        "tmax": 5.0,
                        "baseline_tmin": -0.5,
                        "baseline_tmax": 0.0,
                    }
                }
            },
        }
        # Should not raise TypeError for unexpected kwarg 'per_condition_groups'
        pipeline = _build_pipeline_from_state([entry])
        assert len(pipeline.steps) == 1


# ---------------------------------------------------------------------------
# CA-7: sanity — EpochsExtractionParams metadata entries all registered
# ---------------------------------------------------------------------------


class TestCA7MetadataRegistry:
    """CA-7: All five registered ParamMeta entries for epochs_extraction are present."""

    def test_tmin_registered(self) -> None:
        meta = metadata_for("epochs_extraction", "tmin")
        assert meta is not None
        assert meta.unit == "s"

    def test_tmax_registered(self) -> None:
        meta = metadata_for("epochs_extraction", "tmax")
        assert meta is not None
        assert meta.unit == "s"

    def test_baseline_tmin_registered(self) -> None:
        meta = metadata_for("epochs_extraction", "baseline_tmin")
        assert meta is not None

    def test_baseline_tmax_registered(self) -> None:
        meta = metadata_for("epochs_extraction", "baseline_tmax")
        assert meta is not None

    def test_reject_amplitude_registered(self) -> None:
        meta = metadata_for("epochs_extraction", "reject_amplitude")
        assert meta is not None
        assert meta.unit == "mol/L"

    def test_unknown_field_returns_none(self) -> None:
        assert metadata_for("epochs_extraction", "nonexistent") is None
