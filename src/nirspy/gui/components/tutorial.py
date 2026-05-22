"""Tutorial overlay component for guided onboarding.

Provides a sequential modal overlay that walks new users through the five
core steps of using NIRSPY: loading a SNIRF file, building a pipeline,
configuring parameters, running the pipeline, and viewing results.

Decision D3 confirmed by Lead: overlay modal sequential approach.
"""

from __future__ import annotations

from dataclasses import dataclass

import dash_bootstrap_components as dbc
from dash import html


@dataclass(frozen=True)
class TutorialStep:
    """A single step in the guided tutorial.

    Attributes
    ----------
    step:
        1-based step number.
    title:
        Short heading displayed in the modal.
    body:
        Instructional text for the step.
    target_id:
        DOM ``id`` of the element to visually highlight.
    """

    step: int
    title: str
    body: str
    target_id: str


TUTORIAL_STEPS: list[TutorialStep] = [
    TutorialStep(
        step=1,
        title="Load a SNIRF file",
        body=(
            "Click the ‘Load SNIRF’ block in the catalog "
            "on the left to add it to your pipeline, then select "
            "your SNIRF file using the ‘Select SNIRF file’ "
            "button below the pipeline."
        ),
        target_id="block-catalog",
    ),
    TutorialStep(
        step=2,
        title="Build your pipeline",
        body=(
            "Add preprocessing blocks from the catalog. "
            "Use the arrow buttons to reorder blocks and "
            "the toggle to enable or disable individual steps."
        ),
        target_id="pipeline-view",
    ),
    TutorialStep(
        step=3,
        title="Configure parameters",
        body=(
            "Click any block card to open its parameter editor "
            "on the right panel.  Hover over the (i) icons for "
            "scientific references and recommended ranges."
        ),
        target_id="param-editor",
    ),
    TutorialStep(
        step=4,
        title="Execute the pipeline",
        body=(
            "Click ‘Run Pipeline’ to process your data.  "
            "A progress bar will appear while blocks execute "
            "sequentially."
        ),
        target_id="run-button",
    ),
    TutorialStep(
        step=5,
        title="View results",
        body=(
            "Explore the tabs below: Raw Data shows the loaded "
            "signals, QC displays channel quality (SCI), and HRF "
            "plots the averaged hemodynamic response per condition."
        ),
        target_id="viz-tabs",
    ),
]

# CSS class applied to the highlighted target element
TUTORIAL_HIGHLIGHT_CLASS = "tutorial-highlight"

# Inline CSS injected into the layout; keeps the tutorial self-contained
# without requiring an external stylesheet.
TUTORIAL_CSS = (
    ".tutorial-highlight {"
    "  box-shadow: 0 0 0 4px rgba(63, 81, 181, 0.5) !important;"
    "  position: relative;"
    "  z-index: 1000;"
    "  transition: box-shadow 0.3s ease;"
    "}"
)


def render_tutorial_modal(step_index: int) -> dbc.Modal:
    """Build a Bootstrap modal for a given tutorial step.

    Parameters
    ----------
    step_index:
        Zero-based index into :data:`TUTORIAL_STEPS`.

    Returns
    -------
    dbc.Modal
        A modal dialog with title, body text, and navigation buttons.
    """
    total = len(TUTORIAL_STEPS)
    if step_index < 0 or step_index >= total:
        step_index = 0

    ts = TUTORIAL_STEPS[step_index]

    prev_disabled = step_index == 0
    is_last = step_index == total - 1

    footer_buttons: list[dbc.Button | html.Span] = [
        dbc.Button(
            "Skip tutorial",
            id="tutorial-skip",
            color="link",
            size="sm",
            className="me-auto text-muted",
        ),
        dbc.Button(
            "Previous",
            id="tutorial-prev",
            color="secondary",
            size="sm",
            disabled=prev_disabled,
        ),
    ]

    if is_last:
        footer_buttons.append(
            dbc.Button(
                "Finish",
                id="tutorial-finish",
                color="success",
                size="sm",
            ),
        )
    else:
        footer_buttons.append(
            dbc.Button(
                "Next",
                id="tutorial-next",
                color="primary",
                size="sm",
            ),
        )

    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle(
                    f"Step {ts.step}/{total} — {ts.title}"
                ),
                close_button=False,
            ),
            dbc.ModalBody(ts.body),
            dbc.ModalFooter(footer_buttons),
        ],
        id="tutorial-modal",
        is_open=True,
        centered=True,
        backdrop="static",
        keyboard=False,
    )
