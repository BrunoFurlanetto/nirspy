"""HTML Report block -- v0.4 (T-038).

HTMLReportBlock: sink block that generates a standalone HTML report
summarizing the executed pipeline, with optional embedded Plotly plots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from nirspy.domain.block import BlockResult, BlockSpec
from nirspy.domain.data_types import DataType
from nirspy.domain.exceptions import ValidationError

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HTMLReportParams:
    """Parameters for HTML report generation.

    Attributes
    ----------
    output_path:
        File path for the generated HTML report.
    title:
        Report title displayed in the header.
    include_plots:
        Whether to embed Plotly visualizations.
    include_qc:
        Whether to include quality control section.
    """

    output_path: str = "report.html"
    title: str = "NIRSPY Pipeline Report"
    include_plots: bool = True
    include_qc: bool = True


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

_REPORT_SPEC = BlockSpec(
    block_id="html_report",
    display_name="HTML Report",
    input_type=DataType.ANY,
    output_type=DataType.NONE,
    params_class=HTMLReportParams,
    description=(
        "Generates a standalone HTML report summarizing the pipeline "
        "execution. Sink block -- produces no downstream output."
    ),
)


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------


class HTMLReportBlock:
    """Generate a standalone HTML report from pipeline results.

    Accepts any input type:
    - dict[str, mne.Evoked]: generates HRF plots
    - GLMResult: generates t-stat bar charts
    - pd.DataFrame: renders as HTML table
    - Other: basic metadata report

    Pipeline context (executed steps, params) is read from
    context.extra when available.
    """

    SPEC: ClassVar[BlockSpec] = _REPORT_SPEC

    def __init__(
        self,
        params: HTMLReportParams | None = None,
    ) -> None:
        self.params: HTMLReportParams = params or HTMLReportParams()

    @property
    def spec(self) -> BlockSpec:
        """Return the static block descriptor."""
        return _REPORT_SPEC

    def run(self, context: Any, inputs: dict[str, Any]) -> BlockResult:
        """Execute HTML report generation."""
        import jinja2

        if not inputs:
            raise ValidationError(
                "HTMLReportBlock requires input data. "
                "It cannot be the first block in a pipeline."
            )

        data = next(iter(inputs.values()))

        # Gather template variables
        template_vars = self._build_template_vars(data, context)

        # Render template
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=jinja2.select_autoescape(["html"]),
        )
        template = env.get_template("report.html.j2")
        html_content = template.render(**template_vars)

        # Write output
        output_path = Path(self.params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding="utf-8")

        metadata: dict[str, Any] = {
            "output_file": str(output_path),
            "title": self.params.title,
            "input_type": type(data).__name__,
            "file_size_bytes": output_path.stat().st_size,
        }

        return BlockResult(
            data=None,
            block_id=_REPORT_SPEC.block_id,
            metadata=metadata,
        )

    def _build_template_vars(
        self, data: Any, context: Any
    ) -> dict[str, Any]:
        """Build the dictionary of variables for the Jinja2 template."""
        import pandas as pd

        from nirspy.domain.glm_result import GLMResult

        template_vars: dict[str, Any] = {
            "title": self.params.title,
            "generated_at": datetime.now(tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            ),
            "pipeline_steps": self._get_pipeline_steps(context),
            "table_html": None,
            "plots": [],
            "qc_info": None,
        }

        # Handle different input types
        if isinstance(data, GLMResult):
            template_vars["table_html"] = self._glm_to_table(data)
            if self.params.include_plots:
                template_vars["plots"] = self._glm_to_plots(data)
        elif isinstance(data, pd.DataFrame):
            template_vars["table_html"] = data.to_html(
                classes="dataframe", max_rows=50, max_cols=20
            )
        elif isinstance(data, dict):
            template_vars.update(self._handle_dict_input(data))

        # QC section from context
        if self.params.include_qc:
            template_vars["qc_info"] = self._get_qc_info(context)

        return template_vars

    def _get_pipeline_steps(self, context: Any) -> list[str]:
        """Extract pipeline step names from context.extra."""
        if context is None:
            return []
        extra = getattr(context, "extra", None)
        if extra is None:
            return []
        return list(extra.get("pipeline_steps", []))

    def _get_qc_info(self, context: Any) -> dict[str, str] | None:
        """Extract QC metadata from context."""
        if context is None:
            return None
        extra = getattr(context, "extra", None)
        if extra is None:
            return None
        qc = extra.get("qc", None)
        if qc and isinstance(qc, dict):
            return {str(k): str(v) for k, v in qc.items()}
        return None

    def _glm_to_table(self, glm_result: Any) -> str:
        """Convert GLMResult to HTML summary table."""
        df = glm_result.to_dataframe()
        return df.to_html(classes="dataframe", index=False, max_rows=100)

    def _glm_to_plots(self, glm_result: Any) -> list[str]:
        """Generate Plotly HTML snippets for each regressor."""
        import plotly.graph_objects as go

        plots: list[str] = []
        for reg_name in glm_result.regressor_names:
            # Skip drift/constant regressors
            if reg_name.startswith("drift") or reg_name == "constant":
                continue

            idx = glm_result.regressor_names.index(reg_name)
            values = glm_result.t_stats[idx, :]
            channels = glm_result.channel_names

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=channels,
                        y=values,
                        marker_color="#2196F3",
                    )
                ]
            )
            fig.update_layout(
                title=f"{reg_name} -- t-statistics",
                xaxis_title="Channel",
                yaxis_title="t-stat",
                template="plotly_white",
                height=350,
                margin={"t": 40, "b": 60, "l": 50, "r": 20},
            )
            plots.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

        return plots

    def _handle_dict_input(self, data: dict[str, Any]) -> dict[str, Any]:
        """Handle dict input (assumed dict[str, Evoked])."""
        result: dict[str, Any] = {}
        plots: list[str] = []

        try:
            import mne

            sample = next(iter(data.values()), None)
            if sample is not None and isinstance(sample, mne.Evoked):
                if self.params.include_plots:
                    plots = self._evoked_to_plots(data)
                # Build summary table
                rows = []
                for cond_name, evoked in data.items():
                    rows.append(
                        f"<tr><td>{cond_name}</td>"
                        f"<td>{evoked.nave}</td>"
                        f"<td>{evoked.tmin:.3f}</td>"
                        f"<td>{evoked.tmax:.3f}</td></tr>"
                    )
                table = (
                    "<table><tr><th>Condition</th><th>N averages</th>"
                    "<th>tmin</th><th>tmax</th></tr>"
                    + "".join(rows)
                    + "</table>"
                )
                result["table_html"] = table
        except ImportError:
            pass

        result["plots"] = plots
        return result

    def _evoked_to_plots(self, data: dict[str, Any]) -> list[str]:
        """Generate time-series plots for Evoked data."""
        import plotly.graph_objects as go

        plots: list[str] = []
        for cond_name, evoked in data.items():
            times = evoked.times
            # Plot mean across channels
            mean_data = evoked.data.mean(axis=0)

            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=times,
                        y=mean_data,
                        mode="lines",
                        name=cond_name,
                    )
                ]
            )
            fig.update_layout(
                title=f"HRF -- {cond_name} (mean across channels)",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                template="plotly_white",
                height=300,
                margin={"t": 40, "b": 50, "l": 50, "r": 20},
            )
            plots.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

        return plots
