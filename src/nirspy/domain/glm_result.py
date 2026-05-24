"""GLMResult -- domain-layer container for GLM statistical results.

This dataclass encapsulates the output of a first-level GLM analysis
without importing MNE or mne-nirs.  The engine layer converts MNE-NIRS
``RegressionResults`` into this pure representation.

numpy is accepted in domain/ (it is stdlib-adjacent and already used
implicitly via DataType objects that transport MNE arrays).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class GLMResult:
    """Pure container for first-level GLM results.

    Attributes
    ----------
    theta:
        Coefficient matrix (n_regressors, n_channels).
    t_stats:
        T-statistic matrix (n_regressors, n_channels).
    p_values:
        P-value matrix (n_regressors, n_channels).  Two-tailed.
    mse:
        Mean squared error per channel (n_channels,).
    channel_names:
        List of channel names matching columns of theta/t_stats/p_values.
    regressor_names:
        List of regressor names matching rows of theta/t_stats/p_values.
    design_matrix:
        Design matrix used for the fit (n_timepoints, n_regressors).
    noise_model:
        Noise model used ('ar1', 'ols', etc.).
    metadata:
        Arbitrary extra info (e.g. drift_model, hrf_model used).
    """

    theta: npt.NDArray[np.floating[Any]]
    t_stats: npt.NDArray[np.floating[Any]]
    p_values: npt.NDArray[np.floating[Any]]
    mse: npt.NDArray[np.floating[Any]]
    channel_names: list[str]
    regressor_names: list[str]
    design_matrix: npt.NDArray[np.floating[Any]]
    noise_model: str = "ar1"
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_dataframe(self) -> Any:
        """Return a long-format DataFrame with GLM coefficients and stats.

        Columns: regressor, channel, theta, t_stat, p_value.

        Returns
        -------
        pd.DataFrame
            Long-format table suitable for downstream analysis and export.
        """
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for i, reg in enumerate(self.regressor_names):
            for j, ch in enumerate(self.channel_names):
                rows.append({
                    "regressor": reg,
                    "channel": ch,
                    "theta": float(self.theta[i, j]),
                    "t_stat": float(self.t_stats[i, j]),
                    "p_value": float(self.p_values[i, j]),
                })
        return pd.DataFrame(rows)

    def get_contrast(self, name: str) -> npt.NDArray[np.floating[Any]]:
        """Return t-stats array for a named regressor (across channels).

        Parameters
        ----------
        name:
            Name of the regressor / condition.

        Returns
        -------
        np.ndarray
            Array of shape (n_channels,) with t-statistics.

        Raises
        ------
        KeyError
            If *name* is not in ``regressor_names``.
        """
        if name not in self.regressor_names:
            raise KeyError(
                f"Regressor {name!r} not found. "
                f"Available: {self.regressor_names}"
            )
        idx = self.regressor_names.index(name)
        return self.t_stats[idx, :]
