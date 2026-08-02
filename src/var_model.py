"""
Vector Autoregression estimation, lag selection, and Impulse Response Functions for cross-asset shock transmission analysis.
"""

import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")

# ==========================================================
# Configuration
# ==========================================================

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

DEFAULT_ENDOG = [
    "spy_ret",
    "tlt_ret",
    "gld_ret",
    "dgs2_change",
    "dgs10_change",
    "dxy_ret"
]

CHOLESKY_ORDER = [
    "dgs2_change",
    "dgs10_change",
    "dxy_ret",
    "spy_ret",
    "tlt_ret",
    "gld_ret"
]

MAX_LAGS = 10
IRF_PERIODS = 20


# ==========================================================
# Helper
# ==========================================================

def _validate_columns(df, cols):

    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


# ==========================================================
# Fit VAR
# ==========================================================

def fit_var_model(
    df,
    endog_cols=None,
    maxlags=MAX_LAGS
):

    try:

        endog_cols = endog_cols or DEFAULT_ENDOG

        _validate_columns(df, endog_cols)

        endog = (
            df[endog_cols]
            .dropna()
            .copy()
        )

        model = VAR(endog)

        order_results = model.select_order(
            maxlags=maxlags
        )

        optimal_lag = order_results.aic

        if optimal_lag is None:
            optimal_lag = 1

        optimal_lag = int(optimal_lag)

        fitted = model.fit(optimal_lag)

        with open(
            OUTPUTS_DIR / "var_summary.txt",
            "w",
            encoding="utf-8"
        ) as f:

            f.write(str(fitted.summary()))

        with open(
            OUTPUTS_DIR / "var_model.pkl",
            "wb"
        ) as f:

            pickle.dump(fitted, f)

        return (
            fitted,
            optimal_lag,
            order_results
        )

    except Exception as e:

        print(
            f"VAR estimation failed: {e}"
        )

        return None


# ==========================================================
# IRF
# ==========================================================

def generate_irf(
    fitted_model,
    impulse_var,
    response_vars=None,
    periods=IRF_PERIODS,
    figsize=(14,10)
):

    try:

        response_vars = (
            response_vars
            or CHOLESKY_ORDER
        )

        reordered = fitted_model.reorder(
            CHOLESKY_ORDER
        )

        irf = reordered.irf(
            periods=periods
        )

        stderr = None

        try:
            stderr = irf.stderr()
        except Exception:
            stderr = None

        fig, axes = plt.subplots(
            len(response_vars),
            1,
            figsize=figsize,
            sharex=True
        )

        if len(response_vars) == 1:
            axes = [axes]

        impulse_idx = CHOLESKY_ORDER.index(
            impulse_var
        )

        horizon = np.arange(
            periods + 1
        )

        for ax, response in zip(
            axes,
            response_vars
        ):

            response_idx = CHOLESKY_ORDER.index(
                response
            )

            y = irf.irfs[
                :,
                response_idx,
                impulse_idx
            ]

            ax.plot(
                horizon,
                y,
                lw=2
            )

            if stderr is not None:

                se = stderr[
                    :,
                    response_idx,
                    impulse_idx
                ]

                ax.fill_between(
                    horizon,
                    y - 1.96 * se,
                    y + 1.96 * se,
                    alpha=0.25
                )

            ax.axhline(
                0,
                color="black",
                linestyle="--",
                linewidth=0.8
            )

            ax.set_title(
                f"Response of {response} to 1-sd shock in {impulse_var}"
            )

            ax.grid(
                alpha=0.3
            )

        axes[-1].set_xlabel(
            "Days"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUTS_DIR /
            f"irf_shock_{impulse_var}.png",
            dpi=300,
            bbox_inches="tight"
        )

        return fig

    except Exception as e:

        print(
            f"IRF generation failed: {e}"
        )

        return None


# ==========================================================
# Multiple IRFs
# ==========================================================

def generate_all_irfs(
    fitted_model,
    impulse_vars=None,
    periods=IRF_PERIODS
):

    try:

        impulse_vars = impulse_vars or [
            "dgs2_change",
            "spy_ret",
            "dxy_ret"
        ]

        figures = {}

        for impulse in impulse_vars:

            figures[impulse] = generate_irf(
                fitted_model=fitted_model,
                impulse_var=impulse,
                periods=periods
            )

        return figures

    except Exception as e:

        print(
            f"Failed generating IRFs: {e}"
        )

        return None


# ==========================================================
# Diagnostics
# ==========================================================

def var_diagnostics(
    fitted_model
):

    try:

        diagnostics = {}

        report = []

        try:

            white = fitted_model.test_whiteness()

            diagnostics[
                "whiteness"
            ] = white.pvalue

            report.append(
                f"Whiteness Test p-value: {white.pvalue}"
            )

        except Exception as e:

            diagnostics[
                "whiteness"
            ] = None

            report.append(
                f"Whiteness Test Failed: {e}"
            )

        try:

            normal = fitted_model.test_normality()

            diagnostics[
                "normality"
            ] = normal.pvalue

            report.append(
                f"Normality Test p-value: {normal.pvalue}"
            )

        except Exception as e:

            diagnostics[
                "normality"
            ] = None

            report.append(
                f"Normality Test Failed: {e}"
            )

        try:

            serial = fitted_model.test_serial_correlation()

            diagnostics[
                "serial_correlation"
            ] = serial.pvalue

            report.append(
                f"Serial Correlation Test p-value: {serial.pvalue}"
            )

        except Exception as e:

            diagnostics[
                "serial_correlation"
            ] = None

            report.append(
                f"Serial Correlation Test Failed: {e}"
            )

        with open(
            OUTPUTS_DIR /
            "var_diagnostics.txt",
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n".join(report)
            )

        return diagnostics

    except Exception as e:

        print(
            f"Diagnostics failed: {e}"
        )

        return None
