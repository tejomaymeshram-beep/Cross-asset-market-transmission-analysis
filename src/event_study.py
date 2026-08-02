"""
Event study analysis for macroeconomic announcements: FOMC, CPI, and NFP.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ==========================================================
# Configuration
# ==========================================================

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

EVENT_TYPES = ["FOMC", "CPI", "NFP"]

ASSET_RETS = [
    "spy_ret",
    "tlt_ret",
    "gld_ret",
    "uso_ret",
    "dgs2_change",
    "dgs10_change"
]

WINDOW = (-5, 5)

CONFIDENCE = 0.95


# ==========================================================
# Helper
# ==========================================================

def _validate_columns(df, cols, name="DataFrame"):

    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} missing required columns: {missing}"
        )


# ==========================================================
# Event Study
# ==========================================================

def run_event_study(
    df,
    events_df,
    event_type,
    window=WINDOW
):

    _validate_columns(
        df,
        ["DATE"] + ASSET_RETS,
        "master"
    )

    _validate_columns(
        events_df,
        ["DATE", "EVENT_TYPE"],
        "events"
    )

    df = df.copy()
    events = events_df.copy()

    df["DATE"] = pd.to_datetime(df["DATE"])
    events["DATE"] = pd.to_datetime(events["DATE"])

    df = df.sort_values("DATE").reset_index(drop=True)

    events = events[
        events["EVENT_TYPE"] == event_type
    ]

    results = {}

    days = np.arange(
        window[0],
        window[1] + 1
    )

    for asset in ASSET_RETS:

        paths = []

        for event_date in events["DATE"]:

            idx = df.index[
                df["DATE"] == event_date
            ]

            if len(idx) == 0:
                continue

            idx = idx[0]

            start = idx + window[0]
            end = idx + window[1]

            if start < 0 or end >= len(df):
                continue

            values = (
                df
                .iloc[start:end+1][asset]
                .values
            )

            if len(values) != len(days):
                continue

            paths.append(values)

        if len(paths) == 0:

            results[asset] = {
                "mean": pd.Series(dtype=float),
                "se": pd.Series(dtype=float),
                "ci": pd.Series(dtype=float),
                "days": days,
                "raw": np.empty((0, len(days)))
            }

            continue

        arr = np.vstack(paths)

        mean = arr.mean(axis=0)

        se = stats.sem(
            arr,
            axis=0,
            nan_policy="omit"
        )

        tcrit = stats.t.ppf(
            (1 + CONFIDENCE) / 2,
            arr.shape[0] - 1
        )

        ci = se * tcrit

        results[asset] = {

            "mean": pd.Series(
                mean,
                index=days
            ),

            "se": pd.Series(
                se,
                index=days
            ),

            "ci": pd.Series(
                ci,
                index=days
            ),

            "days": days,

            "raw": arr

        }

    export = []

    for asset in ASSET_RETS:

        if results[asset]["mean"].empty:
            continue

        temp = pd.DataFrame({

            "asset": asset,

            "day": days,

            "mean": results[asset]["mean"].values,

            "se": results[asset]["se"].values,

            "ci": results[asset]["ci"].values

        })

        export.append(temp)

    if export:

        pd.concat(export).to_csv(

            OUTPUTS_DIR /
            f"event_study_{event_type}.csv",

            index=False

        )

    return results


# ==========================================================
# Plot
# ==========================================================

def plot_event_study(
    event_results,
    event_type,
    figsize=(16,12)
):

    fig, axes = plt.subplots(
        3,
        2,
        figsize=figsize
    )

    axes = axes.flatten()

    for ax, asset in zip(
        axes,
        ASSET_RETS
    ):

        res = event_results[asset]

        if res["mean"].empty:
            continue

        x = res["days"]

        cum = np.cumsum(
            res["mean"].values
        )

        ci = np.cumsum(
            res["ci"].values
        )

        ax.plot(
            x,
            cum,
            lw=2
        )

        ax.fill_between(
            x,
            cum - ci,
            cum + ci,
            alpha=0.25
        )

        ax.axvline(
            0,
            color="red",
            linestyle="--"
        )

        ax.axhline(
            0,
            color="black",
            linewidth=0.8
        )

        ax.set_title(
            f"{asset} around {event_type}"
        )

        ax.set_xlabel("Days")

        ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        OUTPUTS_DIR /
        f"event_study_{event_type}.png",

        dpi=300,

        bbox_inches="tight"

    )

    return fig


# ==========================================================
# Regime Analysis
# ==========================================================

def event_study_by_regime(
    df,
    events_df,
    event_type,
    regime_col="regime"
):

    _validate_columns(
        df,
        [regime_col]
    )

    colors = {

        "low_vol": "green",

        "normal_vol": "blue",

        "high_vol": "red"

    }

    regime_results = {}

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(16,12)
    )

    axes = axes.flatten()

    for regime, color in colors.items():

        subset = df[
            df[regime_col] == regime
        ]

        res = run_event_study(

            subset,

            events_df,

            event_type

        )

        regime_results[regime] = res

        for ax, asset in zip(
            axes,
            ASSET_RETS
        ):

            if res[asset]["mean"].empty:
                continue

            ax.plot(

                res[asset]["days"],

                np.cumsum(
                    res[asset]["mean"]
                ),

                color=color,

                linewidth=2,

                label=regime

            )

            ax.axvline(
                0,
                color="black",
                linestyle="--"
            )

            ax.axhline(
                0,
                color="grey"
            )

            ax.set_title(asset)

    for ax in axes:
        ax.legend()

    plt.tight_layout()

    plt.savefig(

        OUTPUTS_DIR /
        f"event_study_{event_type}_by_regime.png",

        dpi=300,

        bbox_inches="tight"

    )

    return regime_results


# ==========================================================
# Significance Summary
# ==========================================================

def summarize_event_significance(
    event_results
):

    rows = []

    for asset in ASSET_RETS:

        res = event_results[asset]

        if res["mean"].empty:
            continue

        raw = res["raw"]

        if raw.shape[0] == 0:
            continue

        day_index = np.where(
            res["days"] == 1
        )[0]

        if len(day_index) == 0:
            continue

        day_index = day_index[0]

        sample = raw[:, day_index]

        sample = sample[
            ~np.isnan(sample)
        ]

        if len(sample) < 2:

            tstat = np.nan
            pvalue = np.nan

        else:

            tstat, pvalue = stats.ttest_1samp(
                sample,
                popmean=0
            )

        rows.append({

            "asset":
            asset,

            "event_type":
            "CURRENT",

            "t+1_mean":
            np.nanmean(sample),

            "t+1_pvalue":
            pvalue,

            "significant":
            (
                False
                if np.isnan(pvalue)
                else pvalue < 0.05
            )

        })

    summary = pd.DataFrame(rows)

    summary.to_csv(

        OUTPUTS_DIR /
        "event_significance_summary.csv",

        index=False

    )

    return summary
