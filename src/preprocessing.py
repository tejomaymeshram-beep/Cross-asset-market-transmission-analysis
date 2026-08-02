"""
Data preprocessing, feature engineering, and regime classification for the market transmission framework.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# ==========================================================
# Configuration
# ==========================================================

DATA_DIR = Path("data")

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

ASSET_COLS = [
    "SPY",
    "TLT",
    "GLD",
    "USO",
    "DX-Y.NYB",
    "DGS2",
    "DGS10"
]

RETURN_COLS = [
    "spy_ret",
    "tlt_ret",
    "gld_ret",
    "uso_ret",
    "dxy_ret",
    "dgs2_change",
    "dgs10_change"
]

REGIME_BOUNDS = {
    "low_vol": (0, 15),
    "normal_vol": (15, 25),
    "high_vol": (25, 999)
}


# ==========================================================
# Helper
# ==========================================================

def _check_columns(df, cols, name="DataFrame"):

    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


# ==========================================================
# Master Dataset
# ==========================================================

def build_master_dataset(
    vix_df,
    fred_df,
    yf_df,
    events_df
):

    required = {
        "vix_df": vix_df,
        "fred_df": fred_df,
        "yf_df": yf_df,
        "events_df": events_df
    }

    for name, df in required.items():

        _check_columns(
            df,
            ["DATE"],
            name
        )

        df["DATE"] = pd.to_datetime(
            df["DATE"]
        )

        if getattr(df["DATE"].dt, "tz", None) is not None:
            df["DATE"] = df["DATE"].dt.tz_localize(None)

    vix = vix_df.set_index("DATE")
    fred = fred_df.set_index("DATE")
    yf = yf_df.set_index("DATE")
    events = events_df.set_index("DATE")

    master = (
        vix
        .join(fred, how="outer")
        .join(yf, how="outer")
        .join(events, how="outer")
    )

    master = (
        master
        .sort_index()
        .reset_index()
    )

    master.to_csv(
        DATA_DIR / "master_merged.csv",
        index=False
    )

    return master


# ==========================================================
# Returns & Features
# ==========================================================

def calculate_returns(df):

    required = [
        "SPY",
        "TLT",
        "GLD",
        "USO",
        "DX-Y.NYB",
        "DGS2",
        "DGS10",
        "T10Y2Y",
        "T10Y3M",
        "SOFR",
        "DFF",
        "VIX",
        "VIX3M",
        "CPIAUCSL",
        "UNRATE"
    ]

    _check_columns(df, required)

    out = df.copy()

    out["spy_ret"] = out["SPY"].pct_change()
    out["tlt_ret"] = out["TLT"].pct_change()
    out["gld_ret"] = out["GLD"].pct_change()
    out["uso_ret"] = out["USO"].pct_change()
    out["dxy_ret"] = out["DX-Y.NYB"].pct_change()

    out["dgs2_change"] = out["DGS2"].diff()
    out["dgs10_change"] = out["DGS10"].diff()

    out["curve_change"] = out["T10Y2Y"].diff()

    out["sofr_effr_spread"] = (
        out["SOFR"] -
        out["DFF"]
    )

    out["vix_term_slope"] = (
        out["VIX3M"] -
        out["VIX"]
    ) / out["VIX"]

    for col in [
        "DGS2",
        "DGS10",
        "T10Y2Y",
        "T10Y3M"
    ]:

        out[col] = out[col].ffill(limit=5)

    for col in [
        "CPIAUCSL",
        "UNRATE"
    ]:

        out[col] = out[col].ffill(limit=35)

    out = out.dropna(
        subset=RETURN_COLS
    )

    out.to_csv(
        DATA_DIR / "master_returns.csv",
        index=False
    )

    return out


# ==========================================================
# Stationarity
# ==========================================================

def test_stationarity(
    df,
    columns=None,
    maxlag=5
):

    columns = columns or RETURN_COLS

    _check_columns(
        df,
        columns
    )

    rows = []

    for col in columns:

        series = df[col].dropna()

        res = adfuller(
            series,
            maxlag=maxlag,
            regression="c",
            autolag="AIC"
        )

        rows.append({

            "variable":
            col,

            "adf_statistic":
            res[0],

            "p_value":
            res[1],

            "critical_value_5pct":
            res[4]["5%"],

            "is_stationary":
            res[1] < 0.05

        })

        if res[1] >= 0.05:

            print(
                f"WARNING: {col} appears non-stationary "
                f"(p={res[1]:.4f})"
            )

    results = pd.DataFrame(rows)

    results.to_csv(
        OUTPUTS_DIR /
        "stationarity_test.csv",
        index=False
    )

    return results


# ==========================================================
# Regimes
# ==========================================================

def create_regimes(
    df,
    vix_col="VIX"
):

    _check_columns(
        df,
        [vix_col]
    )

    out = df.copy()

    conditions = [

        out[vix_col] < 15,

        (
            (out[vix_col] >= 15)
            &
            (out[vix_col] <= 25)
        ),

        out[vix_col] > 25

    ]

    choices = [
        "low_vol",
        "normal_vol",
        "high_vol"
    ]

    out["regime"] = pd.Categorical(

        np.select(
            conditions,
            choices,
            default="unknown"
        ),

        categories=[
            "low_vol",
            "normal_vol",
            "high_vol",
            "unknown"
        ],

        ordered=True

    )

    out.to_csv(
        DATA_DIR /
        "master_regimes.csv",
        index=False
    )

    return out


# ==========================================================
# Daily COT Alignment
# ==========================================================

def align_cot_to_daily(
    df,
    cot_df
):

    cot_required = [

        "Report_Date_as_YYYY_MM_DD",
        "Market_and_Exchange_Names",
        "Open_Interest_All",
        "NonComm_Positions_Long_All",
        "NonComm_Positions_Short_All",
        "Comm_Positions_Long_All",
        "Comm_Positions_Short_All"

    ]

    _check_columns(
        df,
        ["DATE"]
    )

    _check_columns(
        cot_df,
        cot_required,
        "cot_df"
    )

    master = df.copy()

    master["DATE"] = pd.to_datetime(
        master["DATE"]
    )

    master = master.sort_values(
        "DATE"
    )

    results = []

    for market in cot_df[
        "Market_and_Exchange_Names"
    ].unique():

        temp = cot_df[
            cot_df[
                "Market_and_Exchange_Names"
            ] == market
        ].copy()

        temp[
            "Report_Date_as_YYYY_MM_DD"
        ] = pd.to_datetime(
            temp[
                "Report_Date_as_YYYY_MM_DD"
            ]
        )

        temp = temp.sort_values(
            "Report_Date_as_YYYY_MM_DD"
        )

        merged = pd.merge_asof(

            master,

            temp,

            left_on="DATE",

            right_on="Report_Date_as_YYYY_MM_DD",

            direction="backward"

        )

        merged["market"] = market

        merged["noncomm_net"] = (

            merged[
                "NonComm_Positions_Long_All"
            ]

            -

            merged[
                "NonComm_Positions_Short_All"
            ]

        )

        merged["comm_net"] = (

            merged[
                "Comm_Positions_Long_All"
            ]

            -

            merged[
                "Comm_Positions_Short_All"
            ]

        )

        merged["rolling_min"] = (

            merged["noncomm_net"]

            .rolling(
                52 * 5,
                min_periods=20
            )

            .min()

        )

        merged["rolling_max"] = (

            merged["noncomm_net"]

            .rolling(
                52 * 5,
                min_periods=20
            )

            .max()

        )

        merged["COT_Index"] = (

            (

                merged["noncomm_net"]

                -

                merged["rolling_min"]

            )

            /

            (

                merged["rolling_max"]

                -

                merged["rolling_min"]

            )

        ) * 100

        results.append(merged)

    final = pd.concat(
        results,
        ignore_index=True
    )

    final.to_csv(
        DATA_DIR /
        "master_with_cot.csv",
        index=False
    )

    return final
