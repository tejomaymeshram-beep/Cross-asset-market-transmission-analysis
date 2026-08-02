"""
CFTC Commitments of Traders analysis: COT Index, positioning extremes, and contrarian signal evaluation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================================
# Configuration
# ==========================================================

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

LOOKBACK_WEEKS = 52
EXTREME_HIGH = 80
EXTREME_LOW = 20
HOLDING_DAYS = 20


# ==========================================================
# Helpers
# ==========================================================

def _check_columns(df, cols, name="DataFrame"):

    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} missing required columns: {missing}"
        )


# ==========================================================
# COT Index
# ==========================================================

def calculate_cot_index(
    cot_df,
    market_name="E-MINI S&P 500"
):

    required = [

        "Market_and_Exchange_Names",
        "Report_Date_as_YYYY_MM_DD",
        "NonComm_Positions_Long_All",
        "NonComm_Positions_Short_All",
        "Comm_Positions_Long_All",
        "Comm_Positions_Short_All"

    ]

    _check_columns(
        cot_df,
        required,
        "cot_df"
    )

    out = cot_df[
        cot_df["Market_and_Exchange_Names"]
        .str.contains(
            market_name,
            case=False,
            na=False
        )
    ].copy()

    out["DATE"] = pd.to_datetime(
        out["Report_Date_as_YYYY_MM_DD"]
    )

    out = (
        out
        .sort_values("DATE")
        .set_index("DATE")
    )

    out["noncomm_net"] = (

        out["NonComm_Positions_Long_All"]

        -

        out["NonComm_Positions_Short_All"]

    )

    out["comm_net"] = (

        out["Comm_Positions_Long_All"]

        -

        out["Comm_Positions_Short_All"]

    )

    window = LOOKBACK_WEEKS

    rolling_min = (
        out["noncomm_net"]
        .rolling(window, min_periods=5)
        .min()
    )

    rolling_max = (
        out["noncomm_net"]
        .rolling(window, min_periods=5)
        .max()
    )

    out["cot_index"] = (

        (

            out["noncomm_net"]

            -

            rolling_min

        )

        /

        (

            rolling_max

            -

            rolling_min

        )

    ) * 100

    return out


# ==========================================================
# Extreme Signals
# ==========================================================

def identify_extremes(
    cot_indexed,
    high=EXTREME_HIGH,
    low=EXTREME_LOW
):

    _check_columns(
        cot_indexed,
        ["cot_index"]
    )

    df = cot_indexed.copy()

    df["extreme_bullish"] = (
        df["cot_index"] > high
    )

    df["extreme_bearish"] = (
        df["cot_index"] < low
    )

    bullish_flip = (

        df["extreme_bullish"]

        &

        ~df["extreme_bullish"]
        .shift(fill_value=False)

    )

    bearish_flip = (

        df["extreme_bearish"]

        &

        ~df["extreme_bearish"]
        .shift(fill_value=False)

    )

    signals = pd.DataFrame({

        "DATE":
        df.index,

        "cot_index":
        df["cot_index"],

        "extreme_bullish":
        bullish_flip,

        "extreme_bearish":
        bearish_flip

    })

    signals["signal"] = np.select(

        [

            signals["extreme_bullish"],

            signals["extreme_bearish"]

        ],

        [

            "SHORT",

            "LONG"

        ],

        default="NONE"

    )

    signals = signals[
        signals["signal"] != "NONE"
    ].reset_index(drop=True)

    return signals


# ==========================================================
# Contrarian Backtest
# ==========================================================

def backtest_contrarian(
    df_daily,
    signals,
    price_col="SPY",
    holding_days=HOLDING_DAYS
):

    _check_columns(
        df_daily,
        ["DATE", price_col],
        "df_daily"
    )

    _check_columns(
        signals,
        ["DATE", "signal"],
        "signals"
    )

    prices = df_daily.copy()

    prices["DATE"] = pd.to_datetime(
        prices["DATE"]
    )

    prices = (
        prices
        .sort_values("DATE")
        .reset_index(drop=True)
    )

    trades = []

    for _, row in signals.iterrows():

        date = pd.to_datetime(
            row["DATE"]
        )

        idx = prices.index[
            prices["DATE"] >= date
        ]

        if len(idx) == 0:
            continue

        entry = idx[0]

        exit_idx = entry + holding_days

        if exit_idx >= len(prices):
            continue

        entry_price = prices.loc[
            entry,
            price_col
        ]

        exit_price = prices.loc[
            exit_idx,
            price_col
        ]

        raw_return = (

            exit_price -

            entry_price

        ) / entry_price

        if row["signal"] == "SHORT":

            pnl = -raw_return

        else:

            pnl = raw_return

        trades.append({

            "Entry_Date":
            prices.loc[entry, "DATE"],

            "Exit_Date":
            prices.loc[exit_idx, "DATE"],

            "Signal":
            row["signal"],

            "Entry_Price":
            entry_price,

            "Exit_Price":
            exit_price,

            "Return":
            pnl

        })

    trades = pd.DataFrame(trades)

    trades.to_csv(

        OUTPUTS_DIR /
        "cot_contrarian_trades.csv",

        index=False

    )

    if len(trades):

        sharpe = np.nan

        if trades["Return"].std() > 0:

            sharpe = (

                trades["Return"].mean()

                /

                trades["Return"].std()

            ) * np.sqrt(252 / holding_days)

        summary = {

            "win_rate":

            (trades["Return"] > 0)
            .mean(),

            "avg_return":

            trades["Return"]
            .mean(),

            "total_signals":

            len(trades),

            "sharpe":

            sharpe

        }

    else:

        summary = {

            "win_rate": np.nan,

            "avg_return": np.nan,

            "total_signals": 0,

            "sharpe": np.nan

        }

    return summary, trades


# ==========================================================
# Plot
# ==========================================================

def plot_cot_extremes(
    df_daily,
    cot_indexed,
    price_col="SPY"
):

    _check_columns(
        df_daily,
        ["DATE", price_col]
    )

    _check_columns(
        cot_indexed,
        ["cot_index"]
    )

    prices = df_daily.copy()

    prices["DATE"] = pd.to_datetime(
        prices["DATE"]
    )

    prices = prices.sort_values(
        "DATE"
    )

    cot = (
        cot_indexed
        .reset_index()
        .rename(columns={"index": "DATE"})
    )

    cot["DATE"] = pd.to_datetime(
        cot["DATE"]
    )

    merged = pd.merge_asof(

        prices.sort_values("DATE"),

        cot[
            ["DATE", "cot_index"]
        ].sort_values("DATE"),

        on="DATE",

        direction="backward"

    )

    fig, ax1 = plt.subplots(
        figsize=(15, 7)
    )

    ax2 = ax1.twinx()

    ax1.plot(

        merged["DATE"],

        merged[price_col],

        color="black",

        linewidth=2,

        label=price_col

    )

    ax2.fill_between(

        merged["DATE"],

        merged["cot_index"],

        color="steelblue",

        alpha=0.30

    )

    bullish = (
        merged["cot_index"] >
        EXTREME_HIGH
    )

    bearish = (
        merged["cot_index"] <
        EXTREME_LOW
    )

    for i in range(len(merged)):

        if bullish.iloc[i]:

            ax1.axvspan(

                merged.iloc[i]["DATE"],

                merged.iloc[i]["DATE"],

                color="red",

                alpha=0.25

            )

        if bearish.iloc[i]:

            ax1.axvspan(

                merged.iloc[i]["DATE"],

                merged.iloc[i]["DATE"],

                color="green",

                alpha=0.25

            )

    ax1.set_ylabel(price_col)

    ax2.set_ylabel("COT Index")

    plt.title(
        "S&P 500 vs. Speculator Positioning Extremes (COT Index)"
    )

    plt.tight_layout()

    plt.savefig(

        OUTPUTS_DIR /
        "cot_extremes.png",

        dpi=300,

        bbox_inches="tight"

    )

    return fig
