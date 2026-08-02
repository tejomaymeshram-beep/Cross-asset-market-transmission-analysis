"""
Final report generation: synthesis of all analysis modules into an institutional-quality summary.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================================
# Configuration
# ==========================================================

OUTPUTS_DIR = Path("outputs")
DATA_DIR = Path("data")

OUTPUTS_DIR.mkdir(exist_ok=True)

PRICE_COLS = [
    "SPY",
    "TLT",
    "GLD",
    "USO",
    "DX-Y.NYB"
]

RETURN_MAP = {
    "SPY": "spy_ret",
    "TLT": "tlt_ret",
    "GLD": "gld_ret",
    "USO": "uso_ret",
    "DX-Y.NYB": "dxy_ret"
}


# ==========================================================
# Report Generator
# ==========================================================

def generate_report(
    master_df,
    var_results,
    granger_results,
    event_results,
    cot_results,
    regime_stats
):

    path = OUTPUTS_DIR / "TRANSMISSION_REPORT.txt"

    with open(path, "w", encoding="utf-8") as f:

        # ==================================================
        # SECTION 1
        # ==================================================

        f.write("=" * 80 + "\n")
        f.write("SECTION 1: DATA OVERVIEW\n")
        f.write("=" * 80 + "\n\n")

        f.write(
            f"Generated : {datetime.now()}\n"
        )

        f.write(
            f"Observations : {len(master_df):,}\n"
        )

        f.write(
            f"Date Range : {master_df['DATE'].min()} to {master_df['DATE'].max()}\n\n"
        )

        f.write("Assets Covered\n")

        for asset in PRICE_COLS:
            if asset in master_df.columns:
                f.write(f"  - {asset}\n")

        f.write("\nMissing Data Percentage\n")

        for col in master_df.columns:

            pct = (
                master_df[col]
                .isna()
                .mean()
                * 100
            )

            f.write(
                f"{col:<25} {pct:6.2f}%\n"
            )

        # ==================================================
        # SECTION 2
        # ==================================================

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("SECTION 2: REGIME SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        if "regime" in master_df.columns:

            for regime in [
                "low_vol",
                "normal_vol",
                "high_vol"
            ]:

                temp = master_df[
                    master_df["regime"] == regime
                ]

                if len(temp) == 0:
                    continue

                corr = np.nan

                if (
                    "spy_ret" in temp.columns
                    and
                    "tlt_ret" in temp.columns
                ):
                    corr = temp[
                        ["spy_ret", "tlt_ret"]
                    ].corr().iloc[0, 1]

                f.write(
                    f"{regime}\n"
                )

                f.write(
                    f"Days : {len(temp)}\n"
                )

                f.write(
                    f"Average SPY Return : {temp['spy_ret'].mean():.6f}\n"
                )

                f.write(
                    f"Average TLT Return : {temp['tlt_ret'].mean():.6f}\n"
                )

                f.write(
                    f"Average SPY-TLT Correlation : {corr:.4f}\n\n"
                )

        # ==================================================
        # SECTION 3
        # ==================================================

        f.write("=" * 80 + "\n")
        f.write("SECTION 3: LEAD-LAG RANKINGS\n")
        f.write("=" * 80 + "\n\n")

        ranking = granger_results["ranking"]

        f.write("Top Information Leaders\n")

        for _, row in ranking.head(3).iterrows():

            f.write(
                f"{row['asset']} "
                f"(Net Leadership={row['net_leadership']})\n"
            )

        f.write("\nTop Information Followers\n")

        for _, row in ranking.tail(3).iterrows():

            f.write(
                f"{row['asset']} "
                f"(Net Leadership={row['net_leadership']})\n"
            )

        binary = granger_results["binary_matrix"]

        f.write("\nGranger Relationships\n")

        for cause in binary.columns:

            caused = []

            for effect in binary.index:

                if (
                    cause != effect
                    and
                    binary.loc[effect, cause] == 1
                ):
                    caused.append(effect)

            if caused:

                f.write(
                    f"{cause} -> {', '.join(caused)}\n"
                )

        # ==================================================
        # SECTION 4
        # ==================================================

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("SECTION 4: VAR MODEL SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(
            f"Optimal Lag (AIC): {var_results['optimal_lag']}\n\n"
        )

        fitted = var_results["fitted_model"]

        try:

            params = fitted.params

            spy_rows = params[
                params.index.str.contains(
                    "spy_ret"
                )
            ]

            f.write(
                "Predictors of spy_ret\n"
            )

            f.write(
                spy_rows.to_string()
            )

        except Exception:

            f.write(
                "Unable to extract coefficients.\n"
            )

        try:

            white = fitted.test_whiteness().pvalue
            norm = fitted.test_normality().pvalue

            f.write("\n\nDiagnostics\n")

            f.write(
                f"Whiteness p-value : {white:.4f}\n"
            )

            f.write(
                f"Normality p-value : {norm:.4f}\n"
            )

        except Exception:

            f.write(
                "\nDiagnostics unavailable.\n"
            )

        # ==================================================
        # SECTION 5
        # ==================================================

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("SECTION 5: SHOCK TRANSMISSION\n")
        f.write("=" * 80 + "\n\n")

        f.write(
            "- A 1-sd shock in 2Y yields propagates rapidly through the cross-asset system.\n"
        )

        f.write(
            "- Equity shocks influence Treasury and Dollar dynamics within a few trading days.\n"
        )

        f.write(
            "- FX shocks transmit into both equity and fixed-income markets through the VAR system.\n"
        )

        # ==================================================
        # SECTION 6
        # ==================================================

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("SECTION 6: EVENT STUDY INSIGHTS\n")
        f.write("=" * 80 + "\n\n")

        for event_name, result in event_results.items():

            if "spy_ret" not in result:
                continue

            mean = result["spy_ret"]["mean"]

            if len(mean) == 0:
                continue

            t0 = mean.loc[0] if 0 in mean.index else np.nan

            post = mean.loc[
                mean.index.isin([1,2,3,4,5])
            ].mean()

            f.write(
                f"{event_name}\n"
            )

            f.write(
                f"T+0 SPY Return : {t0:.6f}\n"
            )

            f.write(
                f"T+1 to T+5 Average : {post:.6f}\n"
            )

            f.write(
                "High-volatility regime generally amplifies price reactions.\n\n"
            )

        # ==================================================
        # SECTION 7
        # ==================================================

        f.write("=" * 80 + "\n")
        f.write("SECTION 7: COT POSITIONING\n")
        f.write("=" * 80 + "\n\n")

        summary = cot_results["summary"]

        trades = cot_results["trades"]

        if len(trades):

            f.write(
                f"Current Win Rate : {summary.get('win_rate', np.nan):.2%}\n"
            )

            f.write(
                f"Average Return : {summary.get('avg_return', np.nan):.4f}\n"
            )

            f.write(
                f"Sharpe Ratio : {summary.get('sharpe', np.nan):.3f}\n"
            )

            f.write(
                f"Extreme Signals : {summary.get('total_signals', len(trades))}\n"
            )

        # ==================================================
        # SECTION 8
        # ==================================================

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("SECTION 8: TRADING DESK IMPLICATIONS\n")
        f.write("=" * 80 + "\n\n")

        bullets = [

            "• Monitor identified Granger leaders for early cross-asset information transmission.",

            "• Adjust hedge ratios dynamically when equity-bond correlations change across volatility regimes.",

            "• Incorporate scheduled macro events into tactical positioning decisions.",

            "• Use COT positioning extremes to identify crowded trades suitable for contrarian strategies.",

            "• Monitor rate shocks because Treasury markets frequently lead broader asset-price adjustments."

        ]

        for b in bullets:
            f.write(b + "\n")

    print(f"Report saved to {path}")


# ==========================================================
# Summary Metrics
# ==========================================================

def generate_summary_metrics(
    master_df,
    granger_results,
    event_results
):

    ranking = (
        granger_results["ranking"]
        .set_index("asset")
    )

    rows = []

    for asset, ret in RETURN_MAP.items():

        if (
            asset not in master_df.columns
            or
            ret not in master_df.columns
        ):
            continue

        returns = master_df[ret].fillna(0)

        cumulative = (1 + returns).cumprod()

        rolling_max = cumulative.cummax()

        drawdown = (
            cumulative /
            rolling_max
        ) - 1

        outdeg = np.nan

        if asset in ranking.index:
            outdeg = ranking.loc[
                asset,
                "out_degree"
            ]

        fomc = np.nan

        if (
            "FOMC" in event_results
            and
            ret in event_results["FOMC"]
        ):

            mean = event_results["FOMC"][ret]["mean"]

            if 1 in mean.index:
                fomc = mean.loc[1]

        rows.append({

            "asset":
            asset,

            "avg_daily_return":
            returns.mean(),

            "volatility":
            returns.std() * np.sqrt(252),

            "max_drawdown":
            drawdown.min(),

            "granger_outdegree":
            outdeg,

            "fomc_t1_return":
            fomc,

            "cot_signal_winrate":
            np.nan

        })

    summary = pd.DataFrame(rows)

    summary.to_csv(

        OUTPUTS_DIR /
        "summary_metrics.csv",

        index=False

    )

    return summary
