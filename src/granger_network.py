"""
Granger causality testing and network visualization for cross-asset lead-lag relationships.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests

# ==========================================================
# Configuration
# ==========================================================

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

DEFAULT_VARS = [
    "spy_ret",
    "tlt_ret",
    "gld_ret",
    "dgs2_change",
    "dgs10_change",
    "dxy_ret"
]

VAR_LABELS = {
    "spy_ret": "S&P 500",
    "tlt_ret": "20Y Treasury",
    "gld_ret": "Gold",
    "dgs2_change": "2Y Yield",
    "dgs10_change": "10Y Yield",
    "dxy_ret": "Dollar Index",
}

MAX_LAG = 5
SIGNIFICANCE_LEVEL = 0.05


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
# Granger Matrix
# ==========================================================

def build_granger_matrix(
    df,
    variables=None,
    maxlag=MAX_LAG
):

    variables = variables or DEFAULT_VARS

    _validate_columns(df, variables)

    data = df[variables].dropna()

    pvalue_matrix = pd.DataFrame(
        np.nan,
        index=variables,
        columns=variables
    )

    binary_matrix = pd.DataFrame(
        0,
        index=variables,
        columns=variables,
        dtype=int
    )

    for cause in variables:

        for effect in variables:

            if cause == effect:
                continue

            try:

                pair = data[
                    [effect, cause]
                ].values

                result = grangercausalitytests(
                    pair,
                    maxlag=maxlag,
                    verbose=False
                )

                p_value = result[1][0]["ssr_ftest"][1]

                pvalue_matrix.loc[
                    effect,
                    cause
                ] = p_value

                binary_matrix.loc[
                    effect,
                    cause
                ] = int(
                    p_value < SIGNIFICANCE_LEVEL
                )

            except Exception:

                pvalue_matrix.loc[
                    effect,
                    cause
                ] = np.nan

                binary_matrix.loc[
                    effect,
                    cause
                ] = 0

    np.fill_diagonal(
        pvalue_matrix.values,
        np.nan
    )

    np.fill_diagonal(
        binary_matrix.values,
        0
    )

    pvalue_matrix.to_csv(
        OUTPUTS_DIR /
        "granger_pvalue_matrix.csv"
    )

    binary_matrix.to_csv(
        OUTPUTS_DIR /
        "granger_binary_matrix.csv"
    )

    return (
        pvalue_matrix,
        binary_matrix
    )


# ==========================================================
# Heatmaps
# ==========================================================

def plot_granger_heatmaps(
    pvalue_matrix,
    binary_matrix,
    figsize=(16, 6)
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize
    )

    label_names = [
        VAR_LABELS.get(
            c,
            c
        )
        for c in pvalue_matrix.columns
    ]

    sns.heatmap(

        pvalue_matrix,

        annot=True,

        fmt=".3f",

        cmap="RdYlGn_r",

        vmin=0,

        vmax=0.1,

        center=0.05,

        xticklabels=label_names,

        yticklabels=label_names,

        ax=axes[0]

    )

    axes[0].set_title(
        "Granger Causality p-values (lag=1)"
    )

    axes[0].set_xlabel(
        "Cause"
    )

    axes[0].set_ylabel(
        "Effect"
    )

    sns.heatmap(

        binary_matrix.astype(int),

        annot=True,

        fmt="d",

        cmap="binary",

        xticklabels=label_names,

        yticklabels=label_names,

        ax=axes[1],

        cbar=False

    )

    axes[1].set_title(
        "Significant Lead-Lag Relationships (p < 0.05)"
    )

    axes[1].set_xlabel(
        "Cause"
    )

    axes[1].set_ylabel(
        "Effect"
    )

    plt.tight_layout()

    plt.savefig(

        OUTPUTS_DIR /
        "granger_heatmaps.png",

        dpi=300,

        bbox_inches="tight"

    )

    return fig


# ==========================================================
# Network
# ==========================================================

def plot_granger_network(
    binary_matrix,
    figsize=(12, 10)
):

    G = nx.DiGraph()

    for label in VAR_LABELS.values():

        G.add_node(label)

    for effect in binary_matrix.index:

        for cause in binary_matrix.columns:

            if binary_matrix.loc[
                effect,
                cause
            ] == 1:

                G.add_edge(

                    VAR_LABELS[cause],

                    VAR_LABELS[effect]

                )

    pos = nx.spring_layout(
        G,
        seed=42
    )

    colors = {

        "S&P 500": "steelblue",

        "20Y Treasury": "mediumpurple",

        "Gold": "gold",

        "2Y Yield": "firebrick",

        "10Y Yield": "firebrick",

        "Dollar Index": "darkgreen"

    }

    node_colors = []

    node_sizes = []

    for node in G.nodes():

        node_colors.append(
            colors[node]
        )

        node_sizes.append(
            1000 +
            200 * G.out_degree(node)
        )

    fig = plt.figure(
        figsize=figsize
    )

    nx.draw_networkx_nodes(

        G,

        pos,

        node_size=node_sizes,

        node_color=node_colors,

        edgecolors="black",

        linewidths=1.2

    )

    nx.draw_networkx_labels(

        G,

        pos,

        font_size=10,

        font_weight="bold"

    )

    nx.draw_networkx_edges(

        G,

        pos,

        arrows=True,

        arrowstyle="-|>",

        arrowsize=22,

        width=2,

        connectionstyle="arc3,rad=0.08"

    )

    plt.title(
        "Cross-Asset Information Flow Network"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(

        OUTPUTS_DIR /
        "granger_network.png",

        dpi=300,

        bbox_inches="tight"

    )

    return fig


# ==========================================================
# Ranking
# ==========================================================

def rank_lead_lag(
    pvalue_matrix,
    binary_matrix
):

    assets = list(
        binary_matrix.columns
    )

    rows = []

    for asset in assets:

        out_degree = int(
            binary_matrix[
                asset
            ].sum()
        )

        in_degree = int(
            binary_matrix.loc[
                asset
            ].sum()
        )

        rows.append({

            "asset":
            VAR_LABELS.get(
                asset,
                asset
            ),

            "out_degree":
            out_degree,

            "in_degree":
            in_degree,

            "net_leadership":
            out_degree - in_degree

        })

    ranking = (
        pd.DataFrame(rows)
        .sort_values(
            "net_leadership",
            ascending=False
        )
        .reset_index(drop=True)
    )

    ranking.to_csv(

        OUTPUTS_DIR /
        "lead_lag_ranking.csv",

        index=False

    )

    return ranking
  
