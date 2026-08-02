"""
Raw data ingestion for Cross-Asset Market Shock Transmission .
All functions return pandas DataFrames and save raw data to disk.
"""

import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from fredapi import Fred
from cot_reports import cot_all

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================================
# Configuration
# ==========================================================

START_DATE = "2018-04-03"
END_DATE = "2026-07-31"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

FRED_SERIES = {
    "DGS2": "2Y Treasury",
    "DGS10": "10Y Treasury",
    "DFF": "Fed Funds Effective",
    "SOFR": "SOFR",
    "T10Y2Y": "Yield Curve 10Y-2Y",
    "T10Y3M": "Yield Curve 10Y-3M",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "CPI All Urban",
    "DTWEXBGS": "Trade Weighted Dollar Index",
}

YF_TICKERS = {
    "SPY": "S&P 500",
    "TLT": "20Y Treasury ETF",
    "GLD": "Gold ETF",
    "USO": "WTI Oil ETF",
    "DX-Y.NYB": "US Dollar Index",
}

CBOE_URLS = {
    "VIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "VIX3M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
    "VIX6M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX6M_History.csv",
}

# ==========================================================
# CBOE
# ==========================================================

def fetch_cboe_vix(start_date=None, end_date=None):

    start_date = start_date or START_DATE
    end_date = end_date or END_DATE

    try:

        dfs = []

        for name, url in CBOE_URLS.items():

            df = pd.read_csv(
                url,
                parse_dates=["DATE"],
                date_format="%m/%d/%Y"
            )

            df = df[["DATE", "CLOSE"]].rename(
                columns={"CLOSE": name}
            )

            df = df[
                (df["DATE"] >= pd.to_datetime(start_date))
                &
                (df["DATE"] <= pd.to_datetime(end_date))
            ]

            dfs.append(df)

        out = dfs[0]

        for df in dfs[1:]:
            out = out.merge(df, on="DATE", how="outer")

        out = out.sort_values("DATE")

        out.to_csv(
            DATA_DIR / "vix_raw.csv",
            index=False
        )

        return out

    except Exception as e:

        print(f"[fetch_cboe_vix] {e}")

        return pd.DataFrame()

# ==========================================================
# FRED
# ==========================================================

def fetch_fred_data(api_key, start_date=None, end_date=None):

    start_date = start_date or START_DATE
    end_date = end_date or END_DATE

    try:

        fred = Fred(api_key=api_key)

        frames = []

        for sid, _ in FRED_SERIES.items():

            try:

                s = fred.get_series(
                    sid,
                    observation_start=start_date,
                    observation_end=end_date
                )

                df = s.to_frame(name=sid)

                df.index.name = "DATE"

                frames.append(df)

            except Exception as e:

                print(f"[FRED:{sid}] {e}")

        out = pd.concat(
            frames,
            axis=1
        )

        out = out.reset_index()

        out.to_csv(
            DATA_DIR / "fred_raw.csv",
            index=False
        )

        return out

    except Exception as e:

        print(f"[fetch_fred_data] {e}")

        return pd.DataFrame()

# ==========================================================
# Yahoo Finance
# ==========================================================

def fetch_yf_data(start_date=None, end_date=None):

    start_date = start_date or START_DATE
    end_date = end_date or END_DATE

    try:

        data = yf.download(
            list(YF_TICKERS.keys()),
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )

        if isinstance(data.columns, pd.MultiIndex):

            if "Close" in data.columns.levels[0]:
                data = data["Close"]

            elif "Adj Close" in data.columns.levels[0]:
                data = data["Adj Close"]

            else:
                data = data.xs(
                    data.columns.levels[0][0],
                    axis=1,
                    level=0
                )

        rename = {}

        for t in data.columns:

            rename[t] = YF_TICKERS.get(t, t)

        data = data.rename(columns=rename)

        data = data.reset_index()

        data.to_csv(
            DATA_DIR / "yf_raw.csv",
            index=False
        )

        return data

    except Exception as e:

        print(f"[fetch_yf_data] {e}")

        return pd.DataFrame()

# ==========================================================
# COT
# ==========================================================

def fetch_cot_data():

    try:

        cot = cot_all(
            cot_report_type="traders_in_financial_futures_fut"
        )

        mask = (
            cot["Market_and_Exchange_Names"]
            .str.contains(
                "E-MINI S&P 500|10-YEAR U.S. TREASURY NOTES|2-YEAR U.S. TREASURY NOTES",
                case=False,
                na=False,
                regex=True
            )
        )

        cols = [
            "Report_Date_as_YYYY_MM_DD",
            "Market_and_Exchange_Names",
            "Open_Interest_All",
            "NonComm_Positions_Long_All",
            "NonComm_Positions_Short_All",
            "Comm_Positions_Long_All",
            "Comm_Positions_Short_All"
        ]

        cot = cot.loc[mask, cols].copy()

        cot["Report_Date_as_YYYY_MM_DD"] = pd.to_datetime(
            cot["Report_Date_as_YYYY_MM_DD"]
        )

        cot.to_csv(
            DATA_DIR / "cot_raw.csv",
            index=False
        )

        return cot

    except Exception as e:

        print(f"[fetch_cot_data] {e}")

        return pd.DataFrame()

# ==========================================================
# Events Calendar
# ==========================================================

def build_events_calendar():

    try:

        fomc_dates = [

            "2018-01-31","2018-03-21","2018-05-02","2018-06-13",
            "2018-08-01","2018-09-26","2018-11-08","2018-12-19",

            "2019-01-30","2019-03-20","2019-05-01","2019-06-19",
            "2019-07-31","2019-09-18","2019-10-30","2019-12-11",

            "2020-01-29","2020-03-15","2020-04-29","2020-06-10",
            "2020-07-29","2020-09-16","2020-11-05","2020-12-16",

            "2021-01-27","2021-03-17","2021-04-28","2021-06-16",
            "2021-07-28","2021-09-22","2021-11-03","2021-12-15",

            "2022-01-26","2022-03-16","2022-05-04","2022-06-15",
            "2022-07-27","2022-09-21","2022-11-02","2022-12-14",

            "2023-02-01","2023-03-22","2023-05-03","2023-06-14",
            "2023-07-26","2023-09-20","2023-11-01","2023-12-13",

            "2024-01-31","2024-03-20","2024-05-01","2024-06-12",
            "2024-07-31","2024-09-18","2024-11-07","2024-12-18",

            "2025-01-29","2025-03-19","2025-05-07","2025-06-18",
            "2025-07-30","2025-09-17","2025-11-07","2025-12-17",

            "2026-01-28","2026-03-18","2026-04-29","2026-06-17",
            "2026-07-29","2026-09-16","2026-10-28","2026-12-09"
        ]

        fomc = pd.DataFrame({
            "DATE": pd.to_datetime(fomc_dates),
            "EVENT_TYPE": "FOMC"
        })

        months = pd.date_range(
            "2018-04-01",
            "2026-07-01",
            freq="MS"
        )

        cpi = pd.DataFrame({
            "DATE": months + pd.offsets.Day(14),
            "EVENT_TYPE": "CPI"
        })

        nfp_dates = []

        for m in months:

            first = m

            while first.weekday() != 4:
                first += pd.Timedelta(days=1)

            nfp_dates.append(first)

        nfp = pd.DataFrame({
            "DATE": nfp_dates,
            "EVENT_TYPE": "NFP"
        })

        events = pd.concat(
            [fomc, cpi, nfp],
            ignore_index=True
        )

        events = events.sort_values("DATE")

        events.to_csv(
            DATA_DIR / "events_raw.csv",
            index=False
        )

        return events

    except Exception as e:

        print(f"[build_events_calendar] {e}")

        return pd.DataFrame()
