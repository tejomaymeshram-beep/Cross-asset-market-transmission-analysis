# Cross-Asset Macro Shock Transmission 

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Description

This framework reconstructs and quantifies how macroeconomic shocks propagate across asset classes (equities, Treasuries, commodities, FX, volatility) using Vector Autoregression, Granger causality, Impulse Response Functions, and regime-dependent correlation analysis.

Multi-asset trading desks need to understand informational precedence — which markets react first, which lag, and how transmission changes in high-volatility regimes. This framework answers those questions using only free public data.

---

## Data Sources

| Source | Data | Frequency | Coverage | Cost |
|---------|------|-----------|----------|------|
| FRED API | Treasury yields, SOFR, Fed Funds, CPI, Unemployment | Daily / Monthly | 1954-present | Free |
| CBOE | VIX, VIX3M, VIX6M | Daily | 1990/2008-present | Free |
| Yahoo Finance | SPY, TLT, GLD, USO, DXY | Daily | 1993-present | Free |
| CFTC COT | Futures positioning | Weekly | 1986-present | Free |
| Federal Reserve | FOMC meeting dates | Event | 2018-2026 | Free |

---

## Architecture

```text
data_ingestion.py
        │
        ▼
preprocessing.py
        │
        ▼
[var_model.py, granger_network.py, event_study.py, cot_analysis.py]
        │
        ▼
reporting.py
```

---

## Setup Instructions

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Get a free FRED API key:

https://fred.stlouisfed.org/docs/api/api_key.html

4. Run:

```
notebooks/macro_transmission_framework.ipynb
```

using Google Colab or a local Python environment.

5. All generated outputs are automatically saved to:

```
outputs/
```

---

## Key Methodology

- Vector Autoregression (VAR) with AIC lag selection
- Granger causality testing (lag 1, p < 0.05)
- Impulse Response Functions (Cholesky ordering: rates → FX → equities → safe havens)
- Regime-dependent analysis (VIX < 15, 15–25, > 25)
- CFTC COT Index for positioning extremes
- Event studies around FOMC, CPI, NFP

---


## Disclaimer

This is a research and educational framework. It is not investment advice. Past performance does not guarantee future results.

---

## License

MIT License
