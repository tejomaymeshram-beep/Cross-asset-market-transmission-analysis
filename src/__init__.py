"""
Cross-Asset Market Shock Transmission 
"""

from .data_ingestion import (
    fetch_cboe_vix,
    fetch_fred_data,
    fetch_yf_data,
    fetch_cot_data,
    build_events_calendar,
)

from .preprocessing import (
    build_master_dataset,
    calculate_returns,
    test_stationarity,
    create_regimes,
)

from .var_model import (
    fit_var_model,
    generate_irf,
)

from .granger_network import (
    build_granger_matrix,
    plot_granger_network,
)

from .event_study import (
    run_event_study,
)

from .cot_analysis import (
    calculate_cot_index,
)

from .reporting import (
    generate_report,
)

__all__ = [
    "fetch_cboe_vix",
    "fetch_fred_data",
    "fetch_yf_data",
    "fetch_cot_data",
    "build_events_calendar",
    "build_master_dataset",
    "calculate_returns",
    "test_stationarity",
    "create_regimes",
    "fit_var_model",
    "generate_irf",
    "build_granger_matrix",
    "plot_granger_network",
    "run_event_study",
    "calculate_cot_index",
    "generate_report",
]
