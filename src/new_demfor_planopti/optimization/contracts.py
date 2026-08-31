from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================
# INPUT CONTRACTS
# ============================================================


FORECAST_COLUMNS = [
    "week_start",
    "sku_id",
    "forecast_demand",
    "forecast_horizon",
]


SKU_PARAMETER_COLUMNS = [
    "sku_id",
    "initial_inventory",
    "initial_backlog",
    "unit_production_cost",
    "unit_holding_cost",
    "unit_shortage_cost",
    "processing_time",
    "max_production_per_week",
]


CAPACITY_COLUMNS = [
    "week_start",
    "production_capacity",
]


# ============================================================
# PLANNING CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class PlanningConfig:
    """
    Global configuration for the deterministic production-planning
    optimization model.

    The first production-planning formulation is intentionally kept
    linear so that:

        - optimization is fast,
        - solutions are interpretable,
        - infeasibility is easier to diagnose,
        - the model can later be extended into MILP formulations.

    Backlog is allowed in the first formulation and penalized through
    unit_shortage_cost.
    """

    planning_horizon: int = 13

    demand_tolerance: float = 1e-9

    solver_method: str = "highs"


# ============================================================
# BASIC HELPERS
# ============================================================


def _require_columns(
    df: pd.DataFrame,
    required: list[str],
    frame_name: str,
) -> None:
    """
    Raise a clear error if a planning input is missing required
    columns.
    """

    missing = (
        set(required)
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"{frame_name} is missing required "
            f"columns: {sorted(missing)}"
        )


def _require_no_missing(
    df: pd.DataFrame,
    columns: list[str],
    frame_name: str,
) -> None:
    """
    Validate that critical optimization fields contain no missing
    values.
    """

    missing_counts = (
        df[columns]
        .isna()
        .sum()
    )

    missing_counts = (
        missing_counts[
            missing_counts > 0
        ]
    )

    if not missing_counts.empty:

        raise ValueError(
            f"{frame_name} contains missing "
            f"values:\n{missing_counts}"
        )


def _require_nonnegative(
    df: pd.DataFrame,
    columns: list[str],
    frame_name: str,
) -> None:
    """
    Optimization quantities and costs represented by these columns
    must not be negative.
    """

    for column in columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if values.isna().any():

            raise ValueError(
                f"{frame_name}.{column} "
                "must be numeric."
            )

        if (
            values < 0
        ).any():

            raise ValueError(
                f"{frame_name}.{column} "
                "must be nonnegative."
            )


def _require_finite(
    df: pd.DataFrame,
    columns: list[str],
    frame_name: str,
) -> None:
    """
    Prevent NaN/inf values from reaching the optimization solver.
    """

    for column in columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if not np.isfinite(
            values
        ).all():

            raise ValueError(
                f"{frame_name}.{column} "
                "must contain finite numeric values."
            )


# ============================================================
# FORECAST CONTRACT
# ============================================================


def validate_forecast_frame(
    forecast: pd.DataFrame,
    config: PlanningConfig,
) -> pd.DataFrame:
    """
    Validate and normalize the demand forecast consumed by the
    production-planning model.

    Expected granularity:

        one row per SKU per planning week
    """

    _require_columns(
        forecast,
        FORECAST_COLUMNS,
        "forecast",
    )

    df = forecast[
        FORECAST_COLUMNS
    ].copy()

    df[
        "week_start"
    ] = pd.to_datetime(
        df[
            "week_start"
        ],
        errors="raise",
    )

    df[
        "sku_id"
    ] = (
        df[
            "sku_id"
        ]
        .astype(str)
    )

    df[
        "forecast_demand"
    ] = pd.to_numeric(
        df[
            "forecast_demand"
        ],
        errors="raise",
    )

    df[
        "forecast_horizon"
    ] = pd.to_numeric(
        df[
            "forecast_horizon"
        ],
        errors="raise",
    ).astype(int)

    _require_no_missing(
        df,
        FORECAST_COLUMNS,
        "forecast",
    )

    _require_finite(
        df,
        [
            "forecast_demand",
            "forecast_horizon",
        ],
        "forecast",
    )

    if (
        df[
            "forecast_demand"
        ]
        <
        -config.demand_tolerance
    ).any():

        raise ValueError(
            "Forecast demand must be "
            "nonnegative."
        )

    # Remove tiny floating-point negatives.
    df[
        "forecast_demand"
    ] = (
        df[
            "forecast_demand"
        ]
        .clip(
            lower=0.0
        )
    )

    duplicated = (
        df.duplicated(
            subset=[
                "week_start",
                "sku_id",
            ],
            keep=False,
        )
    )

    if duplicated.any():

        duplicate_rows = (
            df.loc[
                duplicated,
                [
                    "week_start",
                    "sku_id",
                ],
            ]
            .sort_values(
                [
                    "week_start",
                    "sku_id",
                ]
            )
        )

        raise ValueError(
            "Forecast must contain exactly "
            "one row per "
            "(week_start, sku_id).\n"
            f"Duplicates:\n"
            f"{duplicate_rows.head(20)}"
        )

    horizons = sorted(
        df[
            "forecast_horizon"
        ]
        .unique()
        .tolist()
    )

    expected_horizons = list(
        range(
            1,
            config.planning_horizon + 1,
        )
    )

    if (
        horizons
        != expected_horizons
    ):

        raise ValueError(
            "Forecast horizons must be "
            f"{expected_horizons}, "
            f"found {horizons}."
        )

    # Every SKU must have every planning horizon.
    horizon_counts = (
        df.groupby(
            "sku_id",
            observed=True,
        )[
            "forecast_horizon"
        ]
        .nunique()
    )

    bad_skus = (
        horizon_counts[
            horizon_counts
            != config.planning_horizon
        ]
    )

    if not bad_skus.empty:

        raise ValueError(
            "Every SKU must contain all "
            "planning horizons.\n"
            f"{bad_skus}"
        )

    return (
        df
        .sort_values(
            [
                "forecast_horizon",
                "sku_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# SKU PARAMETER CONTRACT
# ============================================================


def validate_sku_parameters(
    sku_parameters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate SKU-level production-planning parameters.

    Parameters
    ----------
    initial_inventory:
        Inventory available before horizon 1.

    initial_backlog:
        Outstanding demand before horizon 1.

    unit_production_cost:
        Cost of producing one unit.

    unit_holding_cost:
        Cost of carrying one unit of ending inventory for one week.

    unit_shortage_cost:
        Cost of carrying one unit of backlog for one week.

    processing_time:
        Capacity consumed by one produced unit.

    max_production_per_week:
        SKU-specific weekly production ceiling.
    """

    _require_columns(
        sku_parameters,
        SKU_PARAMETER_COLUMNS,
        "sku_parameters",
    )

    df = sku_parameters[
        SKU_PARAMETER_COLUMNS
    ].copy()

    df[
        "sku_id"
    ] = (
        df[
            "sku_id"
        ]
        .astype(str)
    )

    numeric_columns = [
        column
        for column
        in SKU_PARAMETER_COLUMNS
        if column != "sku_id"
    ]

    for column in numeric_columns:

        df[
            column
        ] = pd.to_numeric(
            df[
                column
            ],
            errors="raise",
        )

    _require_no_missing(
        df,
        SKU_PARAMETER_COLUMNS,
        "sku_parameters",
    )

    _require_finite(
        df,
        numeric_columns,
        "sku_parameters",
    )

    _require_nonnegative(
        df,
        numeric_columns,
        "sku_parameters",
    )

    if (
        df[
            "processing_time"
        ]
        <= 0
    ).any():

        raise ValueError(
            "processing_time must be > 0 "
            "for every SKU."
        )

    if (
        df[
            "unit_shortage_cost"
        ]
        <= 0
    ).any():

        raise ValueError(
            "unit_shortage_cost must be > 0 "
            "for every SKU."
        )

    if (
        df[
            "max_production_per_week"
        ]
        <= 0
    ).any():

        raise ValueError(
            "max_production_per_week must "
            "be > 0 for every SKU."
        )

    if (
        df[
            "sku_id"
        ].duplicated()
    ).any():

        duplicates = (
            df.loc[
                df[
                    "sku_id"
                ].duplicated(
                    keep=False
                ),
                "sku_id",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "SKU parameters must contain "
            "one row per SKU. "
            f"Duplicates: {duplicates}"
        )

    return (
        df
        .sort_values(
            "sku_id"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# CAPACITY CONTRACT
# ============================================================


def validate_capacity_frame(
    capacity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate weekly aggregate production capacity.

    production_capacity is expressed in capacity units.

    SKU production consumes:

        processing_time * production_quantity

    capacity units.
    """

    _require_columns(
        capacity,
        CAPACITY_COLUMNS,
        "capacity",
    )

    df = capacity[
        CAPACITY_COLUMNS
    ].copy()

    df[
        "week_start"
    ] = pd.to_datetime(
        df[
            "week_start"
        ],
        errors="raise",
    )

    df[
        "production_capacity"
    ] = pd.to_numeric(
        df[
            "production_capacity"
        ],
        errors="raise",
    )

    _require_no_missing(
        df,
        CAPACITY_COLUMNS,
        "capacity",
    )

    _require_finite(
        df,
        [
            "production_capacity",
        ],
        "capacity",
    )

    _require_nonnegative(
        df,
        [
            "production_capacity",
        ],
        "capacity",
    )

    if (
        df[
            "week_start"
        ].duplicated()
    ).any():

        raise ValueError(
            "Capacity must contain exactly "
            "one row per planning week."
        )

    return (
        df
        .sort_values(
            "week_start"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# CROSS-FRAME CONTRACT
# ============================================================


def validate_planning_inputs(
    forecast: pd.DataFrame,
    sku_parameters: pd.DataFrame,
    capacity: pd.DataFrame,
    config: PlanningConfig | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Validate the complete set of optimization inputs.

    This function should be called before constructing the LP.

    Returns normalized copies of:

        forecast
        sku_parameters
        capacity
    """

    if config is None:

        config = PlanningConfig()

    forecast_clean = (
        validate_forecast_frame(
            forecast=forecast,
            config=config,
        )
    )

    sku_clean = (
        validate_sku_parameters(
            sku_parameters
        )
    )

    capacity_clean = (
        validate_capacity_frame(
            capacity
        )
    )

    # --------------------------------------------------------
    # SKU COVERAGE
    # --------------------------------------------------------

    forecast_skus = set(
        forecast_clean[
            "sku_id"
        ].unique()
    )

    parameter_skus = set(
        sku_clean[
            "sku_id"
        ].unique()
    )

    missing_sku_parameters = (
        forecast_skus
        - parameter_skus
    )

    if missing_sku_parameters:

        raise ValueError(
            "Missing production parameters "
            "for forecast SKUs: "
            f"{sorted(missing_sku_parameters)}"
        )

    # --------------------------------------------------------
    # WEEKLY CAPACITY COVERAGE
    # --------------------------------------------------------

    forecast_weeks = set(
        forecast_clean[
            "week_start"
        ].unique()
    )

    capacity_weeks = set(
        capacity_clean[
            "week_start"
        ].unique()
    )

    missing_capacity_weeks = (
        forecast_weeks
        - capacity_weeks
    )

    if missing_capacity_weeks:

        formatted = sorted(
            pd.Timestamp(week).strftime(
                "%Y-%m-%d"
            )
            for week
            in missing_capacity_weeks
        )

        raise ValueError(
            "Missing production-capacity "
            "rows for planning weeks: "
            f"{formatted}"
        )

    # Keep only rows relevant to this planning run.
    sku_clean = (
        sku_clean.loc[
            sku_clean[
                "sku_id"
            ].isin(
                forecast_skus
            )
        ]
        .copy()
        .sort_values(
            "sku_id"
        )
        .reset_index(
            drop=True
        )
    )

    capacity_clean = (
        capacity_clean.loc[
            capacity_clean[
                "week_start"
            ].isin(
                forecast_weeks
            )
        ]
        .copy()
        .sort_values(
            "week_start"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # BALANCED HORIZON CONTRACT
    # --------------------------------------------------------

    expected_rows = (
        len(
            forecast_skus
        )
        *
        config.planning_horizon
    )

    if (
        len(
            forecast_clean
        )
        != expected_rows
    ):

        raise ValueError(
            "Forecast planning panel is not "
            "balanced. "
            f"Expected {expected_rows} rows, "
            f"found {len(forecast_clean)}."
        )

    return (
        forecast_clean,
        sku_clean,
        capacity_clean,
    )