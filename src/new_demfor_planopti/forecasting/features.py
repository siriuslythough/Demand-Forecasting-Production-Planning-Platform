from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

# Demand lags.
#
# 1, 2, 4 weeks:
#     recent demand behaviour
#
# 8, 13 weeks:
#     medium-term demand state
#
# 26 weeks:
#     half-year structure
#
# 52 weeks:
#     same period approximately one year ago

LAGS = [
    1,
    2,
    3,
    4,
    8,
    13,
    26,
    52,
]


# Rolling windows.
#
# All rolling features will be shifted by one period BEFORE
# calculating the statistic to prevent target leakage.

ROLLING_WINDOWS = [
    4,
    8,
    13,
    26,
    52,
]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

# One global LightGBM model is trained across all SKUs.
#
# These identifiers allow the pooled model to distinguish:
#     SKU-specific behaviour
#     category behaviour
#     department behaviour

CATEGORICAL_FEATURES = [
    "sku_id",
    "cat_id",
    "dept_id",
]


# ============================================================
# CALENDAR / KNOWN-FUTURE FEATURES
# ============================================================

# These fields are available independently of future realized demand.
#
# Important:
# We intentionally exclude realized future sell_price for now because
# that value may not be known at forecast time.

CALENDAR_FEATURES = [
    "year",
    "quarter",
    "month",
    "week_of_year",

    "snap_days",
    "event_days",

    "cultural_event_days",
    "national_event_days",
    "religious_event_days",
    "sporting_event_days",

    # Cyclical annual encoding
    "week_sin",
    "week_cos",

    # Global chronological trend
    "time_idx",
]

# ============================================================
# LEVEL / TREND FEATURES
# ============================================================
#
# These features explicitly describe whether recent demand is
# running above or below its longer-run level.
#
# This is motivated by the observed late-2015 regime change,
# where the 13-week portfolio mean fell materially below the
# 52-week mean.

TREND_FEATURES = [
    "level_diff_4_13",
    "level_diff_13_52",
    "level_ratio_4_13",
    "level_ratio_13_52",

    "recent_change_4",
    "recent_change_13",

    "yoy_delta",
]


# ============================================================
# FEATURE-NAME HELPERS
# ============================================================


def lag_feature_names() -> list[str]:
    """
    Return all demand-lag feature names.
    """

    return [
        f"demand_lag_{lag}"
        for lag in LAGS
    ]


def rolling_feature_names() -> list[str]:
    """
    Return all rolling-statistic feature names.
    """

    names: list[str] = []

    for window in ROLLING_WINDOWS:

        names.extend(
            [
                f"demand_roll_mean_{window}",
                f"demand_roll_std_{window}",
                f"demand_roll_min_{window}",
                f"demand_roll_max_{window}",
                f"demand_zero_rate_{window}",
            ]
        )

    return names


def model_feature_columns() -> list[str]:
    """
    Full feature set consumed by the global LightGBM model.
    """

    return (
        CATEGORICAL_FEATURES
        + CALENDAR_FEATURES
        + lag_feature_names()
        + rolling_feature_names()
        + TREND_FEATURES
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================


def add_forecasting_features(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add leakage-safe forecasting features to a weekly SKU panel.

    Parameters
    ----------
    panel:
        Weekly balanced panel containing at least:

            week_start
            sku_id
            cat_id
            dept_id
            demand

            year
            quarter
            month
            week_of_year

            snap_days
            event_days

            cultural_event_days
            national_event_days
            religious_event_days
            sporting_event_days

    Returns
    -------
    pd.DataFrame

        Original panel plus:

            time_idx
            week_sin
            week_cos

            demand_lag_*

            demand_roll_mean_*
            demand_roll_std_*
            demand_roll_min_*
            demand_roll_max_*
            demand_zero_rate_*

    Leakage rule
    ------------
    Every demand-derived predictor for week t must use demand from
    week t-1 or earlier only.
    """

    df = panel.copy()

    # --------------------------------------------------------
    # Validate required input columns
    # --------------------------------------------------------

    required_columns = {
        "week_start",
        "sku_id",
        "cat_id",
        "dept_id",
        "demand",

        "year",
        "quarter",
        "month",
        "week_of_year",

        "snap_days",
        "event_days",

        "cultural_event_days",
        "national_event_days",
        "religious_event_days",
        "sporting_event_days",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Forecast feature input is missing columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Basic preparation
    # --------------------------------------------------------

    df["week_start"] = pd.to_datetime(
        df["week_start"]
    )

    df = (
        df
        .sort_values(
            [
                "sku_id",
                "week_start",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Global chronological index
    # --------------------------------------------------------
    #
    # Example:
    #
    # 2011-01-29 -> 0
    # 2011-02-05 -> 1
    # ...
    #
    # Same index is shared across every SKU.

    unique_weeks = (
        df["week_start"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    week_lookup = {
        week: idx
        for idx, week in enumerate(
            unique_weeks
        )
    }

    df["time_idx"] = (
        df["week_start"]
        .map(week_lookup)
        .astype(int)
    )

    # --------------------------------------------------------
    # Cyclical annual seasonality
    # --------------------------------------------------------
    #
    # Raw week number treats:
    #
    # week 52 and week 1
    #
    # as numerically far apart even though they are adjacent.
    #
    # Sine/cosine encoding puts them next to each other on a
    # circular annual representation.

    df["week_sin"] = np.sin(
        2.0
        * np.pi
        * df["week_of_year"].astype(float)
        / 52.0
    )

    df["week_cos"] = np.cos(
        2.0
        * np.pi
        * df["week_of_year"].astype(float)
        / 52.0
    )

    # --------------------------------------------------------
    # Demand lags
    # --------------------------------------------------------

    for lag in LAGS:

        df[
            f"demand_lag_{lag}"
        ] = (
            df
            .groupby(
                "sku_id",
                observed=True,
            )["demand"]
            .shift(lag)
        )

    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------
    #
    # THIS SHIFT(1) IS CRITICAL.
    #
    # Wrong:
    #
    #     s.rolling(13).mean()
    #
    # That includes demand at the row currently being predicted.
    #
    # Correct:
    #
    #     s.shift(1).rolling(13).mean()
    #
    # Therefore all rolling features use historical demand only.

    for window in ROLLING_WINDOWS:

        df[
            f"demand_roll_mean_{window}"
        ] = (
            df
            .groupby(
                "sku_id",
                observed=True,
            )["demand"]
            .transform(
                lambda s:
                    s
                    .shift(1)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .mean()
            )
        )

        df[
            f"demand_roll_std_{window}"
        ] = (
            df
            .groupby(
                "sku_id",
                observed=True,
            )["demand"]
            .transform(
                lambda s:
                    s
                    .shift(1)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .std()
            )
        )

        df[
            f"demand_roll_min_{window}"
        ] = (
            df
            .groupby(
                "sku_id",
                observed=True,
            )["demand"]
            .transform(
                lambda s:
                    s
                    .shift(1)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .min()
            )
        )

        df[
            f"demand_roll_max_{window}"
        ] = (
            df
            .groupby(
                "sku_id",
                observed=True,
            )["demand"]
            .transform(
                lambda s:
                    s
                    .shift(1)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .max()
            )
        )

        df[
            f"demand_zero_rate_{window}"
        ] = (
            df
            .groupby(
                "sku_id",
                observed=True,
            )["demand"]
            .transform(
                lambda s:
                    s
                    .shift(1)
                    .eq(0)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .mean()
            )
        )

    # ========================================================
    # LEVEL / TREND FEATURES
    # ========================================================
    #
    # These features summarize how the recent demand regime
    # differs from longer-run demand.
    #
    # They use only previously shifted demand features, so no
    # current-week target information enters the predictors.


    # --------------------------------------------------------
    # Difference between short- and medium-run levels
    # --------------------------------------------------------

    df["level_diff_4_13"] = (
        df["demand_roll_mean_4"]
        - df["demand_roll_mean_13"]
    )


    # --------------------------------------------------------
    # Difference between recent and annual demand level
    # --------------------------------------------------------

    df["level_diff_13_52"] = (
        df["demand_roll_mean_13"]
        - df["demand_roll_mean_52"]
    )


    # --------------------------------------------------------
    # Relative short-run level
    #
    # > 1 : short-run level above medium-run level
    # < 1 : short-run level below medium-run level
    # --------------------------------------------------------

    df["level_ratio_4_13"] = (
        df["demand_roll_mean_4"]
        / (
            df["demand_roll_mean_13"]
            + 1e-6
        )
    )


    # --------------------------------------------------------
    # Recent vs annual demand regime
    #
    # This is the SKU-level analogue of the portfolio diagnostic
    # we calculated in the notebook.
    # --------------------------------------------------------

    df["level_ratio_13_52"] = (
        df["demand_roll_mean_13"]
        / (
            df["demand_roll_mean_52"]
            + 1e-6
        )
    )


    # --------------------------------------------------------
    # Recent movement
    # --------------------------------------------------------

    df["recent_change_4"] = (
        df["demand_lag_1"]
        - df["demand_lag_4"]
    )


    df["recent_change_13"] = (
        df["demand_lag_1"]
        - df["demand_lag_13"]
    )


    # --------------------------------------------------------
    # Year-over-year level change
    #
    # Compare the most recently observed demand with demand
    # approximately one year earlier.
    # --------------------------------------------------------

    df["yoy_delta"] = (
        df["demand_lag_1"]
        - df["demand_lag_52"]
    )


    return df


# ============================================================
# SUPERVISED TRAINING FRAME
# ============================================================


def make_training_frame(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Turn historical weekly time series into a supervised ML table.

    Rows without enough historical context are discarded.

    Because the longest lag is 52 weeks, approximately the first
    year of each SKU is unavailable for LightGBM training.
    """

    featured = add_forecasting_features(
        history
    )

    required_features = (
        model_feature_columns()
    )

    required_columns = (
        ["demand"]
        + required_features
    )

    training = (
        featured
        .dropna(
            subset=required_columns
        )
        .reset_index(drop=True)
    )

    if training.empty:
        raise ValueError(
            "No LightGBM training rows remain after feature "
            "engineering. At least 52 weeks of historical "
            "demand are required."
        )

    if training[
        required_features
    ].isna().any().any():
        raise ValueError(
            "Missing values remain in LightGBM features."
        )

    return training