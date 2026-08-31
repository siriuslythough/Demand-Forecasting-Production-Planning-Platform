from __future__ import annotations

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from new_demfor_planopti.forecasting.features import (
    CALENDAR_FEATURES,
    CATEGORICAL_FEATURES,
    TREND_FEATURES,
    add_forecasting_features,
    lag_feature_names,
    make_training_frame,
    model_feature_columns,
    rolling_feature_names,
)


# ============================================================
# FEATURE CONTRACTS
# ============================================================


HISTORY_FEATURES = (
    lag_feature_names()
    + rolling_feature_names()
    + TREND_FEATURES
)


def direct_model_feature_columns() -> list[str]:
    """
    Feature contract for direct multi-horizon LightGBM.

    Compared with the recursive one-step model, the direct model
    additionally receives forecast_horizon.

    This allows one pooled model to learn different relationships for:

        horizon = 1
        horizon = 2
        ...
        horizon = 13
    """

    return (
        CATEGORICAL_FEATURES
        + CALENDAR_FEATURES
        + HISTORY_FEATURES
        + [
            "forecast_horizon",
        ]
    )


# ============================================================
# COMMON LIGHTGBM CONFIGURATION
# ============================================================


def _build_lightgbm_regressor(
    random_state: int = 42,
) -> LGBMRegressor:
    """
    Shared LightGBM configuration.

    Tweedie is retained because demand is non-negative and
    heteroscedastic/skewed.
    """

    return LGBMRegressor(
        objective="tweedie",

        tweedie_variance_power=1.2,

        n_estimators=700,
        learning_rate=0.03,

        num_leaves=31,
        min_child_samples=20,

        subsample=0.90,
        colsample_bytree=0.90,

        reg_alpha=0.10,
        reg_lambda=0.10,

        random_state=random_state,

        n_jobs=-1,

        verbosity=-1,
    )


def _build_pipeline(
    feature_columns: list[str],
    random_state: int = 42,
) -> Pipeline:
    """
    Build preprocessing + LightGBM pipeline for a supplied feature
    contract.
    """

    categorical = [
        col
        for col in CATEGORICAL_FEATURES
        if col in feature_columns
    ]

    numeric = [
        col
        for col in feature_columns
        if col not in categorical
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical,
            ),
            (
                "numeric",
                "passthrough",
                numeric,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = _build_lightgbm_regressor(
        random_state=random_state
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# RECURSIVE GLOBAL LIGHTGBM
# ============================================================
#
# This is our existing Experiment-1 model.
#
# We KEEP it so that the direct model can be compared against it.
# ============================================================


def build_global_lightgbm(
    random_state: int = 42,
) -> Pipeline:
    """
    Build the original one-step global LightGBM.

    One model is shared across all SKUs.
    """

    return _build_pipeline(
        feature_columns=(
            model_feature_columns()
        ),
        random_state=random_state,
    )


def fit_global_lightgbm(
    history: pd.DataFrame,
    random_state: int = 42,
) -> Pipeline:
    """
    Train the original recursive/global LightGBM.
    """

    training = make_training_frame(
        history
    )

    X = training[
        model_feature_columns()
    ]

    y = training[
        "demand"
    ]

    model = build_global_lightgbm(
        random_state=random_state
    )

    model.fit(
        X,
        y,
    )

    return model


def forecast_global_lightgbm(
    model: Pipeline,
    history: pd.DataFrame,
    future: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Recursive multi-step LightGBM forecasting.

    At each horizon:

        predict t+1
            ↓
        append prediction to history
            ↓
        predict t+2
            ↓
        ...

    Actual held-out demand is never added to recursive history.

    This remains in the project as the Experiment-1 ML benchmark.
    """

    history = history.copy()
    future = future.copy()

    history[
        "week_start"
    ] = pd.to_datetime(
        history[
            "week_start"
        ]
    )

    future[
        "week_start"
    ] = pd.to_datetime(
        future[
            "week_start"
        ]
    )

    # --------------------------------------------------------
    # Known-future covariates
    # --------------------------------------------------------

    covariate_cols = [
        "week_start",
        "sku_id",

        "cat_id",
        "dept_id",

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
    ]

    work = history[
        covariate_cols
        + [
            "demand",
        ]
    ].copy()

    predictions = []

    future_weeks = (
        future[
            "week_start"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    for week in future_weeks:

        # ----------------------------------------------------
        # Create rows for target week with demand hidden.
        # ----------------------------------------------------

        current_covariates = (
            future.loc[
                future[
                    "week_start"
                ].eq(
                    week
                ),
                covariate_cols,
            ]
            .copy()
        )

        current_covariates[
            "demand"
        ] = np.nan

        temporary = pd.concat(
            [
                work,
                current_covariates,
            ],
            ignore_index=True,
        )

        featured = (
            add_forecasting_features(
                temporary
            )
        )

        current_features = (
            featured.loc[
                featured[
                    "week_start"
                ].eq(
                    week
                )
            ]
            .sort_values(
                "sku_id"
            )
            .copy()
        )

        X = current_features[
            model_feature_columns()
        ]

        if X.isna().any().any():

            missing = (
                X
                .isna()
                .sum()
            )

            raise ValueError(
                "Missing recursive LightGBM "
                "features at forecast time:\n"
                f"{missing[missing > 0]}"
            )

        prediction = (
            model.predict(
                X
            )
        )

        prediction = np.clip(
            prediction,
            0.0,
            None,
        )

        predicted_rows = (
            current_features[
                [
                    "week_start",
                    "sku_id",
                ]
            ]
            .copy()
        )

        predicted_rows[
            "prediction"
        ] = prediction

        predictions.append(
            predicted_rows
        )

        # ----------------------------------------------------
        # Append predictions — NOT actual future demand.
        # ----------------------------------------------------

        prediction_lookup = (
            predicted_rows
            .set_index(
                "sku_id"
            )[
                "prediction"
            ]
            .to_dict()
        )

        current_covariates[
            "demand"
        ] = (
            current_covariates[
                "sku_id"
            ]
            .map(
                prediction_lookup
            )
        )

        work = pd.concat(
            [
                work,
                current_covariates,
            ],
            ignore_index=True,
        )

    output = pd.concat(
        predictions,
        ignore_index=True,
    )

    actual = (
        future[
            [
                "week_start",
                "sku_id",
                "demand",
            ]
        ]
        .copy()
        .rename(
            columns={
                "demand": "actual",
            }
        )
    )

    output = output.merge(
        actual,
        on=[
            "week_start",
            "sku_id",
        ],
        how="left",
        validate="one_to_one",
    )

    # Keep original name for compatibility until pipeline integration.
    output[
        "model"
    ] = "lightgbm"

    output[
        "split"
    ] = split_name

    return output[
        [
            "split",
            "model",
            "week_start",
            "sku_id",
            "actual",
            "prediction",
        ]
    ]


# ============================================================
# DIRECT MULTI-HORIZON TRAINING TABLE
# ============================================================


def make_direct_training_frame(
    history: pd.DataFrame,
    max_horizon: int = 13,
) -> pd.DataFrame:
    """
    Convert historical weekly demand into a direct multi-horizon
    supervised-learning table.

    For each target week, training rows are created for:

        horizon 1
        horizon 2
        ...
        horizon max_horizon

    Example
    -------
    To predict target week t at horizon 3:

        forecast origin = t - 3

    Demand-history features therefore come from information that
    was available at t-3.

    Target-week calendar/event features remain those belonging to t.

    This avoids recursive prediction propagation.
    """

    if max_horizon < 1:
        raise ValueError(
            "max_horizon must be >= 1."
        )

    featured = (
        add_forecasting_features(
            history
        )
        .sort_values(
            [
                "sku_id",
                "week_start",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    frames = []

    # --------------------------------------------------------
    # For target row j and horizon h:
    #
    # history features must come from row:
    #
    #     j - (h - 1)
    #
    # Why?
    #
    # Features stored on row k already use demand through k-1.
    #
    # Therefore for h=1:
    #     target row itself has lag1 = demand at t-1.
    #
    # For h=2:
    #     previous feature row has lag1 = demand at t-2.
    #
    # etc.
    # --------------------------------------------------------

    for horizon in range(
        1,
        max_horizon + 1,
    ):

        frame = featured.copy()

        history_shift = (
            horizon - 1
        )

        for feature in HISTORY_FEATURES:

            if history_shift == 0:
                continue

            frame[
                feature
            ] = (
                frame
                .groupby(
                    "sku_id",
                    observed=True,
                )[
                    feature
                ]
                .shift(
                    history_shift
                )
            )

        frame[
            "forecast_horizon"
        ] = horizon

        frames.append(
            frame
        )

    direct = pd.concat(
        frames,
        ignore_index=True,
    )

    required = (
        [
            "demand",
        ]
        + direct_model_feature_columns()
    )

    direct = (
        direct
        .dropna(
            subset=required
        )
        .sort_values(
            [
                "week_start",
                "sku_id",
                "forecast_horizon",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if direct.empty:
        raise ValueError(
            "No direct LightGBM training rows remain."
        )

    if direct[
        direct_model_feature_columns()
    ].isna().any().any():

        raise ValueError(
            "Missing values remain in direct "
            "LightGBM training features."
        )

    return direct


# ============================================================
# DIRECT MULTI-HORIZON LIGHTGBM
# ============================================================


def build_direct_lightgbm(
    random_state: int = 42,
) -> Pipeline:
    """
    Build direct multi-horizon global LightGBM.
    """

    return _build_pipeline(
        feature_columns=(
            direct_model_feature_columns()
        ),
        random_state=random_state,
    )


def fit_direct_lightgbm(
    history: pd.DataFrame,
    max_horizon: int = 13,
    random_state: int = 42,
) -> Pipeline:
    """
    Train direct multi-horizon LightGBM.

    One model learns all:

        SKU × target week × horizon

    combinations.
    """

    training = (
        make_direct_training_frame(
            history=history,
            max_horizon=max_horizon,
        )
    )

    X = training[
        direct_model_feature_columns()
    ]

    y = training[
        "demand"
    ]

    model = build_direct_lightgbm(
        random_state=random_state
    )

    model.fit(
        X,
        y,
    )

    return model


# ============================================================
# DIRECT FORECAST FEATURE BUILDER
# ============================================================


def make_direct_forecast_frame(
    history: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build direct-model inference features.

    All target horizons use the SAME historical forecast origin.

    No earlier future prediction is inserted into later horizons.

    Actual values in future['demand'] are never used to construct
    predictors.
    """

    history = history.copy()
    future = future.copy()

    history[
        "week_start"
    ] = pd.to_datetime(
        history[
            "week_start"
        ]
    )

    future[
        "week_start"
    ] = pd.to_datetime(
        future[
            "week_start"
        ]
    )

    future_weeks = (
        future[
            "week_start"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not future_weeks:
        raise ValueError(
            "Future forecast horizon is empty."
        )

    first_future_week = (
        future_weeks[0]
    )

    # --------------------------------------------------------
    # Raw columns required by add_forecasting_features()
    # --------------------------------------------------------

    raw_feature_columns = [
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
    ]

    # --------------------------------------------------------
    # Construct exactly one synthetic row after the forecast origin.
    #
    # This row's lag/rolling features therefore summarize all demand
    # available at the forecast origin.
    # --------------------------------------------------------

    origin_rows = (
        future.loc[
            future[
                "week_start"
            ].eq(
                first_future_week
            ),
            raw_feature_columns,
        ]
        .copy()
    )

    # Hide actual first-future-week demand.
    origin_rows[
        "demand"
    ] = np.nan

    origin_input = pd.concat(
        [
            history[
                raw_feature_columns
            ],
            origin_rows,
        ],
        ignore_index=True,
    )

    origin_featured = (
        add_forecasting_features(
            origin_input
        )
    )

    origin_history_features = (
        origin_featured.loc[
            origin_featured[
                "week_start"
            ].eq(
                first_future_week
            ),
            [
                "sku_id",
                *HISTORY_FEATURES,
            ],
        ]
        .copy()
    )

    if origin_history_features[
        HISTORY_FEATURES
    ].isna().any().any():

        missing = (
            origin_history_features[
                HISTORY_FEATURES
            ]
            .isna()
            .sum()
        )

        raise ValueError(
            "Missing history features at direct forecast origin:\n"
            f"{missing[missing > 0]}"
        )

    # --------------------------------------------------------
    # Target-week covariates
    # --------------------------------------------------------

    target = (
        future[
            [
                "week_start",
                "sku_id",

                "cat_id",
                "dept_id",

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
            ]
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Annual cyclical encoding for target week
    # --------------------------------------------------------

    target[
        "week_sin"
    ] = np.sin(
        2.0
        * np.pi
        * target[
            "week_of_year"
        ].astype(
            float
        )
        / 52.0
    )

    target[
        "week_cos"
    ] = np.cos(
        2.0
        * np.pi
        * target[
            "week_of_year"
        ].astype(
            float
        )
        / 52.0
    )

    # --------------------------------------------------------
    # Global chronological index.
    #
    # This reproduces the time_idx convention used during training.
    # --------------------------------------------------------

    all_weeks = sorted(
        set(
            history[
                "week_start"
            ].tolist()
        )
        |
        set(
            future[
                "week_start"
            ].tolist()
        )
    )

    week_index = {
        week: idx
        for idx, week in enumerate(
            all_weeks
        )
    }

    target[
        "time_idx"
    ] = (
        target[
            "week_start"
        ]
        .map(
            week_index
        )
        .astype(
            int
        )
    )

    # --------------------------------------------------------
    # Forecast horizon: 1 ... H
    # --------------------------------------------------------

    horizon_lookup = {
        week: horizon
        for horizon, week in enumerate(
            future_weeks,
            start=1,
        )
    }

    target[
        "forecast_horizon"
    ] = (
        target[
            "week_start"
        ]
        .map(
            horizon_lookup
        )
        .astype(
            int
        )
    )

    # --------------------------------------------------------
    # Every future target receives the same origin-history state
    # for its SKU.
    # --------------------------------------------------------

    target = target.merge(
        origin_history_features,
        on="sku_id",
        how="left",
        validate="many_to_one",
    )

    missing_features = (
        target[
            direct_model_feature_columns()
        ]
        .isna()
        .sum()
    )

    missing_features = (
        missing_features[
            missing_features > 0
        ]
    )

    if not missing_features.empty:

        raise ValueError(
            "Missing direct LightGBM inference features:\n"
            f"{missing_features}"
        )

    return (
        target
        .sort_values(
            [
                "week_start",
                "sku_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# DIRECT MULTI-HORIZON FORECAST
# ============================================================


def forecast_direct_lightgbm(
    model: Pipeline,
    history: pd.DataFrame,
    future: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Generate direct 1...H forecasts from one fixed forecast origin.

    Unlike recursive LightGBM:

        prediction h=1

    is NOT inserted into the predictors for:

        h=2

    and so on.
    """

    forecast_frame = (
        make_direct_forecast_frame(
            history=history,
            future=future,
        )
    )

    X = forecast_frame[
        direct_model_feature_columns()
    ]

    prediction = (
        model.predict(
            X
        )
    )

    prediction = np.clip(
        prediction,
        0.0,
        None,
    )

    output = (
        forecast_frame[
            [
                "week_start",
                "sku_id",
                "forecast_horizon",
            ]
        ]
        .copy()
    )

    output[
        "prediction"
    ] = prediction

    actual = (
        future[
            [
                "week_start",
                "sku_id",
                "demand",
            ]
        ]
        .copy()
        .rename(
            columns={
                "demand": "actual",
            }
        )
    )

    actual[
        "week_start"
    ] = pd.to_datetime(
        actual[
            "week_start"
        ]
    )

    output = output.merge(
        actual,
        on=[
            "week_start",
            "sku_id",
        ],
        how="left",
        validate="one_to_one",
    )

    output[
        "model"
    ] = "lightgbm_direct"

    output[
        "split"
    ] = split_name

    return output[
        [
            "split",
            "model",
            "week_start",
            "sku_id",
            "forecast_horizon",
            "actual",
            "prediction",
        ]
    ]