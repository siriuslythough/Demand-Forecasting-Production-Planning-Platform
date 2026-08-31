from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ============================================================
# COMMON OUTPUT BUILDER
# ============================================================


def _build_output(
    future: pd.DataFrame,
    prediction_records: list[dict],
    model_name: str,
    split_name: str,
) -> pd.DataFrame:
    """
    Convert per-SKU forecasts into the common prediction contract.
    """

    prediction_df = pd.DataFrame(
        prediction_records
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
                "demand": "actual"
            }
        )
    )

    actual["week_start"] = pd.to_datetime(
        actual["week_start"]
    )

    output = actual.merge(
        prediction_df,
        on=[
            "week_start",
            "sku_id",
        ],
        how="left",
        validate="one_to_one",
    )

    if output["prediction"].isna().any():
        raise ValueError(
            f"{model_name} produced missing forecasts."
        )

    # Unit demand cannot be negative.
    output["prediction"] = (
        output["prediction"]
        .clip(lower=0.0)
    )

    output["model"] = model_name
    output["split"] = split_name

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
# DRIFT
# ============================================================


def forecast_drift(
    history: pd.DataFrame,
    future: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Random-walk-with-drift forecast.

    Forecast:
        y_(T+h) =
        y_T + h * (y_T - y_1) / (T - 1)

    Useful benchmark when demand has a persistent long-term trend.
    """

    records = []

    future_weeks = sorted(
        pd.to_datetime(
            future["week_start"]
        ).unique()
    )

    for sku_id, group in history.groupby(
        "sku_id",
        observed=True,
    ):

        group = group.sort_values(
            "week_start"
        )

        values = (
            group["demand"]
            .astype(float)
            .to_numpy()
        )

        if len(values) < 2:
            raise ValueError(
                f"Insufficient history for drift: {sku_id}"
            )

        slope = (
            values[-1] - values[0]
        ) / (
            len(values) - 1
        )

        for horizon, week in enumerate(
            future_weeks,
            start=1,
        ):

            prediction = (
                values[-1]
                + horizon * slope
            )

            records.append(
                {
                    "week_start": pd.Timestamp(week),
                    "sku_id": sku_id,
                    "prediction": prediction,
                }
            )

    return _build_output(
        future,
        records,
        model_name="drift",
        split_name=split_name,
    )


# ============================================================
# ETS / HOLT-WINTERS
# ============================================================


def forecast_ets(
    history: pd.DataFrame,
    future: pd.DataFrame,
    split_name: str,
    season_length: int = 52,
) -> pd.DataFrame:
    """
    Holt-Winters exponential smoothing.

    Components:
        additive trend
        damped trend
        additive seasonality

    This is a strong classical benchmark for weekly retail demand.
    """

    records = []

    future_weeks = sorted(
        pd.to_datetime(
            future["week_start"]
        ).unique()
    )

    horizon = len(
        future_weeks
    )

    for sku_id, group in history.groupby(
        "sku_id",
        observed=True,
    ):

        values = (
            group
            .sort_values("week_start")
            ["demand"]
            .astype(float)
            .to_numpy()
        )

        if len(values) < 2 * season_length:
            raise ValueError(
                f"{sku_id} needs at least "
                f"{2 * season_length} observations for ETS."
            )

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            model = ExponentialSmoothing(
                values,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=season_length,
                initialization_method="estimated",
            )

            fitted = model.fit(
                optimized=True,
                remove_bias=False,
            )

        forecast = fitted.forecast(
            horizon
        )

        for week, prediction in zip(
            future_weeks,
            forecast,
        ):

            records.append(
                {
                    "week_start": pd.Timestamp(week),
                    "sku_id": sku_id,
                    "prediction": float(prediction),
                }
            )

    return _build_output(
        future,
        records,
        model_name="ets_additive_52",
        split_name=split_name,
    )


# ============================================================
# THETA
# ============================================================


def forecast_theta(
    history: pd.DataFrame,
    future: pd.DataFrame,
    split_name: str,
    season_length: int = 52,
) -> pd.DataFrame:
    """
    Theta forecasting method.

    Theta is a strong lightweight classical benchmark and often
    performs competitively with substantially more complex models.
    """

    records = []

    future_weeks = sorted(
        pd.to_datetime(
            future["week_start"]
        ).unique()
    )

    horizon = len(
        future_weeks
    )

    for sku_id, group in history.groupby(
        "sku_id",
        observed=True,
    ):

        values = (
            group
            .sort_values("week_start")
            ["demand"]
            .astype(float)
        )

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            model = ThetaModel(
                values,
                period=season_length,
                deseasonalize=True,
                use_test=True,
            )

            fitted = model.fit()

        forecast = fitted.forecast(
            horizon
        )

        for week, prediction in zip(
            future_weeks,
            forecast,
        ):

            records.append(
                {
                    "week_start": pd.Timestamp(week),
                    "sku_id": sku_id,
                    "prediction": float(prediction),
                }
            )

    return _build_output(
        future,
        records,
        model_name="theta_52",
        split_name=split_name,
    )


# ============================================================
# ARIMA
# ============================================================


def forecast_arima(
    history: pd.DataFrame,
    future: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Small AIC-selected ARIMA benchmark.

    We deliberately search only a compact candidate family rather
    than running an expensive brute-force Auto-ARIMA search.
    """

    candidate_orders = [
        (1, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
        (2, 1, 1),
        (1, 1, 2),
    ]

    records = []

    future_weeks = sorted(
        pd.to_datetime(
            future["week_start"]
        ).unique()
    )

    horizon = len(
        future_weeks
    )

    for sku_id, group in history.groupby(
        "sku_id",
        observed=True,
    ):

        values = (
            group
            .sort_values("week_start")
            ["demand"]
            .astype(float)
            .to_numpy()
        )

        best_fit = None
        best_order = None
        best_aic = np.inf

        for order in candidate_orders:

            try:

                with warnings.catch_warnings():

                    warnings.simplefilter(
                        "ignore"
                    )

                    fitted = ARIMA(
                        values,
                        order=order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()

                if np.isfinite(
                    fitted.aic
                ) and fitted.aic < best_aic:

                    best_fit = fitted
                    best_order = order
                    best_aic = fitted.aic

            except Exception:
                continue

        if best_fit is None:
            raise RuntimeError(
                f"All ARIMA candidates failed for {sku_id}."
            )

        forecast = best_fit.forecast(
            steps=horizon
        )

        for week, prediction in zip(
            future_weeks,
            forecast,
        ):

            records.append(
                {
                    "week_start": pd.Timestamp(week),
                    "sku_id": sku_id,
                    "prediction": float(prediction),
                }
            )

    return _build_output(
        future,
        records,
        model_name="arima_aic",
        split_name=split_name,
    )


# ============================================================
# SARIMA
# ============================================================


def forecast_sarima(
    history: pd.DataFrame,
    future: pd.DataFrame,
    split_name: str,
    season_length: int = 52,
) -> pd.DataFrame:
    """
    Seasonal ARIMA benchmark.

    Small candidate set only.

    This is intentionally constrained because a broad SARIMA grid over
    20 SKUs x multiple rolling folds would be unnecessarily expensive.
    """

    candidate_specs = [
        (
            (1, 1, 0),
            (1, 0, 0, season_length),
        ),
        (
            (0, 1, 1),
            (1, 0, 0, season_length),
        ),
        (
            (1, 1, 1),
            (1, 0, 0, season_length),
        ),
    ]

    records = []

    future_weeks = sorted(
        pd.to_datetime(
            future["week_start"]
        ).unique()
    )

    horizon = len(
        future_weeks
    )

    for sku_id, group in history.groupby(
        "sku_id",
        observed=True,
    ):

        values = (
            group
            .sort_values("week_start")
            ["demand"]
            .astype(float)
            .to_numpy()
        )

        best_fit = None
        best_spec = None
        best_aic = np.inf

        for (
            order,
            seasonal_order,
        ) in candidate_specs:

            try:

                with warnings.catch_warnings():

                    warnings.simplefilter(
                        "ignore"
                    )

                    model = SARIMAX(
                        values,

                        order=order,
                        seasonal_order=seasonal_order,

                        trend=None,

                        enforce_stationarity=False,
                        enforce_invertibility=False,

                        simple_differencing=False,
                    )

                    fitted = model.fit(
                        disp=False,
                        maxiter=100,
                    )

                if (
                    np.isfinite(fitted.aic)
                    and fitted.aic < best_aic
                ):

                    best_fit = fitted
                    best_spec = (
                        order,
                        seasonal_order,
                    )

                    best_aic = fitted.aic

            except Exception:
                continue

        if best_fit is None:
            raise RuntimeError(
                f"All SARIMA candidates failed for {sku_id}."
            )

        forecast = best_fit.forecast(
            steps=horizon
        )

        forecast = np.asarray(
            forecast,
            dtype=float,
        )

        if not np.isfinite(
            forecast
        ).all():
            raise RuntimeError(
                f"SARIMA produced non-finite "
                f"forecasts for {sku_id}."
            )
        forecast = np.clip(
            forecast,
            0.0,
            None,
        )
        for week, prediction in zip(
            future_weeks,
            forecast,
        ):

            records.append(
                {
                    "week_start": pd.Timestamp(week),
                    "sku_id": sku_id,
                    "prediction": float(prediction),
                }
            )

    return _build_output(
        future,
        records,
        model_name="sarima_52",
        split_name=split_name,
    )