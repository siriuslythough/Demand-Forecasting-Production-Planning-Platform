from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-8


def calculate_metrics(
    actual: np.ndarray | pd.Series,
    prediction: np.ndarray | pd.Series,
) -> dict[str, float]:
    """
    Calculate regression metrics used for demand forecasting.

    Metrics
    -------
    MAE:
        Mean absolute error.

    RMSE:
        Root mean squared error.

    WAPE:
        Total absolute error divided by total actual demand.

    sMAPE:
        Symmetric mean absolute percentage error.

    bias_pct:
        Aggregate directional error as a percentage of actual demand.
        Positive = over-forecasting.
    """

    y = np.asarray(actual, dtype=float)
    yhat = np.asarray(prediction, dtype=float)

    error = yhat - y
    abs_error = np.abs(error)

    mae = np.mean(abs_error)

    rmse = np.sqrt(
        np.mean(np.square(error))
    )

    demand_total = np.sum(np.abs(y))

    wape = (
        100.0 * np.sum(abs_error) / demand_total
        if demand_total > EPS
        else np.nan
    )

    smape = 100.0 * np.mean(
        2.0
        * abs_error
        / (
            np.abs(y)
            + np.abs(yhat)
            + EPS
        )
    )

    bias_pct = (
        100.0 * np.sum(error) / demand_total
        if demand_total > EPS
        else np.nan
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "wape": float(wape),
        "smape": float(smape),
        "bias_pct": float(bias_pct),
    }


def portfolio_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate portfolio metrics for every split/model pair.
    """

    rows = []

    for (split, model), group in predictions.groupby(
        ["split", "model"],
        observed=True,
    ):
        metrics = calculate_metrics(
            group["actual"],
            group["prediction"],
        )

        rows.append(
            {
                "split": split,
                "model": model,
                **metrics,
                "rows": len(group),
                "actual_demand": group["actual"].sum(),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["split", "wape", "mae"]
        )
        .reset_index(drop=True)
    )


def sku_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate metrics independently for each SKU.
    """

    rows = []

    for (
        split,
        model,
        sku_id,
    ), group in predictions.groupby(
        ["split", "model", "sku_id"],
        observed=True,
    ):
        metrics = calculate_metrics(
            group["actual"],
            group["prediction"],
        )

        rows.append(
            {
                "split": split,
                "model": model,
                "sku_id": sku_id,
                **metrics,
                "actual_demand": group["actual"].sum(),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["split", "model", "wape"]
        )
        .reset_index(drop=True)
    )