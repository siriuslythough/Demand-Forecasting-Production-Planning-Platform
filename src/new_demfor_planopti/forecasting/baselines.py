from __future__ import annotations

import numpy as np
import pandas as pd


def _prepare_output(
    future: pd.DataFrame,
    prediction_map: dict[
        tuple[str, pd.Timestamp],
        float,
    ],
    model_name: str,
    split_name: str,
) -> pd.DataFrame:

    output = future[
        [
            "week_start",
            "sku_id",
            "demand",
        ]
    ].copy()

    output["week_start"] = pd.to_datetime(
        output["week_start"]
    )

    output["prediction"] = [
        prediction_map[
            (
                sku,
                week,
            )
        ]
        for sku, week in zip(
            output["sku_id"],
            output["week_start"],
        )
    ]

    output = output.rename(
        columns={"demand": "actual"}
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


def forecast_naive(
    history: pd.DataFrame,
    future: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Fixed-origin naive forecast.

    Every forecast horizon receives the last observed demand
    available at the forecast origin.

    We DO NOT use actual demand from earlier held-out weeks.
    """

    prediction_map = {}

    future_weeks = sorted(
        pd.to_datetime(
            future["week_start"]
        ).unique()
    )

    for sku_id, sku_history in history.groupby(
        "sku_id",
        observed=True,
    ):

        last_demand = (
            sku_history
            .sort_values("week_start")
            ["demand"]
            .iloc[-1]
        )

        for week in future_weeks:

            prediction_map[
                (
                    sku_id,
                    pd.Timestamp(week),
                )
            ] = float(last_demand)

    return _prepare_output(
        future,
        prediction_map,
        model_name="naive",
        split_name=split_name,
    )


def forecast_seasonal_naive(
    history: pd.DataFrame,
    future: pd.DataFrame,
    split_name: str,
    season_length: int = 52,
) -> pd.DataFrame:
    """
    Seasonal-naive fixed-origin forecast.

    For horizon h:
        forecast = demand from 52 weeks before that target week.

    With a 13-week horizon, all required seasonal observations
    remain safely inside the history window.
    """

    prediction_map = {}

    future_weeks = sorted(
        pd.to_datetime(
            future["week_start"]
        ).unique()
    )

    for sku_id, sku_history in history.groupby(
        "sku_id",
        observed=True,
    ):

        history_values = (
            sku_history
            .sort_values("week_start")
            ["demand"]
            .astype(float)
            .tolist()
        )

        if len(history_values) < season_length:
            raise ValueError(
                f"{sku_id} has fewer than "
                f"{season_length} history periods."
            )

        for horizon_idx, week in enumerate(
            future_weeks
        ):

            source_index = (
                len(history_values)
                - season_length
                + horizon_idx
            )

            if source_index >= len(history_values):
                raise ValueError(
                    "Forecast horizon is too long for fixed-origin "
                    "seasonal naive using available history."
                )

            prediction = history_values[
                source_index
            ]

            prediction_map[
                (
                    sku_id,
                    pd.Timestamp(week),
                )
            ] = float(prediction)

    return _prepare_output(
        future,
        prediction_map,
        model_name=f"seasonal_naive_{season_length}",
        split_name=split_name,
    )


def forecast_moving_average(
    history: pd.DataFrame,
    future: pd.DataFrame,
    split_name: str,
    window: int,
) -> pd.DataFrame:
    """
    Recursive moving-average forecast.

    After forecasting the first held-out week, its prediction is
    appended to history and used when forecasting the next week.

    This mimics real multi-step forecasting.
    """

    prediction_map = {}

    future_weeks = sorted(
        pd.to_datetime(
            future["week_start"]
        ).unique()
    )

    for sku_id, sku_history in history.groupby(
        "sku_id",
        observed=True,
    ):

        values = (
            sku_history
            .sort_values("week_start")
            ["demand"]
            .astype(float)
            .tolist()
        )

        if len(values) < window:
            raise ValueError(
                f"{sku_id} has fewer than "
                f"{window} observations."
            )

        for week in future_weeks:

            prediction = float(
                np.mean(
                    values[-window:]
                )
            )

            prediction_map[
                (
                    sku_id,
                    pd.Timestamp(week),
                )
            ] = prediction

            # Recursive:
            # append forecast, NOT actual held-out demand.
            values.append(prediction)

    return _prepare_output(
        future,
        prediction_map,
        model_name=f"moving_average_{window}",
        split_name=split_name,
    )