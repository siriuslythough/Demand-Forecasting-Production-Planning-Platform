from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from new_demfor_planopti.forecasting.baselines import (
    forecast_moving_average,
    forecast_naive,
    forecast_seasonal_naive,
)

from new_demfor_planopti.forecasting.metrics import (
    portfolio_metrics,
    sku_metrics,
)

from new_demfor_planopti.forecasting.models import (
    fit_direct_lightgbm,
    fit_global_lightgbm,
    forecast_direct_lightgbm,
    forecast_global_lightgbm,
)

from new_demfor_planopti.forecasting.statistical import (
    forecast_arima,
    forecast_drift,
    forecast_ets,
    forecast_sarima,
    forecast_theta,
)


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class ForecastConfig:
    """
    Configuration for the complete forecasting benchmark.

    Experiment design
    -----------------
    - expanding-window rolling-origin validation
    - 13-week production-planning horizon
    - final untouched test period
    """

    panel_path: Path = Path(
        "data/processed/m5_weekly_panel.parquet"
    )

    output_dir: Path = Path(
        "data/processed/forecasting"
    )

    forecast_horizon: int = 13

    validation_folds: int = 3

    test_weeks: int = 13

    moving_average_windows: tuple[int, ...] = (
        4,
        13,
    )

    season_length: int = 52

    random_state: int = 42


# ============================================================
# LOAD / VALIDATE STAGE-1 PANEL
# ============================================================


def load_forecasting_panel(
    panel_path: Path,
) -> pd.DataFrame:
    """
    Load the Stage-1 weekly SKU panel and enforce its contract.
    """

    if not panel_path.exists():
        raise FileNotFoundError(
            f"Forecasting panel not found: "
            f"{panel_path.resolve()}"
        )

    panel = pd.read_parquet(
        panel_path
    )

    panel["week_start"] = pd.to_datetime(
        panel["week_start"]
    )

    panel = (
        panel
        .sort_values(
            [
                "week_start",
                "sku_id",
            ]
        )
        .reset_index(drop=True)
    )

    key = [
        "week_start",
        "sku_id",
    ]

    if panel.duplicated(key).any():
        raise ValueError(
            "Forecast panel violates unique "
            "(week_start, sku_id) contract."
        )

    if panel["demand"].isna().any():
        raise ValueError(
            "Forecast panel contains missing demand."
        )

    if panel["demand"].lt(0).any():
        raise ValueError(
            "Forecast panel contains negative demand."
        )

    n_skus = panel["sku_id"].nunique()
    n_weeks = panel["week_start"].nunique()

    expected_rows = (
        n_skus
        * n_weeks
    )

    if len(panel) != expected_rows:
        raise ValueError(
            "Forecast input must be a balanced SKU-week panel. "
            f"Expected {expected_rows} rows, "
            f"found {len(panel)}."
        )

    return panel


# ============================================================
# ROLLING-ORIGIN VALIDATION
# ============================================================


def build_rolling_validation_folds(
    panel: pd.DataFrame,
    horizon: int = 13,
    n_folds: int = 3,
    test_weeks: int = 13,
) -> tuple[
    list[
        tuple[
            pd.DataFrame,
            pd.DataFrame,
            str,
        ]
    ],
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Create expanding-window rolling-origin validation folds.

    Final test weeks are excluded completely from model selection.

    Example
    -------
    With:

        horizon = 13
        n_folds = 3
        test_weeks = 13

    the timeline is:

        train_1 -> validation_1
        train_2 --------> validation_2
        train_3 ----------------> validation_3
        development --------------------> test

    Training history expands after every validation fold.
    """

    weeks = (
        panel["week_start"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    total_weeks = len(
        weeks
    )

    validation_weeks_required = (
        horizon
        * n_folds
    )

    # We want at least two annual cycles before the first validation
    # origin because ETS/SARIMA use 52-week seasonality.
    minimum_initial_training_weeks = (
        2 * 52
    )

    required_total = (
        minimum_initial_training_weeks
        + validation_weeks_required
        + test_weeks
    )

    if total_weeks < required_total:
        raise ValueError(
            "Insufficient history for rolling validation. "
            f"Need at least {required_total} weeks, "
            f"found {total_weeks}."
        )

    # --------------------------------------------------------
    # Final untouched test period
    # --------------------------------------------------------

    test_start_idx = (
        total_weeks
        - test_weeks
    )

    development_weeks = weeks[
        :test_start_idx
    ]

    test_week_values = set(
        weeks[
            test_start_idx:
        ]
    )

    # --------------------------------------------------------
    # Validation folds are placed immediately before test
    # --------------------------------------------------------

    first_validation_start = (
        len(development_weeks)
        - validation_weeks_required
    )

    folds = []

    for fold_idx in range(
        n_folds
    ):
        validation_start = (
            first_validation_start
            + fold_idx * horizon
        )

        validation_end = (
            validation_start
            + horizon
        )

        train_week_values = set(
            development_weeks[
                :validation_start
            ]
        )

        validation_week_values = set(
            development_weeks[
                validation_start:
                validation_end
            ]
        )

        train = (
            panel.loc[
                panel["week_start"].isin(
                    train_week_values
                )
            ]
            .copy()
        )

        validation = (
            panel.loc[
                panel["week_start"].isin(
                    validation_week_values
                )
            ]
            .copy()
        )

        fold_name = (
            f"validation_{fold_idx + 1}"
        )

        if (
            train["week_start"].max()
            >= validation["week_start"].min()
        ):
            raise ValueError(
                f"Temporal leakage detected in "
                f"{fold_name}."
            )

        folds.append(
            (
                train,
                validation,
                fold_name,
            )
        )

    # --------------------------------------------------------
    # Development = everything before test
    # --------------------------------------------------------

    development = (
        panel.loc[
            ~panel[
                "week_start"
            ].isin(
                test_week_values
            )
        ]
        .copy()
    )

    test = (
        panel.loc[
            panel[
                "week_start"
            ].isin(
                test_week_values
            )
        ]
        .copy()
    )

    if (
        development["week_start"].max()
        >= test["week_start"].min()
    ):
        raise ValueError(
            "Development/test temporal split is invalid."
        )

    return (
        folds,
        development,
        test,
    )


# ============================================================
# RUN COMPLETE MODEL LADDER
# ============================================================


def run_all_models(
    history: pd.DataFrame,
    future: pd.DataFrame,
    split_name: str,
    config: ForecastConfig,
) -> tuple[
    pd.DataFrame,
    dict[str, object],
]:
    """
    Fit and forecast every candidate model for one forecast origin.

    Model ladder
    ------------
    Tier 1 — Simple baselines
        naive
        seasonal naive
        drift
        moving averages

    Tier 2 — Classical statistical models
        ETS
        Theta
        ARIMA
        SARIMA

    Tier 3 — Machine learning
        recursive global LightGBM
        direct multi-horizon global LightGBM

    Returns
    -------
    predictions:
        Long-form predictions from every model.

    fitted_models:
        Fitted ML models that may later be persisted.
    """

    outputs: list[pd.DataFrame] = []

    fitted_models: dict[
        str,
        object,
    ] = {}


    # ========================================================
    # TIER 1 — SIMPLE BASELINES
    # ========================================================

    print(
        f"    [{split_name}] Naive"
    )

    outputs.append(
        forecast_naive(
            history=history,
            future=future,
            split_name=split_name,
        )
    )


    print(
        f"    [{split_name}] Seasonal naive"
    )

    outputs.append(
        forecast_seasonal_naive(
            history=history,
            future=future,
            split_name=split_name,
            season_length=(
                config.season_length
            ),
        )
    )


    print(
        f"    [{split_name}] Drift"
    )

    outputs.append(
        forecast_drift(
            history=history,
            future=future,
            split_name=split_name,
        )
    )


    for window in (
        config.moving_average_windows
    ):

        print(
            f"    [{split_name}] "
            f"Moving average {window}"
        )

        outputs.append(
            forecast_moving_average(
                history=history,
                future=future,
                split_name=split_name,
                window=window,
            )
        )


    # ========================================================
    # TIER 2 — CLASSICAL TIME SERIES
    # ========================================================

    print(
        f"    [{split_name}] ETS"
    )

    outputs.append(
        forecast_ets(
            history=history,
            future=future,
            split_name=split_name,
            season_length=(
                config.season_length
            ),
        )
    )


    print(
        f"    [{split_name}] Theta"
    )

    outputs.append(
        forecast_theta(
            history=history,
            future=future,
            split_name=split_name,
            season_length=(
                config.season_length
            ),
        )
    )


    print(
        f"    [{split_name}] ARIMA"
    )

    outputs.append(
        forecast_arima(
            history=history,
            future=future,
            split_name=split_name,
        )
    )


    print(
        f"    [{split_name}] SARIMA"
    )

    outputs.append(
        forecast_sarima(
            history=history,
            future=future,
            split_name=split_name,
            season_length=(
                config.season_length
            ),
        )
    )


    # ========================================================
    # TIER 3A — RECURSIVE GLOBAL LIGHTGBM
    # ========================================================

    print(
        f"    [{split_name}] "
        f"Global LightGBM — recursive"
    )

    recursive_model = (
        fit_global_lightgbm(
            history=history,
            random_state=(
                config.random_state
            ),
        )
    )

    recursive_predictions = (
        forecast_global_lightgbm(
            model=recursive_model,
            history=history,
            future=future,
            split_name=split_name,
        )
    )

    # The original models.py keeps the legacy name "lightgbm".
    # Relabel it here so our model comparison is unambiguous.
    recursive_predictions[
        "model"
    ] = "lightgbm_recursive"

    outputs.append(
        recursive_predictions
    )

    fitted_models[
        "lightgbm_recursive"
    ] = recursive_model


    # ========================================================
    # TIER 3B — DIRECT MULTI-HORIZON GLOBAL LIGHTGBM
    # ========================================================

    print(
        f"    [{split_name}] "
        f"Global LightGBM — direct"
    )

    direct_model = (
        fit_direct_lightgbm(
            history=history,
            max_horizon=(
                config.forecast_horizon
            ),
            random_state=(
                config.random_state
            ),
        )
    )

    direct_predictions = (
        forecast_direct_lightgbm(
            model=direct_model,
            history=history,
            future=future,
            split_name=split_name,
        )
    )

    outputs.append(
        direct_predictions
    )

    fitted_models[
        "lightgbm_direct"
    ] = direct_model


    # ========================================================
    # COMBINE
    # ========================================================

    predictions = pd.concat(
        outputs,
        ignore_index=True,
    )


    # --------------------------------------------------------
    # Common prediction contract
    # --------------------------------------------------------

    required_columns = {
        "split",
        "model",
        "week_start",
        "sku_id",
        "actual",
        "prediction",
    }

    missing = (
        required_columns
        - set(
            predictions.columns
        )
    )

    if missing:

        raise ValueError(
            "Combined model predictions are "
            f"missing columns: {sorted(missing)}"
        )


    if predictions[
        "prediction"
    ].isna().any():

        raise ValueError(
            "One or more forecasting models "
            "produced missing predictions."
        )


    if predictions[
        "prediction"
    ].lt(0).any():

        raise ValueError(
            "One or more forecasting models "
            "produced negative demand forecasts."
        )


    return (
        predictions,
        fitted_models,
    )


# ============================================================
# CROSS-VALIDATION SUMMARY
# ============================================================


def summarize_cv_metrics(
    cv_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate model performance across rolling-origin folds.

    Primary selection metric:
        mean WAPE

    Secondary robustness diagnostic:
        standard deviation of WAPE
    """

    summary = (
        cv_metrics
        .groupby(
            "model",
            as_index=False,
        )
        .agg(
            mean_mae=(
                "mae",
                "mean",
            ),
            mean_rmse=(
                "rmse",
                "mean",
            ),
            mean_wape=(
                "wape",
                "mean",
            ),
            std_wape=(
                "wape",
                "std",
            ),
            min_wape=(
                "wape",
                "min",
            ),
            max_wape=(
                "wape",
                "max",
            ),
            mean_smape=(
                "smape",
                "mean",
            ),
            mean_bias_pct=(
                "bias_pct",
                "mean",
            ),
            validation_folds=(
                "split",
                "nunique",
            ),
        )
        .sort_values(
            [
                "mean_wape",
                "std_wape",
            ]
        )
        .reset_index(drop=True)
    )

    return summary


# ============================================================
# RESUME BULLET
# ============================================================


def build_resume_bullet(
    panel: pd.DataFrame,
    champion: str,
    test_metrics: pd.DataFrame,
    config: ForecastConfig,
) -> str:
    """
    Generate a resume bullet using actual measured results.
    """

    n_skus = (
        panel["sku_id"]
        .nunique()
    )

    n_weeks = (
        panel["week_start"]
        .nunique()
    )

    champion_row = (
        test_metrics.loc[
            test_metrics[
                "model"
            ].eq(
                champion
            )
        ]
        .iloc[0]
    )

    seasonal_name = (
        f"seasonal_naive_{config.season_length}"
    )

    champion_wape = float(
        champion_row["wape"]
    )


    if champion == seasonal_name:

        improvement_text = ""

    else:

        seasonal_rows = (
            test_metrics.loc[
                test_metrics[
                    "model"
                ].eq(
                    seasonal_name
                )
            ]
        )

        if seasonal_rows.empty:

            improvement_text = ""

        else:

            seasonal_wape = float(
                seasonal_rows.iloc[0][
                    "wape"
                ]
            )

            improvement = (
                100.0
                * (
                    seasonal_wape
                    - champion_wape
                )
                / seasonal_wape
            )

            improvement_text = (
                f", a {improvement:.1f}% "
                f"WAPE improvement versus "
                f"seasonal naive"
        )

    return (
        f"Built a rolling-origin weekly demand forecasting "
        f"framework for {n_skus} M5 SKUs across "
        f"{n_weeks} weeks, benchmarking naive/seasonal "
        f"baselines, ETS/Theta, ARIMA/SARIMA and pooled "
        f"LightGBM on a {config.forecast_horizon}-week "
        f"production-planning horizon; the CV-selected "
        f"{champion} model achieved "
        f"{champion_wape:.1f}% test WAPE"
        f"{improvement_text}."
    )


# ============================================================
# COMPLETE PIPELINE
# ============================================================


def run_forecasting_pipeline(
    config: ForecastConfig | None = None,
) -> dict:
    """
    Run the complete forecasting experiment.

    Selection protocol
    ------------------
    1. Create rolling-origin validation folds.
    2. Evaluate every candidate on every validation origin.
    3. Select champion using mean validation WAPE.
    4. Refit/evaluate on final untouched 13-week test horizon.
    5. Persist predictions, metrics and model artifacts.
    """

    config = (
        config
        or ForecastConfig()
    )

    print()
    print("=" * 72)
    print("DEMAND FORECASTING PIPELINE")
    print("=" * 72)

    # ========================================================
    # 1. LOAD
    # ========================================================

    print(
        "\n[1/7] Loading Stage-1 weekly panel..."
    )

    panel = load_forecasting_panel(
        config.panel_path
    )

    print(
        f"Rows:  {len(panel):,}"
    )

    print(
        f"SKUs:  "
        f"{panel['sku_id'].nunique()}"
    )

    print(
        f"Weeks: "
        f"{panel['week_start'].nunique()}"
    )

    print(
        "Range:",
        panel["week_start"].min(),
        "->",
        panel["week_start"].max(),
    )

    # ========================================================
    # 2. BUILD ROLLING VALIDATION
    # ========================================================

    print(
        "\n[2/7] Building rolling-origin "
        "validation folds..."
    )

    (
        folds,
        development,
        test,
    ) = build_rolling_validation_folds(
        panel=panel,
        horizon=config.forecast_horizon,
        n_folds=config.validation_folds,
        test_weeks=config.test_weeks,
    )

    for (
        train,
        validation,
        fold_name,
    ) in folds:
        print()
        print(
            f"{fold_name}:"
        )

        print(
            "    train      :",
            train["week_start"].min(),
            "->",
            train["week_start"].max(),
        )

        print(
            "    validation :",
            validation["week_start"].min(),
            "->",
            validation["week_start"].max(),
        )

    print()
    print(
        "Final test:",
        test["week_start"].min(),
        "->",
        test["week_start"].max(),
    )

    # ========================================================
    # 3. ROLLING CV
    # ========================================================

    print(
        "\n[3/7] Running rolling-origin model benchmark..."
    )

    validation_prediction_frames = []
    validation_metric_frames = []

    for (
        fold_number,
        (
            train,
            validation,
            fold_name,
        ),
    ) in enumerate(
        folds,
        start=1,
    ):
        print()
        print(
            "-" * 72
        )

        print(
            f"VALIDATION FOLD "
            f"{fold_number}/"
            f"{config.validation_folds}"
        )

        print(
            "-" * 72
        )

        fold_predictions, _ = (
            run_all_models(
                history=train,
                future=validation,
                split_name=fold_name,
                config=config,
            )
        )

        fold_metrics = (
            portfolio_metrics(
                fold_predictions
            )
        )

        validation_prediction_frames.append(
            fold_predictions
        )

        validation_metric_frames.append(
            fold_metrics
        )

        print()
        print(
            fold_metrics[
                [
                    "model",
                    "mae",
                    "rmse",
                    "wape",
                    "smape",
                    "bias_pct",
                ]
            ]
            .sort_values(
                "wape"
            )
            .round(3)
            .to_string(
                index=False
            )
        )

    validation_predictions = (
        pd.concat(
            validation_prediction_frames,
            ignore_index=True,
        )
    )

    cv_metrics = (
        pd.concat(
            validation_metric_frames,
            ignore_index=True,
        )
    )

    # ========================================================
    # 4. SELECT CHAMPION
    # ========================================================

    print(
        "\n[4/7] Selecting model using "
        "rolling-validation performance..."
    )

    cv_summary = (
        summarize_cv_metrics(
            cv_metrics
        )
    )

    print()
    print(
        "CROSS-VALIDATION SUMMARY"
    )

    print(
        cv_summary[
            [
                "model",
                "mean_wape",
                "std_wape",
                "mean_mae",
                "mean_rmse",
                "mean_smape",
                "mean_bias_pct",
            ]
        ]
        .round(3)
        .to_string(
            index=False
        )
    )

    champion = (
        cv_summary.iloc[
            0
        ]["model"]
    )

    print()
    print(
        f"CV champion: {champion}"
    )

    # ========================================================
    # 5. FINAL TEST
    # ========================================================

    print(
        "\n[5/7] Fitting on full development history "
        "and evaluating untouched test horizon..."
    )

    (
        test_predictions,
        final_ml_models,
    ) = run_all_models(
        history=development,
        future=test,
        split_name="test",
        config=config,
    )

    test_metrics = (
        portfolio_metrics(
            test_predictions
        )
    )

    test_metrics_by_sku = (
        sku_metrics(
            test_predictions
        )
    )

    print()
    print(
        "FINAL TEST RESULTS"
    )

    print(
        test_metrics[
            [
                "model",
                "mae",
                "rmse",
                "wape",
                "smape",
                "bias_pct",
            ]
        ]
        .sort_values(
            "wape"
        )
        .round(3)
        .to_string(
            index=False
        )
    )

    champion_test_row = (
        test_metrics.loc[
            test_metrics[
                "model"
            ].eq(
                champion
            )
        ]
    )

    if champion_test_row.empty:
        raise ValueError(
            f"CV champion {champion!r} "
            "was not evaluated on test."
        )

    champion_test_row = (
        champion_test_row.iloc[
            0
        ]
    )

    # ========================================================
    # 6. SAVE ARTIFACTS
    # ========================================================

    print(
        "\n[6/7] Saving forecasting artifacts..."
    )

    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_predictions.to_parquet(
        config.output_dir
        / "validation_predictions.parquet",
        index=False,
    )

    cv_metrics.to_csv(
        config.output_dir
        / "cv_metrics.csv",
        index=False,
    )

    cv_summary.to_csv(
        config.output_dir
        / "cv_summary.csv",
        index=False,
    )

    test_predictions.to_parquet(
        config.output_dir
        / "test_predictions.parquet",
        index=False,
    )

    test_metrics.to_csv(
        config.output_dir
        / "test_metrics.csv",
        index=False,
    )

    test_metrics_by_sku.to_csv(
        config.output_dir
        / "test_metrics_by_sku.csv",
        index=False,
    )

    # ============================================================
    # SAVE FITTED ML MODELS
    # ============================================================

    for (
        model_name,
        fitted_model,
    ) in final_ml_models.items():

        model_path = (
            config.output_dir
            / f"{model_name}_development.joblib"
        )

        joblib.dump(
            fitted_model,
            model_path,
        )

    champion_payload = {
        "champion": champion,
        "selection_metric": "mean_cv_wape",
        "forecast_horizon": (
            config.forecast_horizon
        ),
        "validation_folds": (
            config.validation_folds
        ),
        "test_weeks": (
            config.test_weeks
        ),
        "mean_cv_wape": float(
            cv_summary.iloc[
                0
            ]["mean_wape"]
        ),
        "cv_wape_std": float(
            cv_summary.iloc[
                0
            ]["std_wape"]
        ),
        "test_wape": float(
            champion_test_row[
                "wape"
            ]
        ),
    }

    with open(
        config.output_dir
        / "champion.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            champion_payload,
            file,
            indent=2,
        )

    # ========================================================
    # 7. RESUME RESULT
    # ========================================================

    print(
        "\n[7/7] Generating resume-ready result..."
    )

    resume_bullet = (
        build_resume_bullet(
            panel=panel,
            champion=champion,
            test_metrics=test_metrics,
            config=config,
        )
    )

    (
        config.output_dir
        / "resume_bullet.txt"
    ).write_text(
        resume_bullet,
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("FORECASTING PIPELINE COMPLETE")
    print("=" * 72)

    print(
        f"CV-selected champion: "
        f"{champion}"
    )

    print(
        "Mean validation WAPE:",
        f"{cv_summary.iloc[0]['mean_wape']:.2f}%",
    )

    print(
        "Validation WAPE std:",
        f"{cv_summary.iloc[0]['std_wape']:.2f}",
    )

    print(
        "Champion test WAPE:",
        f"{champion_test_row['wape']:.2f}%",
    )

    print()
    print(
        "RESUME BULLET"
    )

    print(
        resume_bullet
    )

    print("=" * 72)

    return {
        "panel": panel,

        "folds": folds,

        "development": development,
        "test": test,

        "validation_predictions": (
            validation_predictions
        ),

        "cv_metrics": (
            cv_metrics
        ),

        "cv_summary": (
            cv_summary
        ),

        "test_predictions": (
            test_predictions
        ),

        "test_metrics": (
            test_metrics
        ),

        "test_metrics_by_sku": (
            test_metrics_by_sku
        ),

        "champion": champion,

        "resume_bullet": (
            resume_bullet
        ),
    }