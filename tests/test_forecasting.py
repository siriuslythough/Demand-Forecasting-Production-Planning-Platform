from __future__ import annotations

import numpy as np
import pandas as pd

from new_demfor_planopti.forecasting.baselines import (
    forecast_moving_average,
    forecast_naive,
    forecast_seasonal_naive,
)

from new_demfor_planopti.forecasting.features import (
    TREND_FEATURES,
    add_forecasting_features,
    model_feature_columns,
)

from new_demfor_planopti.forecasting.metrics import (
    calculate_metrics,
)

from new_demfor_planopti.forecasting.pipeline import (
    build_rolling_validation_folds,
)

from new_demfor_planopti.forecasting.models import (
    direct_model_feature_columns,
    make_direct_forecast_frame,
    make_direct_training_frame,
)


# ============================================================
# SYNTHETIC TEST DATA
# ============================================================


def make_panel(
    n_weeks: int = 170,
) -> pd.DataFrame:
    """
    Create a small balanced weekly panel for forecasting tests.

    Two SKUs are generated with deterministic increasing demand.

    The synthetic panel includes every column currently required by
    add_forecasting_features(), so tests exercise the same data contract
    used by the real M5 panel.
    """

    weeks = pd.date_range(
        "2020-01-04",
        periods=n_weeks,
        freq="7D",
    )

    rows = []

    for sku_idx, sku_id in enumerate(
        [
            "SKU_A",
            "SKU_B",
        ]
    ):

        for time_idx, week in enumerate(
            weeks
        ):

            # Deterministic positive demand.
            #
            # SKU_A:
            # 100, 101, 102, ...
            #
            # SKU_B:
            # 120, 121, 122, ...
            demand = (
                100
                + 20 * sku_idx
                + time_idx
            )

            rows.append(
                {
                    "week_start": week,

                    "sku_id": sku_id,

                    "cat_id": "FOODS",
                    "dept_id": "FOODS_1",

                    "demand": float(demand),

                    "year": week.year,
                    "quarter": week.quarter,
                    "month": week.month,
                    "week_of_year": int(
                        week.isocalendar().week
                    ),

                    "snap_days": 0,
                    "event_days": 0,

                    "cultural_event_days": 0,
                    "national_event_days": 0,
                    "religious_event_days": 0,
                    "sporting_event_days": 0,
                }
            )

    panel = pd.DataFrame(
        rows
    )

    return (
        panel
        .sort_values(
            [
                "week_start",
                "sku_id",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# METRIC TESTS
# ============================================================


def test_metrics_perfect_forecast():
    """
    Perfect forecasts must produce zero error.
    """

    actual = np.array(
        [
            10.0,
            20.0,
            30.0,
        ]
    )

    prediction = actual.copy()

    metrics = calculate_metrics(
        actual,
        prediction,
    )

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["wape"] == 0.0
    assert metrics["smape"] == 0.0
    assert metrics["bias_pct"] == 0.0


def test_metrics_have_expected_wape():
    """
    Verify WAPE using a simple hand-calculable example.

    actual:
        100 + 200 = 300

    absolute error:
        |90-100| + |220-200|
        = 10 + 20
        = 30

    WAPE:
        30 / 300 * 100
        = 10%
    """

    actual = np.array(
        [
            100.0,
            200.0,
        ]
    )

    prediction = np.array(
        [
            90.0,
            220.0,
        ]
    )

    metrics = calculate_metrics(
        actual,
        prediction,
    )

    assert np.isclose(
        metrics["wape"],
        10.0,
    )


# ============================================================
# FEATURE-ENGINEERING TESTS
# ============================================================


def test_lag_one_is_previous_demand():
    """
    demand_lag_1 for week t must equal demand at week t-1.
    """

    panel = make_panel(
        n_weeks=100
    )

    featured = add_forecasting_features(
        panel
    )

    sku = (
        featured.loc[
            featured["sku_id"].eq(
                "SKU_A"
            )
        ]
        .sort_values(
            "week_start"
        )
        .reset_index(
            drop=True
        )
    )

    idx = 70

    assert (
        sku.loc[
            idx,
            "demand_lag_1",
        ]
        ==
        sku.loc[
            idx - 1,
            "demand",
        ]
    )


def test_lag_52_is_one_year_back():
    """
    Weekly annual lag must reference demand 52 rows earlier.
    """

    panel = make_panel(
        n_weeks=100
    )

    featured = add_forecasting_features(
        panel
    )

    sku = (
        featured.loc[
            featured["sku_id"].eq(
                "SKU_A"
            )
        ]
        .sort_values(
            "week_start"
        )
        .reset_index(
            drop=True
        )
    )

    idx = 70

    assert (
        sku.loc[
            idx,
            "demand_lag_52",
        ]
        ==
        sku.loc[
            idx - 52,
            "demand",
        ]
    )


def test_rolling_mean_excludes_current_target():
    """
    Rolling demand statistics must use t-1 and earlier.

    This explicitly protects against target leakage.
    """

    panel = make_panel(
        n_weeks=100
    )

    featured = add_forecasting_features(
        panel
    )

    sku = (
        featured.loc[
            featured["sku_id"].eq(
                "SKU_A"
            )
        ]
        .sort_values(
            "week_start"
        )
        .reset_index(
            drop=True
        )
    )

    idx = 70

    expected_mean = (
        sku.loc[
            idx - 13:
            idx - 1,
            "demand",
        ]
        .mean()
    )

    actual_feature = (
        sku.loc[
            idx,
            "demand_roll_mean_13",
        ]
    )

    assert np.isclose(
        actual_feature,
        expected_mean,
    )


def test_regime_features_are_created():
    """
    Every Iteration-2 regime feature must exist in the output.
    """

    panel = make_panel(
        n_weeks=100
    )

    featured = add_forecasting_features(
        panel
    )

    for feature in TREND_FEATURES:

        assert feature in featured.columns


def test_regime_features_are_in_model_contract():
    """
    Creating a feature is insufficient if the ML model does not
    receive it.

    Every trend feature must therefore also appear in the formal
    LightGBM feature contract.
    """

    features = (
        model_feature_columns()
    )

    for feature in TREND_FEATURES:

        assert feature in features


def test_recent_level_ratio_uses_prior_history():
    """
    level_ratio_13_52 should equal:

        previous 13-week mean
        ---------------------
        previous 52-week mean

    The current target week must not participate.
    """

    panel = make_panel(
        n_weeks=100
    )

    featured = add_forecasting_features(
        panel
    )

    sku = (
        featured.loc[
            featured["sku_id"].eq(
                "SKU_A"
            )
        ]
        .sort_values(
            "week_start"
        )
        .reset_index(
            drop=True
        )
    )

    # Mature observation where both windows are available.
    idx = 70

    expected_13 = (
        sku.loc[
            idx - 13:
            idx - 1,
            "demand",
        ]
        .mean()
    )

    expected_52 = (
        sku.loc[
            idx - 52:
            idx - 1,
            "demand",
        ]
        .mean()
    )

    expected_ratio = (
        expected_13
        / expected_52
    )

    actual_ratio = (
        sku.loc[
            idx,
            "level_ratio_13_52",
        ]
    )

    assert np.isclose(
        actual_ratio,
        expected_ratio,
    )


def test_yoy_delta_is_previous_week_vs_annual_lag():
    """
    Current yoy_delta definition is:

        demand_lag_1 - demand_lag_52

    Both values must be historical relative to the target row.
    """

    panel = make_panel(
        n_weeks=100
    )

    featured = add_forecasting_features(
        panel
    )

    sku = (
        featured.loc[
            featured["sku_id"].eq(
                "SKU_A"
            )
        ]
        .sort_values(
            "week_start"
        )
        .reset_index(
            drop=True
        )
    )

    idx = 70

    expected = (
        sku.loc[
            idx - 1,
            "demand",
        ]
        -
        sku.loc[
            idx - 52,
            "demand",
        ]
    )

    actual = (
        sku.loc[
            idx,
            "yoy_delta",
        ]
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_mature_regime_features_have_no_missing_values():
    """
    Missing values are expected at the start of each series because
    52 weeks of history do not yet exist.

    After enough history has accumulated, every regime feature should
    be available.
    """

    panel = make_panel(
        n_weeks=120
    )

    featured = add_forecasting_features(
        panel
    )

    mature = (
        featured
        .groupby(
            "sku_id",
            group_keys=False,
        )
        .tail(60)
    )

    assert mature[
        TREND_FEATURES
    ].notna().all().all()


# ============================================================
# BASELINE FORECAST TESTS
# ============================================================


def test_naive_uses_last_observed_history_value():
    """
    Fixed-origin naive forecasting must use only the final observed
    demand from the training/history window.

    It must not read actual values from future weeks.
    """

    panel = make_panel(
        n_weeks=80
    )

    weeks = (
        panel["week_start"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    history_weeks = set(
        weeks[:-5]
    )

    future_weeks = set(
        weeks[-5:]
    )

    history = panel.loc[
        panel["week_start"].isin(
            history_weeks
        )
    ].copy()

    future = panel.loc[
        panel["week_start"].isin(
            future_weeks
        )
    ].copy()

    output = forecast_naive(
        history=history,
        future=future,
        split_name="test",
    )

    for (
        sku_id,
        sku_predictions,
    ) in output.groupby(
        "sku_id"
    ):

        expected = (
            history.loc[
                history["sku_id"].eq(
                    sku_id
                )
            ]
            .sort_values(
                "week_start"
            )
            ["demand"]
            .iloc[-1]
        )

        assert (
            sku_predictions[
                "prediction"
            ]
            .eq(expected)
            .all()
        )


def test_moving_average_first_forecast_is_historical_mean():
    """
    First recursive MA forecast must equal the average of the last
    observed historical values.
    """

    panel = make_panel(
        n_weeks=80
    )

    weeks = (
        panel["week_start"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    history = panel.loc[
        panel["week_start"].isin(
            set(
                weeks[:-3]
            )
        )
    ].copy()

    future = panel.loc[
        panel["week_start"].isin(
            set(
                weeks[-3:]
            )
        )
    ].copy()

    output = forecast_moving_average(
        history=history,
        future=future,
        split_name="test",
        window=4,
    )

    sku_id = "SKU_A"

    expected = (
        history.loc[
            history["sku_id"].eq(
                sku_id
            )
        ]
        .sort_values(
            "week_start"
        )
        ["demand"]
        .tail(4)
        .mean()
    )

    first_prediction = (
        output.loc[
            output["sku_id"].eq(
                sku_id
            )
        ]
        .sort_values(
            "week_start"
        )
        ["prediction"]
        .iloc[0]
    )

    assert np.isclose(
        first_prediction,
        expected,
    )


def test_moving_average_is_recursive():
    """
    Second moving-average forecast must use the first prediction,
    rather than the true first held-out observation.
    """

    panel = make_panel(
        n_weeks=80
    )

    weeks = (
        panel["week_start"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    history = panel.loc[
        panel["week_start"].isin(
            set(
                weeks[:-3]
            )
        )
    ].copy()

    future = panel.loc[
        panel["week_start"].isin(
            set(
                weeks[-3:]
            )
        )
    ].copy()

    output = forecast_moving_average(
        history=history,
        future=future,
        split_name="test",
        window=4,
    )

    sku_id = "SKU_A"

    historical_values = (
        history.loc[
            history["sku_id"].eq(
                sku_id
            )
        ]
        .sort_values(
            "week_start"
        )
        ["demand"]
        .astype(float)
        .tolist()
    )

    expected_first = np.mean(
        historical_values[
            -4:
        ]
    )

    recursive_history = (
        historical_values
        + [
            expected_first
        ]
    )

    expected_second = np.mean(
        recursive_history[
            -4:
        ]
    )

    sku_predictions = (
        output.loc[
            output["sku_id"].eq(
                sku_id
            )
        ]
        .sort_values(
            "week_start"
        )
        ["prediction"]
        .tolist()
    )

    assert np.isclose(
        sku_predictions[0],
        expected_first,
    )

    assert np.isclose(
        sku_predictions[1],
        expected_second,
    )


def test_seasonal_naive_uses_52_week_history():
    """
    For the first forecast horizon, seasonal naive should use demand
    exactly 52 weeks before the target.
    """

    panel = make_panel(
        n_weeks=100
    )

    weeks = (
        panel["week_start"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    history = panel.loc[
        panel["week_start"].isin(
            set(
                weeks[:-5]
            )
        )
    ].copy()

    future = panel.loc[
        panel["week_start"].isin(
            set(
                weeks[-5:]
            )
        )
    ].copy()

    output = forecast_seasonal_naive(
        history=history,
        future=future,
        split_name="test",
        season_length=52,
    )

    sku_id = "SKU_A"

    sku_history = (
        history.loc[
            history["sku_id"].eq(
                sku_id
            )
        ]
        .sort_values(
            "week_start"
        )
        ["demand"]
        .tolist()
    )

    expected_first = (
        sku_history[-52]
    )

    actual_first = (
        output.loc[
            output["sku_id"].eq(
                sku_id
            )
        ]
        .sort_values(
            "week_start"
        )
        ["prediction"]
        .iloc[0]
    )

    assert actual_first == expected_first


# ============================================================
# ROLLING-ORIGIN VALIDATION TESTS
# ============================================================


def test_rolling_validation_fold_count():
    """
    Requested number of expanding-window validation folds must be
    returned.
    """

    panel = make_panel(
        n_weeks=170
    )

    folds, development, test = (
        build_rolling_validation_folds(
            panel=panel,
            horizon=13,
            n_folds=3,
            test_weeks=13,
        )
    )

    assert len(folds) == 3


def test_rolling_validation_horizon_lengths():
    """
    Every validation fold must contain exactly 13 weeks.
    Final test must also contain exactly 13 weeks.
    """

    panel = make_panel(
        n_weeks=170
    )

    folds, development, test = (
        build_rolling_validation_folds(
            panel=panel,
            horizon=13,
            n_folds=3,
            test_weeks=13,
        )
    )

    for (
        train,
        validation,
        fold_name,
    ) in folds:

        assert (
            validation[
                "week_start"
            ].nunique()
            == 13
        )

    assert (
        test[
            "week_start"
        ].nunique()
        == 13
    )


def test_rolling_validation_is_temporally_ordered():
    """
    No validation observation may occur before or during its
    training history.
    """

    panel = make_panel(
        n_weeks=170
    )

    folds, development, test = (
        build_rolling_validation_folds(
            panel=panel,
            horizon=13,
            n_folds=3,
            test_weeks=13,
        )
    )

    for (
        train,
        validation,
        fold_name,
    ) in folds:

        assert (
            train[
                "week_start"
            ].max()
            <
            validation[
                "week_start"
            ].min()
        )

    assert (
        development[
            "week_start"
        ].max()
        <
        test[
            "week_start"
        ].min()
    )


def test_rolling_training_window_expands():
    """
    Expanding-window CV means each subsequent fold has more training
    history than the preceding fold.
    """

    panel = make_panel(
        n_weeks=170
    )

    folds, _, _ = (
        build_rolling_validation_folds(
            panel=panel,
            horizon=13,
            n_folds=3,
            test_weeks=13,
        )
    )

    training_lengths = [
        train[
            "week_start"
        ].nunique()
        for (
            train,
            _,
            _,
        ) in folds
    ]

    assert (
        training_lengths[0]
        <
        training_lengths[1]
        <
        training_lengths[2]
    )

    assert (
        training_lengths[1]
        - training_lengths[0]
        == 13
    )

    assert (
        training_lengths[2]
        - training_lengths[1]
        == 13
    )
    # ============================================================
# DIRECT MULTI-HORIZON LIGHTGBM TESTS
# ============================================================


def test_direct_training_frame_contains_all_horizons():
    """
    Direct training data must represent every requested forecast horizon.
    """

    panel = make_panel(
        n_weeks=120
    )

    training = (
        make_direct_training_frame(
            history=panel,
            max_horizon=13,
        )
    )

    assert set(
        training[
            "forecast_horizon"
        ].unique()
    ) == set(
        range(
            1,
            14,
        )
    )


def test_direct_training_history_is_aligned_to_origin():
    """
    For target index t and horizon h=3:

        origin = t - 3

    Therefore demand_lag_1 in the direct feature row should equal
    demand observed at that origin.
    """

    panel = make_panel(
        n_weeks=120
    )

    training = (
        make_direct_training_frame(
            history=panel,
            max_horizon=13,
        )
    )

    sku_panel = (
        panel.loc[
            panel[
                "sku_id"
            ].eq(
                "SKU_A"
            )
        ]
        .sort_values(
            "week_start"
        )
        .reset_index(
            drop=True
        )
    )

    target_idx = 80

    target_week = (
        sku_panel.loc[
            target_idx,
            "week_start",
        ]
    )

    row = (
        training.loc[
            training[
                "sku_id"
            ].eq(
                "SKU_A"
            )
            &
            training[
                "week_start"
            ].eq(
                target_week
            )
            &
            training[
                "forecast_horizon"
            ].eq(
                3
            )
        ]
        .iloc[0]
    )

    expected_latest_known_demand = (
        sku_panel.loc[
            target_idx - 3,
            "demand",
        ]
    )

    assert np.isclose(
        row[
            "demand_lag_1"
        ],
        expected_latest_known_demand,
    )


def test_direct_feature_contract_contains_horizon():
    """
    The direct model must explicitly receive forecast horizon.
    """

    features = (
        direct_model_feature_columns()
    )

    assert (
        "forecast_horizon"
        in features
    )


def test_direct_forecast_frame_ignores_future_actual_demand():
    """
    Direct inference features must not depend on realized future demand.

    Changing the future target values should not change the model inputs.
    """

    panel = make_panel(
        n_weeks=120
    )

    weeks = (
        panel[
            "week_start"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    history_weeks = set(
        weeks[:-13]
    )

    future_weeks = set(
        weeks[-13:]
    )

    history = panel.loc[
        panel[
            "week_start"
        ].isin(
            history_weeks
        )
    ].copy()

    future_a = panel.loc[
        panel[
            "week_start"
        ].isin(
            future_weeks
        )
    ].copy()

    future_b = (
        future_a.copy()
    )

    # Deliberately corrupt held-out actual demand.
    #
    # If forecasting features use it, the test will fail.
    future_b[
        "demand"
    ] = (
        future_b[
            "demand"
        ]
        * 1000.0
        + 99999.0
    )

    features_a = (
        make_direct_forecast_frame(
            history=history,
            future=future_a,
        )
    )

    features_b = (
        make_direct_forecast_frame(
            history=history,
            future=future_b,
        )
    )

    cols = (
        direct_model_feature_columns()
    )

    pd.testing.assert_frame_equal(
        features_a[
            cols
        ].reset_index(
            drop=True
        ),
        features_b[
            cols
        ].reset_index(
            drop=True
        ),
    )