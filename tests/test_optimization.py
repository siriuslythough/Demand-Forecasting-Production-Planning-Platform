from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from new_demfor_planopti.optimization.contracts import (
    PlanningConfig,
    validate_capacity_frame,
    validate_forecast_frame,
    validate_planning_inputs,
    validate_sku_parameters,
)

from new_demfor_planopti.optimization.templates import (
    PlanningTemplateConfig,
    make_default_capacity,
    make_default_planning_inputs,
    make_default_sku_parameters,
    validate_template_config,
)

from new_demfor_planopti.optimization.solver import (
    build_production_model,
    get_highs_solver,
    solve_production_plan,
)

from new_demfor_planopti.optimization.reporting import (
    build_planning_report,
    summarize_capacity,
    summarize_costs,
    summarize_skus,
)

from new_demfor_planopti.optimization.scenarios import (
    build_jit_baseline,
    compare_to_jit_baseline,
    run_capacity_sensitivity,
    summarize_plan_metrics,
)


# ============================================================
# SYNTHETIC OPTIMIZATION INPUTS
# ============================================================


def make_forecast(
    n_skus: int = 2,
    horizon: int = 13,
) -> pd.DataFrame:
    """
    Balanced SKU-week forecast panel.

    One row per:

        sku_id × forecast_horizon
    """

    weeks = pd.date_range(
        "2026-01-03",
        periods=horizon,
        freq="7D",
    )

    rows = []

    for sku_idx in range(n_skus):

        sku_id = f"SKU_{sku_idx + 1}"

        for horizon_idx, week in enumerate(
            weeks,
            start=1,
        ):

            rows.append(
                {
                    "week_start": week,
                    "sku_id": sku_id,
                    "forecast_demand": (
                        100.0
                        + 10.0 * sku_idx
                        + float(horizon_idx)
                    ),
                    "forecast_horizon": horizon_idx,
                }
            )

    return pd.DataFrame(rows)


def make_sku_parameters(
    n_skus: int = 2,
) -> pd.DataFrame:
    """
    Valid SKU-level planning assumptions.
    """

    rows = []

    for sku_idx in range(n_skus):

        rows.append(
            {
                "sku_id": f"SKU_{sku_idx + 1}",
                "initial_inventory": 20.0,
                "initial_backlog": 0.0,
                "unit_production_cost": (
                    5.0
                    + sku_idx
                ),
                "unit_holding_cost": 0.50,
                "unit_shortage_cost": 15.0,
                "processing_time": (
                    1.0
                    + 0.25 * sku_idx
                ),
                "max_production_per_week": 200.0,
            }
        )

    return pd.DataFrame(rows)


def make_capacity(
    horizon: int = 13,
) -> pd.DataFrame:
    """
    Valid aggregate weekly production capacity.
    """

    weeks = pd.date_range(
        "2026-01-03",
        periods=horizon,
        freq="7D",
    )

    return pd.DataFrame(
        {
            "week_start": weeks,
            "production_capacity": (
                np.repeat(
                    400.0,
                    horizon,
                )
            ),
        }
    )


def make_config(
    horizon: int = 13,
) -> PlanningConfig:

    return PlanningConfig(
        planning_horizon=horizon,
    )


# ============================================================
# FORECAST CONTRACT TESTS
# ============================================================


def test_valid_forecast_passes():

    forecast = make_forecast()

    clean = validate_forecast_frame(
        forecast=forecast,
        config=make_config(),
    )

    assert len(clean) == 26

    assert clean["sku_id"].nunique() == 2

    assert (
        clean["forecast_horizon"].nunique()
        == 13
    )


def test_forecast_demand_must_be_nonnegative():

    forecast = make_forecast()

    forecast.loc[
        forecast.index[0],
        "forecast_demand",
    ] = -10.0

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):

        validate_forecast_frame(
            forecast=forecast,
            config=make_config(),
        )


def test_tiny_negative_forecast_is_clipped():

    forecast = make_forecast()

    forecast.loc[
        forecast.index[0],
        "forecast_demand",
    ] = -1e-12

    clean = validate_forecast_frame(
        forecast=forecast,
        config=make_config(),
    )

    assert (
        clean["forecast_demand"].min()
        >= 0.0
    )


def test_duplicate_forecast_rows_are_rejected():

    forecast = make_forecast()

    duplicate = (
        forecast.iloc[
            [0]
        ]
        .copy()
    )

    forecast = pd.concat(
        [
            forecast,
            duplicate,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="exactly one row",
    ):

        validate_forecast_frame(
            forecast=forecast,
            config=make_config(),
        )


def test_forecast_requires_complete_horizon():

    forecast = make_forecast()

    forecast = forecast.loc[
        ~(
            forecast["sku_id"].eq(
                "SKU_1"
            )
            &
            forecast[
                "forecast_horizon"
            ].eq(13)
        )
    ].copy()

    with pytest.raises(
        ValueError
    ):

        validate_forecast_frame(
            forecast=forecast,
            config=make_config(),
        )


def test_forecast_rejects_wrong_horizon_values():

    forecast = make_forecast()

    forecast.loc[
        forecast[
            "forecast_horizon"
        ].eq(13),
        "forecast_horizon",
    ] = 14

    with pytest.raises(
        ValueError,
        match="Forecast horizons",
    ):

        validate_forecast_frame(
            forecast=forecast,
            config=make_config(),
        )


def test_forecast_missing_required_column():

    forecast = (
        make_forecast()
        .drop(
            columns=[
                "forecast_demand",
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):

        validate_forecast_frame(
            forecast=forecast,
            config=make_config(),
        )


# ============================================================
# SKU PARAMETER CONTRACT TESTS
# ============================================================


def test_valid_sku_parameters_pass():

    parameters = (
        make_sku_parameters()
    )

    clean = validate_sku_parameters(
        parameters
    )

    assert len(clean) == 2

    assert (
        clean["sku_id"].nunique()
        == 2
    )


def test_negative_production_cost_is_rejected():

    parameters = (
        make_sku_parameters()
    )

    parameters.loc[
        0,
        "unit_production_cost",
    ] = -1.0

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):

        validate_sku_parameters(
            parameters
        )


def test_processing_time_must_be_positive():

    parameters = (
        make_sku_parameters()
    )

    parameters.loc[
        0,
        "processing_time",
    ] = 0.0

    with pytest.raises(
        ValueError,
        match="processing_time",
    ):

        validate_sku_parameters(
            parameters
        )


def test_shortage_cost_must_be_positive():

    parameters = (
        make_sku_parameters()
    )

    parameters.loc[
        0,
        "unit_shortage_cost",
    ] = 0.0

    with pytest.raises(
        ValueError,
        match="unit_shortage_cost",
    ):

        validate_sku_parameters(
            parameters
        )


def test_max_production_must_be_positive():

    parameters = (
        make_sku_parameters()
    )

    parameters.loc[
        0,
        "max_production_per_week",
    ] = 0.0

    with pytest.raises(
        ValueError,
        match="max_production_per_week",
    ):

        validate_sku_parameters(
            parameters
        )


def test_duplicate_sku_parameters_are_rejected():

    parameters = (
        make_sku_parameters()
    )

    duplicate = (
        parameters.iloc[
            [0]
        ]
        .copy()
    )

    parameters = pd.concat(
        [
            parameters,
            duplicate,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="one row per SKU",
    ):

        validate_sku_parameters(
            parameters
        )


def test_nonfinite_parameter_is_rejected():

    parameters = (
        make_sku_parameters()
    )

    parameters.loc[
        0,
        "unit_holding_cost",
    ] = np.inf

    with pytest.raises(
        ValueError,
        match="finite",
    ):

        validate_sku_parameters(
            parameters
        )


# ============================================================
# CAPACITY CONTRACT TESTS
# ============================================================


def test_valid_capacity_passes():

    capacity = make_capacity()

    clean = validate_capacity_frame(
        capacity
    )

    assert len(clean) == 13

    assert (
        clean[
            "production_capacity"
        ].min()
        >= 0.0
    )


def test_negative_capacity_is_rejected():

    capacity = make_capacity()

    capacity.loc[
        0,
        "production_capacity",
    ] = -100.0

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):

        validate_capacity_frame(
            capacity
        )


def test_duplicate_capacity_week_is_rejected():

    capacity = make_capacity()

    duplicate = (
        capacity.iloc[
            [0]
        ]
        .copy()
    )

    capacity = pd.concat(
        [
            capacity,
            duplicate,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="one row per planning week",
    ):

        validate_capacity_frame(
            capacity
        )


# ============================================================
# CROSS-FRAME CONTRACT TESTS
# ============================================================


def test_complete_planning_inputs_pass():

    forecast = make_forecast()

    parameters = (
        make_sku_parameters()
    )

    capacity = make_capacity()

    (
        forecast_clean,
        parameters_clean,
        capacity_clean,
    ) = validate_planning_inputs(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_config(),
    )

    assert len(
        forecast_clean
    ) == 26

    assert len(
        parameters_clean
    ) == 2

    assert len(
        capacity_clean
    ) == 13


def test_missing_sku_parameters_are_rejected():

    forecast = make_forecast()

    parameters = (
        make_sku_parameters()
        .loc[
            lambda df:
            ~df[
                "sku_id"
            ].eq(
                "SKU_2"
            )
        ]
        .copy()
    )

    capacity = make_capacity()

    with pytest.raises(
        ValueError,
        match="Missing production parameters",
    ):

        validate_planning_inputs(
            forecast=forecast,
            sku_parameters=parameters,
            capacity=capacity,
            config=make_config(),
        )


def test_missing_capacity_week_is_rejected():

    forecast = make_forecast()

    parameters = (
        make_sku_parameters()
    )

    capacity = (
        make_capacity()
        .iloc[:-1]
        .copy()
    )

    with pytest.raises(
        ValueError,
        match="Missing production-capacity",
    ):

        validate_planning_inputs(
            forecast=forecast,
            sku_parameters=parameters,
            capacity=capacity,
            config=make_config(),
        )


def test_extra_sku_parameters_are_filtered():

    forecast = make_forecast(
        n_skus=2
    )

    parameters = (
        make_sku_parameters(
            n_skus=3
        )
    )

    capacity = make_capacity()

    (
        _,
        parameters_clean,
        _,
    ) = validate_planning_inputs(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_config(),
    )

    assert set(
        parameters_clean[
            "sku_id"
        ]
    ) == {
        "SKU_1",
        "SKU_2",
    }


def test_extra_capacity_weeks_are_filtered():

    forecast = make_forecast(
        horizon=13
    )

    parameters = (
        make_sku_parameters()
    )

    capacity = make_capacity(
        horizon=15
    )

    (
        _,
        _,
        capacity_clean,
    ) = validate_planning_inputs(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_config(),
    )

    assert len(
        capacity_clean
    ) == 13


def test_forecast_panel_must_be_balanced():

    forecast = make_forecast()

    # Remove one SKU-week row and replace its
    # horizon number elsewhere so the initial
    # horizon-set validation does not hide the
    # cross-frame balance requirement.
    forecast = forecast.loc[
        ~(
            forecast[
                "sku_id"
            ].eq(
                "SKU_2"
            )
            &
            forecast[
                "forecast_horizon"
            ].eq(7)
        )
    ].copy()

    parameters = (
        make_sku_parameters()
    )

    capacity = make_capacity()

    with pytest.raises(
        ValueError
    ):

        validate_planning_inputs(
            forecast=forecast,
            sku_parameters=parameters,
            capacity=capacity,
            config=make_config(),
        )
        # ============================================================
# PLANNING TEMPLATE TESTS
# ============================================================


def test_default_sku_parameters_have_expected_shape():
    """
    Default parameter generation should produce exactly one row
    per forecast SKU.
    """

    forecast = make_forecast(
        n_skus=2,
        horizon=13,
    )

    parameters = (
        make_default_sku_parameters(
            forecast=forecast,
            planning_config=make_config(),
        )
    )

    assert len(parameters) == 2

    assert set(
        parameters["sku_id"]
    ) == {
        "SKU_1",
        "SKU_2",
    }


def test_default_initial_inventory_uses_mean_demand():
    """
    Default initial inventory is defined as:

        mean forecast demand
        × initial_inventory_weeks
    """

    forecast = make_forecast(
        n_skus=2,
        horizon=13,
    )

    template_config = (
        PlanningTemplateConfig(
            initial_inventory_weeks=0.50,
        )
    )

    parameters = (
        make_default_sku_parameters(
            forecast=forecast,
            planning_config=make_config(),
            template_config=template_config,
        )
    )

    sku_1_mean_demand = (
        forecast.loc[
            forecast["sku_id"].eq(
                "SKU_1"
            ),
            "forecast_demand",
        ]
        .mean()
    )

    expected_inventory = (
        sku_1_mean_demand
        * 0.50
    )

    actual_inventory = (
        parameters.loc[
            parameters["sku_id"].eq(
                "SKU_1"
            ),
            "initial_inventory",
        ]
        .iloc[0]
    )

    assert np.isclose(
        actual_inventory,
        expected_inventory,
        atol=1e-4,
    )


def test_default_holding_cost_uses_production_cost_rate():
    """
    Holding cost should equal:

        unit production cost
        × weekly holding rate
    """

    forecast = make_forecast()

    template_config = (
        PlanningTemplateConfig(
            unit_production_cost=10.0,
            weekly_holding_rate=0.03,
        )
    )

    parameters = (
        make_default_sku_parameters(
            forecast=forecast,
            planning_config=make_config(),
            template_config=template_config,
        )
    )

    assert np.allclose(
        parameters[
            "unit_holding_cost"
        ],
        0.30,
    )


def test_default_shortage_cost_uses_multiplier():
    """
    Shortage cost should equal:

        production cost
        × shortage multiplier
    """

    forecast = make_forecast()

    template_config = (
        PlanningTemplateConfig(
            unit_production_cost=7.0,
            shortage_cost_multiplier=5.0,
        )
    )

    parameters = (
        make_default_sku_parameters(
            forecast=forecast,
            planning_config=make_config(),
            template_config=template_config,
        )
    )

    assert np.allclose(
        parameters[
            "unit_shortage_cost"
        ],
        35.0,
    )


def test_default_max_production_covers_peak_demand():
    """
    Default SKU production ceilings should exceed the SKU's
    largest forecast demand when multiplier > 1.
    """

    forecast = make_forecast()

    template_config = (
        PlanningTemplateConfig(
            max_production_multiplier=1.50,
        )
    )

    parameters = (
        make_default_sku_parameters(
            forecast=forecast,
            planning_config=make_config(),
            template_config=template_config,
        )
    )

    peak_demand = (
        forecast
        .groupby(
            "sku_id"
        )[
            "forecast_demand"
        ]
        .max()
    )

    parameter_lookup = (
        parameters
        .set_index(
            "sku_id"
        )
    )

    for sku_id in peak_demand.index:

        assert (
            parameter_lookup.loc[
                sku_id,
                "max_production_per_week",
            ]
            >=
            peak_demand.loc[
                sku_id
            ]
        )


def test_default_capacity_contains_all_planning_weeks():
    """
    Capacity template must cover the complete planning horizon.
    """

    forecast = make_forecast(
        horizon=13
    )

    parameters = (
        make_default_sku_parameters(
            forecast=forecast,
            planning_config=make_config(),
        )
    )

    capacity = (
        make_default_capacity(
            forecast=forecast,
            sku_parameters=parameters,
            planning_config=make_config(),
        )
    )

    assert len(capacity) == 13

    assert (
        capacity[
            "week_start"
        ].nunique()
        == 13
    )


def test_default_capacity_is_constant():
    """
    Default plant capacity is deliberately constant across weeks.

    This creates an intertemporal production-planning problem instead
    of simply matching capacity to weekly demand.
    """

    forecast = make_forecast()

    parameters = (
        make_default_sku_parameters(
            forecast=forecast,
            planning_config=make_config(),
        )
    )

    capacity = (
        make_default_capacity(
            forecast=forecast,
            sku_parameters=parameters,
            planning_config=make_config(),
        )
    )

    assert (
        capacity[
            "production_capacity"
        ].nunique()
        == 1
    )


def test_default_capacity_matches_average_workload():
    """
    Capacity should equal:

        mean weekly forecast workload
        × plant_capacity_multiplier

    where:

        workload
        =
        forecast demand
        × processing time
    """

    forecast = make_forecast()

    parameters = (
        make_default_sku_parameters(
            forecast=forecast,
            planning_config=make_config(),
        )
    )

    template_config = (
        PlanningTemplateConfig(
            plant_capacity_multiplier=1.05,
        )
    )

    capacity = (
        make_default_capacity(
            forecast=forecast,
            sku_parameters=parameters,
            planning_config=make_config(),
            template_config=template_config,
        )
    )

    workload = (
        forecast
        .merge(
            parameters[
                [
                    "sku_id",
                    "processing_time",
                ]
            ],
            on="sku_id",
            how="left",
        )
    )

    workload[
        "required_capacity"
    ] = (
        workload[
            "forecast_demand"
        ]
        *
        workload[
            "processing_time"
        ]
    )

    average_weekly_workload = (
        workload
        .groupby(
            "week_start"
        )[
            "required_capacity"
        ]
        .sum()
        .mean()
    )

    expected_capacity = (
        average_weekly_workload
        * 1.05
    )

    actual_capacity = (
        capacity[
            "production_capacity"
        ]
        .iloc[0]
    )

    assert np.isclose(
        actual_capacity,
        expected_capacity,
        atol=1e-4,
    )


def test_capacity_uses_processing_time():
    """
    Capacity generation must respect SKU-specific processing-time
    requirements rather than treating all units as equally costly
    in plant capacity.
    """

    forecast = make_forecast(
        n_skus=2,
        horizon=13,
    )

    parameters = (
        make_default_sku_parameters(
            forecast=forecast,
            planning_config=make_config(),
        )
    )

    # Make SKU_2 twice as resource intensive.
    parameters.loc[
        parameters[
            "sku_id"
        ].eq(
            "SKU_2"
        ),
        "processing_time",
    ] = 2.0

    template_config = (
        PlanningTemplateConfig(
            plant_capacity_multiplier=1.0,
        )
    )

    capacity = (
        make_default_capacity(
            forecast=forecast,
            sku_parameters=parameters,
            planning_config=make_config(),
            template_config=template_config,
        )
    )

    expected = (
        forecast
        .merge(
            parameters[
                [
                    "sku_id",
                    "processing_time",
                ]
            ],
            on="sku_id",
            how="left",
        )
        .assign(
            required_capacity=lambda df:
            (
                df[
                    "forecast_demand"
                ]
                *
                df[
                    "processing_time"
                ]
            )
        )
        .groupby(
            "week_start"
        )[
            "required_capacity"
        ]
        .sum()
        .mean()
    )

    assert np.isclose(
        capacity[
            "production_capacity"
        ].iloc[0],
        expected,
        atol=1e-4,
    )


def test_default_planning_inputs_are_deterministic():
    """
    Running the same template twice with identical inputs must produce
    identical planning assumptions.
    """

    forecast = make_forecast()

    result_a = (
        make_default_planning_inputs(
            forecast=forecast,
            planning_config=make_config(),
        )
    )

    result_b = (
        make_default_planning_inputs(
            forecast=forecast,
            planning_config=make_config(),
        )
    )

    for frame_a, frame_b in zip(
        result_a,
        result_b,
    ):

        pd.testing.assert_frame_equal(
            frame_a,
            frame_b,
        )


def test_custom_template_parameters_propagate():
    """
    Dashboard/user-supplied scenario assumptions must propagate into
    the generated optimization parameters.
    """

    forecast = make_forecast()

    template_config = (
        PlanningTemplateConfig(
            initial_inventory_weeks=1.0,
            initial_backlog=3.0,
            unit_production_cost=8.0,
            weekly_holding_rate=0.05,
            shortage_cost_multiplier=6.0,
            processing_time=1.25,
            max_production_multiplier=2.0,
            plant_capacity_multiplier=1.10,
        )
    )

    (
        _,
        parameters,
        capacity,
    ) = make_default_planning_inputs(
        forecast=forecast,
        planning_config=make_config(),
        template_config=template_config,
    )

    assert np.allclose(
        parameters[
            "initial_backlog"
        ],
        3.0,
    )

    assert np.allclose(
        parameters[
            "unit_production_cost"
        ],
        8.0,
    )

    assert np.allclose(
        parameters[
            "unit_holding_cost"
        ],
        0.40,
    )

    assert np.allclose(
        parameters[
            "unit_shortage_cost"
        ],
        48.0,
    )

    assert np.allclose(
        parameters[
            "processing_time"
        ],
        1.25,
    )

    assert (
        capacity[
            "production_capacity"
        ].min()
        > 0.0
    )


def test_invalid_template_multiplier_is_rejected():
    """
    Invalid scenario configuration should fail before any optimizer
    input is generated.
    """

    config = (
        PlanningTemplateConfig(
            plant_capacity_multiplier=0.0,
        )
    )

    with pytest.raises(
        ValueError,
        match="plant_capacity_multiplier",
    ):

        validate_template_config(
            config
        )
        # ============================================================
# PRODUCTION-PLANNING SOLVER TEST HELPERS
# ============================================================


def make_solver_forecast(
    demands: dict[str, list[float]],
) -> pd.DataFrame:
    """
    Create a compact forecast panel for deterministic LP tests.

    Example
    -------
    {
        "SKU_A": [10, 12],
        "SKU_B": [8, 9],
    }
    """

    horizon = len(
        next(
            iter(
                demands.values()
            )
        )
    )

    weeks = pd.date_range(
        "2026-01-03",
        periods=horizon,
        freq="7D",
    )

    rows = []

    for (
        sku_id,
        sku_demand,
    ) in demands.items():

        assert (
            len(sku_demand)
            == horizon
        )

        for (
            forecast_horizon,
            (
                week,
                demand,
            ),
        ) in enumerate(
            zip(
                weeks,
                sku_demand,
            ),
            start=1,
        ):

            rows.append(
                {
                    "week_start":
                        week,

                    "sku_id":
                        sku_id,

                    "forecast_demand":
                        float(
                            demand
                        ),

                    "forecast_horizon":
                        forecast_horizon,
                }
            )

    return pd.DataFrame(
        rows
    )


def make_solver_parameters(
    sku_ids: list[str],
    *,
    initial_inventory: float = 0.0,
    initial_backlog: float = 0.0,
    production_cost: float = 5.0,
    holding_cost: float = 0.5,
    shortage_cost: float = 20.0,
    processing_time: float = 1.0,
    max_production: float = 100.0,
) -> pd.DataFrame:
    """
    Create uniform SKU parameters for deterministic solver tests.
    """

    return pd.DataFrame(
        {
            "sku_id":
                sku_ids,

            "initial_inventory":
                initial_inventory,

            "initial_backlog":
                initial_backlog,

            "unit_production_cost":
                production_cost,

            "unit_holding_cost":
                holding_cost,

            "unit_shortage_cost":
                shortage_cost,

            "processing_time":
                processing_time,

            "max_production_per_week":
                max_production,
        }
    )


def make_solver_capacity(
    capacities: list[float],
) -> pd.DataFrame:
    """
    Create week-level plant capacity for deterministic solver tests.
    """

    weeks = pd.date_range(
        "2026-01-03",
        periods=len(
            capacities
        ),
        freq="7D",
    )

    return pd.DataFrame(
        {
            "week_start":
                weeks,

            "production_capacity":
                capacities,
        }
    )


def make_solver_config(
    horizon: int,
) -> PlanningConfig:

    return PlanningConfig(
        planning_horizon=horizon,
    )


# ============================================================
# SOLVER AVAILABILITY
# ============================================================


def test_highs_solver_is_available():
    """
    The project's configured optimization engine must be available
    through Pyomo.
    """

    solver = get_highs_solver()

    assert solver.available(
        exception_flag=False
    )


# ============================================================
# MODEL-CONSTRUCTION TESTS
# ============================================================


def test_production_model_has_expected_dimensions():
    """
    Two SKUs across three planning weeks should create:

        6 production variables
        6 inventory variables
        6 backlog variables
        6 inventory-balance constraints
        3 shared-capacity constraints
    """

    forecast = (
        make_solver_forecast(
            {
                "SKU_A": [
                    10,
                    10,
                    10,
                ],
                "SKU_B": [
                    8,
                    8,
                    8,
                ],
            }
        )
    )

    parameters = (
        make_solver_parameters(
            [
                "SKU_A",
                "SKU_B",
            ]
        )
    )

    capacity = (
        make_solver_capacity(
            [
                30,
                30,
                30,
            ]
        )
    )

    model = (
        build_production_model(
            forecast=forecast,
            sku_parameters=parameters,
            capacity=capacity,
            config=make_solver_config(
                horizon=3
            ),
        )
    )

    assert len(
        model.production
    ) == 6

    assert len(
        model.inventory
    ) == 6

    assert len(
        model.backlog
    ) == 6

    assert len(
        model.inventory_balance
    ) == 6

    assert len(
        model.capacity_constraint
    ) == 3


# ============================================================
# BASIC OPTIMAL-SOLUTION TEST
# ============================================================


def test_solver_meets_demand_when_capacity_is_sufficient():
    """
    With:

        zero initial inventory
        sufficient capacity
        positive holding cost
        expensive shortages

    the cheapest solution is to produce each week's demand in that
    week, leaving zero inventory and zero backlog.
    """

    forecast = (
        make_solver_forecast(
            {
                "SKU_A": [
                    10,
                    12,
                ],
            }
        )
    )

    parameters = (
        make_solver_parameters(
            [
                "SKU_A",
            ],
            initial_inventory=0.0,
            production_cost=5.0,
            holding_cost=0.5,
            shortage_cost=20.0,
            max_production=20.0,
        )
    )

    capacity = (
        make_solver_capacity(
            [
                20,
                20,
            ]
        )
    )

    result = (
        solve_production_plan(
            forecast=forecast,
            sku_parameters=parameters,
            capacity=capacity,
            config=make_solver_config(
                horizon=2
            ),
        )
    )

    plan = (
        result.plan
        .sort_values(
            "forecast_horizon"
        )
        .reset_index(
            drop=True
        )
    )

    assert (
        result.solver_status
        == "ok"
    )

    assert (
        result.termination_condition
        == "optimal"
    )

    assert np.allclose(
        plan[
            "production_quantity"
        ],
        [
            10.0,
            12.0,
        ],
    )

    assert np.allclose(
        plan[
            "ending_inventory"
        ],
        0.0,
        atol=1e-7,
    )

    assert np.allclose(
        plan[
            "ending_backlog"
        ],
        0.0,
        atol=1e-7,
    )


# ============================================================
# INITIAL INVENTORY TEST
# ============================================================


def test_solver_uses_initial_inventory_before_production():
    """
    If week-1 demand is already fully covered by initial inventory,
    producing units in week 1 would add unnecessary production and
    holding cost.

    The optimizer should consume inventory first.
    """

    forecast = (
        make_solver_forecast(
            {
                "SKU_A": [
                    10,
                    10,
                ],
            }
        )
    )

    parameters = (
        make_solver_parameters(
            [
                "SKU_A",
            ],
            initial_inventory=10.0,
            production_cost=5.0,
            holding_cost=1.0,
            shortage_cost=30.0,
            max_production=20.0,
        )
    )

    capacity = (
        make_solver_capacity(
            [
                20,
                20,
            ]
        )
    )

    result = (
        solve_production_plan(
            forecast=forecast,
            sku_parameters=parameters,
            capacity=capacity,
            config=make_solver_config(
                horizon=2
            ),
        )
    )

    plan = (
        result.plan
        .sort_values(
            "forecast_horizon"
        )
        .reset_index(
            drop=True
        )
    )

    assert np.isclose(
        plan.loc[
            0,
            "production_quantity",
        ],
        0.0,
        atol=1e-7,
    )

    assert np.isclose(
        plan.loc[
            0,
            "ending_inventory",
        ],
        0.0,
        atol=1e-7,
    )

    assert np.isclose(
        plan.loc[
            1,
            "production_quantity",
        ],
        10.0,
        atol=1e-7,
    )


# ============================================================
# INVENTORY PREBUILD TEST
# ============================================================


def test_solver_prebuilds_inventory_before_capacity_bottleneck():
    """
    Week 2 has zero production capacity.

    Therefore the optimizer must produce week-2 demand during week 1
    and carry it as inventory.

    This is the core intertemporal behavior that makes the production
    planning LP useful.
    """

    forecast = (
        make_solver_forecast(
            {
                "SKU_A": [
                    10,
                    10,
                ],
            }
        )
    )

    parameters = (
        make_solver_parameters(
            [
                "SKU_A",
            ],
            initial_inventory=0.0,
            production_cost=5.0,
            holding_cost=0.5,
            shortage_cost=50.0,
            max_production=20.0,
        )
    )

    capacity = (
        make_solver_capacity(
            [
                20,
                0,
            ]
        )
    )

    result = (
        solve_production_plan(
            forecast=forecast,
            sku_parameters=parameters,
            capacity=capacity,
            config=make_solver_config(
                horizon=2
            ),
        )
    )

    plan = (
        result.plan
        .sort_values(
            "forecast_horizon"
        )
        .reset_index(
            drop=True
        )
    )

    # Produce both weeks' demand during week 1.
    assert np.isclose(
        plan.loc[
            0,
            "production_quantity",
        ],
        20.0,
        atol=1e-7,
    )

    # Carry second week's requirement.
    assert np.isclose(
        plan.loc[
            0,
            "ending_inventory",
        ],
        10.0,
        atol=1e-7,
    )

    # No production possible in week 2.
    assert np.isclose(
        plan.loc[
            1,
            "production_quantity",
        ],
        0.0,
        atol=1e-7,
    )

    # Inventory serves demand.
    assert np.isclose(
        plan.loc[
            1,
            "ending_inventory",
        ],
        0.0,
        atol=1e-7,
    )

    assert np.isclose(
        plan.loc[
            1,
            "ending_backlog",
        ],
        0.0,
        atol=1e-7,
    )


# ============================================================
# INSUFFICIENT CAPACITY / BACKLOG TEST
# ============================================================


def test_solver_creates_backlog_when_capacity_is_insufficient():
    """
    Backlog keeps the LP feasible when available production capacity
    cannot satisfy all forecast demand.

    Demand:
        10 + 10 = 20

    Maximum capacity:
         5 +  5 = 10

    Final backlog must therefore be 10 units.
    """

    forecast = (
        make_solver_forecast(
            {
                "SKU_A": [
                    10,
                    10,
                ],
            }
        )
    )

    parameters = (
        make_solver_parameters(
            [
                "SKU_A",
            ],
            initial_inventory=0.0,
            production_cost=5.0,
            holding_cost=0.5,
            shortage_cost=30.0,
            max_production=20.0,
        )
    )

    capacity = (
        make_solver_capacity(
            [
                5,
                5,
            ]
        )
    )

    result = (
        solve_production_plan(
            forecast=forecast,
            sku_parameters=parameters,
            capacity=capacity,
            config=make_solver_config(
                horizon=2
            ),
        )
    )

    plan = (
        result.plan
        .sort_values(
            "forecast_horizon"
        )
        .reset_index(
            drop=True
        )
    )

    assert np.isclose(
        plan.loc[
            0,
            "ending_backlog",
        ],
        5.0,
        atol=1e-7,
    )

    assert np.isclose(
        plan.loc[
            1,
            "ending_backlog",
        ],
        10.0,
        atol=1e-7,
    )

    assert (
        result.termination_condition
        == "optimal"
    )


# ============================================================
# SHARED CAPACITY TEST
# ============================================================


def test_multiple_skus_share_plant_capacity():
    """
    Aggregate capacity must be respected across SKUs rather than
    independently for each SKU.
    """

    forecast = (
        make_solver_forecast(
            {
                "SKU_A": [
                    10,
                ],
                "SKU_B": [
                    10,
                ],
            }
        )
    )

    parameters = (
        make_solver_parameters(
            [
                "SKU_A",
                "SKU_B",
            ],
            shortage_cost=30.0,
            processing_time=1.0,
            max_production=20.0,
        )
    )

    capacity = (
        make_solver_capacity(
            [
                15,
            ]
        )
    )

    result = (
        solve_production_plan(
            forecast=forecast,
            sku_parameters=parameters,
            capacity=capacity,
            config=make_solver_config(
                horizon=1
            ),
        )
    )

    total_production = (
        result.plan[
            "production_quantity"
        ].sum()
    )

    total_backlog = (
        result.plan[
            "ending_backlog"
        ].sum()
    )

    assert np.isclose(
        total_production,
        15.0,
        atol=1e-7,
    )

    assert np.isclose(
        total_backlog,
        5.0,
        atol=1e-7,
    )

    assert (
        result.capacity_usage[
            "capacity_used"
        ].iloc[0]
        <=
        15.0
        + 1e-7
    )


# ============================================================
# CAPACITY PROCESSING-TIME TEST
# ============================================================


def test_solver_capacity_uses_processing_time():
    """
    Capacity is measured in processing units, not finished-product
    units.

    SKU_A:
        processing time = 1

    SKU_B:
        processing time = 2

    Therefore producing:

        5 A + 5 B

    consumes:

        5(1) + 5(2) = 15 capacity units.
    """

    forecast = (
        make_solver_forecast(
            {
                "SKU_A": [
                    5,
                ],
                "SKU_B": [
                    5,
                ],
            }
        )
    )

    parameters = pd.DataFrame(
        {
            "sku_id": [
                "SKU_A",
                "SKU_B",
            ],

            "initial_inventory": [
                0.0,
                0.0,
            ],

            "initial_backlog": [
                0.0,
                0.0,
            ],

            "unit_production_cost": [
                5.0,
                5.0,
            ],

            "unit_holding_cost": [
                0.5,
                0.5,
            ],

            "unit_shortage_cost": [
                50.0,
                50.0,
            ],

            "processing_time": [
                1.0,
                2.0,
            ],

            "max_production_per_week": [
                20.0,
                20.0,
            ],
        }
    )

    capacity = (
        make_solver_capacity(
            [
                15,
            ]
        )
    )

    result = (
        solve_production_plan(
            forecast=forecast,
            sku_parameters=parameters,
            capacity=capacity,
            config=make_solver_config(
                horizon=1
            ),
        )
    )

    assert np.isclose(
        result.capacity_usage.loc[
            0,
            "capacity_used",
        ],
        15.0,
        atol=1e-7,
    )

    assert np.isclose(
        result.capacity_usage.loc[
            0,
            "utilization_pct",
        ],
        100.0,
        atol=1e-7,
    )


# ============================================================
# ECONOMIC PRIORITIZATION TEST
# ============================================================


def test_higher_shortage_cost_sku_gets_capacity_priority():
    """
    When capacity is insufficient, the optimizer should allocate
    production toward the SKU with the larger shortage penalty.

    This verifies that the model is actually optimizing economics,
    rather than simply distributing capacity mechanically.
    """

    forecast = (
        make_solver_forecast(
            {
                "SKU_HIGH": [
                    10,
                ],
                "SKU_LOW": [
                    10,
                ],
            }
        )
    )

    parameters = pd.DataFrame(
        {
            "sku_id": [
                "SKU_HIGH",
                "SKU_LOW",
            ],

            "initial_inventory": [
                0.0,
                0.0,
            ],

            "initial_backlog": [
                0.0,
                0.0,
            ],

            "unit_production_cost": [
                5.0,
                5.0,
            ],

            "unit_holding_cost": [
                0.0,
                0.0,
            ],

            "unit_shortage_cost": [
                100.0,
                10.0,
            ],

            "processing_time": [
                1.0,
                1.0,
            ],

            "max_production_per_week": [
                20.0,
                20.0,
            ],
        }
    )

    capacity = (
        make_solver_capacity(
            [
                10,
            ]
        )
    )

    result = (
        solve_production_plan(
            forecast=forecast,
            sku_parameters=parameters,
            capacity=capacity,
            config=make_solver_config(
                horizon=1
            ),
        )
    )

    plan = (
        result.plan
        .set_index(
            "sku_id"
        )
    )

    assert np.isclose(
        plan.loc[
            "SKU_HIGH",
            "production_quantity",
        ],
        10.0,
        atol=1e-7,
    )

    assert np.isclose(
        plan.loc[
            "SKU_HIGH",
            "ending_backlog",
        ],
        0.0,
        atol=1e-7,
    )

    assert np.isclose(
        plan.loc[
            "SKU_LOW",
            "production_quantity",
        ],
        0.0,
        atol=1e-7,
    )

    assert np.isclose(
        plan.loc[
            "SKU_LOW",
            "ending_backlog",
        ],
        10.0,
        atol=1e-7,
    )


# ============================================================
# INVENTORY-BALANCE CONTRACT TEST
# ============================================================


def test_extracted_plan_satisfies_inventory_balance():
    """
    Independently reconstruct the stock-flow equation from the
    extracted solution.

    This verifies that the tidy output preserves the solved Pyomo
    mathematics correctly.
    """

    forecast = (
        make_solver_forecast(
            {
                "SKU_A": [
                    9,
                    14,
                    11,
                ],
            }
        )
    )

    parameters = (
        make_solver_parameters(
            [
                "SKU_A",
            ],
            initial_inventory=4.0,
            initial_backlog=0.0,
            shortage_cost=40.0,
            max_production=20.0,
        )
    )

    capacity = (
        make_solver_capacity(
            [
                20,
                10,
                20,
            ]
        )
    )

    result = (
        solve_production_plan(
            forecast=forecast,
            sku_parameters=parameters,
            capacity=capacity,
            config=make_solver_config(
                horizon=3
            ),
        )
    )

    plan = (
        result.plan
        .sort_values(
            "forecast_horizon"
        )
        .reset_index(
            drop=True
        )
    )

    previous_inventory = 4.0
    previous_backlog = 0.0

    for row in plan.itertuples(
        index=False
    ):

        expected_net_inventory = (
            previous_inventory
            -
            previous_backlog
            +
            row.production_quantity
            -
            row.forecast_demand
        )

        actual_net_inventory = (
            row.ending_inventory
            -
            row.ending_backlog
        )

        assert np.isclose(
            actual_net_inventory,
            expected_net_inventory,
            atol=1e-7,
        )

        previous_inventory = (
            row.ending_inventory
        )

        previous_backlog = (
            row.ending_backlog
        )


# ============================================================
# OBJECTIVE EXTRACTION TEST
# ============================================================


def test_extracted_cost_matches_objective():
    """
    The sum of SKU-week cost components must equal the Pyomo
    objective returned by the optimizer.
    """

    forecast = (
        make_solver_forecast(
            {
                "SKU_A": [
                    10,
                    15,
                ],
                "SKU_B": [
                    8,
                    12,
                ],
            }
        )
    )

    parameters = (
        make_solver_parameters(
            [
                "SKU_A",
                "SKU_B",
            ],
            production_cost=6.0,
            holding_cost=0.8,
            shortage_cost=25.0,
            max_production=30.0,
        )
    )

    capacity = (
        make_solver_capacity(
            [
                30,
                30,
            ]
        )
    )

    result = (
        solve_production_plan(
            forecast=forecast,
            sku_parameters=parameters,
            capacity=capacity,
            config=make_solver_config(
                horizon=2
            ),
        )
    )

    extracted_total = float(
        result.plan[
            "total_cost"
        ].sum()
    )

    assert np.isclose(
        extracted_total,
        result.objective_value,
        rtol=1e-7,
        atol=1e-6,
    )


# ============================================================
# SKU PRODUCTION-LIMIT TEST
# ============================================================


def test_sku_production_ceiling_is_respected():
    """
    SKU-specific production limits must bind even when plant capacity
    itself is abundant.
    """

    forecast = (
        make_solver_forecast(
            {
                "SKU_A": [
                    20,
                ],
            }
        )
    )

    parameters = (
        make_solver_parameters(
            [
                "SKU_A",
            ],
            shortage_cost=50.0,
            max_production=7.0,
        )
    )

    capacity = (
        make_solver_capacity(
            [
                100,
            ]
        )
    )

    result = (
        solve_production_plan(
            forecast=forecast,
            sku_parameters=parameters,
            capacity=capacity,
            config=make_solver_config(
                horizon=1
            ),
        )
    )

    row = (
        result.plan.iloc[0]
    )

    assert np.isclose(
        row[
            "production_quantity"
        ],
        7.0,
        atol=1e-7,
    )

    assert np.isclose(
        row[
            "ending_backlog"
        ],
        13.0,
        atol=1e-7,
    )
    # ============================================================
# OPTIMIZATION REPORTING TESTS
# ============================================================


def test_cost_summary_matches_solver_objective():
    """
    Reported cost components must reconcile exactly to the solved
    Pyomo objective.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [10, 12],
            "SKU_B": [8, 11],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
            "SKU_B",
        ],
        production_cost=6.0,
        holding_cost=0.5,
        shortage_cost=25.0,
        max_production=30.0,
    )

    capacity = make_solver_capacity(
        [
            30,
            30,
        ]
    )

    result = solve_production_plan(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=2
        ),
    )

    summary = summarize_costs(
        plan=result.plan,
        objective_value=result.objective_value,
    )

    total_row = summary.loc[
        summary[
            "cost_component"
        ].eq(
            "total"
        )
    ].iloc[0]

    assert np.isclose(
        total_row["cost"],
        result.objective_value,
        atol=1e-6,
    )

    component_total = (
        summary.loc[
            summary[
                "cost_component"
            ].ne(
                "total"
            ),
            "cost",
        ]
        .sum()
    )

    assert np.isclose(
        component_total,
        result.objective_value,
        atol=1e-6,
    )


def test_cost_component_shares_sum_to_100():
    """
    Production, holding, and shortage shares should sum to 100%
    whenever total cost is positive.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [10, 15],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
        ],
        production_cost=5.0,
        holding_cost=1.0,
        shortage_cost=20.0,
        max_production=20.0,
    )

    capacity = make_solver_capacity(
        [
            20,
            20,
        ]
    )

    result = solve_production_plan(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=2
        ),
    )

    summary = summarize_costs(
        plan=result.plan,
        objective_value=result.objective_value,
    )

    component_rows = summary.loc[
        summary[
            "cost_component"
        ].ne(
            "total"
        )
    ]

    assert np.isclose(
        component_rows[
            "share_of_total_pct"
        ].sum(),
        100.0,
        atol=1e-7,
    )


def test_capacity_summary_detects_constrained_week():
    """
    A week operating at full capacity should be classified as
    capacity-constrained.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [10],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
        ],
        shortage_cost=50.0,
        max_production=20.0,
    )

    capacity = make_solver_capacity(
        [
            10,
        ]
    )

    result = solve_production_plan(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=1
        ),
    )

    summary, detail = summarize_capacity(
        result.capacity_usage,
        constrained_threshold_pct=99.0,
    )

    assert bool(
        detail.loc[
            0,
            "is_capacity_constrained",
        ]
    )

    assert (
        summary.loc[
            0,
            "constrained_weeks",
        ]
        == 1
    )

    assert np.isclose(
        summary.loc[
            0,
            "peak_utilization_pct",
        ],
        100.0,
        atol=1e-7,
    )


def test_capacity_summary_detects_slack_week():
    """
    A week with excess capacity should retain positive slack and not
    be classified as constrained.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [5],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
        ],
        shortage_cost=50.0,
        max_production=20.0,
    )

    capacity = make_solver_capacity(
        [
            20,
        ]
    )

    result = solve_production_plan(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=1
        ),
    )

    summary, detail = summarize_capacity(
        result.capacity_usage,
        constrained_threshold_pct=99.0,
    )

    assert not bool(
        detail.loc[
            0,
            "is_capacity_constrained",
        ]
    )

    assert (
        detail.loc[
            0,
            "capacity_slack",
        ]
        > 0
    )

    assert (
        summary.loc[
            0,
            "constrained_weeks",
        ]
        == 0
    )


def test_sku_service_is_100_when_final_backlog_zero():
    """
    If all demand obligations are satisfied by horizon end, SKU service
    should equal 100%.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [10, 10],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
        ],
        shortage_cost=50.0,
        max_production=20.0,
    )

    capacity = make_solver_capacity(
        [
            20,
            20,
        ]
    )

    result = solve_production_plan(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=2
        ),
    )

    sku_summary = summarize_skus(
        plan=result.plan,
        sku_parameters=parameters,
    )

    row = sku_summary.iloc[0]

    assert np.isclose(
        row["final_backlog"],
        0.0,
        atol=1e-7,
    )

    assert np.isclose(
        row["service_pct"],
        100.0,
        atol=1e-7,
    )


def test_sku_service_reflects_remaining_backlog():
    """
    Remaining final backlog should lower the horizon service metric.

    Demand:
        10 + 10 = 20

    Production capacity:
         5 +  5 = 10

    Final backlog:
        10

    Service:
        1 - 10 / 20 = 50%
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [10, 10],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
        ],
        shortage_cost=30.0,
        max_production=20.0,
    )

    capacity = make_solver_capacity(
        [
            5,
            5,
        ]
    )

    result = solve_production_plan(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=2
        ),
    )

    sku_summary = summarize_skus(
        plan=result.plan,
        sku_parameters=parameters,
    )

    row = sku_summary.iloc[0]

    assert np.isclose(
        row["final_backlog"],
        10.0,
        atol=1e-7,
    )

    assert np.isclose(
        row["service_pct"],
        50.0,
        atol=1e-7,
    )


def test_sku_summary_uses_final_horizon_state():
    """
    final_inventory and final_backlog must come from the final planning
    period rather than from an average or maximum over the horizon.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [10, 10],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
        ],
        initial_inventory=10.0,
        shortage_cost=50.0,
        max_production=20.0,
    )

    capacity = make_solver_capacity(
        [
            20,
            20,
        ]
    )

    result = solve_production_plan(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=2
        ),
    )

    sku_summary = summarize_skus(
        plan=result.plan,
        sku_parameters=parameters,
    )

    final_plan_row = (
        result.plan
        .sort_values(
            "forecast_horizon"
        )
        .iloc[-1]
    )

    summary_row = (
        sku_summary.iloc[0]
    )

    assert np.isclose(
        summary_row[
            "final_inventory"
        ],
        final_plan_row[
            "ending_inventory"
        ],
        atol=1e-7,
    )

    assert np.isclose(
        summary_row[
            "final_backlog"
        ],
        final_plan_row[
            "ending_backlog"
        ],
        atol=1e-7,
    )


def test_sku_summary_costs_reconcile_to_plan():
    """
    SKU-level cost summaries should sum back to the detailed plan.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [10, 12],
            "SKU_B": [8, 9],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
            "SKU_B",
        ],
        production_cost=5.0,
        holding_cost=0.5,
        shortage_cost=25.0,
        max_production=30.0,
    )

    capacity = make_solver_capacity(
        [
            30,
            30,
        ]
    )

    result = solve_production_plan(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=2
        ),
    )

    sku_summary = summarize_skus(
        plan=result.plan,
        sku_parameters=parameters,
    )

    assert np.isclose(
        sku_summary[
            "total_cost"
        ].sum(),
        result.plan[
            "total_cost"
        ].sum(),
        atol=1e-6,
    )

    assert np.isclose(
        sku_summary[
            "total_forecast_demand"
        ].sum(),
        result.plan[
            "forecast_demand"
        ].sum(),
        atol=1e-7,
    )

    assert np.isclose(
        sku_summary[
            "total_production"
        ].sum(),
        result.plan[
            "production_quantity"
        ].sum(),
        atol=1e-7,
    )


def test_complete_planning_report_builds_all_artifacts():
    """
    Main reporting entry point should produce all dashboard-facing
    report tables.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [10, 12],
            "SKU_B": [8, 10],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
            "SKU_B",
        ],
        shortage_cost=30.0,
        max_production=30.0,
    )

    capacity = make_solver_capacity(
        [
            30,
            30,
        ]
    )

    result = solve_production_plan(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=2
        ),
    )

    report = build_planning_report(
        result=result,
        sku_parameters=parameters,
    )

    assert len(
        report.executive_summary
    ) == 1

    assert set(
        report.cost_summary[
            "cost_component"
        ]
    ) == {
        "production",
        "holding",
        "shortage",
        "total",
    }

    assert len(
        report.capacity_summary
    ) == 1

    assert len(
        report.capacity_detail
    ) == 2

    assert (
        report.sku_summary[
            "sku_id"
        ].nunique()
        == 2
    )


def test_executive_summary_reconciles_with_solution():
    """
    Executive KPIs must be derivable from the underlying optimization
    result rather than being independently computed inconsistently.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [10, 10],
            "SKU_B": [5, 5],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
            "SKU_B",
        ],
        shortage_cost=40.0,
        max_production=30.0,
    )

    capacity = make_solver_capacity(
        [
            30,
            30,
        ]
    )

    result = solve_production_plan(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=2
        ),
    )

    report = build_planning_report(
        result=result,
        sku_parameters=parameters,
    )

    executive = (
        report.executive_summary
        .iloc[0]
    )

    assert np.isclose(
        executive[
            "objective_value"
        ],
        result.objective_value,
        atol=1e-6,
    )

    assert np.isclose(
        executive[
            "total_forecast_demand"
        ],
        result.plan[
            "forecast_demand"
        ].sum(),
        atol=1e-7,
    )

    assert np.isclose(
        executive[
            "total_production"
        ],
        result.plan[
            "production_quantity"
        ].sum(),
        atol=1e-7,
    )

    assert (
        executive[
            "planning_weeks"
        ]
        == 2
    )

    assert (
        executive[
            "skus"
        ]
        == 2
    )


def test_portfolio_service_is_100_with_no_final_backlog():
    """
    Full fulfillment by the end of the horizon should produce 100%
    portfolio service.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [10, 10],
            "SKU_B": [8, 8],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
            "SKU_B",
        ],
        shortage_cost=50.0,
        max_production=30.0,
    )

    capacity = make_solver_capacity(
        [
            40,
            40,
        ]
    )

    result = solve_production_plan(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=2
        ),
    )

    report = build_planning_report(
        result=result,
        sku_parameters=parameters,
    )

    executive = (
        report.executive_summary
        .iloc[0]
    )

    assert np.isclose(
        executive[
            "final_backlog"
        ],
        0.0,
        atol=1e-7,
    )

    assert np.isclose(
        executive[
            "portfolio_service_pct"
        ],
        100.0,
        atol=1e-7,
    )

# ============================================================
# SCENARIO / POLICY TESTS
# ============================================================


def test_jit_baseline_respects_shared_capacity():
    """
    The non-anticipatory JIT policy must remain physically feasible
    under the same shared plant-capacity constraint as the optimizer.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [10],
            "SKU_B": [10],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
            "SKU_B",
        ],
        processing_time=1.0,
        max_production=20.0,
    )

    capacity = make_solver_capacity(
        [
            15,
        ]
    )

    baseline = build_jit_baseline(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=1
        ),
    )

    capacity_used = float(
        baseline[
            "capacity_used"
        ].sum()
    )

    assert (
        capacity_used
        <= 15.0 + 1e-7
    )

    assert np.isclose(
        capacity_used,
        15.0,
        atol=1e-7,
    )


def test_jit_baseline_does_not_look_ahead():
    """
    Future demand must not change today's JIT production decision.

    The two forecasts are identical in week 1 but have very different
    week-2 demand.

    Since the JIT policy is non-anticipatory, week-1 production must
    remain identical.
    """

    forecast_a = make_solver_forecast(
        {
            "SKU_A": [
                5,
                5,
            ],
        }
    )

    forecast_b = make_solver_forecast(
        {
            "SKU_A": [
                5,
                50,
            ],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
        ],
        max_production=100.0,
    )

    capacity = make_solver_capacity(
        [
            100,
            100,
        ]
    )

    baseline_a = build_jit_baseline(
        forecast=forecast_a,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=2
        ),
    )

    baseline_b = build_jit_baseline(
        forecast=forecast_b,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=2
        ),
    )

    week_1_a = float(
        baseline_a.loc[
            baseline_a[
                "forecast_horizon"
            ].eq(1),
            "production_quantity",
        ].iloc[0]
    )

    week_1_b = float(
        baseline_b.loc[
            baseline_b[
                "forecast_horizon"
            ].eq(1),
            "production_quantity",
        ].iloc[0]
    )

    assert np.isclose(
        week_1_a,
        week_1_b,
        atol=1e-7,
    )

    assert np.isclose(
        week_1_a,
        5.0,
        atol=1e-7,
    )


def test_jit_baseline_stock_flow_reconciles():
    """
    JIT inventory/backlog state must obey the same stock-flow identity
    as the Pyomo optimization model.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [
                10,
                10,
                10,
            ],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
        ],
        initial_inventory=3.0,
        max_production=20.0,
    )

    capacity = make_solver_capacity(
        [
            8,
            6,
            7,
        ]
    )

    baseline = build_jit_baseline(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=3
        ),
    )

    plan = (
        baseline
        .sort_values(
            "forecast_horizon"
        )
        .reset_index(
            drop=True
        )
    )

    previous_inventory = 3.0
    previous_backlog = 0.0

    for row in plan.itertuples(
        index=False
    ):

        expected_net_position = (
            previous_inventory
            - previous_backlog
            + row.production_quantity
            - row.forecast_demand
        )

        actual_net_position = (
            row.ending_inventory
            - row.ending_backlog
        )

        assert np.isclose(
            actual_net_position,
            expected_net_position,
            atol=1e-7,
        )

        previous_inventory = (
            row.ending_inventory
        )

        previous_backlog = (
            row.ending_backlog
        )


def test_plan_metric_summary_reconciles_stock_flow():
    """
    Shared scenario metrics should correctly summarize any feasible
    production plan.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [
                10,
                10,
            ],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
        ],
        initial_inventory=2.0,
        max_production=20.0,
    )

    capacity = make_solver_capacity(
        [
            10,
            10,
        ]
    )

    baseline = build_jit_baseline(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=2
        ),
    )

    metrics = summarize_plan_metrics(
        plan=baseline,
        sku_parameters=parameters,
        capacity=capacity,
    )

    expected_final_net_inventory = (
        metrics[
            "initial_inventory"
        ]
        - metrics[
            "initial_backlog"
        ]
        + metrics[
            "total_production"
        ]
        - metrics[
            "total_forecast_demand"
        ]
    )

    actual_final_net_inventory = (
        metrics[
            "final_inventory"
        ]
        - metrics[
            "final_backlog"
        ]
    )

    assert np.isclose(
        expected_final_net_inventory,
        actual_final_net_inventory,
        atol=1e-7,
    )


def test_optimization_outperforms_jit_when_lookahead_has_value():
    """
    Construct a case where anticipatory production is economically
    valuable.

    Week 1:
        capacity = 20

    Week 2:
        capacity = 0

    Demand:
        10 each week

    JIT produces only week-1 demand and therefore cannot satisfy week 2.

    The optimizer sees the bottleneck in advance and prebuilds inventory.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [
                10,
                10,
            ],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
        ],
        initial_inventory=0.0,
        production_cost=5.0,
        holding_cost=0.5,
        shortage_cost=50.0,
        max_production=20.0,
    )

    capacity = make_solver_capacity(
        [
            20,
            0,
        ]
    )

    config = make_solver_config(
        horizon=2
    )

    optimized_result = solve_production_plan(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=config,
    )

    comparison_result = compare_to_jit_baseline(
        optimized_result=optimized_result,
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=config,
    )

    comparison = (
        comparison_result.comparison
        .set_index(
            "strategy"
        )
    )

    baseline_cost = float(
        comparison.loc[
            "JIT baseline",
            "total_cost",
        ]
    )

    optimized_cost = float(
        comparison.loc[
            "Pyomo optimized",
            "total_cost",
        ]
    )

    baseline_backlog = float(
        comparison.loc[
            "JIT baseline",
            "final_backlog",
        ]
    )

    optimized_backlog = float(
        comparison.loc[
            "Pyomo optimized",
            "final_backlog",
        ]
    )

    assert (
        optimized_cost
        < baseline_cost
    )

    assert (
        optimized_backlog
        < baseline_backlog
    )

    assert np.isclose(
        optimized_backlog,
        0.0,
        atol=1e-7,
    )


def test_baseline_comparison_impact_metrics_are_consistent():
    """
    Impact metrics must reconcile exactly with the strategy comparison.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [
                10,
                10,
            ],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
        ],
        production_cost=5.0,
        holding_cost=0.5,
        shortage_cost=50.0,
        max_production=20.0,
    )

    capacity = make_solver_capacity(
        [
            20,
            0,
        ]
    )

    config = make_solver_config(
        horizon=2
    )

    optimized_result = solve_production_plan(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=config,
    )

    result = compare_to_jit_baseline(
        optimized_result=optimized_result,
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=config,
    )

    impact = (
        result.impact_summary
        .iloc[0]
    )

    assert (
        impact[
            "cost_savings"
        ]
        > 0
    )

    assert (
        impact[
            "cost_savings_pct"
        ]
        > 0
    )

    assert (
        impact[
            "backlog_reduction"
        ]
        > 0
    )

    assert (
        impact[
            "shortage_cost_reduction"
        ]
        > 0
    )


# ============================================================
# CAPACITY-SENSITIVITY TESTS
# ============================================================


def test_capacity_sensitivity_returns_requested_scenarios():
    """
    Every requested capacity scenario should appear exactly once.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [
                10,
                10,
            ],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
        ],
        shortage_cost=30.0,
        max_production=20.0,
    )

    capacity = make_solver_capacity(
        [
            10,
            10,
        ]
    )

    scenarios = {
        "Low":
            0.95,

        "Baseline":
            1.00,

        "High":
            1.05,
    }

    result = run_capacity_sensitivity(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=2
        ),
        scenarios=scenarios,
    )

    assert set(
        result.summary[
            "scenario"
        ]
    ) == set(
        scenarios
    )

    assert set(
        result.plans
    ) == set(
        scenarios
    )


def test_lower_capacity_can_create_backlog():
    """
    With demand exactly matching baseline capacity, reducing capacity
    by 5% should create backlog while baseline capacity can fully serve
    demand.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [
                10,
                10,
            ],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
        ],
        initial_inventory=0.0,
        shortage_cost=50.0,
        max_production=20.0,
    )

    capacity = make_solver_capacity(
        [
            10,
            10,
        ]
    )

    result = run_capacity_sensitivity(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=2
        ),
        scenarios={
            "Capacity -5%":
                0.95,

            "Baseline":
                1.00,
        },
    )

    summary = (
        result.summary
        .set_index(
            "scenario"
        )
    )

    assert (
        summary.loc[
            "Capacity -5%",
            "final_backlog",
        ]
        > 0
    )

    assert np.isclose(
        summary.loc[
            "Baseline",
            "final_backlog",
        ],
        0.0,
        atol=1e-7,
    )


def test_optimal_cost_is_nonincreasing_with_more_capacity():
    """
    Additional free capacity expands the LP feasible region.

    Therefore the optimal objective cannot increase as plant capacity
    increases.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [
                10,
                15,
                10,
            ],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
        ],
        initial_inventory=0.0,
        production_cost=5.0,
        holding_cost=0.5,
        shortage_cost=40.0,
        max_production=30.0,
    )

    capacity = make_solver_capacity(
        [
            10,
            10,
            10,
        ]
    )

    result = run_capacity_sensitivity(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=3
        ),
        scenarios={
            "80%":
                0.80,

            "100%":
                1.00,

            "120%":
                1.20,
        },
    )

    summary = (
        result.summary
        .sort_values(
            "capacity_multiplier"
        )
    )

    costs = (
        summary[
            "total_cost"
        ]
        .to_numpy()
    )

    assert (
        np.diff(
            costs
        )
        <= 1e-7
    ).all()


def test_capacity_sensitivity_preserves_forecast_demand():
    """
    Capacity experiments change only plant capacity.

    Forecast demand must remain identical across scenario solutions.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [
                10,
                15,
            ],
            "SKU_B": [
                8,
                12,
            ],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
            "SKU_B",
        ],
        max_production=30.0,
    )

    capacity = make_solver_capacity(
        [
            30,
            30,
        ]
    )

    result = run_capacity_sensitivity(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=2
        ),
        scenarios={
            "Low":
                0.90,

            "Baseline":
                1.00,

            "High":
                1.10,
        },
    )

    expected_demand = float(
        forecast[
            "forecast_demand"
        ].sum()
    )

    for plan in result.plans.values():

        assert np.isclose(
            plan[
                "forecast_demand"
            ].sum(),
            expected_demand,
            atol=1e-7,
        )


def test_capacity_sensitivity_is_deterministic():
    """
    Identical scenario inputs should generate identical scenario
    summaries and plans.
    """

    forecast = make_solver_forecast(
        {
            "SKU_A": [
                10,
                12,
            ],
        }
    )

    parameters = make_solver_parameters(
        [
            "SKU_A",
        ],
        shortage_cost=40.0,
        max_production=20.0,
    )

    capacity = make_solver_capacity(
        [
            10,
            10,
        ]
    )

    scenarios = {
        "Low":
            0.95,

        "Baseline":
            1.00,

        "High":
            1.05,
    }

    result_a = run_capacity_sensitivity(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=2
        ),
        scenarios=scenarios,
    )

    result_b = run_capacity_sensitivity(
        forecast=forecast,
        sku_parameters=parameters,
        capacity=capacity,
        config=make_solver_config(
            horizon=2
        ),
        scenarios=scenarios,
    )

    pd.testing.assert_frame_equal(
        result_a.summary,
        result_b.summary,
    )

    for scenario_name in scenarios:

        pd.testing.assert_frame_equal(
            result_a.plans[
                scenario_name
            ],
            result_b.plans[
                scenario_name
            ],
        )
