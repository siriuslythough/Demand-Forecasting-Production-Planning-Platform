from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from new_demfor_planopti.optimization.contracts import (
    PlanningConfig,
    validate_planning_inputs,
    validate_sku_parameters,
)

from new_demfor_planopti.optimization.solver import (
    ProductionPlanResult,
    solve_production_plan,
)


# ============================================================
# RESULT CONTRACTS
# ============================================================


@dataclass(frozen=True)
class BaselineComparisonResult:
    """
    Comparison between the optimized Pyomo plan and a feasible
    non-anticipatory JIT production policy.
    """

    baseline_plan: pd.DataFrame

    comparison: pd.DataFrame

    impact_summary: pd.DataFrame


@dataclass(frozen=True)
class CapacitySensitivityResult:
    """
    Results from resolving the production-planning LP under multiple
    plant-capacity scenarios.
    """

    summary: pd.DataFrame

    plans: dict[
        str,
        pd.DataFrame,
    ]


# ============================================================
# INTERNAL HELPERS
# ============================================================


def _safe_pct(
    numerator: float,
    denominator: float,
) -> float:
    """
    Safe percentage calculation.
    """

    if denominator <= 0:

        return 0.0

    return (
        100.0
        * numerator
        / denominator
    )


def _validate_plan_frame(
    plan: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate the common plan structure used by optimized and baseline
    production policies.
    """

    required_columns = [
        "week_start",
        "forecast_horizon",
        "sku_id",
        "forecast_demand",
        "production_quantity",
        "ending_inventory",
        "ending_backlog",
        "capacity_used",
        "production_cost",
        "holding_cost",
        "shortage_cost",
        "total_cost",
    ]

    missing = (
        set(required_columns)
        - set(plan.columns)
    )

    if missing:

        raise ValueError(
            "Plan is missing required columns: "
            f"{sorted(missing)}"
        )

    output = plan[
        required_columns
    ].copy()

    output[
        "week_start"
    ] = pd.to_datetime(
        output[
            "week_start"
        ],
        errors="raise",
    )

    numeric_columns = [
        "forecast_demand",
        "production_quantity",
        "ending_inventory",
        "ending_backlog",
        "capacity_used",
        "production_cost",
        "holding_cost",
        "shortage_cost",
        "total_cost",
    ]

    for column in numeric_columns:

        output[
            column
        ] = pd.to_numeric(
            output[
                column
            ],
            errors="raise",
        )

        if not np.isfinite(
            output[
                column
            ]
        ).all():

            raise ValueError(
                f"Plan column {column} "
                "contains non-finite values."
            )

    for column in [
        "forecast_demand",
        "production_quantity",
        "ending_inventory",
        "ending_backlog",
        "capacity_used",
    ]:

        if (
            output[
                column
            ]
            < -1e-7
        ).any():

            raise ValueError(
                f"Plan column {column} "
                "contains negative values."
            )

    duplicated = (
        output.duplicated(
            subset=[
                "week_start",
                "sku_id",
            ],
            keep=False,
        )
    )

    if duplicated.any():

        raise ValueError(
            "Plan contains duplicate "
            "(week_start, sku_id) rows."
        )

    return (
        output
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
# NON-ANTICIPATORY JIT BASELINE
# ============================================================


def build_jit_baseline(
    forecast: pd.DataFrame,
    sku_parameters: pd.DataFrame,
    capacity: pd.DataFrame,
    config: PlanningConfig | None = None,
) -> pd.DataFrame:
    """
    Construct a feasible non-anticipatory production-planning baseline.

    Policy
    ------
    For each planning week:

        1. consume available inventory,
        2. include outstanding backlog in current requirements,
        3. attempt to produce only current requirements,
        4. respect SKU production ceilings,
        5. respect shared plant capacity,
        6. allocate insufficient shared capacity proportionally across
           current SKU production requirements,
        7. carry any unsatisfied requirement as backlog.

    Crucially, this policy does NOT use future demand when choosing
    current production.

    It therefore provides a reasonable benchmark for measuring the
    value of multi-period look-ahead optimization.
    """

    if config is None:

        config = PlanningConfig()

    (
        forecast_clean,
        sku_clean,
        capacity_clean,
    ) = validate_planning_inputs(
        forecast=forecast,
        sku_parameters=sku_parameters,
        capacity=capacity,
        config=config,
    )

    parameters = (
        sku_clean
        .set_index(
            "sku_id"
        )
    )

    skus = sorted(
        forecast_clean[
            "sku_id"
        ]
        .unique()
        .tolist()
    )

    weeks = (
        forecast_clean[
            [
                "week_start",
                "forecast_horizon",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "forecast_horizon"
        )
        .reset_index(
            drop=True
        )
    )

    capacity_lookup = (
        capacity_clean
        .set_index(
            "week_start"
        )[
            "production_capacity"
        ]
        .to_dict()
    )

    inventory = {
        sku_id:
            float(
                parameters.loc[
                    sku_id,
                    "initial_inventory",
                ]
            )

        for sku_id
        in skus
    }

    backlog = {
        sku_id:
            float(
                parameters.loc[
                    sku_id,
                    "initial_backlog",
                ]
            )

        for sku_id
        in skus
    }

    rows: list[
        dict[
            str,
            object,
        ]
    ] = []


    # ========================================================
    # WEEK-BY-WEEK NON-ANTICIPATORY POLICY
    # ========================================================

    for week_row in weeks.itertuples(
        index=False
    ):

        week = pd.Timestamp(
            week_row.week_start
        )

        horizon = int(
            week_row.forecast_horizon
        )

        week_forecast = (
            forecast_clean.loc[
                forecast_clean[
                    "week_start"
                ].eq(
                    week
                )
            ]
            .set_index(
                "sku_id"
            )[
                "forecast_demand"
            ]
        )


        # ----------------------------------------------------
        # CURRENT-WEEK PRODUCTION REQUIREMENTS
        # ----------------------------------------------------

        desired_production: dict[
            str,
            float,
        ] = {}

        for sku_id in skus:

            demand = float(
                week_forecast.loc[
                    sku_id
                ]
            )

            current_requirement = max(
                0.0,
                demand
                + backlog[
                    sku_id
                ]
                - inventory[
                    sku_id
                ],
            )

            production_limit = float(
                parameters.loc[
                    sku_id,
                    "max_production_per_week",
                ]
            )

            desired_production[
                sku_id
            ] = min(
                current_requirement,
                production_limit,
            )


        # ----------------------------------------------------
        # SHARED PLANT CAPACITY
        # ----------------------------------------------------

        desired_capacity = sum(
            desired_production[
                sku_id
            ]
            *
            float(
                parameters.loc[
                    sku_id,
                    "processing_time",
                ]
            )

            for sku_id
            in skus
        )

        available_capacity = float(
            capacity_lookup[
                week
            ]
        )

        if desired_capacity <= 0:

            capacity_scale = 0.0

        else:

            capacity_scale = min(
                1.0,
                available_capacity
                /
                desired_capacity,
            )


        # ----------------------------------------------------
        # EXECUTE POLICY
        # ----------------------------------------------------

        for sku_id in skus:

            demand = float(
                week_forecast.loc[
                    sku_id
                ]
            )

            production = (
                desired_production[
                    sku_id
                ]
                *
                capacity_scale
            )

            previous_inventory = (
                inventory[
                    sku_id
                ]
            )

            previous_backlog = (
                backlog[
                    sku_id
                ]
            )

            net_inventory = (
                previous_inventory
                -
                previous_backlog
                +
                production
                -
                demand
            )

            ending_inventory = max(
                net_inventory,
                0.0,
            )

            ending_backlog = max(
                -net_inventory,
                0.0,
            )

            inventory[
                sku_id
            ] = ending_inventory

            backlog[
                sku_id
            ] = ending_backlog

            production_cost = (
                production
                *
                float(
                    parameters.loc[
                        sku_id,
                        "unit_production_cost",
                    ]
                )
            )

            holding_cost = (
                ending_inventory
                *
                float(
                    parameters.loc[
                        sku_id,
                        "unit_holding_cost",
                    ]
                )
            )

            shortage_cost = (
                ending_backlog
                *
                float(
                    parameters.loc[
                        sku_id,
                        "unit_shortage_cost",
                    ]
                )
            )

            processing_time = float(
                parameters.loc[
                    sku_id,
                    "processing_time",
                ]
            )

            capacity_used = (
                production
                *
                processing_time
            )

            rows.append(
                {
                    "week_start":
                        week,

                    "forecast_horizon":
                        horizon,

                    "sku_id":
                        sku_id,

                    "forecast_demand":
                        demand,

                    "production_quantity":
                        production,

                    "ending_inventory":
                        ending_inventory,

                    "ending_backlog":
                        ending_backlog,

                    "capacity_used":
                        capacity_used,

                    "production_cost":
                        production_cost,

                    "holding_cost":
                        holding_cost,

                    "shortage_cost":
                        shortage_cost,

                    "total_cost":
                        (
                            production_cost
                            +
                            holding_cost
                            +
                            shortage_cost
                        ),
                }
            )


    baseline = (
        _validate_plan_frame(
            pd.DataFrame(
                rows
            )
        )
    )


    # ========================================================
    # FEASIBILITY CHECK
    # ========================================================

    weekly_capacity = (
        baseline
        .groupby(
            "week_start",
            as_index=False,
            observed=True,
        )
        .agg(
            capacity_used=(
                "capacity_used",
                "sum",
            )
        )
        .merge(
            capacity_clean,
            on="week_start",
            how="left",
            validate="one_to_one",
        )
    )

    if (
        weekly_capacity[
            "capacity_used"
        ]
        >
        weekly_capacity[
            "production_capacity"
        ]
        + 1e-6
    ).any():

        raise RuntimeError(
            "JIT baseline violates shared "
            "plant capacity."
        )

    return baseline


# ============================================================
# COMMON PLAN METRICS
# ============================================================


def summarize_plan_metrics(
    plan: pd.DataFrame,
    sku_parameters: pd.DataFrame,
    capacity: pd.DataFrame,
) -> dict[
    str,
    float,
]:
    """
    Compute common metrics for any feasible production plan.

    Works for both:

        optimized Pyomo plan
        JIT baseline plan
    """

    plan_clean = (
        _validate_plan_frame(
            plan
        )
    )

    sku_clean = (
        validate_sku_parameters(
            sku_parameters
        )
    )

    capacity_clean = (
        capacity.copy()
    )

    capacity_clean[
        "week_start"
    ] = pd.to_datetime(
        capacity_clean[
            "week_start"
        ]
    )


    # --------------------------------------------------------
    # FINAL SKU STATE
    # --------------------------------------------------------

    final_state = (
        plan_clean
        .sort_values(
            [
                "sku_id",
                "forecast_horizon",
            ]
        )
        .groupby(
            "sku_id",
            observed=True,
        )
        .tail(1)
    )

    total_demand = float(
        plan_clean[
            "forecast_demand"
        ].sum()
    )

    total_production = float(
        plan_clean[
            "production_quantity"
        ].sum()
    )

    initial_inventory = float(
        sku_clean[
            "initial_inventory"
        ].sum()
    )

    initial_backlog = float(
        sku_clean[
            "initial_backlog"
        ].sum()
    )

    final_inventory = float(
        final_state[
            "ending_inventory"
        ].sum()
    )

    final_backlog = float(
        final_state[
            "ending_backlog"
        ].sum()
    )

    obligation = (
        total_demand
        +
        initial_backlog
    )

    if obligation <= 0:

        service_pct = (
            100.0
            if final_backlog <= 1e-9
            else 0.0
        )

    else:

        service_pct = (
            100.0
            *
            (
                1.0
                -
                final_backlog
                /
                obligation
            )
        )

        service_pct = float(
            np.clip(
                service_pct,
                0.0,
                100.0,
            )
        )


    # --------------------------------------------------------
    # BACKLOG-FREE SKU-WEEK RATE
    # --------------------------------------------------------

    backlog_free_rows = int(
        (
            plan_clean[
                "ending_backlog"
            ]
            <= 1e-8
        ).sum()
    )

    backlog_free_sku_week_pct = (
        _safe_pct(
            backlog_free_rows,
            len(
                plan_clean
            ),
        )
    )


    # --------------------------------------------------------
    # CAPACITY
    # --------------------------------------------------------

    total_available_capacity = float(
        capacity_clean[
            "production_capacity"
        ].sum()
    )

    total_capacity_used = float(
        plan_clean[
            "capacity_used"
        ].sum()
    )

    capacity_utilization_pct = (
        _safe_pct(
            total_capacity_used,
            total_available_capacity,
        )
    )


    # --------------------------------------------------------
    # STOCK-FLOW RECONCILIATION
    # --------------------------------------------------------

    expected_final_net_inventory = (
        initial_inventory
        -
        initial_backlog
        +
        total_production
        -
        total_demand
    )

    actual_final_net_inventory = (
        final_inventory
        -
        final_backlog
    )

    if not np.isclose(
        expected_final_net_inventory,
        actual_final_net_inventory,
        rtol=1e-7,
        atol=1e-5,
    ):

        raise RuntimeError(
            "Plan does not reconcile over "
            "the complete planning horizon. "
            f"Expected final net inventory "
            f"{expected_final_net_inventory}, "
            f"found "
            f"{actual_final_net_inventory}."
        )


    return {
        "total_cost":
            float(
                plan_clean[
                    "total_cost"
                ].sum()
            ),

        "production_cost":
            float(
                plan_clean[
                    "production_cost"
                ].sum()
            ),

        "holding_cost":
            float(
                plan_clean[
                    "holding_cost"
                ].sum()
            ),

        "shortage_cost":
            float(
                plan_clean[
                    "shortage_cost"
                ].sum()
            ),

        "total_forecast_demand":
            total_demand,

        "initial_inventory":
            initial_inventory,

        "initial_backlog":
            initial_backlog,

        "total_production":
            total_production,

        "final_inventory":
            final_inventory,

        "final_backlog":
            final_backlog,

        "service_pct":
            service_pct,

        "backlog_free_sku_week_pct":
            backlog_free_sku_week_pct,

        "capacity_utilization_pct":
            capacity_utilization_pct,
    }


# ============================================================
# OPTIMIZED VS JIT COMPARISON
# ============================================================


def compare_to_jit_baseline(
    optimized_result: ProductionPlanResult,
    forecast: pd.DataFrame,
    sku_parameters: pd.DataFrame,
    capacity: pd.DataFrame,
    config: PlanningConfig | None = None,
) -> BaselineComparisonResult:
    """
    Compare the solved multi-period optimization model against the
    feasible non-anticipatory JIT policy.
    """

    if config is None:

        config = PlanningConfig()

    baseline_plan = (
        build_jit_baseline(
            forecast=forecast,
            sku_parameters=sku_parameters,
            capacity=capacity,
            config=config,
        )
    )

    baseline_metrics = (
        summarize_plan_metrics(
            plan=baseline_plan,
            sku_parameters=sku_parameters,
            capacity=capacity,
        )
    )

    optimized_metrics = (
        summarize_plan_metrics(
            plan=optimized_result.plan,
            sku_parameters=sku_parameters,
            capacity=capacity,
        )
    )

    comparison = pd.DataFrame(
        [
            {
                "strategy":
                    "JIT baseline",

                **baseline_metrics,
            },
            {
                "strategy":
                    "Pyomo optimized",

                **optimized_metrics,
            },
        ]
    )


    # ========================================================
    # IMPACT METRICS
    # ========================================================

    baseline_cost = float(
        baseline_metrics[
            "total_cost"
        ]
    )

    optimized_cost = float(
        optimized_metrics[
            "total_cost"
        ]
    )

    cost_savings = (
        baseline_cost
        -
        optimized_cost
    )

    cost_savings_pct = (
        _safe_pct(
            cost_savings,
            baseline_cost,
        )
    )

    backlog_reduction = (
        baseline_metrics[
            "final_backlog"
        ]
        -
        optimized_metrics[
            "final_backlog"
        ]
    )

    service_improvement = (
        optimized_metrics[
            "service_pct"
        ]
        -
        baseline_metrics[
            "service_pct"
        ]
    )

    shortage_cost_reduction = (
        baseline_metrics[
            "shortage_cost"
        ]
        -
        optimized_metrics[
            "shortage_cost"
        ]
    )

    additional_holding_cost = (
        optimized_metrics[
            "holding_cost"
        ]
        -
        baseline_metrics[
            "holding_cost"
        ]
    )

    impact_summary = pd.DataFrame(
        [
            {
                "baseline_total_cost":
                    baseline_cost,

                "optimized_total_cost":
                    optimized_cost,

                "cost_savings":
                    cost_savings,

                "cost_savings_pct":
                    cost_savings_pct,

                "baseline_final_backlog":
                    baseline_metrics[
                        "final_backlog"
                    ],

                "optimized_final_backlog":
                    optimized_metrics[
                        "final_backlog"
                    ],

                "backlog_reduction":
                    backlog_reduction,

                "baseline_service_pct":
                    baseline_metrics[
                        "service_pct"
                    ],

                "optimized_service_pct":
                    optimized_metrics[
                        "service_pct"
                    ],

                "service_improvement_pp":
                    service_improvement,

                "shortage_cost_reduction":
                    shortage_cost_reduction,

                "additional_holding_cost":
                    additional_holding_cost,
            }
        ]
    )

    return BaselineComparisonResult(
        baseline_plan=(
            baseline_plan
        ),

        comparison=(
            comparison
        ),

        impact_summary=(
            impact_summary
        ),
    )


# ============================================================
# CAPACITY SENSITIVITY
# ============================================================


def run_capacity_sensitivity(
    forecast: pd.DataFrame,
    sku_parameters: pd.DataFrame,
    capacity: pd.DataFrame,
    config: PlanningConfig | None = None,
    scenarios: dict[
        str,
        float,
    ] | None = None,
) -> CapacitySensitivityResult:
    """
    Resolve the production-planning LP under multiple plant-capacity
    scenarios.

    Defaults
    --------
        Capacity -5%
        Baseline
        Capacity +5%
        Capacity +10%

    All non-capacity assumptions remain fixed.
    """

    if config is None:

        config = PlanningConfig()

    if scenarios is None:

        scenarios = {
            "Capacity -5%":
                0.95,

            "Baseline":
                1.00,

            "Capacity +5%":
                1.05,

            "Capacity +10%":
                1.10,
        }

    if not scenarios:

        raise ValueError(
            "At least one capacity scenario "
            "must be supplied."
        )

    for (
        scenario_name,
        multiplier,
    ) in scenarios.items():

        if (
            not isinstance(
                scenario_name,
                str,
            )
            or not scenario_name.strip()
        ):

            raise ValueError(
                "Capacity scenario names "
                "must be non-empty strings."
            )

        if (
            not np.isfinite(
                multiplier
            )
            or multiplier <= 0
        ):

            raise ValueError(
                "Capacity multipliers must "
                "be finite and > 0."
            )


    # ========================================================
    # VALIDATE BASE INPUTS ONCE
    # ========================================================

    (
        forecast_clean,
        sku_clean,
        capacity_clean,
    ) = validate_planning_inputs(
        forecast=forecast,
        sku_parameters=sku_parameters,
        capacity=capacity,
        config=config,
    )


    scenario_rows = []

    plans: dict[
        str,
        pd.DataFrame,
    ] = {}


    # ========================================================
    # SOLVE EACH CAPACITY SCENARIO
    # ========================================================

    for (
        scenario_name,
        multiplier,
    ) in scenarios.items():

        scenario_capacity = (
            capacity_clean.copy()
        )

        scenario_capacity[
            "production_capacity"
        ] = (
            scenario_capacity[
                "production_capacity"
            ]
            *
            float(
                multiplier
            )
        )

        result = (
            solve_production_plan(
                forecast=forecast_clean,
                sku_parameters=sku_clean,
                capacity=scenario_capacity,
                config=config,
                tee=False,
            )
        )

        metrics = (
            summarize_plan_metrics(
                plan=result.plan,
                sku_parameters=sku_clean,
                capacity=scenario_capacity,
            )
        )

        capacity_detail = (
            result.capacity_usage
        )

        constrained_weeks = int(
            (
                capacity_detail[
                    "utilization_pct"
                ]
                >= 99.0
            ).sum()
        )

        peak_utilization_pct = float(
            capacity_detail[
                "utilization_pct"
            ].max()
        )

        positive_shadow_values = (
            capacity_detail[
                "capacity_marginal_value"
            ]
            if (
                "capacity_marginal_value"
                in capacity_detail.columns
            )
            else pd.Series(
                dtype=float
            )
        )

        max_capacity_marginal_value = (
            float(
                positive_shadow_values.max()
            )
            if not positive_shadow_values.empty
            else 0.0
        )

        scenario_rows.append(
            {
                "scenario":
                    scenario_name,

                "capacity_multiplier":
                    float(
                        multiplier
                    ),

                **metrics,

                "peak_utilization_pct":
                    peak_utilization_pct,

                "constrained_weeks":
                    constrained_weeks,

                "max_capacity_marginal_value":
                    max_capacity_marginal_value,
            }
        )

        plans[
            scenario_name
        ] = (
            result.plan.copy()
        )


    summary = pd.DataFrame(
        scenario_rows
    )


    # ========================================================
    # BASELINE-RELATIVE COST CHANGE
    # ========================================================

    baseline_rows = (
        summary.loc[
            np.isclose(
                summary[
                    "capacity_multiplier"
                ],
                1.0,
            )
        ]
    )

    if not baseline_rows.empty:

        baseline_cost = float(
            baseline_rows.iloc[0][
                "total_cost"
            ]
        )

        summary[
            "cost_change_vs_baseline"
        ] = (
            summary[
                "total_cost"
            ]
            -
            baseline_cost
        )

        summary[
            "cost_change_pct_vs_baseline"
        ] = np.where(
            baseline_cost > 0,

            100.0
            *
            summary[
                "cost_change_vs_baseline"
            ]
            /
            baseline_cost,

            0.0,
        )

    else:

        summary[
            "cost_change_vs_baseline"
        ] = np.nan

        summary[
            "cost_change_pct_vs_baseline"
        ] = np.nan


    return CapacitySensitivityResult(
        summary=(
            summary
            .sort_values(
                "capacity_multiplier"
            )
            .reset_index(
                drop=True
            )
        ),

        plans=plans,
    )