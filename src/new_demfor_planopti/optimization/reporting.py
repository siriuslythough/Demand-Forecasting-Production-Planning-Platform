from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from new_demfor_planopti.optimization.contracts import (
    validate_sku_parameters,
)

from new_demfor_planopti.optimization.solver import (
    ProductionPlanResult,
)


# ============================================================
# REPORT CONTRACT
# ============================================================


@dataclass(frozen=True)
class PlanningReport:
    """
    Decision-support outputs derived from an optimized production plan.

    Attributes
    ----------
    executive_summary:
        One-row table containing the main planning KPIs.

    cost_summary:
        Cost decomposition.

    capacity_summary:
        One-row plant utilization summary.

    capacity_detail:
        Week-level capacity utilization.

    sku_summary:
        SKU-level production, inventory, backlog, service, and cost
        metrics.
    """

    executive_summary: pd.DataFrame

    cost_summary: pd.DataFrame

    capacity_summary: pd.DataFrame

    capacity_detail: pd.DataFrame

    sku_summary: pd.DataFrame


# ============================================================
# REQUIRED SOLUTION COLUMNS
# ============================================================


PLAN_REQUIRED_COLUMNS = [
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


CAPACITY_REQUIRED_COLUMNS = [
    "week_start",
    "production_capacity",
    "capacity_used",
    "capacity_slack",
    "utilization_pct",
]


# ============================================================
# INTERNAL HELPERS
# ============================================================


def _require_columns(
    df: pd.DataFrame,
    required: list[str],
    frame_name: str,
) -> None:
    """
    Validate dataframe columns before reporting.
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


def _require_finite(
    df: pd.DataFrame,
    columns: list[str],
    frame_name: str,
) -> None:
    """
    Reporting should never silently propagate NaN or infinity.
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
                "contains non-finite values."
            )


def _safe_pct(
    numerator: float,
    denominator: float,
) -> float:
    """
    Safe percentage helper.
    """

    if denominator <= 0:

        return 0.0

    return (
        100.0
        * numerator
        / denominator
    )


def _service_pct(
    total_demand: float,
    initial_backlog: float,
    final_backlog: float,
) -> float:
    """
    Horizon-level service measure.

    Demand obligation during the planning horizon is:

        forecast demand
        + backlog already outstanding at horizon start

    Remaining unsatisfied demand at horizon end is:

        final backlog

    Therefore:

        service %
        =
        1
        -
        final backlog
        /
        total demand obligation

    clipped to [0, 100].
    """

    obligation = (
        float(total_demand)
        +
        float(initial_backlog)
    )

    if obligation <= 0:

        return (
            100.0
            if final_backlog <= 1e-9
            else 0.0
        )

    service = (
        100.0
        *
        (
            1.0
            -
            float(final_backlog)
            /
            obligation
        )
    )

    return float(
        np.clip(
            service,
            0.0,
            100.0,
        )
    )


# ============================================================
# SOLUTION VALIDATION
# ============================================================


def validate_reporting_inputs(
    result: ProductionPlanResult,
    sku_parameters: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Validate and normalize reporting inputs.

    Returns
    -------
    plan
    capacity_usage
    sku_parameters
    """

    if not isinstance(
        result,
        ProductionPlanResult,
    ):

        raise TypeError(
            "result must be a "
            "ProductionPlanResult."
        )

    plan = (
        result.plan
        .copy()
    )

    capacity_usage = (
        result.capacity_usage
        .copy()
    )

    sku_clean = (
        validate_sku_parameters(
            sku_parameters
        )
    )

    _require_columns(
        plan,
        PLAN_REQUIRED_COLUMNS,
        "plan",
    )

    _require_columns(
        capacity_usage,
        CAPACITY_REQUIRED_COLUMNS,
        "capacity_usage",
    )

    numeric_plan_columns = [
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

    _require_finite(
        plan,
        numeric_plan_columns,
        "plan",
    )

    _require_finite(
        capacity_usage,
        [
            "production_capacity",
            "capacity_used",
            "capacity_slack",
            "utilization_pct",
        ],
        "capacity_usage",
    )

    plan[
        "week_start"
    ] = pd.to_datetime(
        plan[
            "week_start"
        ]
    )

    capacity_usage[
        "week_start"
    ] = pd.to_datetime(
        capacity_usage[
            "week_start"
        ]
    )

    result_skus = set(
        plan[
            "sku_id"
        ].astype(str)
    )

    parameter_skus = set(
        sku_clean[
            "sku_id"
        ].astype(str)
    )

    missing_parameters = (
        result_skus
        - parameter_skus
    )

    if missing_parameters:

        raise ValueError(
            "Missing SKU parameters for "
            "reporting: "
            f"{sorted(missing_parameters)}"
        )

    sku_clean = (
        sku_clean.loc[
            sku_clean[
                "sku_id"
            ].isin(
                result_skus
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    plan = (
        plan
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

    capacity_usage = (
        capacity_usage
        .sort_values(
            "week_start"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        plan,
        capacity_usage,
        sku_clean,
    )


# ============================================================
# COST REPORT
# ============================================================


def summarize_costs(
    plan: pd.DataFrame,
    objective_value: float | None = None,
) -> pd.DataFrame:
    """
    Produce cost decomposition for the optimized plan.

    Output contains:

        production
        inventory holding
        shortage/backlog
        total
    """

    _require_columns(
        plan,
        PLAN_REQUIRED_COLUMNS,
        "plan",
    )

    production_cost = float(
        plan[
            "production_cost"
        ].sum()
    )

    holding_cost = float(
        plan[
            "holding_cost"
        ].sum()
    )

    shortage_cost = float(
        plan[
            "shortage_cost"
        ].sum()
    )

    total_cost = (
        production_cost
        +
        holding_cost
        +
        shortage_cost
    )

    if (
        objective_value
        is not None
        and not np.isclose(
            total_cost,
            objective_value,
            rtol=1e-7,
            atol=1e-6,
        )
    ):

        raise ValueError(
            "Reported total cost does not "
            "match optimization objective. "
            f"Reported={total_cost}, "
            f"objective={objective_value}."
        )

    rows = [
        {
            "cost_component":
                "production",

            "cost":
                production_cost,
        },
        {
            "cost_component":
                "holding",

            "cost":
                holding_cost,
        },
        {
            "cost_component":
                "shortage",

            "cost":
                shortage_cost,
        },
        {
            "cost_component":
                "total",

            "cost":
                total_cost,
        },
    ]

    output = pd.DataFrame(
        rows
    )

    output[
        "share_of_total_pct"
    ] = 0.0

    non_total_mask = (
        output[
            "cost_component"
        ]
        .ne(
            "total"
        )
    )

    if total_cost > 0:

        output.loc[
            non_total_mask,
            "share_of_total_pct",
        ] = (
            100.0
            *
            output.loc[
                non_total_mask,
                "cost",
            ]
            /
            total_cost
        )

    output.loc[
        output[
            "cost_component"
        ].eq(
            "total"
        ),
        "share_of_total_pct",
    ] = (
        100.0
        if total_cost > 0
        else 0.0
    )

    return output


# ============================================================
# CAPACITY REPORT
# ============================================================


def summarize_capacity(
    capacity_usage: pd.DataFrame,
    constrained_threshold_pct: float = 99.0,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Create week-level and aggregate plant-capacity metrics.

    A week is labeled constrained when utilization is at or above
    constrained_threshold_pct.
    """

    _require_columns(
        capacity_usage,
        CAPACITY_REQUIRED_COLUMNS,
        "capacity_usage",
    )

    if (
        not np.isfinite(
            constrained_threshold_pct
        )
        or constrained_threshold_pct < 0
        or constrained_threshold_pct > 100
    ):

        raise ValueError(
            "constrained_threshold_pct "
            "must lie between 0 and 100."
        )

    detail = (
        capacity_usage
        .copy()
        .sort_values(
            "week_start"
        )
        .reset_index(
            drop=True
        )
    )

    detail[
        "is_capacity_constrained"
    ] = (
        detail[
            "utilization_pct"
        ]
        >= constrained_threshold_pct
    )

    total_available_capacity = float(
        detail[
            "production_capacity"
        ].sum()
    )

    total_used_capacity = float(
        detail[
            "capacity_used"
        ].sum()
    )

    overall_utilization_pct = (
        _safe_pct(
            total_used_capacity,
            total_available_capacity,
        )
    )

    constrained_weeks = int(
        detail[
            "is_capacity_constrained"
        ].sum()
    )

    summary = pd.DataFrame(
        [
            {
                "planning_weeks":
                    int(
                        len(
                            detail
                        )
                    ),

                "total_available_capacity":
                    total_available_capacity,

                "total_capacity_used":
                    total_used_capacity,

                "overall_utilization_pct":
                    overall_utilization_pct,

                "average_weekly_utilization_pct":
                    float(
                        detail[
                            "utilization_pct"
                        ].mean()
                    ),

                "peak_utilization_pct":
                    float(
                        detail[
                            "utilization_pct"
                        ].max()
                    ),

                "minimum_utilization_pct":
                    float(
                        detail[
                            "utilization_pct"
                        ].min()
                    ),

                "constrained_weeks":
                    constrained_weeks,

                "constrained_weeks_pct":
                    _safe_pct(
                        constrained_weeks,
                        len(
                            detail
                        ),
                    ),

                "total_capacity_slack":
                    float(
                        detail[
                            "capacity_slack"
                        ].sum()
                    ),
            }
        ]
    )

    return (
        summary,
        detail,
    )


# ============================================================
# SKU REPORT
# ============================================================


def summarize_skus(
    plan: pd.DataFrame,
    sku_parameters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create SKU-level optimization metrics.

    Metrics include:

        total forecast demand
        total production
        initial inventory
        initial backlog
        final inventory
        final backlog
        peak backlog
        average inventory
        service %
        backlog-free weeks %
        total costs
    """

    _require_columns(
        plan,
        PLAN_REQUIRED_COLUMNS,
        "plan",
    )

    sku_clean = (
        validate_sku_parameters(
            sku_parameters
        )
    )

    plan_sorted = (
        plan
        .sort_values(
            [
                "sku_id",
                "forecast_horizon",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    aggregate = (
        plan_sorted
        .groupby(
            "sku_id",
            observed=True,
        )
        .agg(
            total_forecast_demand=(
                "forecast_demand",
                "sum",
            ),

            total_production=(
                "production_quantity",
                "sum",
            ),

            average_weekly_production=(
                "production_quantity",
                "mean",
            ),

            average_ending_inventory=(
                "ending_inventory",
                "mean",
            ),

            peak_ending_inventory=(
                "ending_inventory",
                "max",
            ),

            average_ending_backlog=(
                "ending_backlog",
                "mean",
            ),

            peak_ending_backlog=(
                "ending_backlog",
                "max",
            ),

            production_cost=(
                "production_cost",
                "sum",
            ),

            holding_cost=(
                "holding_cost",
                "sum",
            ),

            shortage_cost=(
                "shortage_cost",
                "sum",
            ),

            total_cost=(
                "total_cost",
                "sum",
            ),

            planning_weeks=(
                "forecast_horizon",
                "nunique",
            ),
        )
        .reset_index()
    )

    final_state = (
        plan_sorted
        .groupby(
            "sku_id",
            observed=True,
        )
        .tail(1)[
            [
                "sku_id",
                "ending_inventory",
                "ending_backlog",
            ]
        ]
        .rename(
            columns={
                "ending_inventory":
                    "final_inventory",

                "ending_backlog":
                    "final_backlog",
            }
        )
    )

    backlog_free_weeks = (
        plan_sorted
        .assign(
            backlog_free=lambda df:
            (
                df[
                    "ending_backlog"
                ]
                <= 1e-8
            )
        )
        .groupby(
            "sku_id",
            observed=True,
        )
        .agg(
            backlog_free_weeks=(
                "backlog_free",
                "sum",
            )
        )
        .reset_index()
    )

    parameters = (
        sku_clean[
            [
                "sku_id",
                "initial_inventory",
                "initial_backlog",
                "unit_production_cost",
                "unit_holding_cost",
                "unit_shortage_cost",
                "processing_time",
                "max_production_per_week",
            ]
        ]
        .copy()
    )

    output = (
        aggregate
        .merge(
            final_state,
            on="sku_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            backlog_free_weeks,
            on="sku_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            parameters,
            on="sku_id",
            how="left",
            validate="one_to_one",
        )
    )

    output[
        "service_pct"
    ] = output.apply(
        lambda row:
        _service_pct(
            total_demand=(
                row[
                    "total_forecast_demand"
                ]
            ),
            initial_backlog=(
                row[
                    "initial_backlog"
                ]
            ),
            final_backlog=(
                row[
                    "final_backlog"
                ]
            ),
        ),
        axis=1,
    )

    output[
        "backlog_free_weeks_pct"
    ] = (
        100.0
        *
        output[
            "backlog_free_weeks"
        ]
        /
        output[
            "planning_weeks"
        ]
    )

    return (
        output
        .sort_values(
            [
                "service_pct",
                "total_forecast_demand",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================


def build_executive_summary(
    result: ProductionPlanResult,
    plan: pd.DataFrame,
    capacity_summary: pd.DataFrame,
    sku_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the compact KPI table intended for the dashboard header.
    """

    total_forecast_demand = float(
        plan[
            "forecast_demand"
        ].sum()
    )

    total_production = float(
        plan[
            "production_quantity"
        ].sum()
    )

    final_inventory = float(
        sku_summary[
            "final_inventory"
        ].sum()
    )

    final_backlog = float(
        sku_summary[
            "final_backlog"
        ].sum()
    )

    initial_backlog = float(
        sku_summary[
            "initial_backlog"
        ].sum()
    )

    total_demand_obligation = (
        total_forecast_demand
        +
        initial_backlog
    )

    portfolio_service_pct = (
        _service_pct(
            total_demand=(
                total_forecast_demand
            ),
            initial_backlog=(
                initial_backlog
            ),
            final_backlog=(
                final_backlog
            ),
        )
    )

    backlog_free_rows = int(
        (
            plan[
                "ending_backlog"
            ]
            <= 1e-8
        ).sum()
    )

    backlog_free_sku_week_pct = (
        _safe_pct(
            backlog_free_rows,
            len(
                plan
            ),
        )
    )

    capacity_row = (
        capacity_summary
        .iloc[0]
    )

    summary = pd.DataFrame(
        [
            {
                "solver_status":
                    result.solver_status,

                "termination_condition":
                    result.termination_condition,

                "objective_value":
                    float(
                        result.objective_value
                    ),

                "total_forecast_demand":
                    total_forecast_demand,

                "initial_backlog":
                    initial_backlog,

                "total_demand_obligation":
                    total_demand_obligation,

                "total_production":
                    total_production,

                "final_inventory":
                    final_inventory,

                "final_backlog":
                    final_backlog,

                "portfolio_service_pct":
                    portfolio_service_pct,

                "backlog_free_sku_week_pct":
                    backlog_free_sku_week_pct,

                "overall_capacity_utilization_pct":
                    float(
                        capacity_row[
                            "overall_utilization_pct"
                        ]
                    ),

                "peak_capacity_utilization_pct":
                    float(
                        capacity_row[
                            "peak_utilization_pct"
                        ]
                    ),

                "constrained_weeks":
                    int(
                        capacity_row[
                            "constrained_weeks"
                        ]
                    ),

                "planning_weeks":
                    int(
                        capacity_row[
                            "planning_weeks"
                        ]
                    ),

                "skus":
                    int(
                        plan[
                            "sku_id"
                        ].nunique()
                    ),
            }
        ]
    )

    return summary


# ============================================================
# COMPLETE REPORT
# ============================================================


def build_planning_report(
    result: ProductionPlanResult,
    sku_parameters: pd.DataFrame,
    constrained_threshold_pct: float = 99.0,
) -> PlanningReport:
    """
    Convert a solved production plan into all reporting artifacts.

    This is the primary reporting entry point for scripts and the
    future dashboard.
    """

    (
        plan,
        capacity_usage,
        sku_clean,
    ) = validate_reporting_inputs(
        result=result,
        sku_parameters=sku_parameters,
    )

    cost_summary = (
        summarize_costs(
            plan=plan,
            objective_value=(
                result.objective_value
            ),
        )
    )

    (
        capacity_summary,
        capacity_detail,
    ) = summarize_capacity(
        capacity_usage=capacity_usage,
        constrained_threshold_pct=(
            constrained_threshold_pct
        ),
    )

    sku_summary = (
        summarize_skus(
            plan=plan,
            sku_parameters=sku_clean,
        )
    )

    executive_summary = (
        build_executive_summary(
            result=result,
            plan=plan,
            capacity_summary=(
                capacity_summary
            ),
            sku_summary=(
                sku_summary
            ),
        )
    )

    return PlanningReport(
        executive_summary=(
            executive_summary
        ),

        cost_summary=(
            cost_summary
        ),

        capacity_summary=(
            capacity_summary
        ),

        capacity_detail=(
            capacity_detail
        ),

        sku_summary=(
            sku_summary
        ),
    )