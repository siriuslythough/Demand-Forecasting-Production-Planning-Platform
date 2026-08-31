from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from pyomo.opt import (
    SolverStatus,
    TerminationCondition,
)

from new_demfor_planopti.optimization.contracts import (
    PlanningConfig,
    validate_planning_inputs,
)


# ============================================================
# RESULT CONTRACT
# ============================================================


@dataclass(frozen=True)
class ProductionPlanResult:
    """
    Structured result returned by the production-planning optimizer.

    Attributes
    ----------
    solver_status:
        Solver-level execution status.

    termination_condition:
        Mathematical termination condition reported by HiGHS.

    objective_value:
        Minimum total planning cost.

    plan:
        SKU-week production plan.

    capacity_usage:
        Week-level plant-capacity utilization.
    """

    solver_status: str

    termination_condition: str

    objective_value: float

    plan: pd.DataFrame

    capacity_usage: pd.DataFrame


# ============================================================
# INTERNAL VALIDATION
# ============================================================


def _validate_horizon_week_mapping(
    forecast: pd.DataFrame,
    config: PlanningConfig,
) -> dict[int, pd.Timestamp]:
    """
    Verify that every forecast horizon refers to exactly one calendar
    week and that the horizon/week mapping is chronological.

    Example
    -------
    horizon 1 -> 2026-01-03
    horizon 2 -> 2026-01-10
    ...
    horizon 13 -> 2026-03-28
    """

    horizon_week_counts = (
        forecast
        .groupby(
            "forecast_horizon",
            observed=True,
        )[
            "week_start"
        ]
        .nunique()
    )

    if not horizon_week_counts.eq(1).all():

        raise ValueError(
            "Each forecast_horizon must map "
            "to exactly one week_start."
        )

    week_horizon_counts = (
        forecast
        .groupby(
            "week_start",
            observed=True,
        )[
            "forecast_horizon"
        ]
        .nunique()
    )

    if not week_horizon_counts.eq(1).all():

        raise ValueError(
            "Each planning week must map "
            "to exactly one forecast_horizon."
        )

    mapping_frame = (
        forecast[
            [
                "forecast_horizon",
                "week_start",
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

    expected_horizons = list(
        range(
            1,
            config.planning_horizon + 1,
        )
    )

    actual_horizons = (
        mapping_frame[
            "forecast_horizon"
        ]
        .tolist()
    )

    if actual_horizons != expected_horizons:

        raise ValueError(
            "Planning-horizon mapping is "
            "incomplete or incorrectly ordered."
        )

    weeks = (
        mapping_frame[
            "week_start"
        ]
        .tolist()
    )

    if weeks != sorted(weeks):

        raise ValueError(
            "week_start must increase with "
            "forecast_horizon."
        )

    if len(weeks) > 1:

        week_differences = (
            mapping_frame[
                "week_start"
            ]
            .diff()
            .dropna()
        )

        expected_difference = (
            pd.Timedelta(
                days=7
            )
        )

        if not (
            week_differences
            == expected_difference
        ).all():

            raise ValueError(
                "Production planning currently "
                "requires consecutive 7-day "
                "weekly forecast periods."
            )

    return {
        int(row.forecast_horizon):
        pd.Timestamp(
            row.week_start
        )
        for row
        in mapping_frame.itertuples(
            index=False
        )
    }


# ============================================================
# LOOKUP BUILDERS
# ============================================================


def _make_forecast_lookup(
    forecast: pd.DataFrame,
) -> dict[
    tuple[str, int],
    float,
]:
    """
    Convert the validated forecast panel into:

        (sku_id, horizon) -> forecast demand
    """

    return {
        (
            str(row.sku_id),
            int(
                row.forecast_horizon
            ),
        ):
        float(
            row.forecast_demand
        )
        for row
        in forecast.itertuples(
            index=False
        )
    }


def _make_sku_lookup(
    sku_parameters: pd.DataFrame,
    column: str,
) -> dict[str, float]:
    """
    Create a SKU -> parameter lookup.
    """

    return {
        str(row.sku_id):
        float(
            getattr(
                row,
                column,
            )
        )
        for row
        in sku_parameters.itertuples(
            index=False
        )
    }


def _make_capacity_lookup(
    capacity: pd.DataFrame,
    horizon_week_map: dict[
        int,
        pd.Timestamp,
    ],
) -> dict[int, float]:
    """
    Convert calendar-week capacity into a horizon-indexed lookup.
    """

    week_capacity = {
        pd.Timestamp(
            row.week_start
        ):
        float(
            row.production_capacity
        )
        for row
        in capacity.itertuples(
            index=False
        )
    }

    return {
        horizon:
        week_capacity[
            week
        ]
        for (
            horizon,
            week,
        ) in horizon_week_map.items()
    }


# ============================================================
# PYOMO MODEL BUILDER
# ============================================================


def build_production_model(
    forecast: pd.DataFrame,
    sku_parameters: pd.DataFrame,
    capacity: pd.DataFrame,
    config: PlanningConfig | None = None,
) -> pyo.ConcreteModel:
    """
    Construct the deterministic multi-period production-planning LP.

    Decision variables
    ------------------
    production[i,t]:
        Units of SKU i produced in week t.

    inventory[i,t]:
        Ending on-hand inventory of SKU i after week t.

    backlog[i,t]:
        Ending unmet/backlogged demand for SKU i after week t.


    Objective
    ---------
    Minimize:

        production cost
        + holding cost
        + backlog/shortage cost


    Inventory-flow constraint
    -------------------------

        inventory[i,t] - backlog[i,t]

            =

        previous inventory
        - previous backlog
        + production[i,t]
        - forecast demand[i,t]


    Capacity constraint
    -------------------

        sum_i(
            processing_time[i]
            * production[i,t]
        )

        <= plant_capacity[t]


    SKU production ceiling
    ----------------------

        production[i,t]
        <= max_production_per_week[i]
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

    horizon_week_map = (
        _validate_horizon_week_mapping(
            forecast=forecast_clean,
            config=config,
        )
    )

    skus = sorted(
        forecast_clean[
            "sku_id"
        ]
        .unique()
        .tolist()
    )

    periods = list(
        range(
            1,
            config.planning_horizon + 1,
        )
    )

    demand_lookup = (
        _make_forecast_lookup(
            forecast_clean
        )
    )

    initial_inventory_lookup = (
        _make_sku_lookup(
            sku_clean,
            "initial_inventory",
        )
    )

    initial_backlog_lookup = (
        _make_sku_lookup(
            sku_clean,
            "initial_backlog",
        )
    )

    production_cost_lookup = (
        _make_sku_lookup(
            sku_clean,
            "unit_production_cost",
        )
    )

    holding_cost_lookup = (
        _make_sku_lookup(
            sku_clean,
            "unit_holding_cost",
        )
    )

    shortage_cost_lookup = (
        _make_sku_lookup(
            sku_clean,
            "unit_shortage_cost",
        )
    )

    processing_time_lookup = (
        _make_sku_lookup(
            sku_clean,
            "processing_time",
        )
    )

    max_production_lookup = (
        _make_sku_lookup(
            sku_clean,
            "max_production_per_week",
        )
    )

    capacity_lookup = (
        _make_capacity_lookup(
            capacity=capacity_clean,
            horizon_week_map=horizon_week_map,
        )
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = pyo.ConcreteModel(
        name="multi_period_production_planning"
    )

    # ========================================================
    # LP DUAL VALUES
    # ========================================================
    #
    # Import shadow prices from HiGHS after solving.
    #
    # These are meaningful for the current continuous LP.
    # If the model is later extended to a MILP with binary setup

    # variables, ordinary LP dual interpretation will no longer apply
    # directly.
    # ========================================================

    model.dual = pyo.Suffix(
        direction=pyo.Suffix.IMPORT
    )


    # ========================================================
    # SETS
    # ========================================================

    model.SKUS = pyo.Set(
        initialize=skus,
        ordered=True,
    )

    model.PERIODS = pyo.Set(
        initialize=periods,
        ordered=True,
    )


    # ========================================================
    # PARAMETERS
    # ========================================================

    model.demand = pyo.Param(
        model.SKUS,
        model.PERIODS,
        initialize=demand_lookup,
        within=pyo.NonNegativeReals,
    )

    model.initial_inventory = pyo.Param(
        model.SKUS,
        initialize=initial_inventory_lookup,
        within=pyo.NonNegativeReals,
    )

    model.initial_backlog = pyo.Param(
        model.SKUS,
        initialize=initial_backlog_lookup,
        within=pyo.NonNegativeReals,
    )

    model.unit_production_cost = pyo.Param(
        model.SKUS,
        initialize=production_cost_lookup,
        within=pyo.NonNegativeReals,
    )

    model.unit_holding_cost = pyo.Param(
        model.SKUS,
        initialize=holding_cost_lookup,
        within=pyo.NonNegativeReals,
    )

    model.unit_shortage_cost = pyo.Param(
        model.SKUS,
        initialize=shortage_cost_lookup,
        within=pyo.NonNegativeReals,
    )

    model.processing_time = pyo.Param(
        model.SKUS,
        initialize=processing_time_lookup,
        within=pyo.PositiveReals,
    )

    model.max_production = pyo.Param(
        model.SKUS,
        initialize=max_production_lookup,
        within=pyo.PositiveReals,
    )

    model.production_capacity = pyo.Param(
        model.PERIODS,
        initialize=capacity_lookup,
        within=pyo.NonNegativeReals,
    )


    # ========================================================
    # DECISION VARIABLES
    # ========================================================

    def production_bounds(
        current_model: pyo.ConcreteModel,
        sku_id: str,
        period: int,
    ) -> tuple[
        float,
        float,
    ]:

        return (
            0.0,
            float(
                pyo.value(
                    current_model.max_production[
                        sku_id
                    ]
                )
            ),
        )

    model.production = pyo.Var(
        model.SKUS,
        model.PERIODS,
        domain=pyo.NonNegativeReals,
        bounds=production_bounds,
    )

    model.inventory = pyo.Var(
        model.SKUS,
        model.PERIODS,
        domain=pyo.NonNegativeReals,
    )

    model.backlog = pyo.Var(
        model.SKUS,
        model.PERIODS,
        domain=pyo.NonNegativeReals,
    )


    # ========================================================
    # INVENTORY / BACKLOG BALANCE
    # ========================================================

    def inventory_balance_rule(
        current_model: pyo.ConcreteModel,
        sku_id: str,
        period: int,
    ) -> pyo.Constraint:

        if period == 1:

            previous_net_inventory = (
                current_model.initial_inventory[
                    sku_id
                ]
                -
                current_model.initial_backlog[
                    sku_id
                ]
            )

        else:

            previous_net_inventory = (
                current_model.inventory[
                    sku_id,
                    period - 1,
                ]
                -
                current_model.backlog[
                    sku_id,
                    period - 1,
                ]
            )

        return (
            current_model.inventory[
                sku_id,
                period,
            ]
            -
            current_model.backlog[
                sku_id,
                period,
            ]

            ==

            previous_net_inventory
            +
            current_model.production[
                sku_id,
                period,
            ]
            -
            current_model.demand[
                sku_id,
                period,
            ]
        )

    model.inventory_balance = (
        pyo.Constraint(
            model.SKUS,
            model.PERIODS,
            rule=inventory_balance_rule,
        )
    )


    # ========================================================
    # SHARED PLANT CAPACITY
    # ========================================================

    def capacity_rule(
        current_model: pyo.ConcreteModel,
        period: int,
    ) -> pyo.Constraint:

        return (
            sum(
                current_model.processing_time[
                    sku_id
                ]
                *
                current_model.production[
                    sku_id,
                    period,
                ]

                for sku_id
                in current_model.SKUS
            )

            <=

            current_model.production_capacity[
                period
            ]
        )

    model.capacity_constraint = (
        pyo.Constraint(
            model.PERIODS,
            rule=capacity_rule,
        )
    )


    # ========================================================
    # OBJECTIVE COMPONENT EXPRESSIONS
    # ========================================================

    model.production_cost = (
        pyo.Expression(
            expr=sum(
                model.unit_production_cost[
                    sku_id
                ]
                *
                model.production[
                    sku_id,
                    period,
                ]

                for sku_id
                in model.SKUS

                for period
                in model.PERIODS
            )
        )
    )

    model.holding_cost = (
        pyo.Expression(
            expr=sum(
                model.unit_holding_cost[
                    sku_id
                ]
                *
                model.inventory[
                    sku_id,
                    period,
                ]

                for sku_id
                in model.SKUS

                for period
                in model.PERIODS
            )
        )
    )

    model.shortage_cost = (
        pyo.Expression(
            expr=sum(
                model.unit_shortage_cost[
                    sku_id
                ]
                *
                model.backlog[
                    sku_id,
                    period,
                ]

                for sku_id
                in model.SKUS

                for period
                in model.PERIODS
            )
        )
    )


    # ========================================================
    # OBJECTIVE
    # ========================================================

    model.total_cost = (
        pyo.Objective(
            expr=(
                model.production_cost
                +
                model.holding_cost
                +
                model.shortage_cost
            ),
            sense=pyo.minimize,
        )
    )


    # ========================================================
    # ATTACH CLEAN METADATA FOR EXTRACTION
    # ========================================================
    #
    # These are ordinary Python attributes rather than Pyomo
    # optimization components.
    # ========================================================

    model._horizon_week_map = (
        horizon_week_map
    )

    model._forecast_clean = (
        forecast_clean
    )

    model._sku_parameters_clean = (
        sku_clean
    )

    model._capacity_clean = (
        capacity_clean
    )

    return model


# ============================================================
# SOLVER ACCESS
# ============================================================


def get_highs_solver():
    """
    Create the Pyomo HiGHS solver interface and verify availability.

    `highspy` should be installed in the project environment.
    """

    solver = pyo.SolverFactory(
        "highs"
    )

    if not solver.available(
        exception_flag=False
    ):

        raise RuntimeError(
            "HiGHS is not available through "
            "Pyomo. Install the project "
            "dependencies and verify that "
            "`highspy` is available in the "
            "active Python environment."
        )

    return solver


# ============================================================
# NUMERICAL CLEANUP
# ============================================================


def _clean_solver_value(
    value: float,
    tolerance: float = 1e-8,
) -> float:
    """
    Remove harmless tiny floating-point values returned by the LP
    solver.

    Example:

        -2e-12 -> 0
    """

    value = float(
        value
    )

    if abs(value) <= tolerance:

        return 0.0

    return value


# ============================================================
# SOLUTION EXTRACTION
# ============================================================


def extract_production_plan(
    model: pyo.ConcreteModel,
) -> pd.DataFrame:
    """
    Convert solved Pyomo variables into a tidy SKU-week plan.
    """

    rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    horizon_week_map = (
        model._horizon_week_map
    )

    for period in model.PERIODS:

        week_start = (
            horizon_week_map[
                int(period)
            ]
        )

        for sku_id in model.SKUS:

            production = (
                _clean_solver_value(
                    pyo.value(
                        model.production[
                            sku_id,
                            period,
                        ]
                    )
                )
            )

            inventory = (
                _clean_solver_value(
                    pyo.value(
                        model.inventory[
                            sku_id,
                            period,
                        ]
                    )
                )
            )

            backlog = (
                _clean_solver_value(
                    pyo.value(
                        model.backlog[
                            sku_id,
                            period,
                        ]
                    )
                )
            )

            demand = float(
                pyo.value(
                    model.demand[
                        sku_id,
                        period,
                    ]
                )
            )

            processing_time = float(
                pyo.value(
                    model.processing_time[
                        sku_id
                    ]
                )
            )

            production_cost = (
                production
                *
                float(
                    pyo.value(
                        model.unit_production_cost[
                            sku_id
                        ]
                    )
                )
            )

            holding_cost = (
                inventory
                *
                float(
                    pyo.value(
                        model.unit_holding_cost[
                            sku_id
                        ]
                    )
                )
            )

            shortage_cost = (
                backlog
                *
                float(
                    pyo.value(
                        model.unit_shortage_cost[
                            sku_id
                        ]
                    )
                )
            )

            rows.append(
                {
                    "week_start":
                        pd.Timestamp(
                            week_start
                        ),

                    "forecast_horizon":
                        int(period),

                    "sku_id":
                        str(sku_id),

                    "forecast_demand":
                        demand,

                    "production_quantity":
                        production,

                    "ending_inventory":
                        inventory,

                    "ending_backlog":
                        backlog,

                    "net_inventory":
                        (
                            inventory
                            - backlog
                        ),

                    "capacity_used":
                        (
                            processing_time
                            * production
                        ),

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

    plan = pd.DataFrame(
        rows
    )

    numeric_columns = [
        "forecast_demand",
        "production_quantity",
        "ending_inventory",
        "ending_backlog",
        "net_inventory",
        "capacity_used",
        "production_cost",
        "holding_cost",
        "shortage_cost",
        "total_cost",
    ]

    for column in numeric_columns:

        if not np.isfinite(
            plan[
                column
            ]
        ).all():

            raise RuntimeError(
                "Optimization solution contains "
                f"non-finite values in {column}."
            )

    return (
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


def extract_capacity_usage(
    model: pyo.ConcreteModel,
    plan: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate the SKU-level production plan into weekly capacity usage.
    """

    capacity = (
        model._capacity_clean
        .copy()
    )

    used = (
        plan
        .groupby(
            [
                "week_start",
                "forecast_horizon",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            capacity_used=(
                "capacity_used",
                "sum",
            )
        )
    )

    # ========================================================
    # CAPACITY SHADOW PRICES
    # ========================================================

    dual_rows = []

    for period in model.PERIODS:

        constraint = (
            model.capacity_constraint[
                period
            ]
        )

        raw_dual = (
            model.dual.get(
                constraint,
                0.0,
            )
        )

        if raw_dual is None:
            raw_dual = 0.0

        raw_dual = float(
            raw_dual
        )

        week_start = (
            model._horizon_week_map[
                int(period)
            ]
        )

        # For a minimization problem with an upper-bound resource
        # constraint, the imported LP dual is normally non-positive.
        #
        # Report economic marginal value as a positive cost-saving
        # quantity:
        #
        #     marginal value
        #     =
        #     reduction in objective from one extra capacity unit
        #
        capacity_marginal_value = max(
            0.0,
            -raw_dual,
        )

        dual_rows.append(
            {
                "week_start":
                    pd.Timestamp(
                        week_start
                    ),

                "capacity_dual":
                    raw_dual,

                "capacity_marginal_value":
                    capacity_marginal_value,
            }
        )


    dual_frame = pd.DataFrame(
        dual_rows
    )


    capacity = capacity.merge(
        dual_frame,
        on="week_start",
        how="left",
        validate="one_to_one",
    )

    capacity = (
        capacity
        .merge(
            used,
            on="week_start",
            how="left",
            validate="one_to_one",
        )
    )

    capacity[
        "capacity_used"
    ] = (
        capacity[
            "capacity_used"
        ]
        .fillna(0.0)
    )

    capacity[
        "capacity_slack"
    ] = (
        capacity[
            "production_capacity"
        ]
        -
        capacity[
            "capacity_used"
        ]
    )

    capacity[
        "capacity_slack"
    ] = (
        capacity[
            "capacity_slack"
        ]
        .where(
            capacity[
                "capacity_slack"
            ].abs()
            > 1e-8,
            0.0,
        )
    )

    capacity[
        "utilization_pct"
    ] = np.where(
        capacity[
            "production_capacity"
        ]
        > 0,

        100.0
        *
        capacity[
            "capacity_used"
        ]
        /
        capacity[
            "production_capacity"
        ],

        0.0,
    )

    return (
        capacity
        .sort_values(
            "week_start"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# SOLVE END-TO-END
# ============================================================


def solve_production_plan(
    forecast: pd.DataFrame,
    sku_parameters: pd.DataFrame,
    capacity: pd.DataFrame,
    config: PlanningConfig | None = None,
    tee: bool = False,
) -> ProductionPlanResult:
    """
    Build and solve the deterministic production-planning LP using
    Pyomo + HiGHS.

    Backlog is permitted and penalized, which means insufficient
    capacity does not automatically make the optimization model
    infeasible.

    Returns
    -------
    ProductionPlanResult
        Solver metadata plus SKU-week production and capacity outputs.
    """

    if config is None:

        config = PlanningConfig()

    model = build_production_model(
        forecast=forecast,
        sku_parameters=sku_parameters,
        capacity=capacity,
        config=config,
    )

    solver = get_highs_solver()

    results = solver.solve(
        model,
        tee=tee,
    )


    # ========================================================
    # SOLVER STATUS CHECK
    # ========================================================

    solver_status = (
        results.solver.status
    )

    termination_condition = (
        results.solver.termination_condition
    )

    if (
        solver_status
        != SolverStatus.ok
    ):

        raise RuntimeError(
            "HiGHS did not return solver "
            f"status 'ok'. Status: "
            f"{solver_status}"
        )

    if (
        termination_condition
        != TerminationCondition.optimal
    ):

        raise RuntimeError(
            "Production-planning optimization "
            "did not terminate optimally. "
            "Termination condition: "
            f"{termination_condition}"
        )


    # ========================================================
    # EXTRACT SOLUTION
    # ========================================================

    objective_value = float(
        pyo.value(
            model.total_cost
        )
    )

    if not np.isfinite(
        objective_value
    ):

        raise RuntimeError(
            "Optimization returned a "
            "non-finite objective value."
        )

    plan = (
        extract_production_plan(
            model
        )
    )

    capacity_usage = (
        extract_capacity_usage(
            model=model,
            plan=plan,
        )
    )


    # ========================================================
    # FINAL NUMERICAL CONTRACTS
    # ========================================================

    if (
        plan[
            "production_quantity"
        ]
        < -1e-7
    ).any():

        raise RuntimeError(
            "Solved production plan contains "
            "negative production quantities."
        )

    if (
        plan[
            "ending_inventory"
        ]
        < -1e-7
    ).any():

        raise RuntimeError(
            "Solved production plan contains "
            "negative inventory."
        )

    if (
        plan[
            "ending_backlog"
        ]
        < -1e-7
    ).any():

        raise RuntimeError(
            "Solved production plan contains "
            "negative backlog."
        )

    if (
        capacity_usage[
            "capacity_used"
        ]
        >
        capacity_usage[
            "production_capacity"
        ]
        + 1e-6
    ).any():

        raise RuntimeError(
            "Solved production plan violates "
            "plant capacity."
        )

    plan_cost = float(
        plan[
            "total_cost"
        ].sum()
    )

    if not np.isclose(
        plan_cost,
        objective_value,
        rtol=1e-7,
        atol=1e-6,
    ):

        raise RuntimeError(
            "Extracted plan cost does not "
            "match the Pyomo objective. "
            f"Plan cost={plan_cost}, "
            f"objective={objective_value}."
        )


    return ProductionPlanResult(
        solver_status=str(
            solver_status
        ),

        termination_condition=str(
            termination_condition
        ),

        objective_value=(
            objective_value
        ),

        plan=plan,

        capacity_usage=(
            capacity_usage
        ),
    )