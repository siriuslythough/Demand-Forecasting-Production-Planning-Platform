from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from new_demfor_planopti.optimization.contracts import (
    PlanningConfig,
    validate_capacity_frame,
    validate_forecast_frame,
    validate_planning_inputs,
    validate_sku_parameters,
)


# ============================================================
# DEFAULT SCENARIO CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class PlanningTemplateConfig:
    """
    Reproducible default assumptions for a demonstration production-
    planning scenario.

    IMPORTANT
    ---------
    These values are NOT estimated from the M5 dataset.

    M5 provides retail demand and calendar/price information, but it
    does not provide:

        - manufacturing costs,
        - holding costs,
        - shortage penalties,
        - plant processing times,
        - available production capacity,
        - initial inventory.

    These defaults therefore exist only to:

        1. make the optimization engine runnable end-to-end,
        2. provide sensible dashboard defaults,
        3. support automated testing,
        4. allow users to overwrite assumptions later.

    Parameters
    ----------
    initial_inventory_weeks:
        Initial inventory expressed as a multiple of each SKU's
        average weekly forecast demand.

    initial_backlog:
        Default backlog before the first planning week.

    unit_production_cost:
        Default cost of producing one unit.

    weekly_holding_rate:
        Holding cost as a fraction of production cost per week.

    shortage_cost_multiplier:
        Shortage/backlog cost as a multiple of production cost.

    processing_time:
        Capacity units consumed per produced unit.

    max_production_multiplier:
        SKU-specific production ceiling expressed as a multiple of
        that SKU's maximum weekly forecast demand.

    plant_capacity_multiplier:
        Aggregate plant capacity relative to average forecast-implied
        weekly workload.

        A value slightly above 1.0 creates a useful planning problem:
        average demand is serviceable, while peak weeks may require
        inventory prebuilding.

    minimum_sku_production:
        Prevents zero-demand SKUs from receiving a zero production
        ceiling.

    minimum_plant_capacity:
        Prevents an entirely zero-demand forecast from creating zero
        plant capacity.
    """

    initial_inventory_weeks: float = 0.10

    initial_backlog: float = 0.0

    unit_production_cost: float = 5.0

    weekly_holding_rate: float = 0.02

    shortage_cost_multiplier: float = 4.0

    processing_time: float = 1.0

    max_production_multiplier: float = 1.50

    plant_capacity_multiplier: float = 1.00

    minimum_sku_production: float = 1.0

    minimum_plant_capacity: float = 1.0

    round_digits: int = 4


# ============================================================
# CONFIG VALIDATION
# ============================================================


def validate_template_config(
    config: PlanningTemplateConfig,
) -> PlanningTemplateConfig:
    """
    Validate synthetic planning assumptions before generating
    optimization input tables.
    """

    nonnegative_fields = {
        "initial_inventory_weeks":
            config.initial_inventory_weeks,

        "initial_backlog":
            config.initial_backlog,

        "unit_production_cost":
            config.unit_production_cost,

        "weekly_holding_rate":
            config.weekly_holding_rate,
    }

    positive_fields = {
        "shortage_cost_multiplier":
            config.shortage_cost_multiplier,

        "processing_time":
            config.processing_time,

        "max_production_multiplier":
            config.max_production_multiplier,

        "plant_capacity_multiplier":
            config.plant_capacity_multiplier,

        "minimum_sku_production":
            config.minimum_sku_production,

        "minimum_plant_capacity":
            config.minimum_plant_capacity,
    }

    for (
        name,
        value,
    ) in nonnegative_fields.items():

        if not np.isfinite(value):

            raise ValueError(
                f"{name} must be finite."
            )

        if value < 0:

            raise ValueError(
                f"{name} must be nonnegative."
            )

    for (
        name,
        value,
    ) in positive_fields.items():

        if not np.isfinite(value):

            raise ValueError(
                f"{name} must be finite."
            )

        if value <= 0:

            raise ValueError(
                f"{name} must be > 0."
            )

    if (
        not isinstance(
            config.round_digits,
            int,
        )
        or config.round_digits < 0
    ):

        raise ValueError(
            "round_digits must be a "
            "nonnegative integer."
        )

    return config


# ============================================================
# DEFAULT SKU PARAMETER TEMPLATE
# ============================================================


def make_default_sku_parameters(
    forecast: pd.DataFrame,
    planning_config: PlanningConfig | None = None,
    template_config: PlanningTemplateConfig | None = None,
) -> pd.DataFrame:
    """
    Generate default SKU-level planning parameters.

    The demand forecast determines only SCALE-related quantities:

        initial inventory
        max production per week

    Economic assumptions remain explicit scenario parameters rather
    than being falsely inferred from M5 demand.

    Returns
    -------
    pd.DataFrame
        One row per SKU with the contract required by the optimizer.
    """

    if planning_config is None:

        planning_config = (
            PlanningConfig()
        )

    if template_config is None:

        template_config = (
            PlanningTemplateConfig()
        )

    validate_template_config(
        template_config
    )

    forecast_clean = (
        validate_forecast_frame(
            forecast=forecast,
            config=planning_config,
        )
    )

    demand_summary = (
        forecast_clean
        .groupby(
            "sku_id",
            observed=True,
        )
        .agg(
            mean_forecast_demand=(
                "forecast_demand",
                "mean",
            ),
            peak_forecast_demand=(
                "forecast_demand",
                "max",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Initial inventory
    # --------------------------------------------------------
    #
    # Example:
    #
    #     initial_inventory_weeks = 0.5
    #
    # means the planning horizon starts with approximately half a
    # week of average forecast demand already available.
    # --------------------------------------------------------

    demand_summary[
        "initial_inventory"
    ] = (
        demand_summary[
            "mean_forecast_demand"
        ]
        *
        template_config.initial_inventory_weeks
    )


    # --------------------------------------------------------
    # Economic assumptions
    # --------------------------------------------------------

    demand_summary[
        "initial_backlog"
    ] = (
        template_config.initial_backlog
    )

    demand_summary[
        "unit_production_cost"
    ] = (
        template_config.unit_production_cost
    )

    demand_summary[
        "unit_holding_cost"
    ] = (
        template_config.unit_production_cost
        *
        template_config.weekly_holding_rate
    )

    demand_summary[
        "unit_shortage_cost"
    ] = (
        template_config.unit_production_cost
        *
        template_config.shortage_cost_multiplier
    )


    # --------------------------------------------------------
    # Capacity consumption
    # --------------------------------------------------------

    demand_summary[
        "processing_time"
    ] = (
        template_config.processing_time
    )


    # --------------------------------------------------------
    # SKU-specific production ceiling
    # --------------------------------------------------------
    #
    # We allow production above the largest forecast week so that the
    # LP can intentionally prebuild inventory ahead of capacity peaks.
    # --------------------------------------------------------

    demand_summary[
        "max_production_per_week"
    ] = np.maximum(
        demand_summary[
            "peak_forecast_demand"
        ]
        *
        template_config.max_production_multiplier,

        template_config.minimum_sku_production,
    )


    output_columns = [
        "sku_id",
        "initial_inventory",
        "initial_backlog",
        "unit_production_cost",
        "unit_holding_cost",
        "unit_shortage_cost",
        "processing_time",
        "max_production_per_week",
    ]

    output = (
        demand_summary[
            output_columns
        ]
        .copy()
    )

    numeric_columns = [
        column
        for column
        in output_columns
        if column != "sku_id"
    ]

    output[
        numeric_columns
    ] = output[
        numeric_columns
    ].round(
        template_config.round_digits
    )

    return (
        validate_sku_parameters(
            output
        )
    )


# ============================================================
# DEFAULT WEEKLY CAPACITY TEMPLATE
# ============================================================


def make_default_capacity(
    forecast: pd.DataFrame,
    sku_parameters: pd.DataFrame,
    planning_config: PlanningConfig | None = None,
    template_config: PlanningTemplateConfig | None = None,
) -> pd.DataFrame:
    """
    Generate aggregate weekly plant capacity.

    Default design
    --------------
    Capacity is constant across the planning horizon and based on:

        average forecast-implied weekly workload
        × plant_capacity_multiplier

    where workload is:

        forecast_demand × processing_time

    summed across SKUs.

    Why constant capacity?
    ----------------------
    If capacity were simply set equal to each week's demand workload,
    there would be little intertemporal production-planning problem.

    A roughly fixed plant capacity creates meaningful peak/valley
    behavior:

        low-demand weeks
            -> potential inventory prebuild

        high-demand weeks
            -> inventory drawdown / possible backlog

    This makes the LP useful without manufacturing arbitrary weekly
    capacity shocks.
    """

    if planning_config is None:

        planning_config = (
            PlanningConfig()
        )

    if template_config is None:

        template_config = (
            PlanningTemplateConfig()
        )

    validate_template_config(
        template_config
    )

    forecast_clean = (
        validate_forecast_frame(
            forecast=forecast,
            config=planning_config,
        )
    )

    sku_clean = (
        validate_sku_parameters(
            sku_parameters
        )
    )


    # --------------------------------------------------------
    # Ensure every forecast SKU has processing-time information
    # --------------------------------------------------------

    forecast_skus = set(
        forecast_clean[
            "sku_id"
        ].unique()
    )

    parameter_skus = set(
        sku_clean[
            "sku_id"
        ].unique()
    )

    missing = (
        forecast_skus
        - parameter_skus
    )

    if missing:

        raise ValueError(
            "Cannot generate capacity: "
            "missing SKU parameters for "
            f"{sorted(missing)}"
        )


    # --------------------------------------------------------
    # Convert forecast demand into required capacity workload
    # --------------------------------------------------------

    workload = (
        forecast_clean
        .merge(
            sku_clean[
                [
                    "sku_id",
                    "processing_time",
                ]
            ],
            on="sku_id",
            how="left",
            validate="many_to_one",
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

    weekly_workload = (
        workload
        .groupby(
            "week_start",
            as_index=False,
            observed=True,
        )
        .agg(
            forecast_workload=(
                "required_capacity",
                "sum",
            )
        )
        .sort_values(
            "week_start"
        )
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # Constant baseline plant capacity
    # --------------------------------------------------------

    average_workload = float(
        weekly_workload[
            "forecast_workload"
        ].mean()
    )

    production_capacity = max(
        average_workload
        *
        template_config.plant_capacity_multiplier,

        template_config.minimum_plant_capacity,
    )

    weekly_workload[
        "production_capacity"
    ] = production_capacity

    output = (
        weekly_workload[
            [
                "week_start",
                "production_capacity",
            ]
        ]
        .copy()
    )

    output[
        "production_capacity"
    ] = (
        output[
            "production_capacity"
        ]
        .round(
            template_config.round_digits
        )
    )

    return (
        validate_capacity_frame(
            output
        )
    )


# ============================================================
# COMPLETE DEFAULT SCENARIO
# ============================================================


def make_default_planning_inputs(
    forecast: pd.DataFrame,
    planning_config: PlanningConfig | None = None,
    template_config: PlanningTemplateConfig | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build a complete, validated optimization scenario from a forecast.

    Returns
    -------
    forecast_clean
    sku_parameters
    capacity

    This is the function the future dashboard can call immediately
    after forecasting when the user wants default planning assumptions.
    """

    if planning_config is None:

        planning_config = (
            PlanningConfig()
        )

    if template_config is None:

        template_config = (
            PlanningTemplateConfig()
        )

    validate_template_config(
        template_config
    )

    forecast_clean = (
        validate_forecast_frame(
            forecast=forecast,
            config=planning_config,
        )
    )

    sku_parameters = (
        make_default_sku_parameters(
            forecast=forecast_clean,
            planning_config=planning_config,
            template_config=template_config,
        )
    )

    capacity = (
        make_default_capacity(
            forecast=forecast_clean,
            sku_parameters=sku_parameters,
            planning_config=planning_config,
            template_config=template_config,
        )
    )

    return (
        validate_planning_inputs(
            forecast=forecast_clean,
            sku_parameters=sku_parameters,
            capacity=capacity,
            config=planning_config,
        )
    )