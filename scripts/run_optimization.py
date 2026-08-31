from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from new_demfor_planopti.optimization import (
    PlanningConfig,
    PlanningTemplateConfig,
    build_planning_report,
    compare_to_jit_baseline,
    make_default_planning_inputs,
    run_capacity_sensitivity,
    solve_production_plan,
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================


EXPECTED_HORIZON = 13


ENSEMBLE_MODELS = [
    "sarima_52",
    "theta_52",
    "lightgbm_recursive",
]


FORECAST_ARTIFACT_NAME = (
    "test_predictions.parquet"
)


CAPACITY_SCENARIOS = {
    "Capacity -5%": 0.95,
    "Baseline": 1.00,
    "Capacity +5%": 1.05,
    "Capacity +10%": 1.10,
}


# ============================================================
# PATH HELPERS
# ============================================================


def find_project_root() -> Path:
    """
    Find repository root by searching upward for pyproject.toml.

    This allows the script to work when launched from:

        project root
        scripts/
        notebooks/
        VS Code
    """

    current = Path.cwd().resolve()

    for candidate in [
        current,
        *current.parents,
    ]:

        if (
            candidate
            / "pyproject.toml"
        ).exists():

            return candidate

    script_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    if (
        script_root
        / "pyproject.toml"
    ).exists():

        return script_root

    raise FileNotFoundError(
        "Could not locate project root "
        "containing pyproject.toml."
    )


# ============================================================
# FORECAST LOADING
# ============================================================


def load_forecasting_predictions(
    forecast_dir: Path,
) -> pd.DataFrame:
    """
    Load the latest saved forecasting predictions.

    The historical evaluation artifact may contain realized demand,
    but optimization consumes only:

        model
        week_start
        sku_id
        prediction

    The actual-demand column is never used.
    """

    path = (
        forecast_dir
        / FORECAST_ARTIFACT_NAME
    )

    if not path.exists():

        raise FileNotFoundError(
            "Forecasting artifact not found:\n"
            f"{path}\n\n"
            "Run the forecasting pipeline first."
        )

    predictions = pd.read_parquet(
        path
    )

    required_columns = {
        "model",
        "week_start",
        "sku_id",
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
            "Forecasting artifact is missing "
            f"required columns: {sorted(missing)}"
        )

    predictions[
        "week_start"
    ] = pd.to_datetime(
        predictions[
            "week_start"
        ]
    )

    predictions[
        "sku_id"
    ] = (
        predictions[
            "sku_id"
        ]
        .astype(str)
    )

    predictions[
        "prediction"
    ] = pd.to_numeric(
        predictions[
            "prediction"
        ],
        errors="raise",
    )

    if not np.isfinite(
        predictions[
            "prediction"
        ]
    ).all():

        raise ValueError(
            "Forecast predictions contain "
            "non-finite values."
        )

    return predictions


# ============================================================
# ENSEMBLE FORECAST CONTRACT
# ============================================================


def build_optimization_forecast(
    predictions: pd.DataFrame,
    expected_horizon: int = EXPECTED_HORIZON,
) -> pd.DataFrame:
    """
    Build the optimization-facing demand forecast.

    Final forecasting strategy
    --------------------------
    Equal-weight ensemble of:

        SARIMA
        Theta
        recursive global LightGBM

    Output contract
    ---------------
    week_start
    sku_id
    forecast_demand
    forecast_horizon
    """

    available_models = set(
        predictions[
            "model"
        ].unique()
    )

    missing_models = (
        set(
            ENSEMBLE_MODELS
        )
        - available_models
    )

    if missing_models:

        raise ValueError(
            "Cannot construct final ensemble. "
            "Missing forecasting models: "
            f"{sorted(missing_models)}\n"
            "Available models: "
            f"{sorted(available_models)}"
        )


    # ========================================================
    # SELECT FINAL ENSEMBLE MEMBERS
    # ========================================================

    source = (
        predictions.loc[
            predictions[
                "model"
            ].isin(
                ENSEMBLE_MODELS
            ),
            [
                "model",
                "week_start",
                "sku_id",
                "prediction",
            ],
        ]
        .copy()
    )


    # ========================================================
    # COVERAGE CHECK
    # ========================================================

    coverage = (
        source
        .groupby(
            "model",
            observed=True,
        )
        .agg(
            rows=(
                "prediction",
                "size",
            ),
            skus=(
                "sku_id",
                "nunique",
            ),
            weeks=(
                "week_start",
                "nunique",
            ),
        )
    )

    print(
        "\nEnsemble model coverage:"
    )

    print(
        coverage.to_string()
    )

    unique_weeks = (
        source[
            "week_start"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    unique_skus = (
        source[
            "sku_id"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if (
        len(
            unique_weeks
        )
        != expected_horizon
    ):

        raise ValueError(
            f"Expected {expected_horizon} "
            "forecast weeks, found "
            f"{len(unique_weeks)}."
        )

    expected_rows_per_model = (
        len(
            unique_weeks
        )
        *
        len(
            unique_skus
        )
    )

    if not (
        coverage[
            "rows"
        ]
        .eq(
            expected_rows_per_model
        )
        .all()
    ):

        raise ValueError(
            "One or more ensemble models "
            "do not contain a complete "
            "SKU-week prediction panel."
        )

    if not (
        coverage[
            "weeks"
        ]
        .eq(
            expected_horizon
        )
        .all()
    ):

        raise ValueError(
            "One or more ensemble models "
            "do not contain the complete "
            "planning horizon."
        )

    if not (
        coverage[
            "skus"
        ]
        .eq(
            len(
                unique_skus
            )
        )
        .all()
    ):

        raise ValueError(
            "Ensemble members disagree on "
            "SKU coverage."
        )


    # ========================================================
    # DUPLICATE CHECK
    # ========================================================

    duplicated = (
        source.duplicated(
            subset=[
                "model",
                "week_start",
                "sku_id",
            ],
            keep=False,
        )
    )

    if duplicated.any():

        raise ValueError(
            "Duplicate model/SKU/week forecasts "
            "exist in the forecasting artifact."
        )


    # ========================================================
    # EQUAL-WEIGHT ENSEMBLE
    # ========================================================

    forecast = (
        source
        .groupby(
            [
                "week_start",
                "sku_id",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            forecast_demand=(
                "prediction",
                "mean",
            )
        )
    )

    forecast[
        "forecast_demand"
    ] = (
        forecast[
            "forecast_demand"
        ]
        .clip(
            lower=0.0
        )
    )


    # ========================================================
    # FORECAST HORIZON
    # ========================================================

    horizon_lookup = {
        pd.Timestamp(
            week
        ): horizon

        for (
            horizon,
            week,
        )
        in enumerate(
            unique_weeks,
            start=1,
        )
    }

    forecast[
        "forecast_horizon"
    ] = (
        forecast[
            "week_start"
        ]
        .map(
            horizon_lookup
        )
        .astype(int)
    )

    forecast = (
        forecast[
            [
                "week_start",
                "sku_id",
                "forecast_demand",
                "forecast_horizon",
            ]
        ]
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

    expected_rows = (
        len(
            unique_skus
        )
        *
        expected_horizon
    )

    if (
        len(
            forecast
        )
        != expected_rows
    ):

        raise RuntimeError(
            "Optimization forecast is not "
            "balanced. "
            f"Expected {expected_rows} rows, "
            f"found {len(forecast)}."
        )

    return forecast


# ============================================================
# JSON HELPERS
# ============================================================


def _json_safe(
    value,
):
    """
    Convert NumPy/pandas values into JSON-safe Python values.
    """

    if isinstance(
        value,
        np.generic,
    ):

        return value.item()

    if isinstance(
        value,
        pd.Timestamp,
    ):

        return value.isoformat()

    if isinstance(
        value,
        dict,
    ):

        return {
            str(key):
                _json_safe(
                    item
                )

            for (
                key,
                item,
            )
            in value.items()
        }

    if isinstance(
        value,
        list,
    ):

        return [
            _json_safe(
                item
            )
            for item
            in value
        ]

    return value


# ============================================================
# ARTIFACT SAVING
# ============================================================


def save_optimization_artifacts(
    output_dir: Path,
    forecast: pd.DataFrame,
    sku_parameters: pd.DataFrame,
    capacity: pd.DataFrame,
    result,
    report,
    baseline_comparison,
    capacity_sensitivity,
) -> None:
    """
    Save optimization inputs, optimized outputs, benchmark results,
    scenario sensitivity, and machine-readable summaries.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # OPTIMIZATION INPUTS
    # ========================================================

    forecast.to_parquet(
        output_dir
        / "optimization_forecast.parquet",
        index=False,
    )

    forecast.to_csv(
        output_dir
        / "optimization_forecast.csv",
        index=False,
    )

    sku_parameters.to_csv(
        output_dir
        / "sku_parameters.csv",
        index=False,
    )

    capacity.to_csv(
        output_dir
        / "capacity.csv",
        index=False,
    )


    # ========================================================
    # OPTIMIZED PLAN
    # ========================================================

    result.plan.to_parquet(
        output_dir
        / "production_plan.parquet",
        index=False,
    )

    result.plan.to_csv(
        output_dir
        / "production_plan.csv",
        index=False,
    )


    # ========================================================
    # OPTIMIZATION REPORTS
    # ========================================================

    report.executive_summary.to_csv(
        output_dir
        / "executive_summary.csv",
        index=False,
    )

    report.cost_summary.to_csv(
        output_dir
        / "cost_summary.csv",
        index=False,
    )

    report.capacity_summary.to_csv(
        output_dir
        / "capacity_summary.csv",
        index=False,
    )

    report.capacity_detail.to_csv(
        output_dir
        / "capacity_detail.csv",
        index=False,
    )

    report.sku_summary.to_csv(
        output_dir
        / "sku_summary.csv",
        index=False,
    )


    # ========================================================
    # JIT BASELINE
    # ========================================================

    baseline_comparison.baseline_plan.to_parquet(
        output_dir
        / "jit_baseline_plan.parquet",
        index=False,
    )

    baseline_comparison.baseline_plan.to_csv(
        output_dir
        / "jit_baseline_plan.csv",
        index=False,
    )

    baseline_comparison.comparison.to_csv(
        output_dir
        / "baseline_comparison.csv",
        index=False,
    )

    baseline_comparison.impact_summary.to_csv(
        output_dir
        / "optimization_impact.csv",
        index=False,
    )


    # ========================================================
    # CAPACITY SENSITIVITY
    # ========================================================

    capacity_sensitivity.summary.to_csv(
        output_dir
        / "capacity_sensitivity.csv",
        index=False,
    )


    # ========================================================
    # MACHINE-READABLE SUMMARY
    # ========================================================

    executive_record = (
        report.executive_summary
        .iloc[0]
        .to_dict()
    )

    impact_record = (
        baseline_comparison
        .impact_summary
        .iloc[0]
        .to_dict()
    )

    sensitivity_records = (
        capacity_sensitivity
        .summary
        .to_dict(
            orient="records"
        )
    )

    summary = {
        "solver_status":
            result.solver_status,

        "termination_condition":
            result.termination_condition,

        "objective_value":
            float(
                result.objective_value
            ),

        "forecast_strategy":
            (
                "equal_weight_"
                "sarima_theta_"
                "lightgbm_recursive"
            ),

        "ensemble_models":
            ENSEMBLE_MODELS,

        "planning_horizon":
            int(
                forecast[
                    "forecast_horizon"
                ].nunique()
            ),

        "sku_count":
            int(
                forecast[
                    "sku_id"
                ].nunique()
            ),

        "executive_summary":
            _json_safe(
                executive_record
            ),

        "optimization_impact":
            _json_safe(
                impact_record
            ),

        "capacity_sensitivity":
            _json_safe(
                sensitivity_records
            ),
    }

    with (
        output_dir
        / "optimization_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            default=str,
        )


# ============================================================
# CONSOLE REPORT
# ============================================================


def print_console_summary(
    forecast: pd.DataFrame,
    result,
    report,
    baseline_comparison,
    capacity_sensitivity,
) -> None:
    """
    Compact command-line decision-support report.
    """

    executive = (
        report.executive_summary
        .iloc[0]
    )

    cost_summary = (
        report.cost_summary
        .set_index(
            "cost_component"
        )
    )

    impact = (
        baseline_comparison
        .impact_summary
        .iloc[0]
    )

    sensitivity = (
        capacity_sensitivity
        .summary
    )


    # ========================================================
    # OPTIMIZED PLAN
    # ========================================================

    print()

    print(
        "=" * 72
    )

    print(
        "PRODUCTION PLANNING OPTIMIZATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Solver status: "
        f"{result.solver_status}"
    )

    print(
        f"Termination: "
        f"{result.termination_condition}"
    )

    print()

    print(
        f"SKUs: "
        f"{forecast['sku_id'].nunique()}"
    )

    print(
        f"Planning weeks: "
        f"{forecast['forecast_horizon'].nunique()}"
    )

    print()

    print(
        f"Total forecast demand: "
        f"{executive['total_forecast_demand']:,.2f}"
    )

    print(
        f"Total production: "
        f"{executive['total_production']:,.2f}"
    )

    print(
        f"Final inventory: "
        f"{executive['final_inventory']:,.2f}"
    )

    print(
        f"Final backlog: "
        f"{executive['final_backlog']:,.2f}"
    )

    print(
        f"Portfolio service: "
        f"{executive['portfolio_service_pct']:.2f}%"
    )

    print()

    print(
        f"Overall capacity utilization: "
        f"{executive['overall_capacity_utilization_pct']:.2f}%"
    )

    print(
        f"Peak capacity utilization: "
        f"{executive['peak_capacity_utilization_pct']:.2f}%"
    )

    print(
        f"Capacity-constrained weeks: "
        f"{int(executive['constrained_weeks'])}"
    )


    # ========================================================
    # COST
    # ========================================================

    print()

    print(
        "COST BREAKDOWN"
    )

    print(
        "-" * 72
    )

    print(
        f"Production cost: "
        f"{cost_summary.loc['production', 'cost']:,.2f}"
    )

    print(
        f"Holding cost: "
        f"{cost_summary.loc['holding', 'cost']:,.2f}"
    )

    print(
        f"Shortage cost: "
        f"{cost_summary.loc['shortage', 'cost']:,.2f}"
    )

    print(
        f"Total optimized cost: "
        f"{result.objective_value:,.2f}"
    )


    # ========================================================
    # BUSINESS IMPACT
    # ========================================================

    print()

    print(
        "OPTIMIZATION IMPACT VS JIT BASELINE"
    )

    print(
        "-" * 72
    )

    print(
        f"JIT baseline cost: "
        f"{impact['baseline_total_cost']:,.2f}"
    )

    print(
        f"Optimized cost: "
        f"{impact['optimized_total_cost']:,.2f}"
    )

    print(
        f"Cost savings: "
        f"{impact['cost_savings']:,.2f}"
    )

    print(
        f"Relative cost reduction: "
        f"{impact['cost_savings_pct']:.2f}%"
    )

    print(
        f"Final backlog reduction: "
        f"{impact['backlog_reduction']:,.2f}"
    )

    print(
        f"Horizon service improvement: "
        f"{impact['service_improvement_pp']:.2f} pp"
    )

    print(
        f"Shortage cost avoided: "
        f"{impact['shortage_cost_reduction']:,.2f}"
    )


    # ========================================================
    # CAPACITY SENSITIVITY
    # ========================================================

    print()

    print(
        "CAPACITY SENSITIVITY"
    )

    print(
        "-" * 72
    )

    display_columns = [
        "scenario",
        "total_cost",
        "final_backlog",
        "service_pct",
        "capacity_utilization_pct",
        "constrained_weeks",
    ]

    print(
        sensitivity[
            display_columns
        ]
        .round(2)
        .to_string(
            index=False
        )
    )

    print(
        "=" * 72
    )


# ============================================================
# MAIN PIPELINE
# ============================================================


def main() -> None:
    """
    Run the complete forecasting-to-production-planning workflow.

    Steps
    -----
    1. Load forecasting predictions.
    2. Construct selected ensemble forecast.
    3. Generate default planning assumptions.
    4. Solve production-planning LP.
    5. Build management reports.
    6. Compare against non-anticipatory JIT baseline.
    7. Run capacity sensitivity analysis.
    8. Save all artifacts.
    """

    print()

    print(
        "=" * 72
    )

    print(
        "FORECAST → PRODUCTION PLANNING INTEGRATION"
    )

    print(
        "=" * 72
    )


    # ========================================================
    # PATHS
    # ========================================================

    project_root = (
        find_project_root()
    )

    forecasting_dir = (
        project_root
        / "data"
        / "processed"
        / "forecasting"
    )

    optimization_dir = (
        project_root
        / "data"
        / "processed"
        / "optimization"
    )

    print(
        "\nProject root:"
    )

    print(
        project_root
    )


    # ========================================================
    # 1 — FORECAST
    # ========================================================

    print(
        "\n[1/8] Loading forecasting artifact..."
    )

    predictions = (
        load_forecasting_predictions(
            forecasting_dir
        )
    )


    # ========================================================
    # 2 — ENSEMBLE
    # ========================================================

    print(
        "\n[2/8] Building optimization forecast..."
    )

    forecast = (
        build_optimization_forecast(
            predictions=predictions,
            expected_horizon=(
                EXPECTED_HORIZON
            ),
        )
    )

    print(
        "\nOptimization forecast:"
    )

    print(
        f"  rows: "
        f"{len(forecast):,}"
    )

    print(
        f"  SKUs: "
        f"{forecast['sku_id'].nunique()}"
    )

    print(
        f"  weeks: "
        f"{forecast['forecast_horizon'].nunique()}"
    )

    print(
        f"  total demand: "
        f"{forecast['forecast_demand'].sum():,.2f}"
    )


    # ========================================================
    # 3 — PLANNING ASSUMPTIONS
    # ========================================================

    print(
        "\n[3/8] Generating planning assumptions..."
    )

    planning_config = (
        PlanningConfig(
            planning_horizon=(
                EXPECTED_HORIZON
            ),
        )
    )

    template_config = (
        PlanningTemplateConfig()
    )

    (
        forecast_clean,
        sku_parameters,
        capacity,
    ) = make_default_planning_inputs(
        forecast=forecast,
        planning_config=(
            planning_config
        ),
        template_config=(
            template_config
        ),
    )


    # ========================================================
    # 4 — SOLVE
    # ========================================================

    print(
        "\n[4/8] Solving production-planning LP "
        "with Pyomo + HiGHS..."
    )

    result = (
        solve_production_plan(
            forecast=forecast_clean,
            sku_parameters=(
                sku_parameters
            ),
            capacity=capacity,
            config=planning_config,
            tee=False,
        )
    )


    # ========================================================
    # 5 — REPORT
    # ========================================================

    print(
        "\n[5/8] Building planning reports..."
    )

    report = (
        build_planning_report(
            result=result,
            sku_parameters=(
                sku_parameters
            ),
        )
    )


    # ========================================================
    # 6 — JIT BENCHMARK
    # ========================================================

    print(
        "\n[6/8] Comparing against JIT baseline..."
    )

    baseline_comparison = (
        compare_to_jit_baseline(
            optimized_result=result,
            forecast=forecast_clean,
            sku_parameters=(
                sku_parameters
            ),
            capacity=capacity,
            config=planning_config,
        )
    )


    # ========================================================
    # 7 — CAPACITY SENSITIVITY
    # ========================================================

    print(
        "\n[7/8] Running capacity sensitivity..."
    )

    capacity_sensitivity = (
        run_capacity_sensitivity(
            forecast=forecast_clean,
            sku_parameters=(
                sku_parameters
            ),
            capacity=capacity,
            config=planning_config,
            scenarios=(
                CAPACITY_SCENARIOS
            ),
        )
    )


    # ========================================================
    # 8 — SAVE
    # ========================================================

    print(
        "\n[8/8] Saving optimization artifacts..."
    )

    save_optimization_artifacts(
        output_dir=optimization_dir,
        forecast=forecast_clean,
        sku_parameters=sku_parameters,
        capacity=capacity,
        result=result,
        report=report,
        baseline_comparison=(
            baseline_comparison
        ),
        capacity_sensitivity=(
            capacity_sensitivity
        ),
    )


    # ========================================================
    # CONSOLE RESULT
    # ========================================================

    print_console_summary(
        forecast=forecast_clean,
        result=result,
        report=report,
        baseline_comparison=(
            baseline_comparison
        ),
        capacity_sensitivity=(
            capacity_sensitivity
        ),
    )

    print(
        "\nArtifacts saved to:"
    )

    print(
        optimization_dir
    )


if __name__ == "__main__":
    main()