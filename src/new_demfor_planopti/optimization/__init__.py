from new_demfor_planopti.optimization.contracts import (
    CAPACITY_COLUMNS,
    FORECAST_COLUMNS,
    SKU_PARAMETER_COLUMNS,
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
    ProductionPlanResult,
    build_production_model,
    extract_capacity_usage,
    extract_production_plan,
    get_highs_solver,
    solve_production_plan,
)

from new_demfor_planopti.optimization.reporting import (
    PlanningReport,
    build_executive_summary,
    build_planning_report,
    summarize_capacity,
    summarize_costs,
    summarize_skus,
    validate_reporting_inputs,
)

from new_demfor_planopti.optimization.scenarios import (
    BaselineComparisonResult,
    CapacitySensitivityResult,
    build_jit_baseline,
    compare_to_jit_baseline,
    run_capacity_sensitivity,
    summarize_plan_metrics,
)


__all__ = [
    # ========================================================
    # CONTRACTS
    # ========================================================
    "CAPACITY_COLUMNS",
    "FORECAST_COLUMNS",
    "SKU_PARAMETER_COLUMNS",
    "PlanningConfig",
    "validate_capacity_frame",
    "validate_forecast_frame",
    "validate_planning_inputs",
    "validate_sku_parameters",

    # ========================================================
    # DEFAULT SCENARIO TEMPLATES
    # ========================================================
    "PlanningTemplateConfig",
    "make_default_capacity",
    "make_default_planning_inputs",
    "make_default_sku_parameters",
    "validate_template_config",

    # ========================================================
    # PYOMO + HIGHS SOLVER
    # ========================================================
    "ProductionPlanResult",
    "build_production_model",
    "extract_capacity_usage",
    "extract_production_plan",
    "get_highs_solver",
    "solve_production_plan",

    # ========================================================
    # REPORTING
    # ========================================================
    "PlanningReport",
    "build_executive_summary",
    "build_planning_report",
    "summarize_capacity",
    "summarize_costs",
    "summarize_skus",
    "validate_reporting_inputs",

    # ========================================================
    # SCENARIO ANALYSIS
    # ========================================================
    "BaselineComparisonResult",
    "CapacitySensitivityResult",
    "build_jit_baseline",
    "compare_to_jit_baseline",
    "run_capacity_sensitivity",
    "summarize_plan_metrics",
]