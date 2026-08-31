from pathlib import Path

from new_demfor_planopti.forecasting import (
    ForecastConfig,
    run_forecasting_pipeline,
)


def main() -> None:
    """
    Run the complete forecasting benchmark.

    Evaluation design
    -----------------
    - 13-week forecasting horizon
    - 3 expanding-window rolling validation folds
    - final untouched 13-week test set

    Model ladder
    ------------
    Simple:
        - naive
        - seasonal naive
        - drift
        - moving averages

    Statistical:
        - ETS
        - Theta
        - ARIMA
        - SARIMA

    Machine learning:
        - global LightGBM
    """

    config = ForecastConfig(
        panel_path=Path(
            "data/processed/m5_weekly_panel.parquet"
        ),

        output_dir=Path(
            "data/processed/forecasting"
        ),

        # Forecast 13 weeks ahead at every origin.
        forecast_horizon=13,

        # Three historical forecast origins for model selection.
        validation_folds=5,

        # Final period is untouched until model selection is finished.
        test_weeks=13,

        moving_average_windows=(
            4,
            13,
        ),

        # Approximate annual seasonality for weekly demand.
        season_length=52,

        random_state=42,
    )

    results = run_forecasting_pipeline(
        config
    )

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        "Champion:",
        results["champion"],
    )

    if "resume_bullet" in results:
        print("\nResume bullet:")
        print(results["resume_bullet"])


if __name__ == "__main__":
    main()