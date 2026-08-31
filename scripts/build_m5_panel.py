from pathlib import Path

from new_demfor_planopti.data.m5 import (
    M5Config,
    build_m5_weekly_panel,
)


def main() -> None:
    config = M5Config(
        raw_dir=Path("data/raw"),
        processed_dir=Path("data/processed"),

        target_state="CA",

        max_skus=20,

        # Only this part of history determines SKU selection.
        selection_history_fraction=0.80,

        # Avoid extremely inactive products for the initial project.
        min_active_week_fraction=0.20,

        required_days_per_week=7,

        save_csv_copy=True,
    )

    panel, sku_manifest = build_m5_weekly_panel(
        config
    )

    print("\nPipeline completed successfully.")
    print("\nPanel preview:")
    print(panel.head(20))

    print("\nSKU manifest:")
    print(sku_manifest)


if __name__ == "__main__":
    main()