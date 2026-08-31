import pandas as pd
import pytest

from new_demfor_planopti.data.m5 import (
    validate_processed_panel,
)


def make_valid_panel() -> pd.DataFrame:

    weeks = pd.date_range(
        "2024-01-06",
        periods=3,
        freq="7D",
    )

    rows = []

    for week_idx, week in enumerate(weeks):

        for sku in ["SKU_A", "SKU_B"]:

            rows.append(
                {
                    "week_start": week,
                    "week_end": (
                        week
                        + pd.Timedelta(days=6)
                    ),
                    "wm_yr_wk": 100 + week_idx,
                    "sku_id": sku,

                    "cat_id": "FOODS",
                    "dept_id": "FOODS_1",

                    "demand": 100.0,

                    "year": week.year,
                    "quarter": week.quarter,
                    "month": week.month,
                    "week_of_year": (
                        week.isocalendar().week
                    ),

                    "snap_days": 2,
                    "event_days": 1,
                    "cultural_event_days": 0,
                    "national_event_days": 1,
                    "religious_event_days": 0,
                    "sporting_event_days": 0,

                    "sell_price": 3.5,
                    "sell_price_min": 3.0,
                    "sell_price_max": 4.0,
                    "sell_price_std": 0.2,

                    "price_store_count": 4,
                    "price_coverage": 1.0,
                    "price_available": True,

                    "days_in_week": 7,
                }
            )

    return pd.DataFrame(rows)


def test_valid_panel_passes():

    panel = make_valid_panel()

    validate_processed_panel(
        panel,
        max_skus=20,
    )


def test_duplicate_key_fails():

    panel = make_valid_panel()

    duplicate = panel.iloc[[0]].copy()

    panel = pd.concat(
        [panel, duplicate],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        validate_processed_panel(
            panel,
            max_skus=20,
        )


def test_negative_demand_fails():

    panel = make_valid_panel()

    panel.loc[
        panel.index[0],
        "demand",
    ] = -10

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        validate_processed_panel(
            panel,
            max_skus=20,
        )


def test_too_many_skus_fails():

    panel = make_valid_panel()

    additional_rows = []

    template = panel.iloc[0].to_dict()

    for i in range(21):

        row = template.copy()

        row["sku_id"] = f"SKU_{i:02d}"

        row["week_start"] = (
            pd.Timestamp("2025-01-04")
        )

        row["week_end"] = (
            pd.Timestamp("2025-01-10")
        )

        row["wm_yr_wk"] = 500

        additional_rows.append(row)

    too_many = pd.DataFrame(
        additional_rows
    )

    with pytest.raises(
        ValueError,
        match="maximum",
    ):
        validate_processed_panel(
            too_many,
            max_skus=20,
        )


def test_incomplete_week_fails():

    panel = make_valid_panel()

    panel.loc[
        panel.index[0],
        "days_in_week",
    ] = 6

    with pytest.raises(
        ValueError,
        match="incomplete",
    ):
        validate_processed_panel(
            panel,
            max_skus=20,
        )


def test_invalid_price_coverage_fails():

    panel = make_valid_panel()

    panel.loc[
        panel.index[0],
        "price_coverage",
    ] = 1.5

    with pytest.raises(
        ValueError,
        match="price_coverage",
    ):
        validate_processed_panel(
            panel,
            max_skus=20,
        )