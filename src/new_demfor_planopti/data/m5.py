from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class M5Config:
    """
    Configuration for the M5 ingestion/preprocessing pipeline.

    The pipeline converts raw daily item-store M5 sales into a
    regional weekly SKU panel.

    Current project contract:
        - one row per (week, sku_id)
        - at most 20 selected SKUs
        - regional demand
        - M5 calendar/event/SNAP features
        - regional price statistics
    """

    raw_dir: Path = Path("data/raw/m5")
    processed_dir: Path = Path("data/processed")

    sales_filename: str = "sales_train_evaluation.csv"
    calendar_filename: str = "calendar.csv"
    prices_filename: str = "sell_prices.csv"

    target_state: str = "CA"

    max_skus: int = 20

    # Use only the first portion of history to choose the product universe.
    # This avoids using the final evaluation period to decide which SKUs
    # are interesting.
    selection_history_fraction: float = 0.80

    # Require a SKU to have positive demand in at least this proportion
    # of selection-period weeks. This can be lowered if you deliberately
    # want strongly intermittent SKUs.
    min_active_week_fraction: float = 0.20

    # M5 wm_yr_wk weeks normally contain 7 days.
    required_days_per_week: int = 7

    save_csv_copy: bool = True

    @property
    def sales_path(self) -> Path:
        return self.raw_dir / self.sales_filename

    @property
    def calendar_path(self) -> Path:
        return self.raw_dir / self.calendar_filename

    @property
    def prices_path(self) -> Path:
        return self.raw_dir / self.prices_filename

    @property
    def panel_path(self) -> Path:
        return self.processed_dir / "m5_weekly_panel.parquet"

    @property
    def panel_csv_path(self) -> Path:
        return self.processed_dir / "m5_weekly_panel.csv"

    @property
    def sku_manifest_path(self) -> Path:
        return self.processed_dir / "selected_skus.csv"


# ============================================================
# Constants
# ============================================================


SALES_ID_COLS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]

REQUIRED_SALES_COLS = set(SALES_ID_COLS)

REQUIRED_CALENDAR_COLS = {
    "date",
    "wm_yr_wk",
    "d",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
}

REQUIRED_PRICE_COLS = {
    "store_id",
    "item_id",
    "wm_yr_wk",
    "sell_price",
}


# ============================================================
# Raw-file inspection
# ============================================================


def inspect_raw_files(config: M5Config) -> dict[str, dict]:
    """
    Inspect raw files without loading the full datasets.

    Useful as the first pipeline sanity check.
    """

    paths = {
        "sales": config.sales_path,
        "calendar": config.calendar_path,
        "prices": config.prices_path,
    }

    inspection = {}

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required M5 file: {path.resolve()}"
            )

        sample = pd.read_csv(path, nrows=5)

        inspection[name] = {
            "path": str(path),
            "columns": sample.columns.tolist(),
            "sample_shape": sample.shape,
        }

    return inspection


# ============================================================
# Loading
# ============================================================


def load_m5_raw(
    config: M5Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three required M5 files.
    """

    print("Loading sales...")
    sales = pd.read_csv(config.sales_path)

    print("Loading calendar...")
    calendar = pd.read_csv(
        config.calendar_path,
        parse_dates=["date"],
    )

    print("Loading prices...")
    prices = pd.read_csv(config.prices_path)

    return sales, calendar, prices


# ============================================================
# Schema validation
# ============================================================


def validate_raw_schema(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    """
    Validate important M5 schema and key assumptions.
    """

    missing_sales = REQUIRED_SALES_COLS - set(sales.columns)
    missing_calendar = REQUIRED_CALENDAR_COLS - set(calendar.columns)
    missing_prices = REQUIRED_PRICE_COLS - set(prices.columns)

    if missing_sales:
        raise ValueError(
            f"Sales missing columns: {sorted(missing_sales)}"
        )

    if missing_calendar:
        raise ValueError(
            f"Calendar missing columns: {sorted(missing_calendar)}"
        )

    if missing_prices:
        raise ValueError(
            f"Prices missing columns: {sorted(missing_prices)}"
        )

    day_cols = get_day_columns(sales)

    if not day_cols:
        raise ValueError("No M5 d_* sales columns found.")

    if not sales["id"].is_unique:
        raise ValueError("sales['id'] must be unique.")

    if not calendar["d"].is_unique:
        raise ValueError("calendar['d'] must be unique.")

    price_key = ["store_id", "item_id", "wm_yr_wk"]

    if prices.duplicated(price_key).any():
        duplicates = prices.loc[
            prices.duplicated(price_key, keep=False),
            price_key,
        ].head()

        raise ValueError(
            "sell_prices is not unique on "
            "(store_id, item_id, wm_yr_wk).\n"
            f"Examples:\n{duplicates}"
        )

    calendar_days = set(calendar["d"])
    missing_day_mappings = [
        col for col in day_cols
        if col not in calendar_days
    ]

    if missing_day_mappings:
        raise ValueError(
            "Some sales d_* columns cannot be mapped through calendar.csv. "
            f"Examples: {missing_day_mappings[:10]}"
        )


# ============================================================
# Column utilities
# ============================================================


def get_day_columns(sales: pd.DataFrame) -> list[str]:
    """
    Return M5 daily sales columns in numeric order.

    Example:
        d_1, d_2, ..., d_1941
    """

    day_cols = [
        col
        for col in sales.columns
        if col.startswith("d_")
    ]

    return sorted(
        day_cols,
        key=lambda x: int(x.split("_")[1]),
    )


# ============================================================
# Geography
# ============================================================


def filter_geography(
    sales: pd.DataFrame,
    state_id: str,
) -> pd.DataFrame:
    """
    Restrict M5 sales to the target regional geography.
    """

    region = sales.loc[
        sales["state_id"].eq(state_id)
    ].copy()

    if region.empty:
        available_states = sorted(
            sales["state_id"].dropna().unique().tolist()
        )

        raise ValueError(
            f"No sales found for state_id={state_id!r}. "
            f"Available states: {available_states}"
        )

    return region


# ============================================================
# Product hierarchy validation
# ============================================================


def build_sku_metadata(
    regional_sales: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one metadata row per M5 item_id.

    cat_id and dept_id should have one consistent mapping per item.
    """

    metadata_counts = (
        regional_sales
        .groupby("item_id")
        .agg(
            n_categories=("cat_id", "nunique"),
            n_departments=("dept_id", "nunique"),
        )
    )

    invalid = metadata_counts.loc[
        (metadata_counts["n_categories"] != 1)
        | (metadata_counts["n_departments"] != 1)
    ]

    if not invalid.empty:
        raise ValueError(
            "Some M5 item_ids map to multiple categories/departments."
        )

    metadata = (
        regional_sales[
            ["item_id", "cat_id", "dept_id"]
        ]
        .drop_duplicates()
        .rename(columns={"item_id": "sku_id"})
        .reset_index(drop=True)
    )

    return metadata


# ============================================================
# SKU selection
# ============================================================


def get_selection_day_columns(
    sales: pd.DataFrame,
    fraction: float,
) -> list[str]:
    """
    Return the early-history d_* columns used to choose SKUs.
    """

    if not 0 < fraction <= 1:
        raise ValueError(
            "selection_history_fraction must be in (0, 1]."
        )

    day_cols = get_day_columns(sales)

    cutoff = max(
        1,
        int(len(day_cols) * fraction),
    )

    return day_cols[:cutoff]


def rank_skus_by_training_volume(
    regional_sales: pd.DataFrame,
    selection_day_cols: Iterable[str],
) -> pd.DataFrame:
    """
    Rank SKUs by total demand in the SKU-selection history.

    Important:
    We sum the wide matrix directly rather than melting the entire
    M5 dataset first.
    """

    selection_day_cols = list(selection_day_cols)

    row_totals = regional_sales[
        selection_day_cols
    ].sum(axis=1)

    ranking_frame = regional_sales[
        ["item_id", "cat_id", "dept_id", "store_id"]
    ].copy()

    ranking_frame["selection_demand"] = row_totals

    ranking = (
        ranking_frame
        .groupby(
            ["item_id", "cat_id", "dept_id"],
            as_index=False,
        )
        .agg(
            selection_demand=("selection_demand", "sum"),
            stores=("store_id", "nunique"),
        )
        .sort_values(
            "selection_demand",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return ranking


def calculate_active_week_fraction(
    regional_sales: pd.DataFrame,
    calendar: pd.DataFrame,
    candidate_skus: Iterable[str],
    selection_day_cols: Iterable[str],
) -> pd.DataFrame:
    """
    Estimate how often each candidate SKU has positive weekly demand
    during the SKU-selection period.

    This prevents us from accidentally selecting products that have
    very large total sales but extremely sparse/short histories,
    unless we deliberately lower the threshold.
    """

    candidate_skus = set(candidate_skus)
    selection_day_cols = list(selection_day_cols)

    subset = regional_sales.loc[
        regional_sales["item_id"].isin(candidate_skus),
        ["item_id", "store_id", *selection_day_cols],
    ].copy()

    long_sales = subset.melt(
        id_vars=["item_id", "store_id"],
        value_vars=selection_day_cols,
        var_name="d",
        value_name="demand",
    )

    daily = (
        long_sales
        .groupby(
            ["item_id", "d"],
            as_index=False,
        )
        ["demand"]
        .sum()
    )

    daily = daily.merge(
        calendar[["d", "wm_yr_wk"]],
        on="d",
        how="left",
        validate="many_to_one",
    )

    weekly = (
        daily
        .groupby(
            ["item_id", "wm_yr_wk"],
            as_index=False,
        )
        ["demand"]
        .sum()
    )

    activity = (
        weekly
        .assign(active=lambda x: x["demand"] > 0)
        .groupby("item_id", as_index=False)
        .agg(
            active_week_fraction=("active", "mean"),
            selection_weeks=("wm_yr_wk", "nunique"),
        )
    )

    return activity


def select_skus(
    regional_sales: pd.DataFrame,
    calendar: pd.DataFrame,
    config: M5Config,
) -> pd.DataFrame:
    """
    Select at most config.max_skus using training-period volume
    and minimum weekly activity.

    Returns a SKU manifest.
    """

    selection_days = get_selection_day_columns(
        regional_sales,
        config.selection_history_fraction,
    )

    ranking = rank_skus_by_training_volume(
        regional_sales,
        selection_days,
    )

    # Evaluate more candidates than needed in case some are
    # removed by the minimum-activity requirement.
    candidate_count = min(
        len(ranking),
        max(config.max_skus * 5, config.max_skus),
    )

    candidates = ranking.head(candidate_count)

    activity = calculate_active_week_fraction(
        regional_sales=regional_sales,
        calendar=calendar,
        candidate_skus=candidates["item_id"],
        selection_day_cols=selection_days,
    )

    manifest = candidates.merge(
        activity,
        on="item_id",
        how="left",
        validate="one_to_one",
    )

    manifest = (
        manifest.loc[
            manifest["active_week_fraction"]
            >= config.min_active_week_fraction
        ]
        .head(config.max_skus)
        .copy()
    )

    if manifest.empty:
        raise ValueError(
            "SKU selection returned zero products. "
            "Reduce min_active_week_fraction."
        )

    manifest["selection_rank"] = np.arange(
        1,
        len(manifest) + 1,
    )

    manifest = manifest.rename(
        columns={"item_id": "sku_id"}
    )

    return manifest[
        [
            "selection_rank",
            "sku_id",
            "cat_id",
            "dept_id",
            "selection_demand",
            "active_week_fraction",
            "selection_weeks",
            "stores",
        ]
    ]


# ============================================================
# Wide -> long, store aggregation
# ============================================================


def build_regional_daily_demand(
    regional_sales: pd.DataFrame,
    selected_skus: Iterable[str],
) -> pd.DataFrame:
    """
    Convert selected M5 SKUs from item-store wide data into
    regional item-day demand.

    We only melt selected SKUs, which dramatically reduces memory.
    """

    selected_skus = set(selected_skus)

    day_cols = get_day_columns(regional_sales)

    subset = regional_sales.loc[
        regional_sales["item_id"].isin(selected_skus),
        [
            "item_id",
            "store_id",
            *day_cols,
        ],
    ].copy()

    if subset.empty:
        raise ValueError(
            "No sales rows found for selected SKUs."
        )

    long_sales = subset.melt(
        id_vars=["item_id", "store_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="demand",
    )

    # Manufacturing/regional interpretation:
    # aggregate all stores in the state into one regional demand signal.
    daily = (
        long_sales
        .groupby(
            ["item_id", "d"],
            as_index=False,
            observed=True,
        )
        .agg(
            demand=("demand", "sum"),
            contributing_stores=("store_id", "nunique"),
        )
    )

    daily = daily.rename(
        columns={"item_id": "sku_id"}
    )

    return daily


# ============================================================
# Calendar preparation
# ============================================================


def prepare_daily_calendar(
    calendar: pd.DataFrame,
    target_state: str,
) -> pd.DataFrame:
    """
    Convert M5 calendar into daily features needed by the weekly panel.
    """

    snap_col = f"snap_{target_state}"

    if snap_col not in calendar.columns:
        raise ValueError(
            f"Calendar does not contain {snap_col!r}."
        )

    cal = calendar.copy()

    cal["has_event"] = (
        cal["event_name_1"].notna()
        | cal["event_name_2"].notna()
    ).astype("int8")

    event_types = [
        "Cultural",
        "National",
        "Religious",
        "Sporting",
    ]

    for event_type in event_types:
        feature = f"{event_type.lower()}_event_day"

        cal[feature] = (
            cal["event_type_1"].eq(event_type)
            | cal["event_type_2"].eq(event_type)
        ).astype("int8")

    cal["snap_active"] = (
        cal[snap_col]
        .fillna(0)
        .astype("int8")
    )

    return cal[
        [
            "d",
            "date",
            "wm_yr_wk",
            "snap_active",
            "has_event",
            "cultural_event_day",
            "national_event_day",
            "religious_event_day",
            "sporting_event_day",
        ]
    ].copy()


# ============================================================
# Daily -> weekly demand
# ============================================================


def build_weekly_demand(
    daily_demand: pd.DataFrame,
    daily_calendar: pd.DataFrame,
    required_days_per_week: int = 7,
) -> pd.DataFrame:
    """
    Map M5 d_* keys to Walmart weeks and aggregate daily demand.

    Partial boundary weeks are excluded.
    """

    merged = daily_demand.merge(
        daily_calendar[
            ["d", "date", "wm_yr_wk"]
        ],
        on="d",
        how="left",
        validate="many_to_one",
    )

    if merged["wm_yr_wk"].isna().any():
        raise ValueError(
            "Some daily demand rows failed calendar mapping."
        )

    # Count how many distinct sales days exist in each M5 week.
    # This lets us remove boundary weeks where the raw sales history
    # contains fewer than seven days.
    week_coverage = (
        merged[
            ["wm_yr_wk", "d"]
        ]
        .drop_duplicates()
        .groupby(
            "wm_yr_wk",
            as_index=False,
        )
        .agg(days_in_sales_week=("d", "nunique"))
    )

    complete_weeks = set(
        week_coverage.loc[
            week_coverage["days_in_sales_week"]
            == required_days_per_week,
            "wm_yr_wk",
        ]
    )

    merged = merged.loc[
        merged["wm_yr_wk"].isin(complete_weeks)
    ].copy()

    weekly = (
        merged
        .groupby(
            ["sku_id", "wm_yr_wk"],
            as_index=False,
            observed=True,
        )
        .agg(
            demand=("demand", "sum"),
            week_start=("date", "min"),
            week_end=("date", "max"),
            days_in_week=("d", "nunique"),
        )
    )

    return weekly


# ============================================================
# Weekly calendar features
# ============================================================


def build_weekly_calendar_features(
    daily_calendar: pd.DataFrame,
    valid_weeks: Iterable[int],
) -> pd.DataFrame:
    """
    Aggregate daily M5 calendar/event information to Walmart weeks.
    """

    valid_weeks = set(valid_weeks)

    cal = daily_calendar.loc[
        daily_calendar["wm_yr_wk"].isin(valid_weeks)
    ].copy()

    weekly = (
        cal
        .groupby(
            "wm_yr_wk",
            as_index=False,
        )
        .agg(
            week_start=("date", "min"),
            week_end=("date", "max"),
            calendar_days=("date", "nunique"),

            snap_days=("snap_active", "sum"),
            event_days=("has_event", "sum"),

            cultural_event_days=(
                "cultural_event_day",
                "sum",
            ),
            national_event_days=(
                "national_event_day",
                "sum",
            ),
            religious_event_days=(
                "religious_event_day",
                "sum",
            ),
            sporting_event_days=(
                "sporting_event_day",
                "sum",
            ),
        )
    )

    weekly["year"] = (
        weekly["week_start"].dt.year.astype("int16")
    )

    weekly["month"] = (
        weekly["week_start"].dt.month.astype("int8")
    )

    weekly["quarter"] = (
        weekly["week_start"].dt.quarter.astype("int8")
    )

    weekly["week_of_year"] = (
        weekly["week_start"]
        .dt
        .isocalendar()
        .week
        .astype("int16")
    )

    return weekly


# ============================================================
# Weekly price features
# ============================================================


def build_weekly_price_features(
    prices: pd.DataFrame,
    regional_sales: pd.DataFrame,
    selected_skus: Iterable[str],
) -> pd.DataFrame:
    """
    Aggregate store-level M5 prices into regional SKU-week features.

    We intentionally DO NOT demand-weight prices because doing so would
    require realized same-week demand.
    """

    selected_skus = set(selected_skus)

    region_stores = sorted(
        regional_sales["store_id"]
        .dropna()
        .unique()
        .tolist()
    )

    region_prices = prices.loc[
        prices["store_id"].isin(region_stores)
        & prices["item_id"].isin(selected_skus)
    ].copy()

    if region_prices.empty:
        raise ValueError(
            "No price records found for selected regional SKUs."
        )

    price_features = (
        region_prices
        .groupby(
            ["item_id", "wm_yr_wk"],
            as_index=False,
            observed=True,
        )
        .agg(
            sell_price=("sell_price", "mean"),
            sell_price_min=("sell_price", "min"),
            sell_price_max=("sell_price", "max"),
            sell_price_std=("sell_price", "std"),
            price_store_count=("store_id", "nunique"),
        )
    )

    price_features["price_coverage"] = (
        price_features["price_store_count"]
        / len(region_stores)
    )

    price_features["price_available"] = True

    price_features = price_features.rename(
        columns={"item_id": "sku_id"}
    )

    return price_features


# ============================================================
# Assemble final panel
# ============================================================


def assemble_weekly_panel(
    weekly_demand: pd.DataFrame,
    weekly_calendar: pd.DataFrame,
    weekly_prices: pd.DataFrame,
    sku_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble the final one-row-per-(week, sku_id) dataset.
    """

    panel = weekly_demand.merge(
        weekly_calendar.drop(
            columns=["week_start", "week_end"]
        ),
        on="wm_yr_wk",
        how="left",
        validate="many_to_one",
    )

    panel = panel.merge(
        sku_metadata,
        on="sku_id",
        how="left",
        validate="many_to_one",
    )

    panel = panel.merge(
        weekly_prices,
        on=["sku_id", "wm_yr_wk"],
        how="left",
        validate="one_to_one",
    )

    # Missing prices remain missing.
    # Preserve the missingness explicitly.
    panel["price_available"] = (
        panel["price_available"]
        .fillna(False)
        .astype(bool)
    )

    panel["price_store_count"] = (
        panel["price_store_count"]
        .fillna(0)
        .astype("int16")
    )

    panel["price_coverage"] = (
        panel["price_coverage"]
        .fillna(0.0)
    )

    panel = panel.sort_values(
        ["week_start", "sku_id"]
    ).reset_index(drop=True)

    ordered_cols = [
        # Primary identity
        "week_start",
        "week_end",
        "wm_yr_wk",
        "sku_id",

        # M5 hierarchy
        "cat_id",
        "dept_id",

        # Target
        "demand",

        # Calendar
        "year",
        "quarter",
        "month",
        "week_of_year",
        "snap_days",
        "event_days",
        "cultural_event_days",
        "national_event_days",
        "religious_event_days",
        "sporting_event_days",

        # Price
        "sell_price",
        "sell_price_min",
        "sell_price_max",
        "sell_price_std",
        "price_store_count",
        "price_coverage",
        "price_available",

        # QA
        "days_in_week",
    ]

    return panel[ordered_cols]


# ============================================================
# Final validation
# ============================================================


def validate_processed_panel(
    panel: pd.DataFrame,
    max_skus: int,
    required_days_per_week: int = 7,
) -> None:
    """
    Enforce the Stage 1 output contract.
    """

    key = ["week_start", "sku_id"]

    # --------------------------------------------------------
    # Unique key
    # --------------------------------------------------------

    if panel.duplicated(key).any():
        duplicates = panel.loc[
            panel.duplicated(key, keep=False),
            key,
        ].head(20)

        raise ValueError(
            "Processed panel violates unique "
            "(week_start, sku_id) key.\n"
            f"{duplicates}"
        )

    # --------------------------------------------------------
    # SKU count
    # --------------------------------------------------------

    n_skus = panel["sku_id"].nunique()

    if n_skus > max_skus:
        raise ValueError(
            f"Panel has {n_skus} SKUs; maximum is {max_skus}."
        )

    # --------------------------------------------------------
    # Demand validity
    # --------------------------------------------------------

    if panel["demand"].isna().any():
        raise ValueError(
            "Demand contains missing values."
        )

    if (panel["demand"] < 0).any():
        raise ValueError(
            "Demand contains negative values."
        )

    # --------------------------------------------------------
    # Complete weeks
    # --------------------------------------------------------

    if not panel[
        "days_in_week"
    ].eq(required_days_per_week).all():
        raise ValueError(
            "Panel contains incomplete demand weeks."
        )

    # --------------------------------------------------------
    # Calendar validity
    # --------------------------------------------------------

    calendar_features = [
        "week_start",
        "week_end",
        "wm_yr_wk",
        "year",
        "month",
        "week_of_year",
        "snap_days",
        "event_days",
    ]

    if panel[calendar_features].isna().any().any():
        missing = (
            panel[calendar_features]
            .isna()
            .sum()
        )

        raise ValueError(
            "Missing required calendar values:\n"
            f"{missing[missing > 0]}"
        )

    # --------------------------------------------------------
    # Price QA
    # --------------------------------------------------------

    if (
        (panel["price_coverage"] < 0)
        | (panel["price_coverage"] > 1)
    ).any():
        raise ValueError(
            "price_coverage must be between 0 and 1."
        )

    positive_price_mask = panel["sell_price"].notna()

    if (
        panel.loc[
            positive_price_mask,
            "sell_price",
        ]
        <= 0
    ).any():
        raise ValueError(
            "Observed sell_price must be positive."
        )

    # --------------------------------------------------------
    # Balanced panel check
    # --------------------------------------------------------

    weeks_per_sku = panel.groupby(
        "sku_id"
    )["wm_yr_wk"].nunique()

    expected_weeks = panel["wm_yr_wk"].nunique()

    if not weeks_per_sku.eq(expected_weeks).all():
        raise ValueError(
            "Panel is not balanced: some SKUs are missing weeks."
        )

    expected_rows = n_skus * expected_weeks

    if len(panel) != expected_rows:
        raise ValueError(
            "Unexpected panel shape. "
            f"Expected {expected_rows}, received {len(panel)}."
        )


# ============================================================
# Reporting
# ============================================================


def print_panel_report(
    panel: pd.DataFrame,
    sku_manifest: pd.DataFrame,
) -> None:
    """
    Print concise QA information after preprocessing.
    """

    print("\n" + "=" * 70)
    print("M5 WEEKLY PANEL REPORT")
    print("=" * 70)

    print(f"Rows:             {len(panel):,}")
    print(f"SKUs:             {panel['sku_id'].nunique():,}")
    print(f"Weeks:            {panel['wm_yr_wk'].nunique():,}")
    print(f"Start:            {panel['week_start'].min()}")
    print(f"End:              {panel['week_end'].max()}")

    print(
        "Missing prices:   "
        f"{panel['sell_price'].isna().mean():.2%}"
    )

    print(
        "Zero-demand rows: "
        f"{panel['demand'].eq(0).mean():.2%}"
    )

    print("\nSelected SKUs:")
    print(
        sku_manifest[
            [
                "selection_rank",
                "sku_id",
                "dept_id",
                "selection_demand",
                "active_week_fraction",
            ]
        ].to_string(index=False)
    )

    print("\nDemand summary:")
    print(
        panel["demand"]
        .describe()
        .round(2)
        .to_string()
    )

    print("\nPrice coverage summary:")
    print(
        panel["price_coverage"]
        .describe()
        .round(3)
        .to_string()
    )

    print("=" * 70)


# ============================================================
# Persistence
# ============================================================


def save_processed_outputs(
    panel: pd.DataFrame,
    sku_manifest: pd.DataFrame,
    config: M5Config,
) -> None:
    """
    Save Stage 1 outputs.
    """

    config.processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel.to_parquet(
        config.panel_path,
        index=False,
    )

    sku_manifest.to_csv(
        config.sku_manifest_path,
        index=False,
    )

    if config.save_csv_copy:
        panel.to_csv(
            config.panel_csv_path,
            index=False,
        )

    print(f"Saved: {config.panel_path}")
    print(f"Saved: {config.sku_manifest_path}")

    if config.save_csv_copy:
        print(f"Saved: {config.panel_csv_path}")


# ============================================================
# Complete pipeline
# ============================================================


def build_m5_weekly_panel(
    config: M5Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run complete M5 Stage 1 preprocessing.

    Returns
    -------
    panel:
        One row per (week_start, sku_id).

    sku_manifest:
        Metadata explaining how/why the selected SKU set was chosen.
    """

    config = config or M5Config()

    print("\n[1/10] Inspecting raw files...")
    inspection = inspect_raw_files(config)

    for name, info in inspection.items():
        print(
            f"{name:10s} "
            f"{info['path']} "
            f"({len(info['columns'])} columns)"
        )

    print("\n[2/10] Loading raw M5 tables...")
    sales, calendar, prices = load_m5_raw(config)

    print(
        f"sales shape:    {sales.shape}"
    )
    print(
        f"calendar shape: {calendar.shape}"
    )
    print(
        f"prices shape:   {prices.shape}"
    )

    print("\n[3/10] Validating raw schema...")
    validate_raw_schema(
        sales,
        calendar,
        prices,
    )

    print("\n[4/10] Filtering geography...")
    regional_sales = filter_geography(
        sales,
        config.target_state,
    )

    print(
        f"State:  {config.target_state}"
    )
    print(
        "Stores: "
        f"{regional_sales['store_id'].nunique()}"
    )
    print(
        "Items:  "
        f"{regional_sales['item_id'].nunique()}"
    )

    print("\n[5/10] Selecting SKUs...")
    sku_manifest = select_skus(
        regional_sales=regional_sales,
        calendar=calendar,
        config=config,
    )

    selected_skus = sku_manifest[
        "sku_id"
    ].tolist()

    print(
        f"Selected {len(selected_skus)} SKUs"
    )

    print("\n[6/10] Aggregating store demand...")
    daily_demand = build_regional_daily_demand(
        regional_sales,
        selected_skus,
    )

    daily_calendar = prepare_daily_calendar(
        calendar,
        config.target_state,
    )

    print("\n[7/10] Aggregating daily -> weekly...")
    weekly_demand = build_weekly_demand(
        daily_demand=daily_demand,
        daily_calendar=daily_calendar,
        required_days_per_week=(
            config.required_days_per_week
        ),
    )

    valid_weeks = (
        weekly_demand[
            "wm_yr_wk"
        ]
        .unique()
        .tolist()
    )

    print("\n[8/10] Building calendar + price features...")
    weekly_calendar = build_weekly_calendar_features(
        daily_calendar,
        valid_weeks,
    )

    weekly_prices = build_weekly_price_features(
        prices=prices,
        regional_sales=regional_sales,
        selected_skus=selected_skus,
    )

    sku_metadata = build_sku_metadata(
        regional_sales
    )

    print("\n[9/10] Assembling panel...")
    panel = assemble_weekly_panel(
        weekly_demand=weekly_demand,
        weekly_calendar=weekly_calendar,
        weekly_prices=weekly_prices,
        sku_metadata=sku_metadata,
    )

    print("\n[10/10] Running final QA...")
    validate_processed_panel(
        panel=panel,
        max_skus=config.max_skus,
        required_days_per_week=(
            config.required_days_per_week
        ),
    )

    print_panel_report(
        panel,
        sku_manifest,
    )

    save_processed_outputs(
        panel,
        sku_manifest,
        config,
    )

    return panel, sku_manifest


if __name__ == "__main__":
    build_m5_weekly_panel()