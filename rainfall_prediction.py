"""
INT234 Predictive Analytics - College Project
Title : Rainfall Prediction and Pattern Analysis Using Historical Indian
        Meteorological Data
File  : rainfall_prediction.py

How to run (Python IDLE on macOS):
    1. Open this file in IDLE (File -> Open).
    2. Set DATASET_PATH below (section "USER SETTINGS").
    3. Press Run -> Run Module (or F5).

All graphs and CSV files are saved in a folder called "rainfall_results".
Only the 8 selected key graphs pop up; other plots are saved without opening.
The script never downloads or invents data: it only uses the CSV you give it.
"""

# ----------------------------------------------------------------------------
# 0. IMPORTS
# ----------------------------------------------------------------------------
import os
import sys
import warnings
import traceback

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, silhouette_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# ----------------------------------------------------------------------------
# 1. USER SETTINGS  (edit this section only)
# ----------------------------------------------------------------------------
# Path to your dataset. Automatically detects relative locations (e.g. data/ or current folder)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_PATHS = [
    os.path.join(_BASE_DIR, "data", "Sub_Division_IMD_2017.csv"),
    os.path.join(_BASE_DIR, "Sub_Division_IMD_2017.csv"),
    "/Users/ramcharanteja/Desktop/Sub_Division_IMD_2017.csv",
    "Sub_Division_IMD_2017.csv"
]
DATASET_PATH = next((p for p in _CANDIDATE_PATHS if os.path.exists(p)), os.path.join(_BASE_DIR, "data", "Sub_Division_IMD_2017.csv"))

OUTPUT_FOLDER_NAME = "rainfall_results"   # created automatically
RANDOM_STATE = 42                         # makes results reproducible
TEST_YEAR_FRACTION = 0.20                 # last 20% of years are used as the test set

# Clustering settings
K_RANGE = range(2, 9)                     # values of k tried for K-Means
N_CLUSTERS = None                         # None = choose k with best silhouette score
                                          # (or write a number such as 3 to force it)

# Subdivisions highlighted in some comparison graphs (ignored if not in data)
HIGHLIGHT_SUBDIVISIONS = ["Punjab", "Kerala", "West Rajasthan", "Assam & Meghalaya"]

# Optional manual column configuration. Leave as None for automatic detection.
# Only change these if the script says it could not detect your columns.
SUBDIVISION_COLUMN = None     # e.g. "SUBDIVISION"
YEAR_COLUMN = None            # e.g. "YEAR"
MONTH_COLUMNS = None          # wide format: list of 12 column names, January -> December
MONTH_COLUMN = None           # long format: column holding the month (name or number)
RAINFALL_COLUMN = None        # long format: column holding rainfall values
ANNUAL_COLUMN = None          # optional: column holding annual rainfall (wide format)

# ----------------------------------------------------------------------------
# 2. CONSTANTS AND SMALL HELPERS
# ----------------------------------------------------------------------------
MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
MONTH_FULL = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY",
              "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]

TARGET = "RAINFALL"                                   # monthly rainfall in mm
NUMERIC_FEATURES = ["YEAR", "MONTH_SIN", "MONTH_COS",
                    "LAG_1", "LAG_2", "LAG_12", "ROLL_3", "ROLL_12", "CLIM_PAST"]
CATEGORICAL_FEATURES = ["SUBDIVISION"]

REPORT_LINES = []     # everything logged here is written to project_summary.txt
OUTPUT_FILES = []     # names of files that were really created
OUTPUT_DIR = None     # set in main()


def log(message=""):
    """Print a message and remember it for project_summary.txt."""
    print(message)
    REPORT_LINES.append(str(message))


def section(title):
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


def out_path(filename):
    return os.path.join(OUTPUT_DIR, filename)


def save_csv(df, filename, index=False):
    df.to_csv(out_path(filename), index=index)
    OUTPUT_FILES.append(filename)
    log(f"  saved file: {filename}")


POPUP_PLOTS = {
    "eda_annual_trend.png",
    "eda_monthly_pattern.png",
    "eda_distributions.png",
    "eda_correlation_heatmap.png",
    "eda_subdivision_month_rainfall_heatmap.png",
    "eda_annual_rainfall_heatmap.png",
    "model_actual_vs_predicted.png",
    "model_metrics_heatmap.png",
}

# Set to False or run with --no-popup (or HEADLESS=1) for non-interactive execution
SHOW_POPUPS = os.environ.get("HEADLESS", "").lower() not in ("1", "true") and "--no-popup" not in sys.argv


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(out_path(filename), dpi=150)
    if SHOW_POPUPS and filename in POPUP_PLOTS:
        log(f"  displaying selected plot: {filename}")
        plt.show(block=True)
    plt.close(fig)
    OUTPUT_FILES.append(filename)
    log(f"  saved plot: {filename}")


def safe_run(step_name, function, *args):
    """Run one analysis step. If it fails, report the error and continue."""
    try:
        return function(*args)
    except Exception as error:      # broad on purpose: one failure must not stop everything
        log(f"\n[WARNING] Step '{step_name}' failed and was skipped: {error}")
        traceback.print_exc()
        return None


# ----------------------------------------------------------------------------
# 3. LOADING AND INSPECTING THE DATASET
# ----------------------------------------------------------------------------
def find_dataset(path):
    """Look for the dataset in the given path and a few common folders."""
    candidates = [path]
    try:
        script_folder = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(script_folder, os.path.basename(path)))
    except NameError:
        pass
    home = os.path.expanduser("~")
    for folder in ("", "Downloads", "Documents", "Desktop"):
        candidates.append(os.path.join(home, folder, os.path.basename(path)))
    candidates.append(os.path.join(os.getcwd(), os.path.basename(path)))
    for candidate in candidates:
        if os.path.isfile(os.path.expanduser(candidate)):
            return os.path.expanduser(candidate)
    raise FileNotFoundError(
        f"Dataset not found: '{path}'. Set DATASET_PATH at the top of the script "
        "to the full path of your CSV/Excel file.")


def load_dataset(path):
    full_path = find_dataset(path)
    extension = os.path.splitext(full_path)[1].lower()
    if extension in (".xlsx", ".xls"):
        df = pd.read_excel(full_path)
    else:
        df = pd.read_csv(full_path)
    log(f"Dataset loaded from: {full_path}")
    return df, full_path


def inspect_dataset(df):
    section("STEP 1: DATASET INSPECTION")
    log(f"Data rows (excluding header): {len(df)}")
    log(f"Columns: {df.shape[1]}")
    log("\nColumn names and data types:")
    log(df.dtypes.astype(str).to_string())

    log("\nMissing values per column (only columns with missing values):")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        log("  none")
    else:
        table = pd.DataFrame({"missing": missing,
                              "percent": (missing / len(df) * 100).round(2)})
        log(table.to_string())
    log(f"\nTotal missing cells: {int(df.isna().sum().sum())}")
    log(f"Fully duplicated rows: {int(df.duplicated().sum())}")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    log(f"\nNumerical columns ({len(numeric_cols)}): {numeric_cols}")
    log(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    if numeric_cols:
        log("\nSummary statistics of numerical columns:")
        log(df[numeric_cols].describe().T.round(2).to_string())
    for col in categorical_cols:
        log(f"\nCategorical column '{col}': {df[col].nunique()} unique values")
        log(df[col].value_counts().head(5).to_string())


# ----------------------------------------------------------------------------
# 4. DETECTING FORMAT AND CONVERTING TO LONG FORMAT
# ----------------------------------------------------------------------------
def detect_structure(df):
    """Find subdivision, year and month/rainfall columns. Returns a dict."""
    upper = {col: str(col).strip().upper() for col in df.columns}

    # subdivision (region) column
    sub_col = SUBDIVISION_COLUMN
    if sub_col is None:
        for key in ("SUBDIV", "REGION", "STATE", "DISTRICT"):
            matches = [c for c, u in upper.items() if key in u]
            if matches:
                sub_col = matches[0]
                break
    if sub_col is None:
        text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        sub_col = text_cols[0] if text_cols else None

    # year column
    year_col = YEAR_COLUMN
    if year_col is None:
        matches = [c for c, u in upper.items() if u == "YEAR"] or \
                  [c for c, u in upper.items() if "YEAR" in u]
        year_col = matches[0] if matches else None

    if sub_col is None or year_col is None:
        raise ValueError("Could not detect the subdivision/year columns. "
                         "Set SUBDIVISION_COLUMN and YEAR_COLUMN at the top of the script.")

    # wide format: twelve month columns
    month_cols = {}
    if MONTH_COLUMNS is not None:
        if len(MONTH_COLUMNS) != 12:
            raise ValueError("MONTH_COLUMNS must contain exactly 12 column names.")
        month_cols = {name: i + 1 for i, name in enumerate(MONTH_COLUMNS)}
    else:
        for col, u in upper.items():
            if u in MONTH_ABBR:
                month_cols[col] = MONTH_ABBR.index(u) + 1
            elif u in MONTH_FULL:
                month_cols[col] = MONTH_FULL.index(u) + 1
    if len(month_cols) == 12:
        annual_col = ANNUAL_COLUMN
        if annual_col is None:
            matches = [c for c, u in upper.items() if u == "ANNUAL"]
            annual_col = matches[0] if matches else None
        return {"format": "wide", "subdivision": sub_col, "year": year_col,
                "month_cols": month_cols, "annual": annual_col}

    # long format: one month column and one rainfall column
    month_col = MONTH_COLUMN
    if month_col is None:
        matches = [c for c, u in upper.items() if u == "MONTH"]
        month_col = matches[0] if matches else None
    rain_col = RAINFALL_COLUMN
    if rain_col is None:
        matches = [c for c, u in upper.items() if "RAIN" in u or "PRECIP" in u]
        rain_col = matches[0] if matches else None
    if month_col is not None and rain_col is not None:
        return {"format": "long", "subdivision": sub_col, "year": year_col,
                "month": month_col, "rain": rain_col, "annual": None}

    raise ValueError("Could not recognise the data layout (neither 12 month columns "
                     "nor MONTH + RAINFALL columns). Set MONTH_COLUMNS or "
                     "MONTH_COLUMN / RAINFALL_COLUMN at the top of the script.")


def month_to_number(value):
    """Convert a month name or number to 1..12 (NaN if not recognised)."""
    if pd.isna(value):
        return np.nan
    try:
        number = int(float(value))
        return number if 1 <= number <= 12 else np.nan
    except (ValueError, TypeError):
        text = str(value).strip().upper()
        if text in MONTH_ABBR:
            return MONTH_ABBR.index(text) + 1
        if text in MONTH_FULL:
            return MONTH_FULL.index(text) + 1
        return np.nan


def to_long_format(df, structure):
    """Return a long table with columns SUBDIVISION, YEAR, MONTH, RAINFALL."""
    sub, year = structure["subdivision"], structure["year"]
    if structure["format"] == "wide":
        month_map = structure["month_cols"]
        long_df = df.melt(id_vars=[sub, year], value_vars=list(month_map.keys()),
                          var_name="MONTH_NAME", value_name=TARGET)
        long_df["MONTH"] = long_df["MONTH_NAME"].map(month_map)
        long_df = long_df.drop(columns="MONTH_NAME")
    else:
        long_df = df[[sub, year, structure["month"], structure["rain"]]].copy()
        long_df.columns = [sub, year, "MONTH", TARGET]
        long_df["MONTH"] = long_df["MONTH"].apply(month_to_number)

    long_df = long_df.rename(columns={sub: "SUBDIVISION", year: "YEAR"})
    long_df["SUBDIVISION"] = long_df["SUBDIVISION"].astype(str).str.strip()
    long_df["YEAR"] = pd.to_numeric(long_df["YEAR"], errors="coerce")
    long_df["MONTH"] = pd.to_numeric(long_df["MONTH"], errors="coerce")
    long_df[TARGET] = pd.to_numeric(long_df[TARGET], errors="coerce")
    return long_df


# ----------------------------------------------------------------------------
# 5. CLEANING
# ----------------------------------------------------------------------------
def clean_long_data(long_df):
    section("STEP 2: DATA CLEANING")
    before = len(long_df)

    # rows without a valid subdivision, year or month cannot be used
    long_df = long_df.dropna(subset=["SUBDIVISION", "YEAR", "MONTH"]).copy()
    long_df["YEAR"] = long_df["YEAR"].astype(int)
    long_df["MONTH"] = long_df["MONTH"].astype(int)
    log(f"Rows removed because subdivision/year/month was invalid: {before - len(long_df)}")

    # duplicates: the same subdivision-year-month must appear only once
    duplicates = long_df.duplicated(subset=["SUBDIVISION", "YEAR", "MONTH"]).sum()
    long_df = long_df.drop_duplicates(subset=["SUBDIVISION", "YEAR", "MONTH"], keep="first")
    log(f"Duplicate subdivision-year-month records removed: {int(duplicates)}")

    # rainfall can never be negative: treat negatives as invalid (missing), not as zero
    negatives = int((long_df[TARGET] < 0).sum())
    long_df.loc[long_df[TARGET] < 0, TARGET] = np.nan
    log(f"Negative rainfall values set to missing: {negatives}")

    # Some subdivisions have gaps in their years. Build a COMPLETE year-month grid for
    # every subdivision so that lag features (e.g. 12 months back) always refer to the
    # correct calendar month. Months that were absent become missing (NaN), NOT zero.
    pieces = []
    for name, group in long_df.groupby("SUBDIVISION"):
        years = range(group["YEAR"].min(), group["YEAR"].max() + 1)
        full_index = pd.MultiIndex.from_product([[name], years, range(1, 13)],
                                                names=["SUBDIVISION", "YEAR", "MONTH"])
        group = group.set_index(["SUBDIVISION", "YEAR", "MONTH"]).reindex(full_index)
        pieces.append(group.reset_index())
    complete = pd.concat(pieces, ignore_index=True)
    added = len(complete) - len(long_df)
    log(f"Rows added for years/months missing from the file (left as NaN): {added}")

    missing_rain = int(complete[TARGET].isna().sum())
    log(f"Missing monthly rainfall values after cleaning: {missing_rain} "
        f"of {len(complete)} ({missing_rain / len(complete) * 100:.2f}%)")
    log("Missing rainfall values are NOT replaced by zero and are NOT guessed; "
        "rows with missing values are excluded only where a model needs a complete row.")

    complete = complete.sort_values(["SUBDIVISION", "YEAR", "MONTH"]).reset_index(drop=True)
    log(f"Long-format rows: {len(complete)}  | subdivisions: {complete['SUBDIVISION'].nunique()}"
        f"  | years: {complete['YEAR'].min()}-{complete['YEAR'].max()}")
    return complete


def build_annual_table(df, long_df, structure):
    """Annual rainfall per subdivision-year (uses the dataset's ANNUAL column if present)."""
    computed = (long_df.groupby(["SUBDIVISION", "YEAR"])[TARGET]
                .apply(lambda s: s.sum(min_count=12)).rename("ANNUAL_FROM_MONTHS").reset_index())
    annual_col = structure.get("annual")
    if annual_col is not None:
        provided = df[[structure["subdivision"], structure["year"], annual_col]].copy()
        provided.columns = ["SUBDIVISION", "YEAR", "ANNUAL"]
        provided["SUBDIVISION"] = provided["SUBDIVISION"].astype(str).str.strip()
        provided["ANNUAL"] = pd.to_numeric(provided["ANNUAL"], errors="coerce")
        provided = provided.drop_duplicates(subset=["SUBDIVISION", "YEAR"])
        annual = computed.merge(provided, on=["SUBDIVISION", "YEAR"], how="left")
        both = annual.dropna(subset=["ANNUAL", "ANNUAL_FROM_MONTHS"])
        diff = (both["ANNUAL"] - both["ANNUAL_FROM_MONTHS"]).abs()
        log(f"\nConsistency check: provided ANNUAL vs sum of 12 months "
            f"({len(both)} complete years): median difference {diff.median():.2f} mm, "
            f"maximum difference {diff.max():.2f} mm.")
        annual = annual[annual["ANNUAL"].notna()].copy()
    else:
        annual = computed.rename(columns={"ANNUAL_FROM_MONTHS": "ANNUAL"})
        annual = annual[annual["ANNUAL"].notna()].copy()
        log("\nNo ANNUAL column found: annual rainfall computed as the sum of 12 months "
            "(only for years with all 12 months present).")
    return annual[["SUBDIVISION", "YEAR", "ANNUAL"]].reset_index(drop=True)


# ----------------------------------------------------------------------------
# 6. OUTLIER ANALYSIS
# ----------------------------------------------------------------------------
def analyse_outliers(long_df):
    """Flag unusual values with the IQR rule inside each subdivision-month group.
    Values are only FLAGGED, never deleted: heavy rain months are real observations."""
    section("STEP 3: OUTLIER ANALYSIS (IQR rule, flagged but kept)")
    grouped = long_df.groupby(["SUBDIVISION", "MONTH"])[TARGET]
    q1 = grouped.transform(lambda s: s.quantile(0.25))
    q3 = grouped.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    valid = iqr > 0     # if IQR is 0 (e.g. mostly zeros) the rule is not meaningful
    long_df["OUTLIER_FLAG"] = valid & ((long_df[TARGET] > q3 + 1.5 * iqr) |
                                       (long_df[TARGET] < q1 - 1.5 * iqr))
    extreme = valid & (long_df[TARGET] > q3 + 3.0 * iqr)

    n_obs = int(long_df[TARGET].notna().sum())
    n_flag = int(long_df["OUTLIER_FLAG"].sum())
    log(f"Observations flagged as outliers (1.5*IQR): {n_flag} ({n_flag / n_obs * 100:.2f}%)")
    log(f"Observations beyond 3*IQR (extreme): {int(extreme.sum())}")
    log("Flagged values are kept: they are valid extreme-rainfall observations. "
        "Tree models handle them well; linear models can be influenced by them.")
    top = long_df.nlargest(5, TARGET)[["SUBDIVISION", "YEAR", "MONTH", TARGET]]
    log("\nFive highest monthly rainfall values in the data (kept):")
    log(top.to_string(index=False))
    return long_df


# ----------------------------------------------------------------------------
# 7. FEATURE ENGINEERING
# ----------------------------------------------------------------------------
def engineer_features(long_df):
    """Create features that use ONLY information from earlier months (no leakage)."""
    section("STEP 4: FEATURE ENGINEERING")
    df = long_df.sort_values(["SUBDIVISION", "YEAR", "MONTH"]).reset_index(drop=True)
    grouped = df.groupby("SUBDIVISION")[TARGET]

    df["LAG_1"] = grouped.shift(1)       # rainfall one month earlier
    df["LAG_2"] = grouped.shift(2)       # rainfall two months earlier
    df["LAG_12"] = grouped.shift(12)     # same calendar month, one year earlier
    # rolling means use shift(1) first so the current month is never included
    df["ROLL_3"] = grouped.transform(lambda s: s.shift(1).rolling(3).mean())
    df["ROLL_12"] = grouped.transform(lambda s: s.shift(1).rolling(12).mean())
    # average rainfall of this same subdivision and calendar month in all EARLIER years
    # (expanding mean after shift(1): the current year is never included)
    df["CLIM_PAST"] = df.groupby(["SUBDIVISION", "MONTH"])[TARGET].transform(
        lambda s: s.shift(1).expanding().mean())
    # month is circular (December is next to January): encode it with sine/cosine
    df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"] / 12)

    log("Features created: YEAR, MONTH_SIN, MONTH_COS, LAG_1, LAG_2, LAG_12, ROLL_3, ROLL_12, "
        "CLIM_PAST (+ SUBDIVISION, one-hot encoded inside the models).")
    log("Lags and rolling means only look at PAST months, so a prediction for month t "
        "never uses rainfall from month t or later (one-step-ahead forecasting).")
    return df


# ----------------------------------------------------------------------------
# 8. EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------------------
def choose_highlights(annual_df):
    available = set(annual_df["SUBDIVISION"])
    chosen = [s for s in HIGHLIGHT_SUBDIVISIONS if s in available]
    if not chosen:
        means = annual_df.groupby("SUBDIVISION")["ANNUAL"].mean().sort_values()
        chosen = list(means.index[:2]) + list(means.index[-2:])
    return chosen


def plot_annual_trend(annual_df, highlights):
    national = annual_df.groupby("YEAR")["ANNUAL"].mean()
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    axes[0].plot(national.index, national.values, color="steelblue", alpha=0.6,
                 label="Yearly mean over subdivisions")
    axes[0].plot(national.index, national.rolling(10, center=True).mean(), color="darkred",
                 linewidth=2, label="10-year moving average")
    axes[0].set_title("Annual Rainfall Trend: Simple Mean Across Subdivisions (1901-2017)")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Annual rainfall (mm)")
    axes[0].legend()
    for name in highlights:
        series = annual_df[annual_df["SUBDIVISION"] == name].set_index("YEAR")["ANNUAL"]
        axes[1].plot(series.index, series.rolling(10, center=True, min_periods=5).mean(),
                     label=name)
    axes[1].set_title("Annual Rainfall in Selected Subdivisions (10-year moving average)")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Annual rainfall (mm)")
    axes[1].legend()
    save_figure(fig, "eda_annual_trend.png")


def plot_monthly_pattern(long_df, highlights):
    month_labels = [m.title() for m in MONTH_ABBR]
    overall = long_df.groupby("MONTH")[TARGET].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, 13), overall.values, color="black", linewidth=3,
            label="Mean of all subdivisions")
    for name in highlights:
        part = long_df[long_df["SUBDIVISION"] == name].groupby("MONTH")[TARGET].mean()
        ax.plot(range(1, 13), part.values, marker="o", label=name)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_title("Average Monthly Rainfall Pattern (1901-2017)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean monthly rainfall (mm)")
    ax.legend()
    save_figure(fig, "eda_monthly_pattern.png")


def plot_distributions(long_df, annual_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    monthly = long_df[TARGET].dropna()
    sns.histplot(monthly, bins=60, ax=axes[0], color="steelblue")
    axes[0].set_title("Distribution of Monthly Rainfall")
    axes[0].set_xlabel("Monthly rainfall (mm)")
    axes[0].set_ylabel("Number of observations")
    sns.histplot(np.log1p(monthly), bins=60, ax=axes[1], color="seagreen")
    axes[1].set_title("Monthly Rainfall After log(1 + x)")
    axes[1].set_xlabel("log(1 + monthly rainfall in mm)")
    axes[1].set_ylabel("Number of observations")
    sns.histplot(annual_df["ANNUAL"], bins=50, ax=axes[2], color="darkorange")
    axes[2].set_title("Distribution of Annual Rainfall")
    axes[2].set_xlabel("Annual rainfall (mm)")
    axes[2].set_ylabel("Number of subdivision-years")
    save_figure(fig, "eda_distributions.png")
    skew = monthly.skew()
    log(f"  Skewness of monthly rainfall: {skew:.2f} (strongly right-skewed if well above 1)")


def plot_boxplots(long_df, annual_df):
    fig, ax = plt.subplots(figsize=(11, 6))
    data = long_df.dropna(subset=[TARGET])
    sns.boxplot(data=data, x="MONTH", y=TARGET, ax=ax, color="lightsteelblue", fliersize=2)
    ax.set_xticks(range(12))
    ax.set_xticklabels([m.title() for m in MONTH_ABBR])
    ax.set_title("Boxplot of Monthly Rainfall by Month (all subdivisions)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly rainfall (mm)")
    save_figure(fig, "eda_boxplot_monthly.png")

    order = annual_df.groupby("SUBDIVISION")["ANNUAL"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(10, 11))
    sns.boxplot(data=annual_df, y="SUBDIVISION", x="ANNUAL", order=order, ax=ax,
                color="lightgreen", fliersize=2)
    ax.set_title("Boxplot of Annual Rainfall by Subdivision (sorted by median)")
    ax.set_xlabel("Annual rainfall (mm)")
    ax.set_ylabel("Meteorological subdivision")
    save_figure(fig, "eda_boxplot_annual_by_subdivision.png")


def plot_correlation_heatmap(long_df, annual_df):
    wide = long_df.pivot_table(index=["SUBDIVISION", "YEAR"], columns="MONTH",
                               values=TARGET, aggfunc="first")
    wide.columns = [MONTH_ABBR[m - 1] for m in wide.columns]
    wide = wide.reset_index().merge(annual_df, on=["SUBDIVISION", "YEAR"], how="left")
    corr = wide[MONTH_ABBR + ["ANNUAL"]].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Correlation Between Monthly and Annual Rainfall (Pearson, mm)")
    save_figure(fig, "eda_correlation_heatmap.png")


def plot_regional_comparison(annual_df):
    stats = annual_df.groupby("SUBDIVISION")["ANNUAL"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.barh(stats.index, stats.values, color="teal")
    ax.set_title("Mean Annual Rainfall by Subdivision (1901-2017)")
    ax.set_xlabel("Mean annual rainfall (mm)")
    ax.set_ylabel("Meteorological subdivision")
    save_figure(fig, "eda_regional_comparison.png")


def plot_monthly_rainfall_heatmap(long_df):
    """Heatmap of mean rainfall by subdivision and calendar month."""
    pivot = long_df.pivot_table(index="SUBDIVISION", columns="MONTH",
                                values=TARGET, aggfunc="mean")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = [MONTH_ABBR[int(m) - 1].title() for m in pivot.columns]
    order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(13, max(7, 0.42 * len(pivot))))
    sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.25, ax=ax,
                cbar_kws={"label": "Mean monthly rainfall (mm)"})
    ax.set_title("Rainfall Seasonality Heatmap: Subdivision vs Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Meteorological subdivision")
    save_figure(fig, "eda_subdivision_month_rainfall_heatmap.png")


def plot_annual_rainfall_heatmap(annual_df):
    """Heatmap of annual rainfall across years and subdivisions."""
    pivot = annual_df.pivot_table(index="SUBDIVISION", columns="YEAR",
                                  values="ANNUAL", aggfunc="mean")
    order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(16, max(7, 0.42 * len(pivot))))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, xticklabels=max(1, len(pivot.columns) // 15),
                yticklabels=True, cbar_kws={"label": "Annual rainfall (mm)"})
    ax.set_title("Annual Rainfall Heatmap Across Meteorological Subdivisions")
    ax.set_xlabel("Year")
    ax.set_ylabel("Meteorological subdivision")
    ax.tick_params(axis="x", labelrotation=45)
    save_figure(fig, "eda_annual_rainfall_heatmap.png")


def run_eda(long_df, annual_df):
    section("STEP 5: EXPLORATORY DATA ANALYSIS (graphs)")
    highlights = choose_highlights(annual_df)
    log(f"Subdivisions highlighted in comparison graphs: {highlights}")
    safe_run("annual trend plot", plot_annual_trend, annual_df, highlights)
    safe_run("monthly pattern plot", plot_monthly_pattern, long_df, highlights)
    safe_run("distribution plots", plot_distributions, long_df, annual_df)
    safe_run("boxplots", plot_boxplots, long_df, annual_df)
    safe_run("correlation heatmap", plot_correlation_heatmap, long_df, annual_df)
    safe_run("regional comparison", plot_regional_comparison, annual_df)
    safe_run("subdivision-month rainfall heatmap", plot_monthly_rainfall_heatmap, long_df)
    safe_run("annual rainfall heatmap", plot_annual_rainfall_heatmap, annual_df)


# ----------------------------------------------------------------------------
# 9. SUPERVISED LEARNING: REGRESSION
# ----------------------------------------------------------------------------
def chronological_split(model_df):
    """Train on earlier years, test on the most recent years (no shuffling)."""
    years = sorted(model_df["YEAR"].unique())
    n_test_years = max(1, int(round(len(years) * TEST_YEAR_FRACTION)))
    cutoff_year = years[-n_test_years]
    train = model_df[model_df["YEAR"] < cutoff_year].copy()
    test = model_df[model_df["YEAR"] >= cutoff_year].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set.")
    return train, test, cutoff_year


def make_preprocessor(numeric_cols, use_subdivision=True, polynomial=False):
    """Scale numeric columns and one-hot encode the subdivision.
    These transformers live inside the model pipeline, so they are fitted on the
    TRAINING data only (this prevents data leakage from the test set)."""
    if polynomial:
        numeric_step = Pipeline([("scale", StandardScaler()),
                                 ("poly", PolynomialFeatures(degree=2, include_bias=False))])
    else:
        numeric_step = StandardScaler()
    parts = [("num", numeric_step, numeric_cols)]
    if use_subdivision:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES))
    return ColumnTransformer(parts)


def evaluate(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {"MAE": mean_absolute_error(actual, predicted), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(actual, predicted)}


def climatology_baseline(train, other):
    """Baseline: average rainfall of the same subdivision and month in the TRAINING years."""
    table = train.groupby(["SUBDIVISION", "MONTH"])[TARGET].mean().rename("CLIM")
    merged = other.join(table, on=["SUBDIVISION", "MONTH"])
    month_mean = train.groupby("MONTH")[TARGET].mean()
    merged["CLIM"] = merged["CLIM"].fillna(merged["MONTH"].map(month_mean))
    return merged["CLIM"].fillna(train[TARGET].mean()).values


def run_regression(feature_df):
    section("STEP 6: SUPERVISED LEARNING - REGRESSION")
    log(f"Target variable: {TARGET} (monthly rainfall in mm per subdivision)")

    needed = [TARGET] + NUMERIC_FEATURES
    model_df = feature_df.dropna(subset=needed).copy()
    log(f"Rows available for modelling: {len(model_df)} of {len(feature_df)} "
        f"(rows with a missing target/feature, e.g. the first 12 months of each "
        f"series, are excluded rather than filled with zero)")

    train, test, cutoff = chronological_split(model_df)
    log(f"Chronological split -> training years: {train['YEAR'].min()}-{train['YEAR'].max()} "
        f"({len(train)} rows) | test years: {test['YEAR'].min()}-{test['YEAR'].max()} "
        f"({len(test)} rows)")

    X_train, y_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train[TARGET]
    X_test, y_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test[TARGET]

    # Simple linear regression uses the ONE numeric feature most correlated with the
    # target in the TRAINING data.
    correlations = train[NUMERIC_FEATURES].corrwith(train[TARGET]).abs().sort_values(ascending=False)
    simple_feature = correlations.index[0]
    log(f"Simple Linear Regression feature (highest |correlation| in training data): "
        f"{simple_feature} (r = {train[simple_feature].corr(train[TARGET]):.3f})")

    models = {
        "Simple Linear Regression": Pipeline([
            ("prep", ColumnTransformer([("num", StandardScaler(), [simple_feature])])),
            ("model", LinearRegression())]),
        "Multiple Linear Regression": Pipeline([
            ("prep", make_preprocessor(NUMERIC_FEATURES)),
            ("model", LinearRegression())]),
        "Polynomial Regression (degree 2)": Pipeline([
            ("prep", make_preprocessor(NUMERIC_FEATURES, polynomial=True)),
            ("model", LinearRegression())]),
        "Decision Tree Regressor": Pipeline([
            ("prep", make_preprocessor(NUMERIC_FEATURES)),
            ("model", DecisionTreeRegressor(max_depth=8, min_samples_leaf=10,
                                            random_state=RANDOM_STATE))]),
        "Random Forest Regressor": Pipeline([
            ("prep", make_preprocessor(NUMERIC_FEATURES)),
            ("model", RandomForestRegressor(n_estimators=200, max_depth=12,
                                            min_samples_leaf=5, n_jobs=-1,
                                            random_state=RANDOM_STATE))]),
        "Gradient Boosting Regressor": Pipeline([
            ("prep", make_preprocessor(NUMERIC_FEATURES)),
            ("model", GradientBoostingRegressor(n_estimators=150, learning_rate=0.05,
                                                max_depth=3, loss="squared_error",
                                                random_state=RANDOM_STATE))]),
    }

    rows = []
    predictions = test[["SUBDIVISION", "YEAR", "MONTH"]].copy()
    predictions["Actual"] = y_test.values
    fitted = {}

    for name, pipeline in models.items():
        try:
            log(f"\nTraining: {name} ...")
            pipeline.fit(X_train, y_train)
            # rainfall cannot be negative, so negative predictions are set to 0
            train_pred = np.clip(pipeline.predict(X_train), 0, None)
            test_pred = np.clip(pipeline.predict(X_test), 0, None)
            train_m, test_m = evaluate(y_train, train_pred), evaluate(y_test, test_pred)
            rows.append({"Model": name, "Type": "Model",
                         "Train_MAE": train_m["MAE"], "Train_RMSE": train_m["RMSE"],
                         "Train_R2": train_m["R2"],
                         "Test_MAE": test_m["MAE"], "Test_MSE": test_m["MSE"],
                         "Test_RMSE": test_m["RMSE"], "Test_R2": test_m["R2"]})
            predictions[name] = test_pred
            fitted[name] = pipeline
            log(f"  done. Test RMSE = {test_m['RMSE']:.2f} mm, Test R2 = {test_m['R2']:.3f}")
        except Exception as error:
            log(f"  [WARNING] {name} failed and was skipped: {error}")

    if not rows:
        raise RuntimeError("No regression model could be trained.")

    # Simple reference predictors (not trained models) to judge if the models add value
    baselines = {
        "Baseline: same month last year": test["LAG_12"].values,
        "Baseline: training average of subdivision-month": climatology_baseline(train, test),
    }
    train_clim = climatology_baseline(train, train)
    for name, test_pred in baselines.items():
        train_pred = train["LAG_12"].values if "last year" in name else train_clim
        train_m, test_m = evaluate(y_train, train_pred), evaluate(y_test, test_pred)
        rows.append({"Model": name, "Type": "Baseline",
                     "Train_MAE": train_m["MAE"], "Train_RMSE": train_m["RMSE"],
                     "Train_R2": train_m["R2"],
                     "Test_MAE": test_m["MAE"], "Test_MSE": test_m["MSE"],
                     "Test_RMSE": test_m["RMSE"], "Test_R2": test_m["R2"]})
        predictions[name] = test_pred

    results = pd.DataFrame(rows)
    results["R2_gap_train_minus_test"] = results["Train_R2"] - results["Test_R2"]
    results = results.round(4)

    section("MODEL COMPARISON (test set = most recent years)")
    log(results.to_string(index=False))
    save_csv(results, "model_comparison.csv")
    save_csv(predictions.round(3), "regression_predictions.csv")

    # best model = lowest test RMSE among the trained models (baselines excluded)
    model_rows = results[results["Type"] == "Model"]
    best_name = model_rows.sort_values("Test_RMSE").iloc[0]["Model"]
    best = model_rows[model_rows["Model"] == best_name].iloc[0]
    log(f"\nBest model on the test set (lowest RMSE): {best_name} "
        f"(RMSE {best['Test_RMSE']:.2f} mm, MAE {best['Test_MAE']:.2f} mm, R2 {best['Test_R2']:.3f})")
    for _, base in results[results["Type"] == "Baseline"].iterrows():
        status = "better than" if base["Test_RMSE"] < best["Test_RMSE"] else "not better than"
        log(f"  Reference '{base['Model']}' (RMSE {base['Test_RMSE']:.2f}) is {status} the best model.")

    interpret_fit(model_rows)
    show_feature_information(fitted)

    safe_run("actual vs predicted plots", plot_actual_vs_predicted, predictions, list(fitted), best_name)
    safe_run("residual plots", plot_residuals, predictions, best_name)
    safe_run("model comparison plot", plot_model_comparison, results)
    safe_run("model metrics heatmap", plot_model_metrics_heatmap, results)
    safe_run("best model residual heatmap", plot_best_residual_heatmap, predictions, best_name)
    safe_run("time series plot", plot_best_timeseries, predictions, best_name)
    if "Random Forest Regressor" in fitted:
        safe_run("Random Forest feature importance plot", plot_feature_importance, fitted["Random Forest Regressor"])
    if "Gradient Boosting Regressor" in fitted:
        safe_run("Gradient Boosting feature importance plot", plot_feature_importance,
                 fitted["Gradient Boosting Regressor"])
    return {"results": results, "best": best_name, "train_years": (train['YEAR'].min(), train['YEAR'].max()),
            "test_years": (test['YEAR'].min(), test['YEAR'].max())}


def interpret_fit(model_rows):
    """Describe under/overfitting using the real train and test R2 values."""
    log("\nTraining vs testing performance (rule-of-thumb interpretation):")
    for _, row in model_rows.iterrows():
        gap = row["Train_R2"] - row["Test_R2"]
        if row["Train_R2"] < 0.5 and row["Test_R2"] < 0.5:
            note = "low R2 on both sets -> possible underfitting (high bias)"
        elif gap > 0.10:
            note = "train R2 clearly above test R2 -> possible overfitting (high variance)"
        elif gap > 0.05:
            note = "train R2 somewhat above test R2 -> mild overfitting tendency"
        else:
            note = "train and test R2 are similar -> no strong sign of overfitting"
        log(f"  {row['Model']}: train R2 {row['Train_R2']:.3f}, test R2 {row['Test_R2']:.3f}, "
            f"gap {gap:.3f} -> {note}")
    log("(The 0.5, 0.10 and 0.05 thresholds are only rough guides, not strict rules.)")


def show_feature_information(fitted):
    if "Multiple Linear Regression" in fitted:
        pipe = fitted["Multiple Linear Regression"]
        names = pipe.named_steps["prep"].get_feature_names_out()
        coefs = pd.Series(pipe.named_steps["model"].coef_, index=names)
        numeric = coefs[[n for n in coefs.index if n.startswith("num__")]]
        numeric.index = [n.replace("num__", "") for n in numeric.index]
        log("\nMultiple Linear Regression coefficients of numeric features "
            "(features standardised, so sizes are comparable; association only, not causation):")
        log(numeric.sort_values(key=abs, ascending=False).round(3).to_string())
    if "Random Forest Regressor" in fitted:
        importance = random_forest_importance(fitted["Random Forest Regressor"])
        log("\nRandom Forest feature importance (top 10; SUBDIVISION dummies are summed):")
        log(importance.head(10).round(4).to_string())


def random_forest_importance(pipeline):
    names = pipeline.named_steps["prep"].get_feature_names_out()
    values = pipeline.named_steps["model"].feature_importances_
    series = pd.Series(values, index=names)
    grouped = {}
    for name, value in series.items():
        clean = name.replace("num__", "")
        if clean.startswith("cat__SUBDIVISION"):
            clean = "SUBDIVISION (all dummies)"
        grouped[clean] = grouped.get(clean, 0) + value
    return pd.Series(grouped).sort_values(ascending=False)


def plot_actual_vs_predicted(predictions, model_names, best_name):
    n = len(model_names)
    columns = 3
    rows_n = int(np.ceil(n / columns))
    fig, axes = plt.subplots(rows_n, columns, figsize=(15, 4.8 * rows_n))
    axes = np.atleast_1d(axes).ravel()
    limit = max(predictions["Actual"].max(), 1)
    for ax, name in zip(axes, model_names):
        ax.scatter(predictions["Actual"], predictions[name], s=4, alpha=0.3,
                   color="steelblue")
        ax.plot([0, limit], [0, limit], color="red", linewidth=1, label="Perfect prediction")
        ax.set_title(f"Actual vs Predicted: {name}" + (" (best)" if name == best_name else ""),
                     fontsize=10)
        ax.set_xlabel("Actual monthly rainfall (mm)")
        ax.set_ylabel("Predicted monthly rainfall (mm)")
        ax.legend(loc="upper left", fontsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    save_figure(fig, "model_actual_vs_predicted.png")


def plot_residuals(predictions, best_name):
    residuals = predictions["Actual"] - predictions[best_name]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(predictions[best_name], residuals, s=4, alpha=0.3, color="darkorange")
    axes[0].axhline(0, color="black")
    axes[0].set_title(f"Residuals vs Predicted Rainfall: {best_name}")
    axes[0].set_xlabel("Predicted monthly rainfall (mm)")
    axes[0].set_ylabel("Residual = actual - predicted (mm)")
    sns.histplot(residuals, bins=60, ax=axes[1], color="slateblue")
    axes[1].set_title(f"Distribution of Residuals: {best_name}")
    axes[1].set_xlabel("Residual (mm)")
    axes[1].set_ylabel("Number of test observations")
    save_figure(fig, "model_residuals.png")


def plot_model_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    order = results.sort_values("Test_RMSE")
    axes[0].barh(order["Model"], order["Test_RMSE"], color="steelblue")
    axes[0].set_title("Test-Set RMSE by Model (lower is better)")
    axes[0].set_xlabel("RMSE (mm of monthly rainfall)")
    axes[0].set_ylabel("Model")
    positions = np.arange(len(results))
    axes[1].bar(positions - 0.2, results["Train_R2"], width=0.4, label="Train R2")
    axes[1].bar(positions + 0.2, results["Test_R2"], width=0.4, label="Test R2")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(results["Model"], rotation=40, ha="right", fontsize=8)
    axes[1].set_title("Training vs Test R2 by Model")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("R2 score (unitless)")
    axes[1].legend()
    save_figure(fig, "model_comparison.png")


def plot_model_metrics_heatmap(results):
    """Standardised model metrics heatmap; makes differently scaled metrics comparable."""
    model_results = results[results["Type"] == "Model"].copy()
    if model_results.empty:
        return
    metrics = model_results.set_index("Model")[["Test_MAE", "Test_RMSE", "Test_R2"]]
    # Z-scores are for visual comparison only; raw metrics remain in model_comparison.csv.
    zscores = metrics.apply(lambda col: (col - col.mean()) / col.std(ddof=0)
                            if col.std(ddof=0) > 0 else col * 0, axis=0)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.7 * len(zscores))))
    sns.heatmap(zscores, annot=metrics.round(2), fmt=".2f", cmap="vlag", center=0,
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "Standardised score (colour)"})
    ax.set_title("Regression Model Metrics Heatmap (numbers show raw values)")
    ax.set_xlabel("Test metric: MAE/RMSE lower is better; R² higher is better")
    ax.set_ylabel("Model")
    save_figure(fig, "model_metrics_heatmap.png")


def plot_best_residual_heatmap(predictions, best_name):
    """Show mean signed prediction error by subdivision and month for the best model."""
    data = predictions[["SUBDIVISION", "MONTH", "Actual", best_name]].copy()
    data["Residual"] = data["Actual"] - data[best_name]
    pivot = data.pivot_table(index="SUBDIVISION", columns="MONTH",
                             values="Residual", aggfunc="mean")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = [MONTH_ABBR[int(m) - 1].title() for m in pivot.columns]
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(13, max(7, 0.42 * len(pivot))))
    limit = np.nanmax(np.abs(pivot.to_numpy()))
    if not np.isfinite(limit) or limit == 0:
        limit = 1
    sns.heatmap(pivot, cmap="RdBu_r", center=0, vmin=-limit, vmax=limit,
                ax=ax, cbar_kws={"label": "Mean residual (actual - predicted, mm)"})
    ax.set_title(f"Best Model Error Heatmap by Subdivision and Month: {best_name}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Meteorological subdivision")
    save_figure(fig, "model_best_residual_heatmap.png")


def plot_best_timeseries(predictions, best_name):
    highlights = [s for s in HIGHLIGHT_SUBDIVISIONS if s in set(predictions["SUBDIVISION"])]
    name = highlights[0] if highlights else predictions["SUBDIVISION"].iloc[0]
    part = predictions[predictions["SUBDIVISION"] == name].sort_values(["YEAR", "MONTH"])
    dates = pd.to_datetime(dict(year=part["YEAR"], month=part["MONTH"], day=1))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dates, part["Actual"], label="Actual", color="black")
    ax.plot(dates, part[best_name], label=f"Predicted ({best_name})", color="crimson", alpha=0.8)
    ax.set_title(f"Actual vs Predicted Monthly Rainfall in the Test Period: {name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly rainfall (mm)")
    ax.legend()
    save_figure(fig, "model_timeseries_example.png")


def plot_feature_importance(pipeline):
    importance = random_forest_importance(pipeline).head(12).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance.index, importance.values, color="seagreen")
    model_name = type(pipeline.named_steps["model"]).__name__.replace("Regressor", "")
    ax.set_title(f"{model_name} Feature Importance (top 12)")
    ax.set_xlabel("Importance (share of total impurity reduction)")
    ax.set_ylabel("Feature")
    model_name = type(pipeline.named_steps["model"]).__name__.replace("Regressor", "").lower()
    save_figure(fig, f"model_feature_importance_{model_name}.png")


# ----------------------------------------------------------------------------
# 10. UNSUPERVISED LEARNING: K-MEANS AND PCA
# ----------------------------------------------------------------------------
def run_clustering(long_df):
    section("STEP 7: UNSUPERVISED LEARNING - K-MEANS CLUSTERING AND PCA")
    # One row per subdivision: its average rainfall in each of the 12 months
    profile = long_df.pivot_table(index="SUBDIVISION", columns="MONTH", values=TARGET,
                                  aggfunc="mean")
    profile.columns = [MONTH_ABBR[m - 1].title() for m in profile.columns]
    log(f"Clustering {len(profile)} subdivisions using their {profile.shape[1]} "
        f"average monthly rainfall values (mm).")
    if profile.isna().any().any():
        log("  Some subdivision-month averages are missing; filled with the column median.")
    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(profile)
    scaled = StandardScaler().fit_transform(values)

    # Elbow method (inertia) and silhouette score for several k
    max_k = min(max(K_RANGE), len(profile) - 1)
    k_values = [k for k in K_RANGE if 2 <= k <= max_k]
    if len(k_values) < 2:
        raise ValueError("Not enough subdivisions to test several values of k.")
    inertias, silhouettes = [], []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(scaled)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(scaled, model.labels_))
    scores = pd.DataFrame({"k": k_values, "inertia": inertias, "silhouette": silhouettes})
    log("\nElbow / silhouette results:")
    log(scores.round(3).to_string(index=False))

    best_k = int(scores.sort_values("silhouette", ascending=False).iloc[0]["k"])
    chosen_k = N_CLUSTERS if N_CLUSTERS is not None else best_k
    log(f"\nk with highest silhouette score: {best_k}. Using k = {chosen_k}"
        f"{' (set manually in N_CLUSTERS)' if N_CLUSTERS is not None else ''}.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(k_values, inertias, marker="o")
    axes[0].set_title("Elbow Method for K-Means")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
    axes[1].plot(k_values, silhouettes, marker="o", color="darkorange")
    axes[1].set_title("Silhouette Score for K-Means")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score (-1 to 1)")
    save_figure(fig, "cluster_elbow_silhouette.png")

    final = KMeans(n_clusters=chosen_k, n_init=10, random_state=RANDOM_STATE).fit(scaled)
    profile["CLUSTER"] = final.labels_

    # PCA (optional extension): reduce the 12 standardised features to 2 components
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(scaled)
    profile["PC1"], profile["PC2"] = components[:, 0], components[:, 1]
    explained = pca.explained_variance_ratio_
    log(f"\nPCA: PC1 explains {explained[0] * 100:.1f}% and PC2 explains "
        f"{explained[1] * 100:.1f}% of the variance "
        f"(total {explained.sum() * 100:.1f}%).")

    month_cols = [m.title() for m in MONTH_ABBR]
    profile["MEAN_ANNUAL_MM"] = profile[month_cols].sum(axis=1)
    monsoon_cols = ["Jun", "Jul", "Aug", "Sep"]
    profile["JJAS_SHARE_PERCENT"] = profile[monsoon_cols].sum(axis=1) / profile["MEAN_ANNUAL_MM"] * 100
    profile["PEAK_MONTH"] = profile[month_cols].idxmax(axis=1)

    assignments = profile.reset_index()[["SUBDIVISION", "CLUSTER", "MEAN_ANNUAL_MM",
                                         "JJAS_SHARE_PERCENT", "PEAK_MONTH", "PC1", "PC2"]]
    assignments = assignments.sort_values(["CLUSTER", "SUBDIVISION"]).round(2)
    save_csv(assignments, "cluster_assignments.csv")

    summary_rows = []
    for cluster, group in profile.groupby("CLUSTER"):
        row = {"CLUSTER": cluster, "N_SUBDIVISIONS": len(group),
               "MEAN_ANNUAL_MM": group["MEAN_ANNUAL_MM"].mean(),
               "MIN_ANNUAL_MM": group["MEAN_ANNUAL_MM"].min(),
               "MAX_ANNUAL_MM": group["MEAN_ANNUAL_MM"].max(),
               "JJAS_SHARE_PERCENT": group["JJAS_SHARE_PERCENT"].mean(),
               "MOST_COMMON_PEAK_MONTH": group["PEAK_MONTH"].mode().iloc[0]}
        for month in month_cols:
            row[f"MEAN_{month.upper()}_MM"] = group[month].mean()
        row["SUBDIVISIONS"] = "; ".join(sorted(group.index))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).round(2)
    save_csv(summary, "cluster_summary.csv")

    log("\nCluster characteristics (descriptive, based only on the data):")
    for _, row in summary.iterrows():
        log(f"  Cluster {int(row['CLUSTER'])}: {int(row['N_SUBDIVISIONS'])} subdivisions, "
            f"mean annual rainfall about {row['MEAN_ANNUAL_MM']:.0f} mm "
            f"(range {row['MIN_ANNUAL_MM']:.0f}-{row['MAX_ANNUAL_MM']:.0f}), "
            f"June-September share about {row['JJAS_SHARE_PERCENT']:.0f}% of the year, "
            f"most common peak month {row['MOST_COMMON_PEAK_MONTH']}.")
        log(f"     Members: {row['SUBDIVISIONS']}")
    log("These clusters group subdivisions with similar average monthly rainfall profiles. "
        "They are NOT official climate, drought or flood categories.")

    # Cluster plots
    fig, ax = plt.subplots(figsize=(11, 8))
    scatter = sns.scatterplot(data=profile.reset_index(), x="PC1", y="PC2", hue="CLUSTER",
                              palette="tab10", s=90, ax=ax)
    for name, row in profile.iterrows():
        ax.annotate(name, (row["PC1"], row["PC2"]), fontsize=6, alpha=0.8)
    ax.set_title("PCA Projection of Subdivision Rainfall Profiles Coloured by K-Means Cluster")
    ax.set_xlabel(f"Principal component 1 ({explained[0] * 100:.1f}% of variance)")
    ax.set_ylabel(f"Principal component 2 ({explained[1] * 100:.1f}% of variance)")
    save_figure(fig, "cluster_pca_scatter.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster, group in profile.groupby("CLUSTER"):
        ax.plot(month_cols, group[month_cols].mean(), marker="o",
                label=f"Cluster {cluster} (n={len(group)})")
    ax.set_title("Average Monthly Rainfall Profile of Each Cluster")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean monthly rainfall (mm)")
    ax.legend()
    save_figure(fig, "cluster_monthly_profiles.png")
    return {"k": chosen_k, "best_k": best_k, "silhouette": float(scores.loc[scores["k"] == chosen_k, "silhouette"].iloc[0]),
            "pca_variance": float(explained.sum())}


# ----------------------------------------------------------------------------
# 11. PROJECT SUMMARY FILE
# ----------------------------------------------------------------------------
DISCUSSION_TEXT = """
GENERAL DISCUSSION (concepts; see the numerical results above for this dataset)

Bias-variance trade-off: A model with high bias is too simple and misses real patterns
(underfitting). A model with high variance follows the training data too closely and
does not generalise (overfitting). Comparing training and test scores for each model
shows where it lies on this scale.

Underfitting and overfitting: Underfitting appears as poor scores on both training and
test data. Overfitting appears as clearly better training scores than test scores.
Limiting tree depth and leaf size (used here) reduces overfitting.

Feature importance: Random Forest importance and linear-model coefficients describe
how much a model USES a feature. They show association, not physical cause and effect.

Model limitations:
 - Only past rainfall, calendar information and subdivision identity are used. No
   temperature, humidity, pressure, ENSO or other physical drivers are included.
 - Monthly rainfall is highly skewed and has many near-zero months, which makes
   point prediction of single months difficult.
 - Subdivision-level monthly totals hide local and daily variation.
 - Rows with missing values were excluded from modelling; they were not imputed.
 - The test period is the most recent block of years; results may differ for other periods,
   and climate may change over time.
 - One-step-ahead prediction uses the true rainfall of previous months. Forecasting several
   months ahead would give weaker results.
 - K-Means clusters depend on chosen features, scaling and k; they are descriptive only.

Potential uses and limits: Such models can support exploratory analysis, planning
discussions and as a baseline for more advanced forecasting. They are not a replacement
for operational weather forecasts and should not be used alone for safety-critical
decisions such as flood warnings.
"""


def write_summary(dataset_path, regression_info, cluster_info):
    lines = ["PROJECT SUMMARY: Rainfall Prediction and Pattern Analysis Using Historical "
             "Indian Meteorological Data (INT234 Predictive Analytics)",
             f"Dataset used: {dataset_path}", ""]
    lines += REPORT_LINES
    lines.append(DISCUSSION_TEXT)
    lines.append("FILES CREATED IN THIS RUN:")
    for name in OUTPUT_FILES:
        lines.append(f"  - {name}")
    with open(out_path("project_summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print("  saved file: project_summary.txt")


# ----------------------------------------------------------------------------
# 12. MAIN PROGRAM
# ----------------------------------------------------------------------------
def main():
    global OUTPUT_DIR
    np.random.seed(RANDOM_STATE)
    sns.set_theme(style="whitegrid")

    try:
        base_folder = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_folder = os.getcwd()
    OUTPUT_DIR = os.path.join(base_folder, OUTPUT_FOLDER_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Results will be saved in: {OUTPUT_DIR}")

    # --- load and inspect
    try:
        raw_df, dataset_path = load_dataset(DATASET_PATH)
    except Exception as error:
        print(f"\nERROR: could not load the dataset.\n{error}")
        return
    inspect_dataset(raw_df)

    if len(raw_df) < 30:
        log(f"\nWARNING: the dataset has only {len(raw_df)} rows. "
            "Some plots or models may be unreliable with very small datasets.")
    else:
        log(f"\nDataset row check passed: {len(raw_df)} data rows.")

    # --- detect structure and convert to long format
    try:
        structure = detect_structure(raw_df)
    except Exception as error:
        log(f"\nERROR: {error}")
        return
    log(f"\nDetected data format: {structure['format'].upper()} "
        f"(subdivision column = '{structure['subdivision']}', year column = '{structure['year']}')")
    if structure["format"] == "wide":
        log("Monthly columns found: " + ", ".join(structure["month_cols"].keys()))
        log("The wide table (one row per subdivision-year, one column per month) will be "
            "converted to long format (one row per subdivision-year-month).")

    long_df = to_long_format(raw_df, structure)
    long_df = clean_long_data(long_df)
    annual_df = build_annual_table(raw_df, long_df, structure)
    long_df = analyse_outliers(long_df)
    feature_df = engineer_features(long_df)
    save_csv(feature_df.round(4), "cleaned_rainfall.csv")

    # --- analyses (each one is independent)
    safe_run("EDA", run_eda, long_df, annual_df)
    regression_info = safe_run("regression", run_regression, feature_df)
    cluster_info = safe_run("clustering", run_clustering, long_df)

    section("STEP 8: FINAL NOTES")
    if regression_info:
        log(f"Best regression model: {regression_info['best']}.")
    if cluster_info:
        log(f"K-Means used k = {cluster_info['k']} (silhouette {cluster_info['silhouette']:.3f}).")
    log("All results above were computed from the supplied dataset only.")
    write_summary(dataset_path, regression_info, cluster_info)
    print(f"\nFinished. Open the folder '{OUTPUT_DIR}' to see all results.")


if __name__ == "__main__":
    main()
