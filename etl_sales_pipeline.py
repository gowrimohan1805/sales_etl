import os
import glob
import pandas as pd
from datetime import datetime
from dateutil import parser

RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

OUTPUT_CSV = os.path.join(PROCESSED_DIR, "sales_clean.csv")
OUTPUT_PARQUET = os.path.join(PROCESSED_DIR, "sales_clean.parquet")


def ensure_dirs():
    """Create folders if they don't exist."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def parse_date_safe(x):
    """Safely parse dates; return NaT if invalid."""
    try:
        return parser.parse(str(x)).date()
    except Exception:
        return pd.NaT


def load_raw_files():
    """Load all CSV files from RAW_DATA_DIR and combine into a single DataFrame."""
    pattern = os.path.join(RAW_DATA_DIR, "*.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}. "
                                f"Put your raw sales CSVs there.")

    df_list = []
    for f in files:
        print(f"[INFO] Reading file: {f}")
        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)
        df_list.append(df)

    combined = pd.concat(df_list, ignore_index=True)
    print(f"[INFO] Combined shape from {len(files)} files: {combined.shape}")
    return combined


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column names and required columns exist."""
    # Lowercase + strip column names
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = [
        "order_id", "order_date", "region", "country",
        "customer_id", "product_id", "category", "sub_category",
        "quantity", "unit_price", "discount", "profit"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\n"
                         f"Columns present: {list(df.columns)}")

    return df


def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Main cleaning & KPI computation logic."""
    df = df.copy()

    # Standardize column names and ensure required ones exist
    df = standardize_columns(df)

    # Parse dates
    df["order_date"] = df["order_date"].apply(parse_date_safe)
    df = df.dropna(subset=["order_date"])

    # Basic type conversions
    numeric_cols = ["quantity", "unit_price", "discount", "profit"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["quantity", "unit_price", "discount", "profit"])

    # Ensure quantity > 0 and unit_price >= 0
    df = df[(df["quantity"] > 0) & (df["unit_price"] >= 0)]

    # String cleaning for categorical columns
    for col in ["region", "country", "category", "sub_category"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    # Compute KPIs
    df["gross_sales"] = df["quantity"] * df["unit_price"]
    df["discount_amount"] = df["gross_sales"] * df["discount"]
    df["net_sales"] = df["gross_sales"] - df["discount_amount"]

    # Replace 0 net_sales to avoid division by zero
    df["net_sales_no_zero"] = df["net_sales"].replace({0: pd.NA})
    df["margin_pct"] = df["profit"] / df["net_sales_no_zero"]
    df.drop(columns=["net_sales_no_zero"], inplace=True)

    # Add time dimensions
    df["order_year"] = df["order_date"].apply(lambda d: d.year)
    df["order_month"] = df["order_date"].apply(lambda d: d.month)
    df["order_month_name"] = df["order_date"].apply(lambda d: d.strftime("%b"))
    df["order_quarter"] = df["order_date"].apply(lambda d: f"Q{((d.month - 1)//3) + 1}")

    # Drop duplicates
    df = df.drop_duplicates(subset=["order_id", "product_id"])

    # Reorder columns to be nicer for Tableau
    cols_order = [
        "order_id", "order_date", "order_year", "order_quarter",
        "order_month", "order_month_name",
        "region", "country", "customer_id",
        "product_id", "category", "sub_category",
        "quantity", "unit_price", "discount",
        "gross_sales", "discount_amount", "net_sales",
        "profit", "margin_pct", "source_file"
    ]

    # Ensure we only pick existing columns
    cols_order = [c for c in cols_order if c in df.columns]
    df = df[cols_order]

    print(f"[INFO] Cleaned data shape: {df.shape}")
    return df


def save_outputs(df: pd.DataFrame):
    """Save the cleaned DataFrame to CSV and Parquet for Tableau."""
    print(f"[INFO] Saving cleaned data to:\n  {OUTPUT_CSV}\n  {OUTPUT_PARQUET}")

    df.to_csv(OUTPUT_CSV, index=False)
    try:
        df.to_parquet(OUTPUT_PARQUET, index=False)
    except Exception as e:
        print(f"[WARN] Could not save Parquet (pyarrow issue?): {e}")


def main():
    start_time = datetime.now()
    print(f"[INFO] ETL job started at {start_time}")

    ensure_dirs()
    raw_df = load_raw_files()
    clean_df = clean_and_transform(raw_df)
    save_outputs(clean_df)

    end_time = datetime.now()
    print(f"[INFO] ETL job finished at {end_time}")
    print(f"[INFO] Duration: {end_time - start_time}")


if __name__ == "__main__":
    main()
