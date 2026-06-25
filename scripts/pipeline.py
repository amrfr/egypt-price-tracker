"""
scripts/pipeline.py
Loads raw price data, cleans it, runs validation checks,
and stores results in a local SQLite database.
"""

import pandas as pd
import numpy as np
import sqlite3
import os

RAW_DATA = os.path.join(os.path.dirname(__file__), "../data/prices.csv")
DB_PATH  = os.path.join(os.path.dirname(__file__), "../data/prices.db")


# ── 1. Load ──────────────────────────────────────────────────────────────────
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"[load]  {len(df)} rows loaded")
    return df


# ── 2. Clean ─────────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Drop fully empty rows
    df = df.dropna(how="all")

    # Strip whitespace from string columns
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    # Enforce expected dtypes
    df["price_egp"] = pd.to_numeric(df["price_egp"], errors="coerce")

    # Drop rows where price is missing or negative
    df = df[df["price_egp"].notna() & (df["price_egp"] > 0)]

    # Normalise category capitalisation
    df["category"] = df["category"].str.title()
    df["city"]     = df["city"].str.title()

    print(f"[clean] {before - len(df)} rows dropped  →  {len(df)} rows remaining")
    return df.reset_index(drop=True)


# ── 3. Validate ──────────────────────────────────────────────────────────────
def validate(df: pd.DataFrame) -> dict:
    report = {}

    # Null check
    null_counts = df.isnull().sum()
    report["nulls_per_column"] = null_counts[null_counts > 0].to_dict()

    # Outlier flag: price more than 3 std devs from item mean
    df["z_score"] = df.groupby("item")["price_egp"].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    outliers = df[df["z_score"].abs() > 3][["date", "item", "city", "price_egp"]]
    report["outliers"] = outliers.to_dict(orient="records")

    # Date range
    report["date_range"] = {
        "min": str(df["date"].min().date()),
        "max": str(df["date"].max().date()),
    }

    report["row_count"] = len(df)
    report["unique_items"] = df["item"].nunique()
    report["cities"] = df["city"].unique().tolist()

    print(f"[validate] {report['row_count']} rows | "
          f"{report['unique_items']} items | "
          f"outliers flagged: {len(outliers)}")
    return report


# ── 4. Store in SQLite ────────────────────────────────────────────────────────
def store(df: pd.DataFrame, db_path: str):
    df_store = df.drop(columns=["z_score"], errors="ignore")
    con = sqlite3.connect(db_path)
    df_store.to_sql("prices", con, if_exists="replace", index=False)

    # Create a helpful aggregation view
    con.execute("""
        CREATE VIEW IF NOT EXISTS monthly_avg AS
        SELECT
            strftime('%Y-%m', date)   AS month,
            category,
            city,
            item,
            ROUND(AVG(price_egp), 2)  AS avg_price,
            COUNT(*)                  AS observations
        FROM prices
        GROUP BY month, category, city, item
    """)
    con.commit()
    con.close()
    print(f"[store]  saved to {db_path}")


# ── 5. Sample SQL queries (run and print) ─────────────────────────────────────
def run_sample_queries(db_path: str):
    con = sqlite3.connect(db_path)

    print("\n── Top 5 most expensive items (latest month) ──")
    q1 = """
        SELECT item, city, price_egp
        FROM prices
        WHERE date = (SELECT MAX(date) FROM prices)
        ORDER BY price_egp DESC
        LIMIT 5
    """
    print(pd.read_sql(q1, con).to_string(index=False))

    print("\n── Average price per category (all time) ──")
    q2 = """
        SELECT category,
               ROUND(AVG(price_egp), 2) AS avg_price_egp
        FROM prices
        GROUP BY category
        ORDER BY avg_price_egp DESC
    """
    print(pd.read_sql(q2, con).to_string(index=False))

    print("\n── Month-on-month price change for Chicken (Cairo) ──")
    q3 = """
        SELECT strftime('%Y-%m', date) AS month,
               price_egp
        FROM prices
        WHERE item = 'Chicken (1kg)' AND city = 'Cairo'
        ORDER BY date
    """
    df_chicken = pd.read_sql(q3, con)
    df_chicken["change_%"] = df_chicken["price_egp"].pct_change().mul(100).round(2)
    print(df_chicken.to_string(index=False))

    print("\n── Cairo vs Alexandria avg grocery price ──")
    q4 = """
        SELECT city,
               ROUND(AVG(price_egp), 2) AS avg_grocery_price
        FROM prices
        WHERE category = 'Groceries'
        GROUP BY city
    """
    print(pd.read_sql(q4, con).to_string(index=False))

    con.close()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_raw(RAW_DATA)
    df = clean(df)
    report = validate(df)
    store(df, DB_PATH)
    run_sample_queries(DB_PATH)
    print("\n[done] Validation report summary:")
    for k, v in report.items():
        print(f"  {k}: {v}")
