"""
dashboard/app.py
Streamlit dashboard — price trends, category breakdown,
city comparison, and AI-generated weekly digest via Claude API.
"""

import os
import sqlite3
import json

import pandas as pd
import plotly.express as px
import streamlit as st
import requests

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH     = os.path.join(os.path.dirname(__file__), "../data/prices.db")
API_KEY     = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_URL  = "https://api.anthropic.com/v1/messages"
MODEL       = "claude-sonnet-4-6"

st.set_page_config(page_title="🇪🇬 Egypt Price Tracker", layout="wide")
st.title("🇪🇬 Egypt Cost-of-Living Tracker")
st.caption("Tracking retail price trends across Egyptian cities · Built with Python, SQLite & Streamlit")


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM prices", con, parse_dates=["date"])
    con.close()
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load data. Run `python scripts/pipeline.py` first.\n\n{e}")
    st.stop()


# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")
cities     = st.sidebar.multiselect("City", df["city"].unique().tolist(),
                                    default=df["city"].unique().tolist())
categories = st.sidebar.multiselect("Category", df["category"].unique().tolist(),
                                    default=df["category"].unique().tolist())

filtered = df[df["city"].isin(cities) & df["category"].isin(categories)]


# ── KPI row ───────────────────────────────────────────────────────────────────
latest_month  = filtered["date"].max()
prev_month    = filtered["date"].unique()
prev_month    = sorted(prev_month)[-2] if len(prev_month) > 1 else latest_month

latest_avg = filtered[filtered["date"] == latest_month]["price_egp"].mean()
prev_avg   = filtered[filtered["date"] == prev_month]["price_egp"].mean()
delta      = ((latest_avg - prev_avg) / prev_avg * 100) if prev_avg else 0

col1, col2, col3 = st.columns(3)
col1.metric("Avg Price (Latest Month)", f"EGP {latest_avg:.1f}", f"{delta:+.1f}% vs prev month")
col2.metric("Items Tracked", df["item"].nunique())
col3.metric("Date Range", f"{df['date'].min().strftime('%b %Y')} → {df['date'].max().strftime('%b %Y')}")

st.divider()


# ── Chart 1: Price trend over time ────────────────────────────────────────────
st.subheader("📈 Price Trends Over Time")
item_options = filtered["item"].unique().tolist()
selected_items = st.multiselect("Select items to compare", item_options,
                                default=item_options[:3])

trend_df = filtered[filtered["item"].isin(selected_items)]
fig1 = px.line(trend_df, x="date", y="price_egp", color="item",
               line_dash="city", markers=True,
               labels={"price_egp": "Price (EGP)", "date": "Month", "item": "Item"},
               title="Monthly Price per Item")
st.plotly_chart(fig1, use_container_width=True)


# ── Chart 2: Category breakdown ───────────────────────────────────────────────
st.subheader("🗂️ Average Price by Category")
cat_df = (filtered.groupby("category")["price_egp"]
          .mean().reset_index()
          .rename(columns={"price_egp": "avg_price_egp"}))
fig2 = px.bar(cat_df, x="category", y="avg_price_egp", color="category",
              labels={"avg_price_egp": "Avg Price (EGP)", "category": "Category"},
              title="All-Time Average Price by Category")
st.plotly_chart(fig2, use_container_width=True)


# ── Chart 3: City comparison ──────────────────────────────────────────────────
st.subheader("🏙️ Cairo vs Alexandria — Grocery Price Comparison")
city_df = (filtered[filtered["category"] == "Groceries"]
           .groupby(["date", "city"])["price_egp"]
           .mean().reset_index())
fig3 = px.line(city_df, x="date", y="price_egp", color="city", markers=True,
               labels={"price_egp": "Avg Grocery Price (EGP)", "date": "Month"},
               title="Average Grocery Price: Cairo vs Alexandria")
st.plotly_chart(fig3, use_container_width=True)


# ── Chart 4: Biggest movers ───────────────────────────────────────────────────
st.subheader("🚀 Biggest Price Movers (Jan 2024 → Latest)")
first_month = df["date"].min()
first_prices = df[df["date"] == first_month].groupby("item")["price_egp"].mean()
last_prices  = df[df["date"] == latest_month].groupby("item")["price_egp"].mean()
movers = ((last_prices - first_prices) / first_prices * 100).dropna().sort_values()
movers_df = movers.reset_index()
movers_df.columns = ["item", "change_pct"]
fig4 = px.bar(movers_df, x="change_pct", y="item", orientation="h",
              color="change_pct", color_continuous_scale="RdYlGn_r",
              labels={"change_pct": "% Change", "item": "Item"},
              title="% Price Change Since Jan 2024")
st.plotly_chart(fig4, use_container_width=True)

st.divider()


# ── AI Weekly Digest ──────────────────────────────────────────────────────────
st.subheader("🤖 AI-Generated Weekly Digest")
st.caption("Summarises recent price movements and flags anything unusual.")

if not API_KEY:
    st.warning("Set the `ANTHROPIC_API_KEY` environment variable to enable AI summaries.")
else:
    if st.button("Generate Digest"):
        # Build a compact summary to send to Claude
        recent = (df[df["date"] >= df["date"].max() - pd.DateOffset(months=2)]
                  .groupby(["item", "date"])["price_egp"]
                  .mean().unstack("date")
                  .round(2))
        summary_text = recent.to_string()

        prompt = f"""You are an economic analyst tracking retail prices in Egypt.
Here is the average price (in EGP) for tracked items over the last two months:

{summary_text}

Write a concise stakeholder digest (3–5 bullet points) covering:
1. Key price trends (what went up / down significantly)
2. Any unusual spikes or anomalies worth flagging
3. One practical implication for a household budget

Keep it factual, plain-English, and under 200 words."""

        with st.spinner("Generating digest..."):
            try:
                res = requests.post(
                    CLAUDE_URL,
                    headers={
                        "x-api-key": API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "max_tokens": 400,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=30,
                )
                res.raise_for_status()
                digest = res.json()["content"][0]["text"]
                st.success("Digest generated!")
                st.markdown(digest)
            except Exception as e:
                st.error(f"API error: {e}")

st.divider()
st.caption("Data collected manually from Egyptian retail markets · Project by Amr Elhamady")
