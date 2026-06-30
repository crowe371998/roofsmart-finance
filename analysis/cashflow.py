"""Cash flow analysis, P&L reconstruction, and forecasting for Roof Smart."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

REVENUE_CATS = {"REVENUE"}
COGS_CATS = {"COGS"}
OPEX_CATS = {"OVERHEAD", "PAYROLL", "VEHICLES", "MARKETING", "EQUIPMENT"}
EXCLUDE_CATS = {"TRANSFERS"}


def _monthly_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add year-month column for grouping."""
    df = df.copy()
    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["_date"].dt.to_period("M").astype(str)
    return df


def monthly_cashflow(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly cash in, cash out, and net per account."""
    if df.empty:
        return pd.DataFrame()

    df = _monthly_group(df)
    df_ex = df[~df["category"].isin(EXCLUDE_CATS)]

    summary = df_ex.groupby("month").agg(
        cash_in=("amount", lambda x: x[x > 0].sum()),
        cash_out=("amount", lambda x: abs(x[x < 0].sum())),
    ).reset_index()

    summary["net"] = summary["cash_in"] - summary["cash_out"]
    summary["burn_rate"] = summary["net"].apply(lambda x: x if x < 0 else 0)
    return summary.sort_values("month")


def reconstruct_pl(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct approximate P&L from categorized transactions."""
    if df.empty:
        return pd.DataFrame()

    df = _monthly_group(df)

    months = sorted(df["month"].unique())
    rows = []

    for month in months:
        mdf = df[df["month"] == month]

        gross_revenue = mdf[mdf["category"].isin(REVENUE_CATS)]["amount"].sum()
        cogs = abs(mdf[mdf["category"].isin(COGS_CATS)]["amount"].sum())
        gross_profit = gross_revenue - cogs
        gross_margin = (gross_profit / gross_revenue * 100) if gross_revenue else 0

        opex = abs(mdf[mdf["category"].isin(OPEX_CATS)]["amount"].sum())
        taxes = abs(mdf[mdf["category"] == "TAXES"]["amount"].sum())

        ebitda = gross_profit - opex
        net_income = ebitda - taxes

        rows.append({
            "month": month,
            "gross_revenue": gross_revenue,
            "cogs": cogs,
            "gross_profit": gross_profit,
            "gross_margin_pct": round(gross_margin, 1),
            "operating_expenses": opex,
            "ebitda": ebitda,
            "taxes": taxes,
            "net_income": net_income,
        })

    return pd.DataFrame(rows)


def forecast_13_week(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a 13-week rolling cash flow forecast using trend extrapolation."""
    if df.empty:
        return pd.DataFrame()

    monthly = monthly_cashflow(df)
    if len(monthly) < 2:
        return pd.DataFrame()

    # Use last 3 months as trend basis
    recent = monthly.tail(3)
    avg_in = recent["cash_in"].mean()
    avg_out = recent["cash_out"].mean()

    # Simple trend: slope from last 3 months
    if len(recent) >= 3:
        in_slope = (recent["cash_in"].iloc[-1] - recent["cash_in"].iloc[0]) / max(len(recent) - 1, 1)
        out_slope = (recent["cash_out"].iloc[-1] - recent["cash_out"].iloc[0]) / max(len(recent) - 1, 1)
    else:
        in_slope = 0
        out_slope = 0

    # Generate weekly forecast (13 weeks)
    last_date_str = monthly["month"].iloc[-1]
    try:
        last_date = datetime.strptime(last_date_str, "%Y-%m")
    except ValueError:
        last_date = datetime.now()

    weeks = []
    weekly_in = avg_in / 4.33
    weekly_out = avg_out / 4.33
    weekly_in_slope = in_slope / 4.33
    weekly_out_slope = out_slope / 4.33

    for i in range(13):
        week_start = last_date + timedelta(weeks=i + 1)
        projected_in = max(0, weekly_in + weekly_in_slope * i)
        projected_out = max(0, weekly_out + weekly_out_slope * i)
        weeks.append({
            "week": i + 1,
            "week_start": week_start.strftime("%Y-%m-%d"),
            "projected_cash_in": round(projected_in, 2),
            "projected_cash_out": round(projected_out, 2),
            "projected_net": round(projected_in - projected_out, 2),
        })

    return pd.DataFrame(weeks)


def days_cash_on_hand(df: pd.DataFrame) -> Optional[float]:
    """Calculate days of cash on hand based on current balance and burn rate."""
    if df.empty:
        return None

    # Current cash = latest balance across all accounts
    df_valid = df[df["balance"] > 0].copy()
    if df_valid.empty:
        total_cash = 0.0
    else:
        # Latest balance per account
        df_valid["_date"] = pd.to_datetime(df_valid["date"], errors="coerce")
        latest = df_valid.sort_values("_date").groupby("account_last4")["balance"].last()
        total_cash = latest.sum()

    monthly = monthly_cashflow(df)
    if monthly.empty or monthly["cash_out"].mean() == 0:
        return None

    daily_burn = monthly["cash_out"].mean() / 30
    if daily_burn <= 0:
        return None

    return round(total_cash / daily_burn, 1)


def vendor_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Top 20 vendors by spend with MoM change."""
    if df.empty:
        return pd.DataFrame()

    expense_df = df[df["amount"] < 0].copy()
    if expense_df.empty:
        return pd.DataFrame()

    expense_df = _monthly_group(expense_df)
    months = sorted(expense_df["month"].unique())
    current_month = months[-1] if months else None
    prior_month = months[-2] if len(months) >= 2 else None

    current = expense_df[expense_df["month"] == current_month].groupby("description")["amount"].sum().abs()
    top20 = current.nlargest(20).reset_index()
    top20.columns = ["vendor", "current_spend"]

    if prior_month:
        prior = expense_df[expense_df["month"] == prior_month].groupby("description")["amount"].sum().abs()
        top20["prior_spend"] = top20["vendor"].map(prior).fillna(0)
        top20["change_pct"] = (
            (top20["current_spend"] - top20["prior_spend"]) / top20["prior_spend"].replace(0, np.nan) * 100
        ).round(1)
        top20["flag"] = top20["change_pct"] > 20
    else:
        top20["prior_spend"] = 0
        top20["change_pct"] = 0
        top20["flag"] = False

    return top20.sort_values("current_spend", ascending=False)


def department_health_scores(df: pd.DataFrame) -> dict[str, float]:
    """Score each department 1-10 based on spend efficiency."""
    if df.empty:
        return {}

    monthly = monthly_cashflow(df)
    pl = reconstruct_pl(df)

    scores: dict[str, float] = {}

    if pl.empty or monthly.empty:
        return {dept: 5.0 for dept in ["Materials & COGS", "Marketing", "Vehicles & Fleet", "Labor & Payroll", "Overhead & Admin"]}

    total_revenue = pl["gross_revenue"].sum()
    if total_revenue == 0:
        return {dept: 5.0 for dept in ["Materials & COGS", "Marketing", "Vehicles & Fleet", "Labor & Payroll", "Overhead & Admin"]}

    def cat_spend(cats):
        return abs(df[df["category"].isin(cats)]["amount"].sum())

    def score_from_ratio(ratio: float, target_low: float, target_high: float) -> float:
        """Score 1-10 where target range = 8-10, outside = lower."""
        if ratio <= target_low:
            return min(10.0, 10 - (target_low - ratio) / target_low * 5)
        elif ratio <= target_high:
            return 9.0
        else:
            return max(1.0, 10 - (ratio - target_high) / target_high * 10)

    cogs_ratio = cat_spend({"COGS"}) / total_revenue
    scores["Materials & COGS"] = round(score_from_ratio(cogs_ratio, 0.25, 0.45), 1)

    mkt_ratio = cat_spend({"MARKETING"}) / total_revenue
    scores["Marketing"] = round(score_from_ratio(mkt_ratio, 0.03, 0.08), 1)

    veh_ratio = cat_spend({"VEHICLES"}) / total_revenue
    scores["Vehicles & Fleet"] = round(score_from_ratio(veh_ratio, 0.03, 0.07), 1)

    pay_ratio = cat_spend({"PAYROLL"}) / total_revenue
    scores["Labor & Payroll"] = round(score_from_ratio(pay_ratio, 0.15, 0.35), 1)

    oh_ratio = cat_spend({"OVERHEAD"}) / total_revenue
    scores["Overhead & Admin"] = round(score_from_ratio(oh_ratio, 0.05, 0.12), 1)

    return scores


def job_costing(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to group transactions by job/project using description patterns."""
    if df.empty:
        return pd.DataFrame()

    import re
    job_pattern = re.compile(
        r"(?:job|project|proj|jb|prj)\s*#?\s*([A-Z0-9\-]+)|"
        r"([A-Z][a-z]+ [A-Z][a-z]+)\s+(?:roof|install|repair|replacement)",
        re.IGNORECASE
    )

    df = df.copy()
    df["job_id"] = df["description"].apply(
        lambda d: (m := job_pattern.search(str(d))) and (m.group(1) or m.group(2)) or None
    )

    job_df = df[df["job_id"].notna()].copy()
    if job_df.empty:
        return pd.DataFrame()

    summary = job_df.groupby("job_id").agg(
        total_revenue=("amount", lambda x: x[x > 0].sum()),
        total_cost=("amount", lambda x: abs(x[x < 0].sum())),
        transaction_count=("amount", "count"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    ).reset_index()

    summary["gross_profit"] = summary["total_revenue"] - summary["total_cost"]
    summary["margin_pct"] = (
        summary["gross_profit"] / summary["total_revenue"].replace(0, np.nan) * 100
    ).round(1)

    return summary.sort_values("total_revenue", ascending=False)
