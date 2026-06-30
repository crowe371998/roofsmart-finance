"""Roof Smart Finance — Streamlit Dashboard."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent))

from analysis.alerts import generate_alerts
from analysis.cashflow import (
    department_health_scores,
    forecast_13_week,
    job_costing,
    monthly_cashflow,
    reconstruct_pl,
    vendor_analysis,
)
from analysis.reports import (
    generate_audit_excel,
    generate_category_summary_excel,
    generate_executive_pdf,
)
from ingest.categorizer import categorize_transactions
from ingest.parser import parse_file, parse_all_statements
from ingest.reconciler import reconcile

STATEMENTS_DIR = Path("data/statements")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("data/reports")
TRANSACTIONS_CSV = PROCESSED_DIR / "all_transactions.csv"

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Roof Smart Finance",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    .metric-card {
        background: linear-gradient(135deg, #16213e, #0f3460);
        border-radius: 10px; padding: 20px; margin: 5px;
        border-left: 4px solid #e94560;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #ffffff; }
    .metric-label { font-size: 13px; color: #a0aec0; margin-bottom: 4px; }
    .alert-critical { background-color: #2d1b1b; border-left: 4px solid #e53e3e; padding: 10px; border-radius: 5px; margin: 5px 0; }
    .alert-warning  { background-color: #2d2b1b; border-left: 4px solid #d69e2e; padding: 10px; border-radius: 5px; margin: 5px 0; }
    .alert-info     { background-color: #1b2d1b; border-left: 4px solid #38a169; padding: 10px; border-radius: 5px; margin: 5px 0; }
    .score-card {
        background: #16213e; border-radius: 8px; padding: 15px;
        text-align: center; margin: 5px;
    }
    .score-value { font-size: 36px; font-weight: bold; }
    .score-label { font-size: 12px; color: #a0aec0; }
    h1, h2, h3 { color: #ffffff !important; }
    .stDataFrame { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── State ────────────────────────────────────────────────────────────────────
def load_transactions() -> pd.DataFrame:
    """Load processed transactions from disk."""
    if TRANSACTIONS_CSV.exists():
        try:
            df = pd.read_csv(TRANSACTIONS_CSV, dtype=str)
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
            df["balance"] = pd.to_numeric(df["balance"], errors="coerce").fillna(0)
            df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)
            return df
        except Exception as e:
            st.error(f"Error loading transactions: {e}")
    return pd.DataFrame()


def save_transactions(df: pd.DataFrame):
    """Save transactions DataFrame to disk."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRANSACTIONS_CSV, index=False)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/0f3460/ffffff?text=ROOF+SMART", use_column_width=True)
    st.markdown("---")

    # API Key input — stored in session so no PowerShell needed
    with st.expander("🔑 Anthropic API Key", expanded=not bool(os.environ.get("ANTHROPIC_API_KEY"))):
        api_key_input = st.text_input(
            "API Key",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            type="password",
            placeholder="sk-ant-...",
            help="Required for AI categorization. Get yours at console.anthropic.com",
        )
        if api_key_input:
            os.environ["ANTHROPIC_API_KEY"] = api_key_input
            st.success("API key set for this session")

    st.markdown("---")

    uploaded_files = st.file_uploader(
        "📂 Upload Bank Statements",
        accept_multiple_files=True,
        type=["csv", "xlsx", "xls", "pdf", "ofx", "qbo", "qfx", "jpg", "jpeg", "png"],
        help="Drag & drop your bank/card statements here",
    )

    if uploaded_files:
        STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
        for uf in uploaded_files:
            dest = STATEMENTS_DIR / uf.name
            dest.write_bytes(uf.getvalue())
        st.success(f"Saved {len(uploaded_files)} file(s) to data/statements/")

    if st.button("⚡ Process All Statements", type="primary", use_container_width=True):
        with st.spinner("Parsing statements..."):
            df = parse_all_statements(STATEMENTS_DIR, PROCESSED_DIR)
        if not df.empty:
            with st.spinner("Categorizing with AI..."):
                df = categorize_transactions(df)
            save_transactions(df)
            st.success(f"✅ Processed {len(df)} transactions!")
            st.rerun()
        else:
            st.warning("No transactions found. Add files to data/statements/")

    st.markdown("---")
    df_global = load_transactions()
    if not df_global.empty:
        last_mod = TRANSACTIONS_CSV.stat().st_mtime
        last_mod_str = datetime.fromtimestamp(last_mod).strftime("%Y-%m-%d %H:%M")
        st.caption(f"Last processed: {last_mod_str}")
        st.caption(f"Transactions loaded: {len(df_global):,}")
    else:
        st.caption("No transactions loaded yet")

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "Transactions", "Reconciliation", "P&L & Cash Flow", "Business Health", "Reports"],
        label_visibility="collapsed",
    )


# ── Load Data ─────────────────────────────────────────────────────────────────
df = load_transactions()


def metric_card(label: str, value: str, delta: str = ""):
    """Render an HTML metric card."""
    delta_html = f"<div style='color:#68d391;font-size:12px'>{delta}</div>" if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def score_color(score: float) -> str:
    if score >= 8:
        return "#68d391"
    elif score >= 6:
        return "#f6e05e"
    else:
        return "#fc8181"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("🏠 Roof Smart — Financial Overview")

    if df.empty:
        st.info("👆 Upload bank statements using the sidebar to get started.")
        st.stop()

    # KPIs
    total_cash = df[df["balance"] > 0].groupby("account_last4")["balance"].last().sum()
    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    current_month = df["_date"].max().to_period("M").to_timestamp() if not df["_date"].isna().all() else None

    if current_month:
        month_df = df[df["_date"].dt.to_period("M") == df["_date"].max().to_period("M")]
        monthly_rev = month_df[month_df["category"] == "REVENUE"]["amount"].sum()
        monthly_exp = abs(month_df[month_df["amount"] < 0]["amount"].sum())
        net_income = monthly_rev - monthly_exp
        gross_margin = (net_income / monthly_rev * 100) if monthly_rev > 0 else 0
    else:
        monthly_rev = monthly_exp = net_income = gross_margin = 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Total Cash Position", f"${total_cash:,.0f}")
    with col2:
        metric_card("Monthly Revenue", f"${monthly_rev:,.0f}")
    with col3:
        metric_card("Monthly Expenses", f"${monthly_exp:,.0f}")
    with col4:
        color = "#68d391" if net_income >= 0 else "#fc8181"
        metric_card("Net Income", f"${net_income:,.0f}")
    with col5:
        metric_card("Gross Margin", f"{gross_margin:.1f}%")

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Cash balance over time
        st.subheader("Cash Balance Over Time")
        bal_df = df[df["balance"] > 0].copy()
        if not bal_df.empty:
            bal_df["date_dt"] = pd.to_datetime(bal_df["date"], errors="coerce")
            fig = px.line(
                bal_df.sort_values("date_dt"),
                x="date_dt", y="balance", color="account_last4",
                template="plotly_dark", labels={"date_dt": "Date", "balance": "Balance ($)", "account_last4": "Account"},
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,33,62,0.8)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No balance data available (not all statement formats include running balances).")

        # Revenue vs Expenses by month
        st.subheader("Revenue vs Expenses by Month")
        monthly = monthly_cashflow(df)
        if not monthly.empty:
            fig2 = go.Figure()
            fig2.add_bar(x=monthly["month"], y=monthly["cash_in"], name="Cash In", marker_color="#68d391")
            fig2.add_bar(x=monthly["month"], y=monthly["cash_out"], name="Cash Out", marker_color="#fc8181")
            fig2.update_layout(
                barmode="group", template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,33,62,0.8)",
                xaxis_title="Month", yaxis_title="Amount ($)",
            )
            st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        # Alerts panel
        st.subheader("Active Alerts")
        alerts = generate_alerts(df)
        for alert in alerts[:10]:
            level = alert.level if hasattr(alert, "level") else alert.get("level", "INFO")
            icon = alert.icon if hasattr(alert, "icon") else alert.get("icon", "")
            title = alert.title if hasattr(alert, "title") else alert.get("title", "")
            detail = alert.detail if hasattr(alert, "detail") else alert.get("detail", "")
            css_class = f"alert-{level.lower()}"
            st.markdown(f"""
            <div class="{css_class}">
                <b>{icon} {title}</b><br>
                <small>{detail}</small>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: TRANSACTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Transactions":
    st.title("📋 Transaction Ledger")

    if df.empty:
        st.info("No transactions loaded. Upload statements in the sidebar.")
        st.stop()

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        accounts = ["All"] + sorted(df["account_last4"].dropna().unique().tolist())
        acct_filter = st.selectbox("Account", accounts)
    with col2:
        cats = ["All"] + sorted(df["category"].dropna().unique().tolist())
        cat_filter = st.selectbox("Category", cats)
    with col3:
        search = st.text_input("Search description", "")
    with col4:
        show_flagged = st.checkbox("Only flagged (low confidence)", False)

    col5, col6 = st.columns(2)
    with col5:
        date_from = st.date_input("From date", value=None)
    with col6:
        date_to = st.date_input("To date", value=None)

    fdf = df.copy()
    if acct_filter != "All":
        fdf = fdf[fdf["account_last4"] == acct_filter]
    if cat_filter != "All":
        fdf = fdf[fdf["category"] == cat_filter]
    if search:
        fdf = fdf[fdf["description"].str.contains(search, case=False, na=False)]
    if show_flagged:
        fdf = fdf[fdf["confidence"] < 0.7]
    if date_from:
        fdf = fdf[fdf["date"] >= date_from.strftime("%Y-%m-%d")]
    if date_to:
        fdf = fdf[fdf["date"] <= date_to.strftime("%Y-%m-%d")]

    st.caption(f"Showing {len(fdf):,} of {len(df):,} transactions")

    # Inline category editor
    display_cols = ["date", "description", "amount", "type", "category", "subcategory", "confidence", "account_last4", "source_file"]
    edit_df = st.data_editor(
        fdf[display_cols].reset_index(drop=True),
        column_config={
            "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
            "category": st.column_config.SelectboxColumn("Category", options=list(
                ["REVENUE", "COGS", "OVERHEAD", "PAYROLL", "VEHICLES", "MARKETING", "EQUIPMENT", "TAXES", "TRANSFERS", "UNKNOWN"]
            )),
        },
        use_container_width=True,
        num_rows="fixed",
        height=500,
    )

    col_save, col_export = st.columns(2)
    with col_save:
        if st.button("💾 Save Category Changes"):
            # Merge edits back into main df
            for col in ["category", "subcategory"]:
                if col in edit_df.columns:
                    fdf_idx = fdf.index.tolist()
                    for i, idx in enumerate(fdf_idx):
                        if i < len(edit_df):
                            df.at[idx, col] = edit_df.at[i, col]
            save_transactions(df)
            st.success("Changes saved!")

    with col_export:
        if st.button("📥 Export to Excel"):
            out = REPORTS_DIR / f"filtered_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            fdf.to_excel(str(out), index=False)
            with open(out, "rb") as f:
                st.download_button("⬇️ Download Excel", f, file_name=out.name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: RECONCILIATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Reconciliation":
    st.title("🔍 Reconciliation Status")

    if df.empty:
        st.info("No transactions loaded.")
        st.stop()

    if st.button("Run Reconciliation", type="primary"):
        with st.spinner("Reconciling accounts..."):
            result = reconcile(df)
        st.session_state["recon_result"] = result

    result = st.session_state.get("recon_result")
    if result is None:
        st.info("Click 'Run Reconciliation' to analyze your accounts.")
        st.stop()

    # Account status
    st.subheader("Account Status")
    for account, status in result.status.items():
        color = "green" if "✅" in status else "orange"
        st.markdown(f"**Account ...{account}**: :{color}[{status}]")

    st.markdown(f"**Summary:** {result.summary}")
    st.markdown(f"**Net Position (excl. transfers):** ${result.net_position:,.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Duplicate candidates
        st.subheader("⚠️ Potential Duplicate Transactions")
        if not result.duplicate_candidates.empty:
            st.dataframe(result.duplicate_candidates, use_container_width=True)
        else:
            st.success("No duplicate transactions detected.")

        # Missing periods
        st.subheader("📅 Statement Gaps")
        if result.missing_periods:
            for gap in result.missing_periods:
                st.warning(gap["message"])
        else:
            st.success("No statement period gaps detected.")

    with col2:
        # Intercompany transfers
        st.subheader("🔄 Matched Intercompany Transfers")
        if not result.intercompany_transfers.empty:
            st.dataframe(result.intercompany_transfers, use_container_width=True)
        else:
            st.info("No intercompany transfers detected.")

        # Uncleared checks
        st.subheader("🖊️ Potentially Uncleared Checks")
        if not result.uncleared_checks.empty:
            st.dataframe(result.uncleared_checks, use_container_width=True)
        else:
            st.success("No uncleared checks flagged.")

    if result.balance_discrepancies:
        st.subheader("❌ Balance Discrepancies")
        for disc in result.balance_discrepancies:
            st.error(disc["message"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: P&L & CASH FLOW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "P&L & Cash Flow":
    st.title("📊 P&L & Cash Flow Analysis")

    if df.empty:
        st.info("No transactions loaded.")
        st.stop()

    pl = reconstruct_pl(df)
    monthly = monthly_cashflow(df)
    forecast = forecast_13_week(df)

    # P&L Table
    st.subheader("Reconstructed P&L (Monthly)")
    if not pl.empty:
        display_pl = pl.copy()
        for col in ["gross_revenue", "cogs", "gross_profit", "operating_expenses", "ebitda", "taxes", "net_income"]:
            if col in display_pl.columns:
                display_pl[col] = display_pl[col].apply(lambda x: f"${x:,.0f}")
        st.dataframe(display_pl, use_container_width=True)
    else:
        st.info("Not enough categorized data for P&L reconstruction.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Cash flow waterfall
        st.subheader("Cash Flow by Month")
        if not monthly.empty:
            colors_list = ["#68d391" if v >= 0 else "#fc8181" for v in monthly["net"]]
            fig = go.Figure(go.Bar(
                x=monthly["month"],
                y=monthly["net"],
                marker_color=colors_list,
                name="Net Cash Flow",
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(22,33,62,0.8)",
                xaxis_title="Month",
                yaxis_title="Net Cash ($)",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 13-week forecast
        st.subheader("13-Week Cash Flow Forecast")
        if not forecast.empty:
            fig2 = go.Figure()
            fig2.add_scatter(x=forecast["week_start"], y=forecast["projected_cash_in"],
                             mode="lines+markers", name="Projected In", line=dict(color="#68d391"))
            fig2.add_scatter(x=forecast["week_start"], y=forecast["projected_cash_out"],
                             mode="lines+markers", name="Projected Out", line=dict(color="#fc8181"))
            fig2.add_scatter(x=forecast["week_start"], y=forecast["projected_net"],
                             mode="lines", name="Net", line=dict(color="#63b3ed", dash="dash"))
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(22,33,62,0.8)",
                xaxis_title="Week Start",
                yaxis_title="Amount ($)",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Need at least 2 months of data for forecasting.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: BUSINESS HEALTH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Business Health":
    st.title("💼 Business Health Dashboard")

    if df.empty:
        st.info("No transactions loaded.")
        st.stop()

    scores = department_health_scores(df)
    vendors = vendor_analysis(df)

    # Department Health Score Cards
    st.subheader("Department Health Scores")
    if scores:
        cols = st.columns(len(scores))
        for col, (dept, score) in zip(cols, scores.items()):
            color = score_color(score)
            with col:
                st.markdown(f"""
                <div class="score-card">
                    <div class="score-value" style="color:{color}">{score}</div>
                    <div style="color:{color}; font-size:10px">/ 10</div>
                    <div class="score-label">{dept}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Category breakdown pie
        st.subheader("Spending by Category")
        cat_totals = df[df["amount"] < 0].groupby("category")["amount"].sum().abs()
        if not cat_totals.empty:
            fig = px.pie(
                values=cat_totals.values,
                names=cat_totals.index,
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Vendor analysis
        st.subheader("Top 20 Vendors by Spend")
        if not vendors.empty:
            disp_vendors = vendors.copy()
            disp_vendors["current_spend"] = disp_vendors["current_spend"].apply(lambda x: f"${x:,.0f}")
            disp_vendors["prior_spend"] = disp_vendors["prior_spend"].apply(lambda x: f"${x:,.0f}")
            disp_vendors["change_pct"] = disp_vendors["change_pct"].apply(
                lambda x: f"🔴 +{x:.0f}%" if x > 20 else (f"+{x:.0f}%" if x > 0 else f"{x:.0f}%")
            )
            st.dataframe(disp_vendors[["vendor", "current_spend", "prior_spend", "change_pct"]],
                         use_container_width=True, height=400)
        else:
            st.info("No vendor data available.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Reports":
    st.title("📄 Generate Reports")

    if df.empty:
        st.info("No transactions loaded.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        date_from_r = st.date_input("Report From Date", value=None, key="rep_from")
    with col2:
        date_to_r = st.date_input("Report To Date", value=None, key="rep_to")

    date_from_str = date_from_r.strftime("%Y-%m-%d") if date_from_r else None
    date_to_str = date_to_r.strftime("%Y-%m-%d") if date_to_r else None

    # Filter df for reports
    rdf = df.copy()
    if date_from_str:
        rdf = rdf[rdf["date"] >= date_from_str]
    if date_to_str:
        rdf = rdf[rdf["date"] <= date_to_str]

    pl = reconstruct_pl(rdf)
    alerts = generate_alerts(rdf)
    scores = department_health_scores(rdf)
    alert_dicts = [a.to_dict() if hasattr(a, "to_dict") else a for a in alerts]

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📋 Executive Summary PDF")
        st.caption("Includes: Financial snapshot, P&L, AI insights, health scores, alerts, recommendations")
        if st.button("Generate Executive PDF", use_container_width=True):
            with st.spinner("Generating PDF (calling Claude for insights)..."):
                path = generate_executive_pdf(rdf, pl, alert_dicts, scores, date_from_str, date_to_str)
            if path.exists():
                with open(path, "rb") as f:
                    st.download_button("⬇️ Download PDF", f, file_name=path.name, mime="application/pdf")

    with col2:
        st.subheader("📊 Full Audit Excel")
        st.caption("Includes: All transactions, by-category, by-account, P&L, reconciliation, flagged items")
        if st.button("Generate Audit Excel", use_container_width=True):
            recon = st.session_state.get("recon_result")
            with st.spinner("Building Excel workbook..."):
                path = generate_audit_excel(rdf, pl, recon, date_from_str, date_to_str)
            if path.exists():
                with open(path, "rb") as f:
                    st.download_button("⬇️ Download Excel", f, file_name=path.name,
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with col3:
        st.subheader("📈 Category Summary Excel")
        st.caption("Category-by-month pivot table for quick spend review")
        if st.button("Generate Category Excel", use_container_width=True):
            with st.spinner("Building category summary..."):
                path = generate_category_summary_excel(rdf)
            if path.exists():
                with open(path, "rb") as f:
                    st.download_button("⬇️ Download Excel", f, file_name=path.name,
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
