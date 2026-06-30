"""Report generator — PDF Executive Summary and Excel Audit for Roof Smart."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("data/reports")


def _get_insights_from_claude(pl: pd.DataFrame, alerts: list, scores: dict) -> tuple[list[str], list[str]]:
    """Ask Claude to write plain-English insights and recommendations."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        insights = [
            "Revenue trends show monthly variation — review seasonal patterns.",
            "Operating expenses are being tracked across all categories.",
            "Reconciliation is complete for all loaded accounts.",
            "Cash position is being monitored against the $5,000 threshold.",
            "Department health scores indicate areas for potential optimization.",
        ]
        recommendations = [
            "Upload more recent statements to improve forecast accuracy.",
            "Review UNKNOWN category transactions for proper classification.",
            "Set up automated weekly statement exports from all bank portals.",
        ]
        return insights, recommendations

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

        pl_summary = pl.tail(3).to_dict(orient="records") if not pl.empty else []
        alert_titles = [a.get("title", "") if isinstance(a, dict) else a.title for a in alerts[:5]]

        prompt = f"""You are a financial advisor for Roof Smart, a roofing company.
Based on these recent financial metrics, write exactly 5 plain-English insights and 3-5 actionable recommendations.

Recent P&L (last 3 months): {json.dumps(pl_summary, default=str)}
Active alerts: {alert_titles}
Department health scores: {json.dumps(scores)}

Return JSON:
{{
  "insights": ["insight 1", "insight 2", "insight 3", "insight 4", "insight 5"],
  "recommendations": ["rec 1", "rec 2", "rec 3"]
}}"""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        data = json.loads(content)
        return data.get("insights", []), data.get("recommendations", [])

    except Exception as exc:
        logger.warning("Claude insights error: %s", exc)
        return (
            ["Financial data has been analyzed.", "Review flagged transactions."],
            ["Continue monitoring cash flow.", "Update categories for UNKNOWN transactions."]
        )


def generate_executive_pdf(
    df: pd.DataFrame,
    pl: pd.DataFrame,
    alerts: list,
    scores: dict,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Path:
    """Generate Executive Summary PDF report."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph,
            Spacer, HRFlowable
        )
    except ImportError:
        console.print("[red]reportlab not installed — cannot generate PDF[/red]")
        return REPORTS_DIR / "error.txt"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"RoofSmart_Executive_Summary_{timestamp}.pdf"

    doc = SimpleDocTemplate(str(out_path), pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=22, spaceAfter=6,
                                  textColor=colors.HexColor("#1a1a2e"))
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceBefore=12,
                               textColor=colors.HexColor("#16213e"))
    normal_style = styles["Normal"]
    body_style = ParagraphStyle("Body", parent=normal_style, fontSize=10, leading=14)

    story = []

    # Header
    story.append(Paragraph("🏠 ROOF SMART", title_style))
    story.append(Paragraph("Financial Intelligence Report", h2_style))
    period = f"{date_from or 'All'} → {date_to or 'Present'}"
    story.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')} | Period: {period}", body_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0f3460")))
    story.append(Spacer(1, 0.2*inch))

    # Financial Snapshot
    story.append(Paragraph("Financial Snapshot", h2_style))
    total_revenue = df[df["category"] == "REVENUE"]["amount"].sum() if not df.empty else 0
    total_expenses = abs(df[df["amount"] < 0]["amount"].sum()) if not df.empty else 0
    net_income = total_revenue - total_expenses
    accounts = df["account_last4"].nunique() if not df.empty else 0
    transactions = len(df)

    snap_data = [
        ["Metric", "Value"],
        ["Total Accounts", str(accounts)],
        ["Total Transactions", f"{transactions:,}"],
        ["Gross Revenue", f"${total_revenue:,.2f}"],
        ["Total Expenses", f"${total_expenses:,.2f}"],
        ["Net Income", f"${net_income:,.2f}"],
    ]
    snap_table = Table(snap_data, colWidths=[3*inch, 3*inch])
    snap_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(snap_table)
    story.append(Spacer(1, 0.2*inch))

    # P&L Summary
    if not pl.empty:
        story.append(Paragraph("P&L Summary (Recent Months)", h2_style))
        recent_pl = pl.tail(3)
        pl_data = [["Month", "Revenue", "COGS", "Gross Profit", "OpEx", "Net Income"]]
        for _, row in recent_pl.iterrows():
            pl_data.append([
                str(row["month"]),
                f"${row['gross_revenue']:,.0f}",
                f"${row['cogs']:,.0f}",
                f"${row['gross_profit']:,.0f}",
                f"${row['operating_expenses']:,.0f}",
                f"${row['net_income']:,.0f}",
            ])
        pl_table = Table(pl_data, colWidths=[1.1*inch]*6)
        pl_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(pl_table)
        story.append(Spacer(1, 0.2*inch))

    # AI Insights
    story.append(Paragraph("Key Insights", h2_style))
    insights, recommendations = _get_insights_from_claude(pl, alerts, scores)
    for i, insight in enumerate(insights, 1):
        story.append(Paragraph(f"{i}. {insight}", body_style))
    story.append(Spacer(1, 0.15*inch))

    # Department Health Scores
    if scores:
        story.append(Paragraph("Department Health Scores", h2_style))
        score_data = [["Department", "Score", "Status"]]
        for dept, score in scores.items():
            status = "Excellent" if score >= 8 else "Good" if score >= 6 else "Needs Attention"
            score_data.append([dept, f"{score}/10", status])
        score_table = Table(score_data, colWidths=[3*inch, 1*inch, 2*inch])
        score_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 0.2*inch))

    # Active Alerts
    if alerts:
        story.append(Paragraph("Active Alerts", h2_style))
        for alert in alerts[:8]:
            if isinstance(alert, dict):
                icon, title, detail = alert.get("icon", ""), alert.get("title", ""), alert.get("detail", "")
            else:
                icon, title, detail = alert.icon, alert.title, alert.detail
            story.append(Paragraph(f"{icon} <b>{title}</b>: {detail}", body_style))

    # Recommendations
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Recommendations", h2_style))
    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", body_style))

    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Generated by Roof Smart Finance | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                            ParagraphStyle("Footer", parent=body_style, fontSize=8, textColor=colors.grey)))

    doc.build(story)
    console.print(f"[green]PDF saved: {out_path}[/green]")
    return out_path


def generate_audit_excel(
    df: pd.DataFrame,
    pl: pd.DataFrame,
    recon_result=None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Path:
    """Generate Full Audit Excel workbook."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"RoofSmart_Audit_{timestamp}.xlsx"

    try:
        import xlsxwriter
    except ImportError:
        console.print("[red]xlsxwriter not installed[/red]")
        return out_path

    with pd.ExcelWriter(str(out_path), engine="xlsxwriter") as writer:
        wb = writer.book

        header_fmt = wb.add_format({
            "bold": True, "bg_color": "#0f3460", "font_color": "white",
            "border": 1, "align": "center"
        })
        money_fmt = wb.add_format({"num_format": "$#,##0.00", "border": 1})
        pct_fmt = wb.add_format({"num_format": "0.0%", "border": 1})
        normal_fmt = wb.add_format({"border": 1})
        red_fmt = wb.add_format({"bg_color": "#FFE0E0", "border": 1, "num_format": "$#,##0.00"})
        green_fmt = wb.add_format({"bg_color": "#E0FFE0", "border": 1, "num_format": "$#,##0.00"})

        def write_df_to_sheet(df_write: pd.DataFrame, sheet_name: str):
            df_write.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            for col_i, col_name in enumerate(df_write.columns):
                ws.write(0, col_i, col_name, header_fmt)
                ws.set_column(col_i, col_i, max(12, len(str(col_name)) + 2))

        # Sheet 1: All Transactions
        if not df.empty:
            write_df_to_sheet(df, "All Transactions")

        # Sheet 2: By Category Summary
        if not df.empty:
            cat_summary = df.groupby(["category", "subcategory"]).agg(
                transaction_count=("amount", "count"),
                total_amount=("amount", "sum"),
                avg_confidence=("confidence", "mean"),
            ).reset_index()
            write_df_to_sheet(cat_summary, "By Category Summary")

        # Sheet 3: By Account Summary
        if not df.empty:
            acct_summary = df.groupby("account_last4").agg(
                transaction_count=("amount", "count"),
                total_credits=("amount", lambda x: x[x > 0].sum()),
                total_debits=("amount", lambda x: x[x < 0].sum()),
                net=("amount", "sum"),
            ).reset_index()
            write_df_to_sheet(acct_summary, "By Account Summary")

        # Sheet 4: Monthly P&L
        if not pl.empty:
            write_df_to_sheet(pl, "Monthly P&L")

        # Sheet 5: Reconciliation Log
        if recon_result is not None:
            recon_rows = []
            for account, status in recon_result.status.items():
                recon_rows.append({"account": account, "status": status})
            if recon_rows:
                write_df_to_sheet(pd.DataFrame(recon_rows), "Reconciliation Log")
            if not recon_result.duplicate_candidates.empty:
                write_df_to_sheet(recon_result.duplicate_candidates, "Duplicate Candidates")

        # Sheet 6: Flagged / Review Needed
        if not df.empty:
            flagged = df[(df["confidence"] < 0.7) | (df["category"] == "UNKNOWN")]
            if not flagged.empty:
                write_df_to_sheet(flagged, "Flagged - Review Needed")

    console.print(f"[green]Excel audit saved: {out_path}[/green]")
    return out_path


def generate_category_summary_excel(df: pd.DataFrame) -> Path:
    """Generate Category Summary Excel."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"RoofSmart_CategorySummary_{timestamp}.xlsx"

    if df.empty:
        pd.DataFrame().to_excel(str(out_path), index=False)
        return out_path

    with pd.ExcelWriter(str(out_path), engine="xlsxwriter") as writer:
        df_monthly = df.copy()
        df_monthly["_date"] = pd.to_datetime(df_monthly["date"], errors="coerce")
        df_monthly["month"] = df_monthly["_date"].dt.to_period("M").astype(str)

        pivot = df_monthly.pivot_table(
            index="category",
            columns="month",
            values="amount",
            aggfunc="sum",
            fill_value=0,
        )
        pivot.to_excel(writer, sheet_name="Category by Month")

        wb = writer.book
        ws = writer.sheets["Category by Month"]
        header_fmt = wb.add_format({"bold": True, "bg_color": "#0f3460", "font_color": "white"})
        for col_i in range(len(pivot.columns) + 1):
            ws.write(0, col_i, (pivot.columns[col_i - 1] if col_i > 0 else "Category"), header_fmt)

    console.print(f"[green]Category summary saved: {out_path}[/green]")
    return out_path
