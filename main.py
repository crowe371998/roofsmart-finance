"""Roof Smart Finance — CLI entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()

STATEMENTS_DIR = Path("data/statements")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("data/reports")
TRANSACTIONS_CSV = PROCESSED_DIR / "all_transactions.csv"


def _load_df():
    """Load processed transactions DataFrame."""
    import pandas as pd
    if not TRANSACTIONS_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(TRANSACTIONS_CSV, dtype=str)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
        df["balance"] = pd.to_numeric(df["balance"], errors="coerce").fillna(0)
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)
        return df
    except Exception as exc:
        console.print(f"[red]Error loading transactions: {exc}[/red]")
        return pd.DataFrame()


@click.group()
def cli():
    """Roof Smart Bank Reconciliation & Financial Intelligence System."""
    pass


@cli.command()
@click.option("--file", "file_path", default=None, help="Process a single file instead of the whole statements directory")
def ingest(file_path: str | None):
    """Process all files in data/statements/ (or a single file)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from ingest.parser import parse_file, parse_all_statements
    from ingest.categorizer import categorize_transactions

    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if file_path:
        p = Path(file_path)
        if not p.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
            return
        console.print(f"Parsing [cyan]{p.name}[/cyan]...")
        df = parse_file(p)
        if df.empty:
            console.print("[yellow]No transactions extracted.[/yellow]")
            return
        console.print(f"[green]✓[/green] {len(df)} transactions parsed")
    else:
        df = parse_all_statements(STATEMENTS_DIR, PROCESSED_DIR)

    if df.empty:
        console.print("[yellow]No transactions to process.[/yellow]")
        return

    console.print(f"\nCategorizing {len(df)} transactions...")
    df = categorize_transactions(df)

    df.to_csv(TRANSACTIONS_CSV, index=False)
    console.print(f"[green]✅ Saved {len(df)} transactions to {TRANSACTIONS_CSV}[/green]")


@cli.command()
def report():
    """Generate all reports (PDF + Excel)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from analysis.reports import generate_executive_pdf, generate_audit_excel, generate_category_summary_excel
    from analysis.cashflow import reconstruct_pl, department_health_scores
    from analysis.alerts import generate_alerts
    from ingest.reconciler import reconcile

    df = _load_df()
    if df.empty:
        console.print("[yellow]No transactions loaded. Run `python main.py ingest` first.[/yellow]")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    console.print("Generating reports...")

    pl = reconstruct_pl(df)
    alerts = generate_alerts(df)
    scores = department_health_scores(df)
    alert_dicts = [a.to_dict() if hasattr(a, "to_dict") else a for a in alerts]
    recon = reconcile(df)

    pdf_path = generate_executive_pdf(df, pl, alert_dicts, scores)
    audit_path = generate_audit_excel(df, pl, recon)
    cat_path = generate_category_summary_excel(df)

    console.print(f"\n[green]Reports generated:[/green]")
    console.print(f"  PDF: {pdf_path}")
    console.print(f"  Audit Excel: {audit_path}")
    console.print(f"  Category Excel: {cat_path}")


@cli.command()
def dashboard():
    """Launch the Streamlit web dashboard."""
    console.print("[cyan]Launching Roof Smart Finance Dashboard...[/cyan]")
    console.print("[green]Open your browser to: http://localhost:8501[/green]")
    app_path = Path(__file__).parent / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", "8501"])


@cli.command()
def alerts():
    """Print current alerts to the terminal."""
    sys.path.insert(0, str(Path(__file__).parent))
    from analysis.alerts import generate_alerts

    df = _load_df()
    if df.empty:
        console.print("[yellow]No transactions loaded.[/yellow]")
        return

    alert_list = generate_alerts(df)
    console.print("\n[bold]Active Alerts[/bold]\n")
    for alert in alert_list:
        level_color = {"CRITICAL": "red", "WARNING": "yellow", "INFO": "green"}.get(alert.level, "white")
        console.print(f"  [{level_color}]{alert.icon} [{alert.level}] {alert.title}[/{level_color}]")
        console.print(f"    {alert.detail}")
        console.print()


@cli.command()
def status():
    """Show account balances and last update timestamp."""
    sys.path.insert(0, str(Path(__file__).parent))
    from datetime import datetime

    console.print("\n[bold cyan]Roof Smart Finance — System Status[/bold cyan]\n")

    # Check statements directory
    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    stmt_files = list(STATEMENTS_DIR.iterdir()) if STATEMENTS_DIR.exists() else []
    console.print(f"Statement files in data/statements/: [cyan]{len(stmt_files)}[/cyan]")

    df = _load_df()
    if df.empty:
        console.print("[yellow]No processed transactions found.[/yellow]")
        console.print("\nRun [bold]python main.py ingest[/bold] to process your statements.")
        return

    # Last update
    if TRANSACTIONS_CSV.exists():
        import os
        mtime = os.path.getmtime(TRANSACTIONS_CSV)
        last_update = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"Last processed: [green]{last_update}[/green]")

    console.print(f"Total transactions: [green]{len(df):,}[/green]")

    # Account balances
    table = Table(title="\nAccount Summary", style="cyan")
    table.add_column("Account", style="white")
    table.add_column("Transactions", justify="right")
    table.add_column("Latest Balance", justify="right", style="green")
    table.add_column("Net Amount", justify="right")

    for account in df["account_last4"].unique():
        acct_df = df[df["account_last4"] == account]
        count = len(acct_df)
        bal_df = acct_df[acct_df["balance"] > 0]
        latest_bal = bal_df["balance"].iloc[-1] if not bal_df.empty else 0
        net = acct_df["amount"].sum()
        net_color = "green" if net >= 0 else "red"
        table.add_row(
            f"...{account}",
            str(count),
            f"${latest_bal:,.2f}",
            f"[{net_color}]${net:,.2f}[/{net_color}]",
        )

    console.print(table)

    # Category breakdown
    cat_table = Table(title="\nCategory Breakdown", style="cyan")
    cat_table.add_column("Category", style="white")
    cat_table.add_column("Transactions", justify="right")
    cat_table.add_column("Total Amount", justify="right")

    for cat, group in df.groupby("category"):
        total = group["amount"].sum()
        color = "green" if total >= 0 else "red"
        cat_table.add_row(
            str(cat),
            str(len(group)),
            f"[{color}]${total:,.2f}[/{color}]",
        )

    console.print(cat_table)
    console.print(f"\n[dim]Run [bold]python main.py dashboard[/bold] to open the web UI[/dim]\n")


if __name__ == "__main__":
    cli()
