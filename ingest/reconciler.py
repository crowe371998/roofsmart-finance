"""Cross-account reconciliation engine for Roof Smart Finance."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    """Result of reconciling one or more accounts."""
    accounts: list[str] = field(default_factory=list)
    status: dict[str, str] = field(default_factory=dict)  # account -> status
    balance_discrepancies: list[dict] = field(default_factory=list)
    duplicate_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    intercompany_transfers: pd.DataFrame = field(default_factory=pd.DataFrame)
    uncleared_checks: pd.DataFrame = field(default_factory=pd.DataFrame)
    missing_periods: list[dict] = field(default_factory=list)
    net_position: float = 0.0
    summary: str = ""


def _detect_cross_account_dupes(df: pd.DataFrame, tolerance_days: int = 3) -> pd.DataFrame:
    """Find transactions that appear in multiple accounts (likely same payment)."""
    dupes = []
    df_sorted = df.sort_values(["amount", "date"]).reset_index(drop=True)

    seen = set()
    for i, row_i in df_sorted.iterrows():
        if i in seen:
            continue
        for j, row_j in df_sorted.iloc[i+1:].iterrows():
            if j in seen:
                break
            if abs(row_i["amount"]) != abs(row_j["amount"]):
                break
            if row_i["account_last4"] == row_j["account_last4"]:
                continue
            try:
                d1 = datetime.strptime(row_i["date"], "%Y-%m-%d")
                d2 = datetime.strptime(row_j["date"], "%Y-%m-%d")
                if abs((d1 - d2).days) <= tolerance_days:
                    dupes.append({
                        "date_1": row_i["date"],
                        "date_2": row_j["date"],
                        "description_1": row_i["description"],
                        "description_2": row_j["description"],
                        "amount": row_i["amount"],
                        "account_1": row_i["account_last4"],
                        "account_2": row_j["account_last4"],
                        "source_1": row_i["source_file"],
                        "source_2": row_j["source_file"],
                    })
                    seen.add(j)
            except (ValueError, TypeError):
                continue

    return pd.DataFrame(dupes) if dupes else pd.DataFrame()


def _detect_intercompany_transfers(df: pd.DataFrame) -> pd.DataFrame:
    """Match internal transfers between Roof Smart accounts."""
    transfers = df[df["category"] == "TRANSFERS"].copy()
    if transfers.empty:
        # Also check description keywords
        keywords = ["transfer", "xfer", "sweep", "move funds", "wire to own"]
        mask = df["description"].str.lower().str.contains("|".join(keywords), na=False)
        transfers = df[mask].copy()

    if transfers.empty:
        return pd.DataFrame()

    matched = []
    used = set()

    for i, row_i in transfers.iterrows():
        if i in used:
            continue
        for j, row_j in transfers.iterrows():
            if j <= i or j in used:
                continue
            # Matching transfer: opposite sign, same amount, within 2 days
            if abs(abs(row_i["amount"]) - abs(row_j["amount"])) < 0.01:
                if row_i["amount"] * row_j["amount"] < 0:  # opposite signs
                    try:
                        d1 = datetime.strptime(row_i["date"], "%Y-%m-%d")
                        d2 = datetime.strptime(row_j["date"], "%Y-%m-%d")
                        if abs((d1 - d2).days) <= 2:
                            matched.append({
                                "date_out": row_i["date"] if row_i["amount"] < 0 else row_j["date"],
                                "date_in": row_j["date"] if row_i["amount"] < 0 else row_i["date"],
                                "amount": abs(row_i["amount"]),
                                "from_account": row_i["account_last4"] if row_i["amount"] < 0 else row_j["account_last4"],
                                "to_account": row_j["account_last4"] if row_i["amount"] < 0 else row_i["account_last4"],
                                "description": row_i["description"],
                            })
                            used.add(i)
                            used.add(j)
                    except (ValueError, TypeError):
                        continue

    return pd.DataFrame(matched) if matched else pd.DataFrame()


def _detect_missing_periods(df: pd.DataFrame) -> list[dict]:
    """Detect gaps between statement coverage periods per account."""
    if df.empty or "date" not in df.columns:
        return []

    missing = []
    df_valid = df[df["date"].str.len() == 10].copy()

    for account in df_valid["account_last4"].unique():
        acct_df = df_valid[df_valid["account_last4"] == account].copy()
        acct_df["_date"] = pd.to_datetime(acct_df["date"], errors="coerce")
        acct_df = acct_df.dropna(subset=["_date"]).sort_values("_date")

        if len(acct_df) < 2:
            continue

        dates = acct_df["_date"].tolist()
        for i in range(1, len(dates)):
            gap_days = (dates[i] - dates[i-1]).days
            if gap_days > 45:  # More than 45 days gap
                missing.append({
                    "account_last4": account,
                    "gap_start": dates[i-1].strftime("%Y-%m-%d"),
                    "gap_end": dates[i].strftime("%Y-%m-%d"),
                    "gap_days": gap_days,
                    "message": f"Account ...{account}: {gap_days}-day gap in statements ({dates[i-1].strftime('%b %Y')} → {dates[i].strftime('%b %Y')})"
                })

    return missing


def _detect_uncleared_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Flag checks written but potentially not yet cashed."""
    check_mask = df["description"].str.lower().str.contains(
        r"\bcheck\b|\bchk\b|\bck\b|\b#\d{3,6}\b", na=False, regex=True
    )
    checks = df[check_mask & (df["amount"] < 0)].copy()

    if checks.empty:
        return pd.DataFrame()

    # Flag checks older than 60 days as potentially uncleared
    cutoff = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    old_checks = checks[checks["date"] < cutoff].copy()
    old_checks["flag"] = "Uncleared > 60 days"

    return old_checks[["date", "description", "amount", "account_last4", "flag"]] if not old_checks.empty else pd.DataFrame()


def _verify_balances(df: pd.DataFrame) -> list[dict]:
    """Verify running balances match transaction sums per account."""
    discrepancies = []
    df_valid = df[df["balance"] != 0].copy()

    if df_valid.empty:
        return []

    for account in df_valid["account_last4"].unique():
        acct_df = df_valid[df_valid["account_last4"] == account].sort_values("date")
        if len(acct_df) < 2:
            continue

        # Check if balance changes match transaction amounts
        acct_df["balance_diff"] = acct_df["balance"].diff()
        acct_df["expected_diff"] = acct_df["amount"].shift(1)

        mismatches = acct_df[
            (acct_df["balance_diff"].notna()) &
            (abs(acct_df["balance_diff"] - acct_df["expected_diff"]) > 1.0)
        ]

        if not mismatches.empty:
            discrepancies.append({
                "account": account,
                "mismatch_count": len(mismatches),
                "sample_date": mismatches.iloc[0]["date"],
                "message": f"Account ...{account}: {len(mismatches)} balance verification failures"
            })

    return discrepancies


def reconcile(df: pd.DataFrame) -> ReconciliationResult:
    """Run full reconciliation on all transactions."""
    result = ReconciliationResult()

    if df.empty:
        result.summary = "No transactions to reconcile."
        return result

    result.accounts = df["account_last4"].unique().tolist()

    console.print("\n[bold cyan]Running Reconciliation Engine...[/bold cyan]")

    # 1. Cross-account duplicate detection
    console.print("  Checking for cross-account duplicates...")
    result.duplicate_candidates = _detect_cross_account_dupes(df)
    n_dupes = len(result.duplicate_candidates)
    console.print(f"  Found [yellow]{n_dupes}[/yellow] potential duplicate transactions")

    # 2. Balance verification
    console.print("  Verifying running balances...")
    result.balance_discrepancies = _verify_balances(df)
    n_disc = len(result.balance_discrepancies)
    console.print(f"  Found [yellow]{n_disc}[/yellow] balance discrepancies")

    # 3. Missing period detection
    console.print("  Checking for statement gaps...")
    result.missing_periods = _detect_missing_periods(df)
    n_gaps = len(result.missing_periods)
    console.print(f"  Found [yellow]{n_gaps}[/yellow] period gaps")

    # 4. Intercompany transfers
    console.print("  Matching intercompany transfers...")
    result.intercompany_transfers = _detect_intercompany_transfers(df)
    n_transfers = len(result.intercompany_transfers)
    console.print(f"  Matched [green]{n_transfers}[/green] intercompany transfer pairs")

    # 5. Uncleared checks
    console.print("  Flagging potentially uncleared checks...")
    result.uncleared_checks = _detect_uncleared_checks(df)
    n_checks = len(result.uncleared_checks)
    console.print(f"  Found [yellow]{n_checks}[/yellow] potentially uncleared checks")

    # 6. Net position (excluding transfers)
    non_transfers = df[df["category"] != "TRANSFERS"]
    result.net_position = non_transfers["amount"].sum()

    # 7. Status per account
    for account in result.accounts:
        has_disc = any(d["account"] == account for d in result.balance_discrepancies)
        has_gap = any(g["account_last4"] == account for g in result.missing_periods)
        if has_disc or has_gap:
            result.status[account] = "⚠️ Discrepancy"
        else:
            result.status[account] = "✅ Reconciled"

    result.summary = (
        f"Reconciled {len(result.accounts)} accounts | "
        f"{n_dupes} potential duplicates | "
        f"{n_disc} balance discrepancies | "
        f"{n_gaps} period gaps | "
        f"Net position: ${result.net_position:,.2f}"
    )

    console.print(f"\n[bold green]Reconciliation complete:[/bold green] {result.summary}")
    return result
