"""AI-powered transaction categorizer using Claude API."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.progress import track

console = Console()
logger = logging.getLogger(__name__)

# Use /tmp on cloud (read-only git checkout), local data/ otherwise
_TMP_CACHE = Path("/tmp/roofsmart/processed/categories_cache.json")
_LOCAL_CACHE = Path("data/processed/categories_cache.json")
CACHE_PATH = _TMP_CACHE if Path("/mount/src").exists() else _LOCAL_CACHE

CATEGORIES = {
    "REVENUE": ["Job Deposits", "Final Payments", "Insurance Checks", "Supplements"],
    "COGS": ["Materials", "Shingles", "Gutters", "Flashing", "Underlayment", "Subcontractor Labor", "Equipment Rental", "Permits & Inspections"],
    "OVERHEAD": ["Office Rent", "Utilities", "Insurance", "Software/Subscriptions"],
    "PAYROLL": ["Owner Draw", "W2 Employees", "1099 Contractors"],
    "VEHICLES": ["Fuel", "Maintenance", "Payments", "Registration"],
    "MARKETING": ["Google Ads", "Facebook Ads", "Door Hangers", "Yard Signs", "Lead Services"],
    "EQUIPMENT": ["Tools", "Ladders", "Safety Gear", "Machinery"],
    "TAXES": ["Estimated Taxes", "Sales Tax", "Payroll Tax"],
    "TRANSFERS": ["Internal Transfers"],
    "UNKNOWN": ["Needs Manual Review"],
}

CATEGORY_PROMPT = """You are a financial analyst for Roof Smart, a roofing company. Categorize each transaction into exactly one of these categories and subcategories:

REVENUE: Job Deposits, Final Payments, Insurance Checks, Supplements
COGS: Materials (Shingles/Gutters/Flashing/Underlayment), Subcontractor Labor, Equipment Rental, Permits & Inspections
OVERHEAD: Office Rent, Utilities, Insurance (GL/WC/Vehicle), Software/Subscriptions
PAYROLL: Owner Draw, W2 Employees, 1099 Contractors
VEHICLES: Fuel, Maintenance, Payments, Registration
MARKETING: Google Ads, Facebook Ads, Door Hangers, Yard Signs, Lead Services (HomeAdvisor/Angi/Thumbtack)
EQUIPMENT: Tools, Ladders, Safety Gear, Machinery
TAXES: Estimated Taxes, Sales Tax, Payroll Tax
TRANSFERS: Internal account transfers (flag these — don't double-count)
UNKNOWN: Needs manual review

Rules:
- Credits/deposits over $500 from individuals or insurance companies = REVENUE
- ACH or wire from unknown party over $1000 = likely REVENUE
- Home Depot, Lowe's, 84 Lumber, ABC Supply = COGS/Materials
- QuickBooks, ServiceTitan, Jobber, Google Workspace = OVERHEAD/Software
- Shell, BP, Exxon, Chevron, WEX, Fleetcor = VEHICLES/Fuel
- HomeAdvisor, Angi, Thumbtack, Google Ads, Meta = MARKETING
- Payroll processors (ADP, Gusto, Paychex) = PAYROLL
- IRS, state tax payments = TAXES
- Transfer between own accounts = TRANSFERS

Return a JSON array with one object per transaction, in the same order:
[{"category": "CATEGORY", "subcategory": "Subcategory", "confidence": 0.0-1.0}, ...]

Transactions to categorize:
"""


def _load_cache() -> dict[str, dict]:
    """Load categorization cache from disk."""
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, dict]) -> None:
    """Save categorization cache to disk."""
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not save cache (non-fatal): %s", exc)


def _make_cache_key(row: pd.Series) -> str:
    """Unique key for caching a transaction categorization."""
    return f"{row['description'][:50]}|{row['amount']}"


def _call_claude(transactions: list[dict], client: Any) -> list[dict]:
    """Call Claude API to categorize a batch of transactions."""
    batch_text = json.dumps([
        {"id": i, "date": t["date"], "description": t["description"], "amount": t["amount"]}
        for i, t in enumerate(transactions)
    ], indent=2)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": CATEGORY_PROMPT + batch_text
                }
            ]
        )
        content = response.content[0].text.strip()

        # Extract JSON from response
        json_match = None
        if "```json" in content:
            json_match = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_match = content.split("```")[1].split("```")[0].strip()
        elif content.startswith("["):
            json_match = content
        else:
            # Find first [ to last ]
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1:
                json_match = content[start:end+1]

        if json_match:
            results = json.loads(json_match)
            if isinstance(results, list) and len(results) == len(transactions):
                return results

    except json.JSONDecodeError as exc:
        logger.warning("JSON decode error from Claude: %s", exc)
    except Exception as exc:
        logger.error("Claude API error: %s", exc)

    # Fallback: mark all as UNKNOWN
    return [{"category": "UNKNOWN", "subcategory": "Needs Manual Review", "confidence": 0.0}
            for _ in transactions]


def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize all transactions using Claude API with caching."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[yellow]ANTHROPIC_API_KEY not set — marking all as UNKNOWN[/yellow]")
        df["category"] = "UNKNOWN"
        df["subcategory"] = "Needs Manual Review"
        df["confidence"] = 0.0
        return df

    try:
        from anthropic import Anthropic  # type: ignore
        client = Anthropic(api_key=api_key)
    except ImportError:
        console.print("[red]anthropic package not installed[/red]")
        return df

    cache = _load_cache()
    to_categorize_indices = []
    to_categorize_rows = []

    for idx, row in df.iterrows():
        key = _make_cache_key(row)
        if key in cache:
            df.at[idx, "category"] = cache[key]["category"]
            df.at[idx, "subcategory"] = cache[key]["subcategory"]
            df.at[idx, "confidence"] = cache[key]["confidence"]
        else:
            to_categorize_indices.append(idx)
            to_categorize_rows.append(row.to_dict())

    if not to_categorize_rows:
        console.print(f"[green]All {len(df)} transactions loaded from cache[/green]")
        return df

    console.print(f"Categorizing {len(to_categorize_rows)} transactions via Claude API...")

    batch_size = 50
    batches = [
        to_categorize_rows[i:i+batch_size]
        for i in range(0, len(to_categorize_rows), batch_size)
    ]

    result_map: dict[int, dict] = {}
    processed = 0

    for batch_idx, batch in enumerate(track(batches, description="Calling Claude API...")):
        results = _call_claude(batch, client)
        for local_i, result in enumerate(results):
            global_i = batch_idx * batch_size + local_i
            result_map[global_i] = result

        processed += len(batch)
        console.print(f"  Batch {batch_idx+1}/{len(batches)} complete ({processed}/{len(to_categorize_rows)})")

    # Apply results back to DataFrame
    for list_pos, (idx, row) in enumerate(zip(to_categorize_indices, to_categorize_rows)):
        result = result_map.get(list_pos, {
            "category": "UNKNOWN", "subcategory": "Needs Manual Review", "confidence": 0.0
        })
        cat = result.get("category", "UNKNOWN")
        sub = result.get("subcategory", "")
        conf = float(result.get("confidence", 0.0))

        df.at[idx, "category"] = cat
        df.at[idx, "subcategory"] = sub
        df.at[idx, "confidence"] = conf

        # Cache it
        key = _make_cache_key(pd.Series(row))
        cache[key] = {"category": cat, "subcategory": sub, "confidence": conf}

    _save_cache(cache)

    low_conf = df[df["confidence"] < 0.7]
    if not low_conf.empty:
        console.print(
            f"[yellow]⚠ {len(low_conf)} transactions flagged for manual review (confidence < 0.7)[/yellow]"
        )

    return df
