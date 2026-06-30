"""Transaction categorizer — rule-based engine + optional Claude API for unknowns."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Use /tmp on cloud (read-only git checkout), local data/ otherwise
_TMP_CACHE = Path("/tmp/roofsmart/processed/categories_cache.json")
_LOCAL_CACHE = Path("data/processed/categories_cache.json")
CACHE_PATH = _TMP_CACHE if Path("/mount/src").exists() else _LOCAL_CACHE

CATEGORIES = {
    "REVENUE": ["Job Deposits", "Final Payments", "Insurance Checks", "Supplements"],
    "COGS": ["Materials", "Subcontractor Labor", "Equipment Rental", "Permits"],
    "OVERHEAD": ["Office Rent", "Utilities", "Insurance", "Software/Subscriptions"],
    "PAYROLL": ["Owner Draw", "W2 Employees", "1099 Contractors"],
    "VEHICLES": ["Fuel", "Maintenance", "Payments", "Registration"],
    "MARKETING": ["Google Ads", "Facebook Ads", "Lead Services"],
    "EQUIPMENT": ["Tools", "Machinery"],
    "TAXES": ["Estimated Taxes", "Sales Tax", "Payroll Tax"],
    "TRANSFERS": ["Internal Transfers"],
    "UNKNOWN": ["Needs Manual Review"],
}

# ---------------------------------------------------------------------------
# Rule-based keyword engine
# ---------------------------------------------------------------------------

# Each rule: (regex pattern, category, subcategory, confidence)
# Patterns are matched case-insensitively against the description.
# Rules are evaluated top-to-bottom; first match wins.
_RULES: list[tuple[str, str, str, float]] = [
    # TRANSFERS (check before REVENUE — transfers look like large credits)
    (r"\bwire transfer\b|\bzelle\b|\bvenmo\b|\bsquare cash\b|\bcash app\b|\btransfer (to|from)\b", "TRANSFERS", "Internal Transfers", 0.85),
    (r"\baccount transfer\b|\binternal transfer\b|\bfunds transfer\b", "TRANSFERS", "Internal Transfers", 0.90),

    # TAXES
    (r"\birs\b|\binternal revenue\b|\bu\.s\. treasury\b|\bestimated tax\b", "TAXES", "Estimated Taxes", 0.95),
    (r"\bsales tax\b|\bstate tax\b|\bpayroll tax\b|\bdept of revenue\b|\bdepartment of revenue\b", "TAXES", "Sales Tax", 0.90),

    # PAYROLL
    (r"\badp\b|\bgusto\b|\bpaychex\b|\bpaylocity\b|\brippling\b", "PAYROLL", "W2 Employees", 0.95),
    (r"\bpayroll\b|\bdirect deposit\b.*\bemployee\b", "PAYROLL", "W2 Employees", 0.85),
    (r"\bowner draw\b|\bowner's draw\b", "PAYROLL", "Owner Draw", 0.95),

    # MARKETING
    (r"\bgoogle ads\b|\bgoogle adwords\b|\bgoogle\s+llc\b", "MARKETING", "Google Ads", 0.90),
    (r"\bfacebook\b|\bmeta platforms\b|\binstagram\b", "MARKETING", "Facebook Ads", 0.90),
    (r"\bhomeadvisor\b|\bangi\b|\bthumbtack\b|\byelp\b|\bhouzz\b|\bnetworx\b|\bleadgen\b|\blead gen\b", "MARKETING", "Lead Services", 0.95),
    (r"\bdoor hanger\b|\byard sign\b|\bdirect mail\b|\bpostcard\b", "MARKETING", "Print Marketing", 0.85),

    # VEHICLES
    (r"\bshell\b|\bbp\b|\bexxon\b|\bmobil\b|\bchevron\b|\bsunoco\b|\bmarathon\b|\bcitgo\b|\bwawa\b|\bquiktrip\b|\bquick trip\b|\bcasey\b|\bpilot\b|\bflying j\b|\bta travel\b", "VEHICLES", "Fuel", 0.92),
    (r"\bwex\b|\bfleetcor\b|\bfuel card\b|\bcomdata\b", "VEHICLES", "Fuel", 0.95),
    (r"\bautozone\b|\bo'reilly\b|\boreilly\b|\bnapa auto\b|\badvance auto\b|\bpep boys\b", "VEHICLES", "Maintenance", 0.90),
    (r"\bcar wash\b|\bjiffy lube\b|\bquick lube\b|\boil change\b|\btire kingdom\b|\bdiscounts tire\b|\bgoodyear\b|\bfirestone\b", "VEHICLES", "Maintenance", 0.90),
    (r"\bford motor\b|\bgm financial\b|\btoyota financial\b|\bhyundai finance\b|\bdodge\b.*\bpayment\b|\btruck payment\b|\bvehicle payment\b|\bauto loan\b", "VEHICLES", "Payments", 0.90),
    (r"\bdmv\b|\bvehicle registration\b|\btag renewal\b", "VEHICLES", "Registration", 0.92),

    # COGS — materials & supply houses
    (r"\bhome depot\b|\bhomedepot\b", "COGS", "Materials", 0.92),
    (r"\blowe'?s\b|\blowes\b", "COGS", "Materials", 0.92),
    (r"\b84 lumber\b|\babc supply\b|\bbuilders firstsource\b|\bfactory direct\b|\bfastenal\b", "COGS", "Materials", 0.95),
    (r"\bmenards\b|\btrue value\b|\bace hardware\b|\bdo it best\b", "COGS", "Materials", 0.88),
    (r"\bsupply house\b|\broofing supply\b|\bgaf\b|\bcertainte?ed\b|\bowens corning\b|\biko\b|\btamko\b", "COGS", "Materials", 0.95),
    (r"\bshingle\b|\bunderlayment\b|\bflashing\b|\bice.water\b|\bdeck nail\b|\bcoil nail\b|\bdrip edge\b|\bsoffit\b|\bfascia\b", "COGS", "Materials", 0.95),
    (r"\bgutter\b|\bdownspout\b|\bscreen guard\b|\bleaf guard\b", "COGS", "Materials", 0.90),
    (r"\bsubcontract\b|\bsub contract\b|\blabor only\b|\bcrew\b.*\bpay\b|\binstall crew\b", "COGS", "Subcontractor Labor", 0.85),
    (r"\bpermit\b|\binspection fee\b|\bbuilding dept\b|\bcounty permit\b", "COGS", "Permits", 0.90),
    (r"\bequipment rental\b|\bunited rentals\b|\bsunbelt rental\b|\brunpro\b|\bdumpster\b|\bwaste mgmt\b|\brepublic services\b", "COGS", "Equipment Rental", 0.90),

    # EQUIPMENT (purchase, not rental)
    (r"\bmilwaukee tool\b|\bdewalt\b|\bmakita\b|\bbosch\b|\bstanley\b|\bknaack\b", "EQUIPMENT", "Tools", 0.90),
    (r"\bsafety gear\b|\bharness\b|\bfall protect\b|\bppe\b|\bhard hat\b|\bsafety supply\b", "EQUIPMENT", "Safety Gear", 0.88),
    (r"\bgenerator\b|\bcompressor\b|\bnailer\b|\bsaw\b.*\bpurchase\b", "EQUIPMENT", "Machinery", 0.85),

    # OVERHEAD
    (r"\bquickbooks\b|\bintuit\b|\bservicetitan\b|\bjobber\b|\baccuweather\b|\bgoogle workspace\b|\bmicrosoft 365\b|\boffice 365\b|\badobe\b|\bdropbox\b|\bslack\b|\bzoom\b", "OVERHEAD", "Software/Subscriptions", 0.95),
    (r"\bcomcast\b|\bat&t\b|\bverizon\b|\bt-mobile\b|\bspectrum\b|\bcox comm\b|\bcenlink\b|\bxfinity\b", "OVERHEAD", "Utilities", 0.88),
    (r"\bduke energy\b|\bconsolidated edison\b|\bpge\b|\bpacific gas\b|\butterior electric\b|\belectric\b.*\bservice\b|\bgas service\b|\bwater service\b", "OVERHEAD", "Utilities", 0.85),
    (r"\boffice rent\b|\brent payment\b|\boffice lease\b|\bmonthly rent\b", "OVERHEAD", "Office Rent", 0.90),
    (r"\boffice depot\b|\bstaples\b|\bamazon\b.*\bbusiness\b|\buline\b", "OVERHEAD", "Office Supplies", 0.75),
    (r"\binsurance\b|\bgl insurance\b|\bworkers comp\b|\bw\.?c\.?\b.*\bpremium\b|\bbusiness insurance\b|\bnationwide\b|\btravelers\b|\bliberty mutual\b|\bprogressive\b.*\bcommercial\b", "OVERHEAD", "Insurance", 0.88),

    # REVENUE — large inbound credits (run last so transfer rules can fire first)
    (r"\bstate farm\b|\ballstate\b|\bfarmers\b|\bgeico\b|\busaa\b|\bnationwide\b|\bprogressive\b|\bcountrywide\b|\baig\b|\bchubb\b|\bhartford\b|\btravelers\b.*\bclaim\b|\bclaim payment\b|\binsurance check\b|\binsurance payment\b", "REVENUE", "Insurance Checks", 0.92),
    (r"\bsupplement\b|\broe payment\b|\broof supplement\b", "REVENUE", "Supplements", 0.95),
    (r"\bdeposit\b.*\bjob\b|\bjob deposit\b|\bdown payment\b.*\broof\b|\bcontract deposit\b", "REVENUE", "Job Deposits", 0.88),
]


def _rule_categorize(description: str, amount: float) -> tuple[str, str, float] | None:
    """Return (category, subcategory, confidence) from rules, or None if no match."""
    desc = str(description).lower()
    for pattern, cat, sub, conf in _RULES:
        if re.search(pattern, desc, re.IGNORECASE):
            return cat, sub, conf

    # Heuristic: large positive amounts with no keyword match are likely REVENUE
    # (inbound ACH, customer payments, insurance — hard to keyword-match)
    if isinstance(amount, (int, float)) and amount > 1000:
        return "REVENUE", "Final Payments", 0.60

    return None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict[str, dict]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, dict]) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not save cache (non-fatal): %s", exc)


def _make_cache_key(row: pd.Series) -> str:
    return f"{str(row['description'])[:50]}|{row['amount']}"


# ---------------------------------------------------------------------------
# Optional Claude API (only called for UNKNOWN transactions if key set)
# ---------------------------------------------------------------------------

CATEGORY_PROMPT = """You are a financial analyst for Roof Smart, a roofing company. Categorize each transaction.

Categories: REVENUE, COGS, OVERHEAD, PAYROLL, VEHICLES, MARKETING, EQUIPMENT, TAXES, TRANSFERS, UNKNOWN

Return ONLY a JSON array, one object per transaction, same order:
[{"category": "CATEGORY", "subcategory": "brief label", "confidence": 0.0-1.0}, ...]

Transactions:
"""


def _call_claude(transactions: list[dict], client: Any) -> list[dict]:
    batch_text = json.dumps([
        {"id": i, "date": t.get("date", ""), "description": t.get("description", ""), "amount": t.get("amount", 0)}
        for i, t in enumerate(transactions)
    ], indent=2)

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # cheapest model
            max_tokens=2048,
            messages=[{"role": "user", "content": CATEGORY_PROMPT + batch_text}]
        )
        content = response.content[0].text.strip()
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1:
            results = json.loads(content[start:end+1])
            if isinstance(results, list) and len(results) == len(transactions):
                return results
    except Exception as exc:
        logger.error("Claude API error: %s", exc)

    return [{"category": "UNKNOWN", "subcategory": "Needs Manual Review", "confidence": 0.0}
            for _ in transactions]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize all transactions. Rules run first; Claude API used for leftovers if key set."""

    # Ensure category columns exist with correct dtypes BEFORE any assignment
    for col in ("category", "subcategory"):
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
        df[col] = df[col].astype(object)
    if "confidence" not in df.columns:
        df["confidence"] = pd.Series(dtype="float64")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    cache = _load_cache()
    unknown_indices: list[int] = []
    unknown_rows: list[dict] = []

    for idx, row in df.iterrows():
        # Skip if already categorized (non-null, non-UNKNOWN)
        existing = df.at[idx, "category"]
        if existing and str(existing) not in ("nan", "None", "UNKNOWN", ""):
            continue

        key = _make_cache_key(row)
        if key in cache:
            df.at[idx, "category"] = str(cache[key]["category"])
            df.at[idx, "subcategory"] = str(cache[key].get("subcategory", ""))
            df.at[idx, "confidence"] = float(cache[key].get("confidence", 0.0))
            continue

        result = _rule_categorize(row.get("description", ""), row.get("amount", 0))
        if result:
            cat, sub, conf = result
            df.at[idx, "category"] = cat
            df.at[idx, "subcategory"] = sub
            df.at[idx, "confidence"] = conf
            cache[key] = {"category": cat, "subcategory": sub, "confidence": conf}
        else:
            unknown_indices.append(idx)
            unknown_rows.append(row.to_dict())

    # Optionally call Claude API for true unknowns
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if unknown_rows and api_key:
        try:
            from anthropic import Anthropic  # type: ignore
            client = Anthropic(api_key=api_key)
            batch_size = 50
            for batch_start in range(0, len(unknown_rows), batch_size):
                batch = unknown_rows[batch_start:batch_start + batch_size]
                batch_idx = unknown_indices[batch_start:batch_start + batch_size]
                results = _call_claude(batch, client)
                for idx, row, result in zip(batch_idx, batch, results):
                    cat = str(result.get("category", "UNKNOWN"))
                    sub = str(result.get("subcategory", ""))
                    conf = float(result.get("confidence", 0.0))
                    df.at[idx, "category"] = cat
                    df.at[idx, "subcategory"] = sub
                    df.at[idx, "confidence"] = conf
                    key = _make_cache_key(pd.Series(row))
                    cache[key] = {"category": cat, "subcategory": sub, "confidence": conf}
        except Exception as exc:
            logger.error("Claude API categorization failed: %s", exc)

    # Mark any remaining uncategorized as UNKNOWN
    for idx in unknown_indices:
        if not df.at[idx, "category"] or str(df.at[idx, "category"]) in ("nan", "None", ""):
            df.at[idx, "category"] = "UNKNOWN"
            df.at[idx, "subcategory"] = "Needs Manual Review"
            df.at[idx, "confidence"] = 0.0

    _save_cache(cache)

    categorized = df[df["category"].notna() & (df["category"] != "UNKNOWN")].shape[0]
    logger.info("Categorized %d / %d transactions via rules+API", categorized, len(df))

    return df
