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
    # -----------------------------------------------------------------------
    # OWNER / SHAREHOLDER — must fire before TRANSFERS and REVENUE
    # -----------------------------------------------------------------------
    # Owner contributions (startup capital, vehicle funding, Hamilton expansion)
    (r"\bowner contribution\b|\bchase contribution\b|\bshareholder contribution\b|\bdue to shareholder\b", "TRANSFERS", "Owner Contribution", 0.95),
    # Owner reimbursements back to Chase (reduce shareholder loan balance)
    (r"\bowner reimbursement\b|\bchase reimbursement\b|\breimburse chase\b|\bshareholder reimburs\b", "TRANSFERS", "Owner Reimbursement", 0.95),
    # Partner draws / distributions (Chase + Seth — NOT payroll)
    (r"\bowner draw\b|\bowner'?s draw\b|\bpartner draw\b|\bdistribution\b.*\bchase\b|\bdistribution\b.*\bseth\b|\bchase draw\b|\bseth draw\b", "TRANSFERS", "Owner Draw", 0.95),

    # -----------------------------------------------------------------------
    # TRANSFERS — internal moves that must never become income/expense
    # -----------------------------------------------------------------------
    # PNC Line of Credit proceeds (liability, not income)
    (r"\bline of credit\b|\bloc\b.*\bdraw\b|\bloc\b.*\badvance\b|\bcredit line\b", "TRANSFERS", "Loan Proceeds", 0.92),
    # Loan interest / LOC fee
    (r"\bloc\b.*\binterest\b|\bline of credit\b.*\binterest\b|\bloc\b.*\bfee\b", "OVERHEAD", "Bank Fees/Interest", 0.88),
    # Credit card payment (offsets liability — not an expense)
    (r"\bcredit card payment\b|\bcc payment\b|\bcard payment\b|\bamex payment\b|\bpnc card payment\b", "TRANSFERS", "Credit Card Payment", 0.92),
    # Inter-account transfers
    (r"\bzelle\b|\bwire transfer\b|\baccount transfer\b|\binternal transfer\b|\bfunds transfer\b", "TRANSFERS", "Loan From Partners", 0.85),
    (r"\bvenmo\b|\bsquare cash\b|\bcash app\b|\btransfer (to|from)\b", "TRANSFERS", "Loan From Partners", 0.80),

    # -----------------------------------------------------------------------
    # TAXES
    # -----------------------------------------------------------------------
    (r"\birs\b|\binternal revenue\b|\bu\.s\. treasury\b|\bestimated tax\b", "TAXES", "Sales Tax Paid", 0.95),
    (r"\bsales tax\b|\bstate tax\b|\bpayroll tax\b|\bdept of revenue\b|\bdepartment of revenue\b|\bwv state tax\b", "TAXES", "Sales Tax Paid", 0.90),

    # -----------------------------------------------------------------------
    # PAYROLL (ADP, manual, cash payroll for Joe/Aaron)
    # -----------------------------------------------------------------------
    (r"\badp\b|\bgusto\b|\bpaychex\b|\bpaylocity\b|\brippling\b", "PAYROLL", "Payroll", 0.95),
    (r"\bpayroll\b|\bdirect deposit\b.*\bemployee\b", "PAYROLL", "Payroll", 0.85),

    # -----------------------------------------------------------------------
    # MARKETING
    # -----------------------------------------------------------------------
    (r"\bgoogle ads\b|\bgoogle adwords\b|\bgoogle\s+llc\b", "MARKETING", "Marketing", 0.90),
    (r"\bfacebook\b|\bmeta platforms\b|\binstagram\b", "MARKETING", "Marketing", 0.90),
    (r"\bgrey marketing\b|\bkrager'?s?\b|\bcrager'?s?\b|\bpaper strateg\b|\bfanbasis\b|\bwsaz\b|\bnap\b\b", "MARKETING", "Marketing", 0.95),
    (r"\bhomeadvisor\b|\bangi\b|\bthumbtack\b|\byelp\b|\bhouzz\b|\bnetworx\b|\bleadgen\b|\blead gen\b", "MARKETING", "Marketing", 0.95),
    (r"\bdoor hanger\b|\byard sign\b|\bdirect mail\b|\bpostcard\b", "MARKETING", "Marketing", 0.85),

    # -----------------------------------------------------------------------
    # VEHICLES
    # -----------------------------------------------------------------------
    (r"\bshell\b|\bbp\b|\bexxon\b|\bmobil\b|\bchevron\b|\bsunoco\b|\bmarathon\b|\bcitgo\b|\bwawa\b|\bquiktrip\b|\bquick trip\b|\bcasey\b|\bpilot\b|\bflying j\b|\bta travel\b", "VEHICLES", "Fuel", 0.92),
    (r"\bwex\b|\bfleetcor\b|\bfuel card\b|\bcomdata\b", "VEHICLES", "Fuel", 0.95),
    (r"\bautozone\b|\bo'?reilly\b|\bnapa auto\b|\bnapa\b|\badvance auto\b|\bpep boys\b", "VEHICLES", "Maintenance", 0.90),
    (r"\bcar wash\b|\bjiffy lube\b|\bquick lube\b|\boil change\b|\btire kingdom\b|\bdiscount tire\b|\bgoodyear\b|\bfirestone\b", "VEHICLES", "Maintenance", 0.90),
    (r"\bford motor\b|\bgm financial\b|\btoyota financial\b|\bhyundai finance\b|\bdodge\b.*\bpayment\b|\btruck payment\b|\bvehicle payment\b|\bauto loan\b", "VEHICLES", "Vehicle Payment", 0.90),
    (r"\bdmv\b|\bvehicle registration\b|\btag renewal\b", "VEHICLES", "Registration", 0.92),
    # Truck purchases → Fixed Asset (capitalize)
    (r"\btruck purchase\b|\bvehicle purchase\b|\bpurchase\b.*\btruck\b", "EQUIPMENT", "Fixed Asset - Vehicle", 0.88),

    # -----------------------------------------------------------------------
    # COGS — Supplies and Materials
    # -----------------------------------------------------------------------
    (r"\bhome depot\b|\bhomedepot\b", "COGS", "Supplies and Materials", 0.92),
    (r"\blowe'?s\b|\blowes\b", "COGS", "Supplies and Materials", 0.92),
    (r"\babc supply\b|\b84 lumber\b|\bbuilders firstsource\b|\bfastenal\b|\bgrainger\b|\bferguson\b", "COGS", "Supplies and Materials", 0.95),
    (r"\bharbor freight\b|\btractor supply\b|\bsherwin.williams\b|\bmesser\b|\bcentral hardware\b", "COGS", "Supplies and Materials", 0.92),
    (r"\bsam'?s club\b|\bcostco\b|\bamazon\b", "COGS", "Supplies and Materials", 0.75),
    (r"\bmenards\b|\btrue value\b|\bace hardware\b|\bdo it best\b", "COGS", "Supplies and Materials", 0.88),
    (r"\bsupply house\b|\broofing supply\b|\bgaf\b|\bcertainte?ed\b|\bowens corning\b|\biko\b|\btamko\b", "COGS", "Supplies and Materials", 0.95),
    (r"\bshingle\b|\bunderlayment\b|\bflashing\b|\bice.water\b|\bdeck nail\b|\bcoil nail\b|\bdrip edge\b|\bsoffit\b|\bfascia\b", "COGS", "Supplies and Materials", 0.95),
    (r"\bgutter\b|\bdownspout\b|\bscreen guard\b|\bleaf guard\b", "COGS", "Supplies and Materials", 0.90),
    (r"\bsubcontract\b|\bsub contract\b|\blabor only\b|\bcrew\b.*\bpay\b|\binstall crew\b|\bethan roebuck\b", "COGS", "Subcontractor Labor", 0.85),
    (r"\bpermit\b|\binspection fee\b|\bbuilding dept\b|\bcounty permit\b", "COGS", "Permits", 0.90),
    (r"\bequipment rental\b|\bunited rentals\b|\bsunbelt rental\b|\brunpro\b|\bdumpster\b|\bwaste mgmt\b|\brepublic services\b", "COGS", "Equipment Rental", 0.90),

    # -----------------------------------------------------------------------
    # EQUIPMENT (capitalize spray rigs, trailer, major tools)
    # -----------------------------------------------------------------------
    (r"\bspray rig\b|\btrailer\b.*\bpurchase\b|\bpurchase\b.*\btrailer\b", "EQUIPMENT", "Fixed Asset - Equipment", 0.90),
    (r"\bmilwaukee tool\b|\bdewalt\b|\bmakita\b|\bbosch\b|\bstanley\b|\bknaack\b", "EQUIPMENT", "Tools", 0.90),
    (r"\bsafety gear\b|\bharness\b|\bfall protect\b|\bppe\b|\bhard hat\b|\bsafety supply\b", "EQUIPMENT", "Safety Gear", 0.88),
    (r"\bgenerator\b|\bcompressor\b|\bnailer\b|\bsaw\b.*\bpurchase\b", "EQUIPMENT", "Machinery", 0.85),

    # -----------------------------------------------------------------------
    # OVERHEAD
    # -----------------------------------------------------------------------
    # Insurance Expenses
    (r"\bstate farm\b|\ballstate\b|\bcgi insurance\b|\bfarmers\b|\bgeico\b|\busaa\b|\bnationwide\b|\btravelers\b|\bliberty mutual\b|\bprogressive\b.*\bcommercial\b|\baig\b|\bchubb\b|\bhartford\b", "OVERHEAD", "Insurance Expenses", 0.92),
    (r"\bgl insurance\b|\bworkers comp\b|\bw\.?c\.?\b.*\bpremium\b|\bbusiness insurance\b|\binsurance premium\b|\bdrone insurance\b|\bcommercial auto\b", "OVERHEAD", "Insurance Expenses", 0.90),
    # Software / Subscriptions
    (r"\bquickbooks\b|\bintuit\b|\bcompanycam\b|\bdispatch\b|\broof coach\b|\bgenesis\b|\btsheets\b|\bfinal orbit\b", "OVERHEAD", "Software/Subscriptions", 0.95),
    (r"\bgoogle workspace\b|\bmicrosoft 365\b|\boffice 365\b|\badobe\b|\bdropbox\b|\bslack\b|\bzoom\b", "OVERHEAD", "Software/Subscriptions", 0.95),
    # Phone / Utilities
    (r"\bt-mobile\b|\bverizon\b|\bat&t\b|\bcomcast\b|\bspectrum\b|\bcox comm\b|\bxfinity\b", "OVERHEAD", "Phone/Utilities", 0.88),
    (r"\bduke energy\b|\bconsolidated edison\b|\belectric\b.*\bservice\b|\bgas service\b|\bwater service\b", "OVERHEAD", "Utilities", 0.85),
    # Rent / Office
    (r"\boffice rent\b|\brent payment\b|\boffice lease\b|\bmonthly rent\b", "OVERHEAD", "Office Rent", 0.90),
    (r"\boffice depot\b|\bstaples\b|\buline\b", "OVERHEAD", "Office Supplies", 0.75),
    # Bank fees
    (r"\bbank fee\b|\bservice charge\b|\bmonthly fee\b|\bannual fee\b|\bnsf\b|\boverdraft\b", "OVERHEAD", "Bank Fees/Interest", 0.88),

    # -----------------------------------------------------------------------
    # REVENUE — only fire on confirmed signals; large unknowns go to UNKNOWN
    # -----------------------------------------------------------------------
    # Customer financing platforms
    (r"\bimprovifi\b", "REVENUE", "Customer Financing", 0.95),
    (r"\bwisetack\b", "REVENUE", "Customer Financing", 0.95),
    (r"\bintuit\b.*\bdeposit\b|\bintuit\b.*\bpayment\b|\bsquare\b.*\bdeposit\b", "REVENUE", "Credit Card Deposit", 0.90),
    # Insurance claim payments (inbound)
    (r"\bclaim payment\b|\binsurance check\b|\binsurance loss\b|\bclaim settlement\b", "REVENUE", "Insurance Checks", 0.90),
    (r"\bsupplement\b|\broe payment\b|\broof supplement\b", "REVENUE", "Supplements", 0.95),
    # Confirmed job payments
    (r"\bjob deposit\b|\bcontract deposit\b|\bdown payment\b.*\broof\b|\broof maxx\b", "REVENUE", "Job Payment", 0.88),
    # Lowe's IME / corporate leads
    (r"\blowe'?s\b.*\bime\b|\bime\b.*\blowe'?s\b|\bcorporate lead\b", "REVENUE", "Corporate Lead", 0.90),
]


def _rule_categorize(description: str, amount: float) -> tuple[str, str, float] | None:
    """Return (category, subcategory, confidence) from rules, or None if no match."""
    desc = str(description).lower()
    for pattern, cat, sub, conf in _RULES:
        if re.search(pattern, desc, re.IGNORECASE):
            return cat, sub, conf

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
