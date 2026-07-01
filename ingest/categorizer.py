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
    # OWNER / SHAREHOLDER
    # -----------------------------------------------------------------------
    (r"\bowner contribution\b|\bchase contribution\b|\bshareholder contribution\b|\bdue to shareholder\b", "TRANSFERS", "Owner Contribution", 0.95),
    (r"\bowner reimbursement\b|\bchase reimbursement\b|\breimburse chase\b|\bshareholder reimburs\b", "TRANSFERS", "Owner Reimbursement", 0.95),
    (r"\bowner draw\b|\bowner'?s draw\b|\bpartner draw\b|\bchase draw\b|\bseth draw\b", "TRANSFERS", "Owner Draw", 0.95),

    # -----------------------------------------------------------------------
    # TRANSFERS — credit card payments, inter-account, LOC
    # -----------------------------------------------------------------------
    (r"amex epayment", "TRANSFERS", "Credit Card Payment", 0.98),
    (r"online credit card pmt|thank you for your pmt.*3453|online payment.*thank you|mobile payment.*thank you", "TRANSFERS", "Credit Card Payment", 0.97),
    (r"online transfer to.*1908|xxxxx1908|0000004121731908", "TRANSFERS", "Loan Repayment to Partners", 0.98),
    (r"online transfer (to|from)|online payment to \d", "TRANSFERS", "Inter-Account Transfer", 0.95),
    (r"\bline of credit\b|\bloc\b.*\bdraw\b|\bloc\b.*\badvance\b", "TRANSFERS", "Loan Proceeds", 0.92),
    (r"\bzelle\b|\bwire transfer\b|\baccount transfer\b|\binternal transfer\b", "TRANSFERS", "Loan From Partners", 0.88),
    (r"\bvenmo\b|\bcash app\b", "TRANSFERS", "Loan From Partners", 0.80),
    (r"\bbranch payment\b", "TRANSFERS", "Loan From Partners", 0.80),

    # -----------------------------------------------------------------------
    # TAXES
    # -----------------------------------------------------------------------
    (r"\birs\b|\binternal revenue\b|\bu\.s\. treasury\b|\bestimated tax\b", "TAXES", "Sales Tax Paid", 0.95),
    (r"wvtaxpay|ohio sales (return|pmt)|ohsalestvl|ohsalesutx|\bwv state tax\b|\bky.sec of state\b|\bohio business filing\b", "TAXES", "Sales Tax Paid", 0.90),
    (r"\bsales tax\b|\bstate tax\b|\bdept of revenue\b|\bdepartment of revenue\b", "TAXES", "Sales Tax Paid", 0.90),
    (r"opc tax", "TAXES", "Tax Prep Fee", 0.90),

    # -----------------------------------------------------------------------
    # PAYROLL
    # -----------------------------------------------------------------------
    (r"\badp\b|\bgusto\b|\bpaychex\b|\bpaylocity\b|\brippling\b", "PAYROLL", "Payroll", 0.95),
    (r"eepay.?garn|eepay/garn", "PAYROLL", "Payroll Garnishment", 0.90),
    (r"\bpayroll\b", "PAYROLL", "Payroll", 0.85),

    # -----------------------------------------------------------------------
    # MARKETING
    # -----------------------------------------------------------------------
    # Facebook — all FACEBK * variants (hundreds of transactions)
    (r"facebk\s*\*|facebook|meta platforms|instagram", "MARKETING", "Marketing", 0.95),
    # N0xxx Payment Facebk (PNC debit card Facebook charges)
    (r"n\d{4} \d{4} payment facebk", "MARKETING", "Marketing", 0.95),
    # Google
    (r"\bgoogle ads\b|\bgoogle adwords\b|\bgoogle\s+llc\b", "MARKETING", "Marketing", 0.90),
    # Known marketing vendors
    (r"paper strateg|in \*paper strateg|fanbasis|wsaz|gray media|sendjim|salty'?s media|wave.*salty|saltys media", "MARKETING", "Marketing", 0.95),
    (r"krager|crager|grey marketing", "MARKETING", "Marketing", 0.95),
    (r"ironton tribune|minuteman press|in \*visionary signs|bamko", "MARKETING", "Marketing", 0.90),
    (r"homeadvisor|angi\b|thumbtack|houzz|networx|roofr inc", "MARKETING", "Marketing", 0.95),
    (r"indeed jobs", "MARKETING", "Recruiting", 0.85),
    (r"in \*structure market|paypal \*struc marke", "MARKETING", "Marketing", 0.85),

    # -----------------------------------------------------------------------
    # VEHICLES — Fuel (real gas station names from actual transactions)
    # -----------------------------------------------------------------------
    (r"speedway", "VEHICLES", "Fuel", 0.95),
    (r"sheetz", "VEHICLES", "Fuel", 0.95),
    (r"super qu[ia]k|super quick|super wash.*(?!car)|\bgo.?mart\b|\bgo mart\b", "VEHICLES", "Fuel", 0.92),
    (r"murphy express|circle k|thornton'?s|huck'?s|kash stop|lkb main|seaman 1st|corner market|generations quick|1st stop|locust grove.*gas|terry rd conv", "VEHICLES", "Fuel", 0.92),
    (r"rich oil|woodford oil|clarks (pns|fast lane)|clarks pns", "VEHICLES", "Fuel", 0.92),
    (r"\bmarathon|marathon petro|exxonmobil|7.?eleven|kroger fuel|union 76|meijer express", "VEHICLES", "Fuel", 0.92),
    (r"gillispies|one stop #\d|locust grove.*gas|one stop gas|s webster.*oh.*pos|ironton food|phillip'?s grocery", "VEHICLES", "Fuel", 0.80),
    (r"united dairy farmers|udf\b", "VEHICLES", "Fuel", 0.85),
    (r"\bwex\b|\bfleetcor\b|\bfuel card\b|\bcomdata\b", "VEHICLES", "Fuel", 0.95),
    (r"\bshell\b|\bexxon\b|\bmobil\b|\bchevron\b|\bsunoco\b|\bcitgo\b|\bwawa\b|\bpilot\b|\bflying j\b|\bbp#|\bbp \d", "VEHICLES", "Fuel", 0.92),
    (r"\bkroger\b", "VEHICLES", "Fuel", 0.72),
    (r"wv parkways", "VEHICLES", "Tolls", 0.90),
    # Maintenance
    (r"vioc|valvoline|oil change|jiffy lube|quick lube", "VEHICLES", "Maintenance", 0.92),
    (r"porter tire|dalton towing|auto zone|autozone|o'?reilly|napa\b|advance auto|pep boys|firestone|goodyear|discount tire", "VEHICLES", "Maintenance", 0.90),
    (r"ironton super wash|super wash self serve|car wash", "VEHICLES", "Maintenance", 0.88),
    # Vehicle payments / registration
    (r"ford motor|gm financial|toyota financial|truck payment|vehicle payment|auto loan", "VEHICLES", "Vehicle Payment", 0.90),
    (r"\bdmv\b|vehicle registration|tag renewal", "VEHICLES", "Registration", 0.92),
    # Rentals
    (r"enterprise.*rent|\benterprise\s+\d|\benterprise\s{2,}|alamo rent|hertz\b", "VEHICLES", "Vehicle Rental", 0.88),

    # -----------------------------------------------------------------------
    # COGS — Supplies and Materials
    # -----------------------------------------------------------------------
    (r"home depot|homedepot", "COGS", "Supplies and Materials", 0.92),
    (r"lowe'?s\b|lowes\b", "COGS", "Supplies and Materials", 0.92),
    (r"abc supply|84 lumber|builders firstsource|fastenal|grainger|ferguson\b", "COGS", "Supplies and Materials", 0.95),
    (r"harbor freight|tractor supply|sherwin.?williams|messer\b|central hardwa|central hardware|o'?dell lumber|kenny queen", "COGS", "Supplies and Materials", 0.92),
    (r"rural king|gme supply|gme\*gme|sprayer depot", "COGS", "Supplies and Materials", 0.88),
    (r"roofing supply|gaf\b|certainteed|owens corning|iko\b|tamko", "COGS", "Supplies and Materials", 0.95),
    (r"shingle|underlayment|flashing|ice.water|deck nail|coil nail|drip edge|soffit|fascia", "COGS", "Supplies and Materials", 0.95),
    (r"gutter|downspout|screen guard|leaf guard", "COGS", "Supplies and Materials", 0.90),
    (r"cougar paws|sp \*cougar|sp cougar", "COGS", "Supplies and Materials", 0.90),
    (r"subcontract|sub contract|labor only|install crew|ethan roebuck|zabos customs|in \*zabos|cams auto", "COGS", "Subcontractor Labor", 0.85),
    # Amazon — roofing co buys tools/supplies; moderate confidence, flag for review
    (r"amazon\.com|amazon mktpl|amazonmktpl|amzn\.com", "COGS", "Supplies and Materials", 0.70),
    (r"portsmouth engineering|scioto county public", "COGS", "Permits", 0.80),
    (r"\bpermit\b|inspection fee|building dept|county permit", "COGS", "Permits", 0.90),
    (r"sunbelt rental|united rentals|runpro|dumpster|waste mgmt|republic services", "COGS", "Equipment Rental", 0.90),

    # -----------------------------------------------------------------------
    # EQUIPMENT
    # -----------------------------------------------------------------------
    (r"spray rig|trailer.*purchase|purchase.*trailer", "EQUIPMENT", "Fixed Asset - Equipment", 0.90),
    (r"milwaukee tool|dewalt|makita|bosch\b|knaack|micro center|micro electron", "EQUIPMENT", "Tools", 0.88),
    (r"best buy\b", "EQUIPMENT", "Tools", 0.72),
    (r"www\.dji\.com|dji\.com|drone reg|flylegitllc", "EQUIPMENT", "Drone Equipment", 0.88),
    (r"academy sport|academy sports|bass pro|sportsman", "EQUIPMENT", "Safety Gear", 0.72),
    (r"generator|compressor|nailer", "EQUIPMENT", "Machinery", 0.85),
    (r"truck purchase|vehicle purchase|purchase.*truck", "EQUIPMENT", "Fixed Asset - Vehicle", 0.88),

    # -----------------------------------------------------------------------
    # OVERHEAD
    # -----------------------------------------------------------------------
    # Insurance
    (r"state farm|allstate|cgi insurance|biberk|biberk insurance|compmanagement|ohio farm bureau|progressive.*(?:ins|commercial)|geico|usaa|nationwide|travelers|liberty mutual|aig\b|chubb|hartford", "OVERHEAD", "Insurance Expenses", 0.92),
    (r"gl insurance|workers comp|w\.?c\.?\b.*premium|business insurance|insurance premium|drone insurance|commercial auto", "OVERHEAD", "Insurance Expenses", 0.90),
    (r"pmt\*oh bureau|oh bureau.*workers|bureau of workers|ohio bureau.*comp", "OVERHEAD", "Insurance Expenses", 0.90),
    # Software / subscriptions
    (r"quickbooks|payidw\.com|intuit\b", "OVERHEAD", "Software/Subscriptions", 0.92),
    (r"companycam|dispatch\b|roof coach|www\.roofcoach|roofcoach\.net|genesis\b|tsheets|final orbit", "OVERHEAD", "Software/Subscriptions", 0.95),
    (r"microsoft\*|microsoft 365|office 365|adobe\b|dropbox|slack\b|zoom\b|clickup|docusign|notarize|ideogram|runway standard|n2co\b", "OVERHEAD", "Software/Subscriptions", 0.92),
    (r"namecheap|name.cheap|ipostal|connectedinvestors|prov inc|checkr\b|first advantage|\blovable\b", "OVERHEAD", "Software/Subscriptions", 0.82),
    (r"fyffe jones|cochran.*company|sq \*cochran", "OVERHEAD", "Accounting Fees", 0.92),
    (r"b4wv|labor.*licens|licens.*labor|business licens|contractor licens", "OVERHEAD", "Business Licenses", 0.90),
    (r"ohio univ.*emarket|ohio univ emarket", "OVERHEAD", "Office Rent", 0.88),
    # Phone
    (r"tmobile|t-mobile|tmobile\*auto|n\d{4} \d{4} payment tmobile", "OVERHEAD", "Phone/Utilities", 0.92),
    (r"verizon|at&t|comcast|spectrum|cox comm|xfinity|u\.s\. cellular", "OVERHEAD", "Phone/Utilities", 0.88),
    # Utilities
    (r"duke energy|consolidated edison|electric.*service|gas service|water service", "OVERHEAD", "Utilities", 0.85),
    # Storage / rent
    (r"stone creek stor|ironton self stor|beechmont self sto|beechmont stor|self stor|storage\b", "OVERHEAD", "Storage/Rent", 0.88),
    (r"office rent|rent payment|office lease|monthly rent", "OVERHEAD", "Office Rent", 0.90),
    # Office supplies / shipping
    (r"office depot|officemax|staples\b|uline\b|ups store|usps\b|order for checks", "OVERHEAD", "Office Supplies", 0.82),
    # Bank fees
    (r"counter check fee|order for checks|membership fee|cr adj membership|service charge|nsf\b|overdraft|annual fee|tran fee|fee\b", "OVERHEAD", "Bank Fees/Interest", 0.85),
    # Walmart — could be supplies/materials for crew
    (r"\bwal.?mart\b|wm supercenter|\bsam'?s club\b|\bcostco\b", "COGS", "Supplies and Materials", 0.68),
    # Labor license / contractor license
    (r"labor.*licens|licens.*labor|contractor.*licens|licens.*contract", "OVERHEAD", "Software/Subscriptions", 0.75),
    # Business association / chamber
    (r"briggs lawrence|greater lawrenc|chamber of commerce|business assoc", "OVERHEAD", "Office Supplies", 0.65),
    (r"loc\b.*interest|line of credit.*interest|loc\b.*fee", "OVERHEAD", "Bank Fees/Interest", 0.88),
    # Business meals (crews, clients — low confidence, flag for review)
    (r"tst\*|tst \*|buffalo wild|wendy'?s|subway\b|panera|hardees|burger king|raising cane|longhorn steak|olive garden|penn station\b|chipotle|jersey mike|frisch|mellow mushroom|cattleman|roosters\b|condado taco|topgolf|sweetgreen", "OVERHEAD", "Business Meals", 0.60),
    (r"blue agave|blueagave|archetype.*oh|cjs on the bay|gametime|bt\*gametime|hickies|guthries", "OVERHEAD", "Business Meals", 0.60),
    (r"scioto ribber|skeetos pizza|china wok|pies.*pints|casablanca express|taqueria|toro loco|koi hibachi|charleys philly|tropical smoothie|marco island brew|levy.*osu|salt rock gala|metro beer|great amer bagel|sams bagels|roosters.*bar|shakery|rams dairy|asm global|smg\b|mountain health|tudor'?s|little caesar", "OVERHEAD", "Business Meals", 0.60),
    (r"\bdunham'?s\b", "EQUIPMENT", "Safety Gear", 0.72),
    (r"family dollar|dollar.?general|dollar tree", "COGS", "Supplies and Materials", 0.60),
    (r"cvs pharmacy|cvs/pharmacy|walgreen|rite aid", "OVERHEAD", "Office Supplies", 0.65),
    (r"riverside parking|parking serv|park.*serv|parkmobile", "OVERHEAD", "Travel", 0.78),
    (r"make it yours", "MARKETING", "Marketing", 0.72),
    # Travel / hotel
    (r"hampton inn|hilton\b|marriott|fontainebleau|jw marriott|aaa park", "OVERHEAD", "Travel", 0.75),
    # AMEX travel bookings and point redemptions
    (r"pwp\s+amex|pwp\s+american expr|amextravel|amex fine hot|amex.*travel|fhr redeem|platinum hotel credit", "OVERHEAD", "Travel", 0.80),
    (r"points for amex trvl|points for statement credit|adj redist purchase bal|dr adj redist cadv|redist.*cadv|cadv.*redist", "TRANSFERS", "Inter-Account Transfer", 0.80),
    # Provisional credit / dispute pending = bank holding disputed funds
    (r"provisional credit.*dispute|irregular signature return|missing signature return", "TRANSFERS", "Inter-Account Transfer", 0.85),
    # Chamber / business association
    (r"greater lawrence|lawrence county.*chamber|chamber of commerce", "OVERHEAD", "Office Supplies", 0.70),
    # Golf / entertainment (business meals category)
    (r"sugarwood golf|glf\*|hibiscus golf|topgolf", "OVERHEAD", "Business Meals", 0.65),

    # -----------------------------------------------------------------------
    # REVENUE — confirmed inbound payment signals
    # -----------------------------------------------------------------------
    (r"improvifi", "REVENUE", "Customer Financing", 0.95),
    (r"wisetack", "REVENUE", "Customer Financing", 0.95),
    (r"staxpayments|staxpmtsmerchant|stax", "REVENUE", "Credit Card Deposit", 0.92),
    (r"corporate ach deposit", "REVENUE", "Customer Payment", 0.80),
    (r"mobile deposit|\bdeposit\b", "REVENUE", "Customer Payment", 0.65),
    (r"claim payment|insurance check|insurance loss|claim settlement", "REVENUE", "Insurance Checks", 0.90),
    (r"\bsupplement\b|roe payment|roof supplement", "REVENUE", "Supplements", 0.95),
    (r"job deposit|contract deposit", "REVENUE", "Job Payment", 0.88),
    (r"lowe'?s.*ime|ime.*lowe'?s|corporate lead", "REVENUE", "Corporate Lead", 0.90),
    # ATM / cash withdrawals — likely owner draw or crew pay, flag for review
    (r"atm withdrawal|^withdrawal$|^withdrawal\b", "TRANSFERS", "Cash Withdrawal", 0.70),
]


def _rule_categorize(description: str, amount: float) -> tuple[str, str, float] | None:
    """Return (category, subcategory, confidence) from rules, or None if no match."""
    desc = str(description)
    inflow = float(amount) > 0  # positive = money coming IN

    # ── Sign-aware overrides (checked before general rules) ──────────────────
    # Roof Maxx: outflow = we're paying Roof Maxx for product (COGS)
    #            inflow  = Roof Maxx paying us a commission/payout (REVENUE)
    # Catches: BLS*Roof Maxx, ACH Roof Maxx, ROOF MAXX WESTERVILLE, ACH Credit Roof Maxx Techno
    if re.search(r"roof maxx", desc, re.IGNORECASE):
        if inflow:
            return "REVENUE", "Job Payment", 0.88
        else:
            return "COGS", "Roof Maxx Product", 0.95

    # Stax: positive = customer deposit (REVENUE), negative = processing fee (OVERHEAD)
    if re.search(r"staxpayments|staxpmtsmerchant|stax", desc, re.IGNORECASE):
        if inflow:
            return "REVENUE", "Credit Card Deposit", 0.92
        else:
            return "OVERHEAD", "Payment Processing Fees", 0.92

    # Wisetack: positive = customer financing deposit (REVENUE), negative = fee (OVERHEAD)
    if re.search(r"wisetack", desc, re.IGNORECASE):
        if inflow:
            return "REVENUE", "Customer Financing", 0.95
        else:
            return "OVERHEAD", "Payment Processing Fees", 0.90

    # Improvifi: same pattern
    if re.search(r"improvifi", desc, re.IGNORECASE):
        if inflow:
            return "REVENUE", "Customer Financing", 0.95
        else:
            return "OVERHEAD", "Payment Processing Fees", 0.90

    # Foundation Finance: inflow = customer financing deposit (REVENUE), outflow = garnishment (PAYROLL)
    if re.search(r"fndtn fin|foundation fin", desc, re.IGNORECASE):
        if inflow:
            return "REVENUE", "Customer Financing", 0.85
        else:
            return "PAYROLL", "Payroll Garnishment", 0.90

    # Mission Path Con: outflow = subcontractor payment (COGS)
    if re.search(r"mission path con", desc, re.IGNORECASE):
        return "COGS", "Subcontractor Labor", 0.85

    # Corporate ACH / ACH Credit: handle known sub-patterns before generic fallback
    if re.search(r"corporate ach|ach credit ach pmt", desc, re.IGNORECASE):
        # AMEX bill payment buried in corporate ACH description
        if re.search(r"amex epayment|epayment.*amex|eepayment", desc, re.IGNORECASE):
            return "TRANSFERS", "Credit Card Payment", 0.97
        # Payroll garnishment
        if re.search(r"eepay.?garn|eepay/garn", desc, re.IGNORECASE):
            return "PAYROLL", "Payroll Garnishment", 0.90
        # WV / state tax payments
        if re.search(r"wvtaxpay|ohsalestvl|ohsalesutx|wv tax|oh tax", desc, re.IGNORECASE):
            return "TAXES", "Sales Tax Paid", 0.90
        # ADP payroll tax
        if re.search(r"adp tax|adp.*tax", desc, re.IGNORECASE):
            return "TAXES", "Payroll Tax", 0.90
        # Transaction/service fee
        if re.search(r"tran fee|transaction fee|svc fee|service fee", desc, re.IGNORECASE):
            return "OVERHEAD", "Bank Fees/Interest", 0.92
        # Generic: inflow = customer payment, outflow = needs review
        if inflow:
            return "REVENUE", "Customer Payment", 0.70
        else:
            return "UNKNOWN", "Needs Manual Review", 0.50

    # Ohio Sales Return: outflow = sales tax payment, inflow = tax refund
    if re.search(r"ohio sales return|ohsalestvl|ohsalesutx", desc, re.IGNORECASE):
        if inflow:
            return "TAXES", "Tax Refund", 0.88
        else:
            return "TAXES", "Sales Tax Paid", 0.90

    # Generic deposit on checking = likely customer payment
    if re.search(r"^deposit$|\bmobile deposit\b", desc, re.IGNORECASE) and inflow:
        return "REVENUE", "Customer Payment", 0.65

    # ATM / generic withdrawal
    if re.search(r"atm withdrawal|^withdrawal$", desc, re.IGNORECASE) and not inflow:
        return "TRANSFERS", "Cash Withdrawal", 0.70

    # ── General rules ────────────────────────────────────────────────────────
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
