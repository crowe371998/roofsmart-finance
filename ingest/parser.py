"""Universal bank statement parser for Roof Smart Finance."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import chardet
import pandas as pd
from dateutil import parser as dateparser
from rich.console import Console
from rich.progress import track

console = Console()
logger = logging.getLogger(__name__)

UNIFIED_COLUMNS = [
    "date", "description", "amount", "type", "balance",
    "account_last4", "account_name", "source_file", "category",
    "subcategory", "confidence",
]


def _make_hash(date: str, amount: float, description: str) -> str:
    """Create a dedup hash from transaction key fields."""
    raw = f"{date}|{amount}|{description[:20]}"
    return hashlib.md5(raw.encode()).hexdigest()


def _normalize_amount(value: str | float | int) -> float:
    """Convert any amount string to a signed float (debits negative)."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", "").replace("$", "").replace(" ", "")
    if not s or s in ("-", ""):
        return 0.0
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        result = float(s)
    except ValueError:
        return 0.0
    return -abs(result) if negative else result


def _normalize_date(value: str) -> str:
    """Parse any date string into YYYY-MM-DD format."""
    if not value or str(value).strip() in ("", "nan"):
        return ""
    try:
        return dateparser.parse(str(value), fuzzy=True).strftime("%Y-%m-%d")
    except Exception:
        return str(value).strip()


def _empty_df() -> pd.DataFrame:
    """Return an empty DataFrame with unified columns."""
    return pd.DataFrame(columns=UNIFIED_COLUMNS)


def _detect_encoding(path: Path) -> str:
    """Detect file encoding using chardet."""
    raw = path.read_bytes()[:50_000]
    result = chardet.detect(raw)
    return result.get("encoding") or "utf-8"


# ── CSV Parsers ──────────────────────────────────────────────────────────────

def _parse_csv(path: Path) -> pd.DataFrame:
    """Parse a CSV bank/card export into unified format."""
    encoding = _detect_encoding(path)
    try:
        df = pd.read_csv(path, encoding=encoding, on_bad_lines="skip")
    except Exception as exc:
        logger.warning("CSV read error %s: %s", path.name, exc)
        return _empty_df()

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Map common column name variants
    col_map = {
        "date": ["date", "transaction_date", "trans_date", "posted_date", "posting_date"],
        "description": ["description", "memo", "payee", "transaction_description", "details", "narration"],
        "amount": ["amount", "transaction_amount", "debit_amount", "credit_amount", "value"],
        "debit": ["debit", "withdrawals", "withdrawal", "debit_amount"],
        "credit": ["credit", "deposits", "deposit", "credit_amount"],
        "balance": ["balance", "running_balance", "available_balance", "ending_balance"],
        "account": ["account", "account_number", "account_no", "acct"],
    }

    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    date_col = find_col(col_map["date"])
    desc_col = find_col(col_map["description"])
    amount_col = find_col(col_map["amount"])
    debit_col = find_col(col_map["debit"])
    credit_col = find_col(col_map["credit"])
    balance_col = find_col(col_map["balance"])
    account_col = find_col(col_map["account"])

    rows = []
    for _, row in df.iterrows():
        date = _normalize_date(row.get(date_col, "") if date_col else "")
        desc = str(row.get(desc_col, "")) if desc_col else ""
        balance = _normalize_amount(row.get(balance_col, 0)) if balance_col else 0.0

        if amount_col and amount_col in row:
            amount = _normalize_amount(row[amount_col])
        elif debit_col or credit_col:
            debit = _normalize_amount(row.get(debit_col, 0)) if debit_col else 0.0
            credit = _normalize_amount(row.get(credit_col, 0)) if credit_col else 0.0
            amount = credit - abs(debit)
        else:
            amount = 0.0

        acct_raw = str(row.get(account_col, "")) if account_col else ""
        last4 = re.sub(r"\D", "", acct_raw)[-4:] if acct_raw else ""

        rows.append({
            "date": date,
            "description": desc.strip(),
            "amount": amount,
            "type": "credit" if amount >= 0 else "debit",
            "balance": balance,
            "account_last4": last4,
            "account_name": "",
            "source_file": path.name,
            "category": "",
            "subcategory": "",
            "confidence": 0.0,
        })

    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS)


# ── Excel Parsers ────────────────────────────────────────────────────────────

_HEADER_KEYWORDS = {"date", "description", "amount", "memo", "payee", "debit", "credit", "balance"}


def _find_header_row(raw_df: pd.DataFrame) -> int:
    """Scan rows to find the actual header row (returns row index, 0-based)."""
    for i, row in raw_df.iterrows():
        vals = {str(v).strip().lower() for v in row.values if str(v).strip() not in ("nan", "")}
        if len(vals & _HEADER_KEYWORDS) >= 2:
            return i
    return 0


def _parse_excel_sheet(xl: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    """Parse a single Excel sheet, auto-detecting the header row."""
    # First pass: read raw with no header to locate actual header row
    raw = xl.parse(sheet, header=None, dtype=str)
    if raw.empty:
        return _empty_df()
    header_row = _find_header_row(raw)
    # Second pass: re-read with correct header
    df = xl.parse(sheet, header=header_row, dtype=str)
    return df


def _parse_excel(path: Path) -> pd.DataFrame:
    """Parse an Excel bank export, handling multi-row header formats (e.g. AMEX)."""
    try:
        xl = pd.ExcelFile(path)
        best: pd.DataFrame = _empty_df()
        for sheet in xl.sheet_names:
            raw = _parse_excel_sheet(xl, sheet)
            if len(raw.columns) < 2:
                continue
            candidate = _dataframe_to_unified(raw, path.name)
            if len(candidate) > len(best):
                best = candidate
        return best
    except Exception as exc:
        logger.warning("Excel read error %s: %s", path.name, exc)
        return _empty_df()


def _dataframe_to_unified(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Convert a raw DataFrame (from Excel or CSV) into unified format."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    col_map = {
        "date": ["date", "transaction_date", "trans_date", "posted_date", "posting_date"],
        "description": ["description", "receipt", "memo", "payee", "transaction_description", "details", "narration"],
        "amount": ["amount", "transaction_amount", "debit_amount", "credit_amount", "value"],
        "debit": ["debit", "withdrawals", "withdrawal"],
        "credit": ["credit", "deposits", "deposit"],
        "balance": ["balance", "running_balance", "available_balance", "ending_balance"],
        "account": ["account", "account_#", "account_number", "account_no", "acct", "card_member"],
    }

    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    date_col = find_col(col_map["date"])
    desc_col = find_col(col_map["description"])
    amount_col = find_col(col_map["amount"])
    debit_col = find_col(col_map["debit"])
    credit_col = find_col(col_map["credit"])
    balance_col = find_col(col_map["balance"])
    account_col = find_col(col_map["account"])

    rows = []
    for _, row in df.iterrows():
        date_val = str(row.get(date_col, "") if date_col else "").strip()
        if not date_val or date_val.lower() in ("nan", "none", ""):
            continue
        date = _normalize_date(date_val)
        if not date:
            continue

        desc = str(row.get(desc_col, "") if desc_col else "").strip()
        # Clean multiline descriptions from AMEX (take first line)
        desc = desc.split("\n")[0].strip()
        if not desc or desc.lower() in ("nan", "none"):
            continue

        balance = _normalize_amount(row.get(balance_col, 0) if balance_col else 0)

        if amount_col and amount_col in row:
            amt_raw = str(row[amount_col]).strip()
            amount = _normalize_amount(amt_raw)
        elif debit_col or credit_col:
            debit = _normalize_amount(row.get(debit_col, 0) if debit_col else 0)
            credit = _normalize_amount(row.get(credit_col, 0) if credit_col else 0)
            amount = credit - abs(debit)
        else:
            amount = 0.0

        acct_raw = str(row.get(account_col, "") if account_col else "")
        last4 = re.sub(r"\D", "", acct_raw)[-4:] if acct_raw and acct_raw.lower() not in ("nan", "none") else ""

        rows.append({
            "date": date,
            "description": desc,
            "amount": amount,
            "type": "credit" if amount >= 0 else "debit",
            "balance": balance,
            "account_last4": last4,
            "account_name": "",
            "source_file": source_name,
            "category": "",
            "subcategory": "",
            "confidence": 0.0,
        })

    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS) if rows else _empty_df()


# ── PDF Parsers ──────────────────────────────────────────────────────────────

# PNC checking: "04/03 1,400.00 Deposit 001232152"
_PNC_CHK_TXN = re.compile(
    r"^(\d{2}/\d{2})\s+([\d,]+\.\d{2})\s+(.+?)(?:\s+\d{9,20})?$"
)
# PNC credit card: "04/10 04/10 2490641346K0VMN8P WAVE - *SALTYS MEDIA LLC ... $258.75"
_PNC_CC_TXN = re.compile(
    r"^(\d{2}/\d{2})\s+\d{2}/\d{2}\s+\S+\s+(.+?)\s+\$?([\d,]+\.\d{2})$"
)
# Section headers that tell us debit vs credit context
_DEDUCTION_HEADERS = re.compile(
    r"(checks and other deductions|debit card purchases|pos purchases|"
    r"atm.*debit|ach deductions|service charge|fees|purchases)", re.I
)
_ADDITION_HEADERS = re.compile(
    r"(deposits and other additions|ach additions|atm deposits|credits|"
    r"your transactions)", re.I
)


def _parse_pdf(path: Path) -> pd.DataFrame:
    """Parse a PNC bank statement PDF (checking or credit card)."""
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        logger.warning("pdfplumber not installed — skipping PDF %s", path.name)
        return _empty_df()

    try:
        with pdfplumber.open(path) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as exc:
        logger.warning("PDF open error %s: %s", path.name, exc)
        return _empty_df()

    # Extract account last4
    last4 = ""
    acct_m = re.search(r"(?:Account #|Account Number|XXXX[- ]+)(?:XXXX[- ]+){0,3}(\d{4})", full_text, re.I)
    if acct_m:
        last4 = acct_m.group(1)
    if not last4:
        acct_m2 = re.search(r"XX-XXXX-(\d{4})", full_text)
        if acct_m2:
            last4 = acct_m2.group(1)

    # Extract statement year from "For the Period MM/DD/YYYY" or "closing date MM/DD/YY"
    year = None
    period_m = re.search(r"For the Period \d{2}/\d{2}/(\d{4})", full_text)
    if period_m:
        year = period_m.group(1)
    if not year:
        close_m = re.search(r"(?:closing date|Statement Date)\s+\d{2}/\d{2}/(\d{2,4})", full_text, re.I)
        if close_m:
            y = close_m.group(1)
            year = f"20{y}" if len(y) == 2 else y
    if not year:
        year = str(pd.Timestamp.now().year)

    # Credit card statements have "TRANS DATE POST DATE" header and "Your transactions" section
    # Checking statements may mention "credit card" in cross-sell text — be specific
    is_credit_card = bool(re.search(r"TRANS DATE\s+POST DATE|Your transactions\s*\n.*TRANS DATE", full_text, re.I | re.DOTALL))
    account_name = "Credit Card" if is_credit_card else "Business Checking"

    rows = []

    if is_credit_card:
        rows = _parse_pnc_cc_text(full_text, year, last4, account_name, path.name)
    else:
        rows = _parse_pnc_checking_text(full_text, year, last4, account_name, path.name)

    # Fallback: generic line-by-line if specific parsers got nothing
    if not rows:
        rows = _parse_pdf_generic(full_text, year, last4, account_name, path.name)

    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS) if rows else _empty_df()


def _parse_pnc_checking_text(text: str, year: str, last4: str, acct_name: str, fname: str) -> list:
    """Parse PNC Business Checking PDF text into transaction rows."""
    rows = []
    is_deduction = False
    in_activity = False

    for line in text.split("\n"):
        stripped = line.strip()

        # Track when we enter the Activity Detail section
        if "Activity Detail" in stripped:
            in_activity = True
            continue
        if not in_activity:
            continue

        # Track debit/credit context from section headers
        if _DEDUCTION_HEADERS.search(stripped):
            is_deduction = True
            continue
        if _ADDITION_HEADERS.search(stripped):
            is_deduction = False
            continue

        # Skip header rows like "Date Transaction Reference posted Amount description number"
        if re.match(r"Date\s+Transaction", stripped, re.I):
            continue
        if re.match(r"posted\s+Amount", stripped, re.I):
            continue

        m = _PNC_CHK_TXN.match(stripped)
        if not m:
            continue

        date_mm_dd, amount_str, desc = m.group(1), m.group(2), m.group(3).strip()
        # Skip daily balance rows — description starts with another MM/DD date
        if re.match(r"^\d{2}/\d{2}", desc):
            continue
        # Skip balance summary rows with multiple amounts
        if re.match(r"^\d{2}/\d{2}\s+[\d,]+\.\d{2}\s+[\d,]+\.\d{2}", stripped):
            continue

        try:
            month, day = date_mm_dd.split("/")
            date_str = f"{year}-{month}-{day}"
        except Exception:
            continue

        amount = _normalize_amount(amount_str)
        if is_deduction:
            amount = -abs(amount)

        rows.append({
            "date": date_str,
            "description": desc,
            "amount": amount,
            "type": "credit" if amount >= 0 else "debit",
            "balance": 0.0,
            "account_last4": last4,
            "account_name": acct_name,
            "source_file": fname,
            "category": "",
            "subcategory": "",
            "confidence": 0.0,
        })

    return rows


def _parse_pnc_cc_text(text: str, year: str, last4: str, acct_name: str, fname: str) -> list:
    """Parse PNC Credit Card PDF text into transaction rows."""
    rows = []
    in_transactions = False

    # Extract year from closing date more carefully for credit cards
    # Closing date like "04/18/25" → use that year
    cc_year = year
    close_m = re.search(r"Statement closing date\s+(\d{2})/\d{2}/(\d{2,4})", text, re.I)
    if close_m:
        y = close_m.group(2)
        cc_year = f"20{y}" if len(y) == 2 else y
    # Detect prior-month transactions (closing month determines year boundary)
    close_month = int(close_m.group(1)) if close_m else 12

    for line in text.split("\n"):
        stripped = line.strip()

        if "Your transactions" in stripped or "TRANS DATE" in stripped:
            in_transactions = True
            continue
        if not in_transactions:
            continue

        # Skip card member lines and header rows
        if re.match(r"(TRANS DATE|Card number|MCC:|continued)", stripped, re.I):
            continue

        # Try credit card format: "04/10 04/10 REFNUM DESCRIPTION $AMOUNT"
        m = _PNC_CC_TXN.match(stripped)
        if not m:
            # Try simpler: "MM/DD MM/DD REFNUM DESC AMOUNT" where amount may lack $
            m2 = re.match(
                r"^(\d{2}/\d{2})\s+\d{2}/\d{2}\s+\S+\s+(.+?)\s+([\d,]+\.\d{2})$",
                stripped
            )
            if not m2:
                continue
            date_mm_dd, desc, amount_str = m2.group(1), m2.group(2), m2.group(3)
        else:
            date_mm_dd, desc, amount_str = m.group(1), m.group(2), m.group(3)

        try:
            month, day = date_mm_dd.split("/")
            # If transaction month > closing month, it's from previous year
            txn_month = int(month)
            txn_year = cc_year if txn_month <= close_month else str(int(cc_year) - 1)
            date_str = f"{txn_year}-{month}-{day}"
        except Exception:
            continue

        # Credit card charges are expenses (negative), payments/credits are positive
        desc_lower = desc.lower()
        is_payment = any(w in desc_lower for w in ["payment", "credit", "return", "refund"])
        amount = _normalize_amount(amount_str)
        if not is_payment:
            amount = -abs(amount)

        rows.append({
            "date": date_str,
            "description": desc.strip(),
            "amount": amount,
            "type": "credit" if amount >= 0 else "debit",
            "balance": 0.0,
            "account_last4": last4,
            "account_name": acct_name,
            "source_file": fname,
            "category": "",
            "subcategory": "",
            "confidence": 0.0,
        })

    return rows


def _parse_pdf_generic(text: str, year: str, last4: str, acct_name: str, fname: str) -> list:
    """Generic fallback: find lines with MM/DD date + dollar amount."""
    rows = []
    date_pat = re.compile(r"(\d{2}/\d{2})")
    amount_pat = re.compile(r"\$?([\d,]+\.\d{2})")

    for line in text.split("\n"):
        stripped = line.strip()
        d = date_pat.search(stripped)
        a = amount_pat.findall(stripped.replace(",", ""))
        if not d or not a:
            continue
        try:
            month, day = d.group(1).split("/")
            date_str = f"{year}-{month}-{day}"
        except Exception:
            continue
        desc = stripped
        amount = _normalize_amount(a[-1])
        rows.append({
            "date": date_str,
            "description": desc,
            "amount": amount,
            "type": "credit" if amount >= 0 else "debit",
            "balance": 0.0,
            "account_last4": last4,
            "account_name": acct_name,
            "source_file": fname,
            "category": "",
            "subcategory": "",
            "confidence": 0.0,
        })
    return rows


# ── OFX/QBO/QFX Parsers ─────────────────────────────────────────────────────

def _parse_ofx(path: Path) -> pd.DataFrame:
    """Parse OFX/QBO/QFX files (QuickBooks/Quicken format)."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("OFX read error %s: %s", path.name, exc)
        return _empty_df()

    rows = []
    last4 = ""
    account_name = ""

    # Extract account info
    acct_match = re.search(r"<ACCTID>([^<]+)", content)
    if acct_match:
        acct_id = acct_match.group(1).strip()
        last4 = acct_id[-4:] if len(acct_id) >= 4 else acct_id

    bank_match = re.search(r"<BANKID>([^<]+)", content)
    if bank_match:
        account_name = bank_match.group(1).strip()

    # Extract transactions
    stmttrn_pattern = re.compile(
        r"<STMTTRN>(.*?)</STMTTRN>", re.DOTALL | re.IGNORECASE
    )
    for match in stmttrn_pattern.finditer(content):
        block = match.group(1)

        def get_tag(tag: str) -> str:
            m = re.search(rf"<{tag}>([^<\r\n]+)", block, re.IGNORECASE)
            return m.group(1).strip() if m else ""

        date_raw = get_tag("DTPOSTED") or get_tag("DTUSER")
        # OFX dates: YYYYMMDD or YYYYMMDDHHMMSS
        if date_raw:
            date_raw = date_raw[:8]
            date_str = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
        else:
            date_str = ""

        amount = _normalize_amount(get_tag("TRNAMT"))
        desc = get_tag("NAME") or get_tag("MEMO") or get_tag("PAYEE")
        trn_type = get_tag("TRNTYPE").upper()

        # OFX DEBIT type means money out (negative)
        if trn_type == "DEBIT" and amount > 0:
            amount = -amount
        elif trn_type == "CREDIT" and amount < 0:
            amount = abs(amount)

        rows.append({
            "date": date_str,
            "description": desc,
            "amount": amount,
            "type": "credit" if amount >= 0 else "debit",
            "balance": 0.0,
            "account_last4": last4,
            "account_name": account_name,
            "source_file": path.name,
            "category": "",
            "subcategory": "",
            "confidence": 0.0,
        })

    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS) if rows else _empty_df()


# ── Image (OCR) Parser ───────────────────────────────────────────────────────

def _parse_image(path: Path) -> pd.DataFrame:
    """Parse an image statement using OCR (pytesseract)."""
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        logger.warning("pytesseract/Pillow not installed — skipping image %s", path.name)
        return _empty_df()

    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        # Convert to fake CSV in memory and reuse line parser
        tmp_path = path.with_suffix(".ocr_tmp.csv")
        # Write OCR text as a single-column CSV and use PDF-style line parsing
        rows = []
        date_pattern = re.compile(
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w{3,9}\s+\d{1,2},?\s+\d{4})\b"
        )
        amount_pattern = re.compile(r"-?\$?[\d,]+\.\d{2}")
        for line in text.split("\n"):
            dates = date_pattern.findall(line)
            amounts = amount_pattern.findall(line.replace(",", ""))
            if dates and amounts:
                date_str = dates[0] if isinstance(dates[0], str) else " ".join(dates[0])
                amount = _normalize_amount(amounts[-1])
                desc = re.sub(r"\s+", " ", line).strip()
                rows.append({
                    "date": _normalize_date(date_str),
                    "description": desc,
                    "amount": amount,
                    "type": "credit" if amount >= 0 else "debit",
                    "balance": 0.0,
                    "account_last4": "",
                    "account_name": "",
                    "source_file": path.name,
                    "category": "",
                    "subcategory": "",
                    "confidence": 0.0,
                })
        return pd.DataFrame(rows, columns=UNIFIED_COLUMNS) if rows else _empty_df()
    except Exception as exc:
        logger.warning("Image OCR error %s: %s", path.name, exc)
        return _empty_df()


# ── Main Parser Dispatcher ───────────────────────────────────────────────────

def parse_file(path: Path) -> pd.DataFrame:
    """Auto-detect format and parse a single statement file."""
    suffix = path.suffix.lower()
    parsers = {
        ".csv": _parse_csv,
        ".xlsx": _parse_excel,
        ".xls": _parse_excel,
        ".pdf": _parse_pdf,
        ".ofx": _parse_ofx,
        ".qbo": _parse_ofx,
        ".qfx": _parse_ofx,
        ".jpg": _parse_image,
        ".jpeg": _parse_image,
        ".png": _parse_image,
    }

    parser_fn = parsers.get(suffix)
    if parser_fn is None:
        # Sniff content
        try:
            sample = path.read_bytes()[:2048]
            if b"OFXHEADER" in sample or b"<OFX>" in sample:
                parser_fn = _parse_ofx
            elif b"%PDF" in sample:
                parser_fn = _parse_pdf
            else:
                parser_fn = _parse_csv
        except Exception:
            return _empty_df()

    try:
        df = parser_fn(path)
    except Exception as exc:
        logger.error("Failed to parse %s: %s", path.name, exc)
        return _empty_df()

    # Clean and validate
    if df.empty:
        return df

    # Deduplicate columns (Excel files sometimes have repeated headers)
    df = df.loc[:, ~df.columns.duplicated()]

    df = df[df["date"].astype(str).str.len() > 0]
    df = df[df["description"].astype(str).str.len() > 0]
    df["_hash"] = [
        _make_hash(str(r["date"]), float(r["amount"]) if r["amount"] else 0.0, str(r["description"]))
        for _, r in df.iterrows()
    ]
    df = df.drop_duplicates(subset=["_hash"]).drop(columns=["_hash"])
    df = df.reset_index(drop=True)
    return df


def parse_all_statements(statements_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """Parse all statement files in a directory and return combined DataFrame."""
    files = [
        f for f in statements_dir.iterdir()
        if f.is_file() and f.suffix.lower() in
        {".csv", ".xlsx", ".xls", ".pdf", ".ofx", ".qbo", ".qfx", ".jpg", ".jpeg", ".png"}
    ]

    if not files:
        console.print("[yellow]No statement files found in data/statements/[/yellow]")
        return _empty_df()

    frames = []
    for f in track(files, description="Parsing statements..."):
        console.print(f"  Parsing [cyan]{f.name}[/cyan]...")
        df = parse_file(f)
        if not df.empty:
            frames.append(df)
            console.print(f"  [green]OK[/green] {len(df)} transactions from {f.name}")
        else:
            console.print(f"  [yellow]WARN[/yellow] No transactions extracted from {f.name}")

    if not frames:
        return _empty_df()

    combined = pd.concat(frames, ignore_index=True)

    # Global dedup across all statements
    combined["_hash"] = combined.apply(
        lambda r: _make_hash(r["date"], r["amount"], r["description"]), axis=1
    )
    before = len(combined)
    combined = combined.drop_duplicates(subset=["_hash"]).drop(columns=["_hash"])
    dupes = before - len(combined)
    if dupes:
        console.print(f"[yellow]Removed {dupes} duplicate transactions across statements[/yellow]")

    combined = combined.sort_values("date").reset_index(drop=True)

    out_path = processed_dir / "all_transactions.csv"
    combined.to_csv(out_path, index=False)
    console.print(f"[green]Saved {len(combined)} transactions to {out_path}[/green]")
    return combined
