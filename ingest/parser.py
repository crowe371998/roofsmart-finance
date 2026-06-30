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

def _parse_excel(path: Path) -> pd.DataFrame:
    """Parse an Excel bank export."""
    try:
        xl = pd.ExcelFile(path)
        frames = []
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            if len(df.columns) >= 2:
                frames.append(df)
        if not frames:
            return _empty_df()
        # Try the largest sheet
        df = max(frames, key=len)
    except Exception as exc:
        logger.warning("Excel read error %s: %s", path.name, exc)
        return _empty_df()

    # Save to temp CSV and reuse CSV parser logic
    tmp = path.with_suffix(".tmp_csv")
    try:
        df.to_csv(tmp, index=False)
        result = _parse_csv(tmp)
        result["source_file"] = path.name
        return result
    finally:
        if tmp.exists():
            tmp.unlink()


# ── PDF Parsers ──────────────────────────────────────────────────────────────

def _parse_pdf(path: Path) -> pd.DataFrame:
    """Parse a PDF bank statement using pdfplumber."""
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        logger.warning("pdfplumber not installed — skipping PDF %s", path.name)
        return _empty_df()

    rows = []
    date_pattern = re.compile(
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w{3,9}\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})\b"
    )
    amount_pattern = re.compile(r"-?\$?[\d,]+\.\d{2}")
    account_pattern = re.compile(r"(?:x+|[*]+|ending in)\s*(\d{4})", re.IGNORECASE)

    last4 = ""
    account_name = ""

    try:
        with pdfplumber.open(path) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

            # Try to extract account number
            m = account_pattern.search(full_text)
            if m:
                last4 = m.group(1)

            # Try to detect bank/account name from header
            first_lines = full_text[:500].split("\n")
            for line in first_lines[:5]:
                if any(w in line.lower() for w in ["checking", "savings", "credit", "account"]):
                    account_name = line.strip()[:50]
                    break

            for page in pdf.pages:
                # Try table extraction first
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    for table_row in table[1:]:  # skip header row
                        if not table_row:
                            continue
                        cells = [str(c or "").strip() for c in table_row]
                        # Need at least date + description + amount
                        if len(cells) < 3:
                            continue

                        # Find date cell
                        date_cell = ""
                        for c in cells:
                            if date_pattern.search(c):
                                date_cell = c
                                break
                        if not date_cell:
                            continue

                        # Find amounts
                        amounts = []
                        for c in cells:
                            matches = amount_pattern.findall(c.replace(",", ""))
                            amounts.extend(matches)

                        # Description: non-date, non-amount cell
                        desc = ""
                        for c in cells:
                            if c != date_cell and c not in amounts and len(c) > 3:
                                desc = c
                                break

                        if not amounts:
                            continue

                        # Determine debit/credit from position
                        amount_vals = [_normalize_amount(a) for a in amounts]
                        if len(amount_vals) == 1:
                            amount = amount_vals[0]
                        elif len(amount_vals) == 2:
                            # debit column, credit column
                            amount = amount_vals[1] if amount_vals[0] == 0 else -abs(amount_vals[0])
                        else:
                            amount = amount_vals[0]
                            balance = amount_vals[-1]

                        balance = amount_vals[-1] if len(amount_vals) > 1 else 0.0

                        rows.append({
                            "date": _normalize_date(date_cell),
                            "description": desc,
                            "amount": amount,
                            "type": "credit" if amount >= 0 else "debit",
                            "balance": balance,
                            "account_last4": last4,
                            "account_name": account_name,
                            "source_file": path.name,
                            "category": "",
                            "subcategory": "",
                            "confidence": 0.0,
                        })

                # Fallback: line-by-line parse if no tables found
                if not rows:
                    text = page.extract_text() or ""
                    for line in text.split("\n"):
                        dates = date_pattern.findall(line)
                        amounts = amount_pattern.findall(line.replace(",", ""))
                        if dates and amounts:
                            date_str = dates[0] if isinstance(dates[0], str) else " ".join(dates[0])
                            amount_str = amounts[-1]
                            # Remove date and amounts from line to get description
                            desc = line
                            for d in dates:
                                desc = desc.replace(d if isinstance(d, str) else " ".join(d), "")
                            for a in amounts:
                                desc = desc.replace(a, "")
                            desc = re.sub(r"\s+", " ", desc).strip()

                            amount = _normalize_amount(amount_str)
                            rows.append({
                                "date": _normalize_date(date_str),
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
    except Exception as exc:
        logger.warning("PDF parse error %s: %s", path.name, exc)
        return _empty_df()

    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS) if rows else _empty_df()


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
    console.print(f"[green]Saved {len(combined)} transactions → {out_path}[/green]")
    return combined
