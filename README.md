# Roof Smart Finance — Bank Reconciliation & Financial Intelligence System

A local, AI-powered financial intelligence system built specifically for Roof Smart. Drop in your bank statements and get instant reconciliation, AI-categorized transactions, P&L reconstruction, cash flow forecasts, and beautiful dashboards — all running on your own machine with no data leaving it (except Claude API calls for categorization).

---

## Setup Instructions

### 1. Install Python 3.10+
Download from https://python.org/downloads — check "Add Python to PATH" during install.

### 2. Install Dependencies
```bash
cd roofsmart-finance
pip install -r requirements.txt
```

### 3. Set Your Anthropic API Key
The AI categorization requires a Claude API key. Set it as an environment variable:

**Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=sk-ant-...your-key-here...
```

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-...your-key-here..."
```

**To make it permanent:** Search "Environment Variables" in Windows → System Properties → Environment Variables → New User Variable.

Get your API key at https://console.anthropic.com

---

## Adding Bank Statements

Place your exported statement files in the `data/statements/` folder. The system auto-detects the format.

**Supported formats:**
- `.csv` — CSV exports from any bank portal
- `.xlsx` / `.xls` — Excel exports
- `.pdf` — PDF bank statements (any bank)
- `.ofx` / `.qbo` / `.qfx` — QuickBooks/Quicken format
- `.jpg` / `.png` — Images of statements (requires Tesseract OCR)

---

## How to Run

### Launch Web Dashboard (recommended)
```bash
python main.py dashboard
```
Then open your browser to **http://localhost:8501**

### Other Commands
```bash
python main.py ingest              # Process all files in data/statements/
python main.py ingest --file X     # Process a single file
python main.py report              # Generate all reports (PDF + Excel)
python main.py alerts              # Print current alerts to terminal
python main.py status              # Show account balances + last update
```

---

## File Format Guide

### What to download from each bank:

| Bank | Format | Where to find |
|------|--------|---------------|
| Chase | CSV | Accounts → Download → Activity (.csv) |
| Bank of America | CSV or OFX | Accounts → Download → Set date range |
| Wells Fargo | CSV | Account Activity → Download |
| Truist | CSV | Statements & Documents → Export |
| Capital One | CSV | Transactions → Download |
| QuickBooks | QBO/IIF | Reports → Export |
| Any bank | PDF | Download your monthly statement PDF |

**Pro tip:** For the most complete data, download both CSV (for transactions) and PDF (for statements with running balances) from each account.

---

## Category Reference

| Category | What it includes |
|----------|-----------------|
| **REVENUE** | Job deposits, final payments, insurance checks, supplements |
| **COGS** | Materials (shingles, gutters, flashing), subcontractor labor, equipment rental, permits |
| **OVERHEAD** | Office rent, utilities, insurance (GL/WC/vehicle), software subscriptions |
| **PAYROLL** | Owner draws, W2 employees, 1099 contractors |
| **VEHICLES** | Fuel, maintenance, truck payments, registration |
| **MARKETING** | Google Ads, Facebook, door hangers, lead services (HomeAdvisor, Angi, Thumbtack) |
| **EQUIPMENT** | Tools, ladders, safety gear, machinery |
| **TAXES** | Estimated taxes, sales tax, payroll tax |
| **TRANSFERS** | Internal account transfers (flagged, not double-counted) |
| **UNKNOWN** | Needs manual review (confidence < 70%) |

---

## Correcting Wrong Categories

**Via the Dashboard:**
1. Open the **Transactions** page
2. Click the **Category** cell on any row to change it
3. Click **Save Category Changes**

**Via the Cache:**
Categories are cached in `data/processed/categories_cache.json`. You can edit this file directly — the key is `"description_prefix|amount"` and the value is `{"category": "...", "subcategory": "...", "confidence": 1.0}`.

After correcting categories, re-run `python main.py ingest` to rebuild the reports.

---

## Data Privacy

- All bank statement files stay on your machine in `data/statements/`
- Only transaction descriptions and amounts are sent to the Claude API for categorization
- Categorization results are cached locally — re-runs don't re-call the API
- No data is stored externally

---

## Troubleshooting

**"No transactions extracted" from a PDF:**
- Some bank PDFs use image-based rendering. Install Tesseract OCR and pytesseract for image support.
- Try exporting as CSV from your bank portal instead.

**Categories all showing UNKNOWN:**
- Check that `ANTHROPIC_API_KEY` is set: `echo $env:ANTHROPIC_API_KEY` (PowerShell)
- Verify your API key at https://console.anthropic.com

**Dashboard won't start:**
- Run `pip install streamlit` and try again
- Make sure you're in the `roofsmart-finance/` directory
