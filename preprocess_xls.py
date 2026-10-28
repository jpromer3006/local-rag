#!/usr/bin/env python3
"""Preprocess .xls/.xlsx financial files into clean structured text for RAG.

Reads files from RAG_DOCS_DIR (default ~/Desktop/Germany), extracts key
financial sheets (Income Statement, Balance Sheet, Cash Flow, Key Stats,
Ratios, Segments), and writes clean text summaries to RAG_PROCESSED_DIR
(default ~/Desktop/Germany_processed).

Usage:
    python preprocess_xls.py                      # process all files
    python preprocess_xls.py --file "path/to.xls" # process one file
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Config — overridden by .env if present
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

def _resolve_docs_dir() -> Path:
    """Resolve the docs directory, handling the trailing-space folder name."""
    env_val = os.getenv("RAG_DOCS_DIR", "")
    if env_val:
        p = Path(env_val).expanduser()
        if p.exists():
            return p
        # Try with trailing space (common macOS quirk)
        p_space = Path(str(p) + " ")
        if p_space.exists():
            return p_space
        return p
    return Path.home() / "Desktop" / "Germany "


DOCS_DIR = _resolve_docs_dir()
PROCESSED_DIR = Path(os.getenv("RAG_PROCESSED_DIR", os.path.expanduser("~/Desktop/Germany_processed")))

TARGET_SHEETS = [
    "Income Statement",
    "Balance Sheet",
    "Cash Flow",
    "Key Stats",
    "Ratios",
    "Segments",
]

# Key financial line items to extract from each sheet
KEY_ITEMS = {
    "Income Statement": [
        "Total Revenue", "Revenue", "Gross Profit", "Operating Income",
        "EBITDA", "EBIT", "Net Income", "Net Income to Common",
        "EPS (Basic)", "EPS Basic", "EPS (Diluted)", "EPS Diluted",
        "Selling General & Admin", "Cost Of Goods Sold",
        "Depreciation & Amort", "Interest Expense", "Income Tax",
    ],
    "Balance Sheet": [
        "Total Assets", "Total Current Assets", "Cash & Short Term",
        "Cash & Equivalents", "Total Debt", "Long Term Debt",
        "Total Current Liabilities", "Total Liabilities",
        "Total Equity", "Total Common Equity", "Book Value Per Share",
        "Accounts Receivable", "Inventories", "Total Shareholders",
    ],
    "Cash Flow": [
        "Cash from Ops", "Cash from Operations", "Operating Activities",
        "Capital Expenditure", "Capex", "Free Cash Flow",
        "Cash from Investing", "Cash from Financing",
        "Dividends Paid", "Net Change in Cash", "Total Cash",
    ],
    "Key Stats": [
        "Total Revenue", "Revenue", "EBITDA", "EBIT",
        "Net Income", "EPS", "Market Capitalization",
        "Total Enterprise Value", "TEV", "Total Debt",
        "Return on Equity", "ROE", "Return on Assets", "ROA",
        "Dividend Yield", "Current Ratio", "Debt / Equity",
    ],
    "Ratios": [
        "Return on Equity", "ROE", "Return on Assets", "ROA",
        "Net Margin", "Operating Margin", "Gross Margin",
        "Current Ratio", "Quick Ratio", "Debt / Equity",
        "Debt/Equity", "Interest Coverage", "Asset Turnover",
        "Revenue / Employee", "Dividend Yield", "Payout Ratio",
    ],
    "Segments": None,  # capture all non-blank rows
}

# Recent years we care about
RECENT_YEARS = 6  # last N data columns


def parse_company_name(filename: str) -> Tuple[str, str]:
    """Extract company name and ticker from filename like 'KBR Inc NYSE KBR Financials (1).xls'."""
    base = Path(filename).stem
    # Remove trailing " (1)", " (2)" etc
    base = re.sub(r"\s*\(\d+\)\s*$", "", base)
    # Remove trailing "Financials"
    base = re.sub(r"\s*Financials\s*$", "", base, flags=re.I)

    # Try to find exchange:ticker pattern
    m = re.search(r"(NYSE|NASDAQ|NYSEAM)\s+(\S+)", base)
    if m:
        exchange = m.group(1)
        ticker = m.group(2)
        company = base[: m.start()].strip()
        return company, f"{exchange}:{ticker}"

    return base.strip(), ""


def extract_years(period_row: pd.Series) -> List[str]:
    """Extract year labels from the period header row."""
    years = []
    for val in period_row:
        if pd.isna(val):
            years.append("")
            continue
        s = str(val)
        # Look for year patterns like Dec-31-2023, Jan-03-2025, 2024, etc
        m = re.search(r"(\d{4})", s)
        if m:
            years.append(m.group(1))
        else:
            years.append("")
    return years


def clean_value(val) -> str:
    """Clean a cell value — return empty string for NaN, '-', blanks."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s in ("-", "\xa0", "NM", "NA", ""):
        return ""
    # Remove commas from numbers
    s = s.replace(",", "")
    return s


def find_header_row(df: pd.DataFrame) -> int:
    """Find the row containing 'For the Fiscal Period Ending' or 'Fiscal Year'."""
    for i in range(min(20, len(df))):
        first_cell = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""
        if any(kw in first_cell.lower() for kw in
               ["fiscal period", "fiscal year", "for the fiscal", "period ending"]):
            return i
    return -1


def extract_sheet(df: pd.DataFrame, sheet_name: str, key_items: Optional[List[str]]) -> List[str]:
    """Extract financial rows from a sheet DataFrame.

    Captures ALL rows that have numeric data in recent year columns.
    This gives the RAG system maximum context for financial analysis.
    """
    header_idx = find_header_row(df)
    if header_idx < 0:
        # For Segments or sheets without standard headers, just grab non-blank rows
        lines = []
        for i in range(len(df)):
            cells = [clean_value(c) for c in df.iloc[i]]
            label = cells[0] if cells else ""
            vals = [c for c in cells[1:] if c]
            if label and vals:
                lines.append(f"  {label}: {', '.join(vals[-RECENT_YEARS:])}")
        return lines

    years = extract_years(df.iloc[header_idx])

    # Get recent year labels (last N non-empty)
    year_cols = [(idx, yr) for idx, yr in enumerate(years) if yr and idx > 0]
    recent = year_cols[-RECENT_YEARS:] if len(year_cols) > RECENT_YEARS else year_cols

    if not recent:
        return []

    year_header = "  Years: " + ", ".join(yr for _, yr in recent)
    lines = [year_header]

    # Skip section header rows (e.g., "Income Statement", "Currency")
    skip_labels = {"income statement", "balance sheet", "cash flow", "key stats",
                   "ratios", "segments", "currency", "multiples", "supplemental",
                   "for the fiscal period ending", "period type", "order",
                   "in millions", "template", "restatement", "conversion"}

    data_start = header_idx + 1
    for i in range(data_start, len(df)):
        label_raw = df.iloc[i, 0]
        if pd.isna(label_raw):
            continue
        label = str(label_raw).strip()
        if not label or label == "\xa0":
            continue

        # Skip non-data rows
        if any(skip in label.lower() for skip in skip_labels):
            continue

        # Extract values for recent year columns
        vals = []
        has_data = False
        for col_idx, yr in recent:
            v = clean_value(df.iloc[i, col_idx]) if col_idx < len(df.columns) else ""
            if v:
                has_data = True
                vals.append(f"{yr}={v}")
            else:
                vals.append(f"{yr}=-")

        if has_data:
            lines.append(f"  {label}: {', '.join(vals)}")

    return lines


def process_file(filepath: Path) -> Optional[str]:
    """Process a single .xls/.xlsx file into structured text."""
    company, ticker = parse_company_name(filepath.name)
    ticker_str = f" ({ticker})" if ticker else ""

    ext = filepath.suffix.lower()
    engine = "xlrd" if ext == ".xls" else "openpyxl"

    try:
        xl = pd.ExcelFile(str(filepath), engine=engine)
    except Exception as e:
        print(f"  ERROR reading {filepath.name}: {e}")
        return None

    available = xl.sheet_names
    sections: List[str] = []

    sections.append(f"COMPANY: {company}{ticker_str}")
    sections.append(f"Source File: {filepath.name}")
    sections.append("")

    # Detect currency and period from first available target sheet
    for sname in TARGET_SHEETS:
        if sname in available:
            df = xl.parse(sname, header=None)
            for i in range(min(10, len(df))):
                cell = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""
                if "millions" in cell.lower() or "currency" in cell.lower():
                    sections.append(f"Reporting: {cell.strip()}")
                    break
            break

    sections.append("")

    for target_sheet in TARGET_SHEETS:
        if target_sheet not in available:
            continue

        df = xl.parse(target_sheet, header=None)
        key_items = KEY_ITEMS.get(target_sheet)
        lines = extract_sheet(df, target_sheet, key_items)

        if lines:
            sections.append(f"{'='*50}")
            sections.append(f"SHEET: {target_sheet}")
            sections.append(f"{'='*50}")
            sections.extend(lines)
            sections.append("")

    if len(sections) <= 4:  # only header, no data
        return None

    return "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess XLS financial files for RAG")
    parser.add_argument("--file", type=str, help="Process a single file instead of entire directory")
    parser.add_argument("--docs-dir", type=str, default=None, help="Override source directory")
    parser.add_argument("--out-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir) if args.docs_dir else DOCS_DIR
    out_dir = Path(args.out_dir) if args.out_dir else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        files = [Path(args.file)]
    else:
        files = sorted([
            f for f in docs_dir.iterdir()
            if f.suffix.lower() in (".xls", ".xlsx") and not f.name.startswith("~")
        ])

    if not files:
        print(f"No .xls/.xlsx files found in {docs_dir}")
        return 1

    print(f"Processing {len(files)} file(s) from {docs_dir}")
    print(f"Output to {out_dir}")
    print()

    processed = 0
    skipped = 0

    for filepath in files:
        print(f"  Processing: {filepath.name}...")
        result = process_file(filepath)

        if result:
            company, ticker = parse_company_name(filepath.name)
            safe_name = re.sub(r"[^\w\s-]", "", company).strip().replace(" ", "_")
            out_path = out_dir / f"{safe_name}.txt"
            out_path.write_text(result, encoding="utf-8")
            processed += 1

            # Count lines with data (lines starting with spaces that contain year=value)
            data_lines = [l for l in result.split("\n") if l.startswith("  ") and "=" in l and "Years:" not in l]
            print(f"    -> {out_path.name} ({len(data_lines)} data points)")
        else:
            skipped += 1
            print(f"    -> SKIPPED (no extractable data)")

    print()
    print(f"Summary: {processed} files processed, {skipped} skipped")
    print(f"Output directory: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
