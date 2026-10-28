#!/usr/bin/env python3
"""Scan Gmail for emails with subject 'JAIA', download all supported
attachments (Excel, PDF, DOCX, etc.) AND save the email body text,
ingest into LocalRAG, and trigger OpenClaw to activate the LLM."""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Set

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# --- Configuration -----------------------------------------------------------

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
SCRIPT_DIR = Path(__file__).resolve().parent
CREDENTIALS_FILE = SCRIPT_DIR / "gmail_credentials.json"
TOKEN_FILE = SCRIPT_DIR / "gmail_token.json"
PROCESSED_IDS_FILE = SCRIPT_DIR / "gmail_processed_ids.json"
IMPORT_DIR = Path.home() / "CentralizedLibrary" / "gmail_imports"
SUBJECT_FILTER = "JAIA"
OPENCLAW_GATEWAY = "http://127.0.0.1:18789"
OPENCLAW_TOKEN = "ollama"

# File extensions the RAG system can ingest (matches rag_config.json)
SUPPORTED_EXTENSIONS = {
    ".xlsx", ".pdf", ".docx", ".pptx", ".csv", ".txt", ".md",
    ".json", ".jsonl", ".xml", ".html", ".htm",
    ".py", ".js", ".ts", ".java", ".go", ".rs", ".sql", ".log",
}


# --- HTML stripping ----------------------------------------------------------

class _HTMLStripper(HTMLParser):
    """Simple HTML-to-text converter."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def strip_html(html: str) -> str:
    stripper = _HTMLStripper()
    stripper.feed(html)
    return stripper.get_text()


# --- Authentication ----------------------------------------------------------

def authenticate() -> Credentials:
    """Authenticate with Gmail API via OAuth2. Opens browser on first run."""
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    elif not creds or not creds.valid:
        if not CREDENTIALS_FILE.exists():
            print(f"ERROR: OAuth credentials file not found at {CREDENTIALS_FILE}")
            print("Download it from Google Cloud Console and save it there.")
            print("See: https://console.cloud.google.com/apis/credentials")
            sys.exit(1)
        flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
        creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
        print(f"Token saved to {TOKEN_FILE}")

    return creds


# --- Processed IDs tracking --------------------------------------------------

def load_processed_ids() -> Set[str]:
    if PROCESSED_IDS_FILE.exists():
        data = json.loads(PROCESSED_IDS_FILE.read_text(encoding="utf-8"))
        return set(data)
    return set()


def save_processed_ids(ids: Set[str]) -> None:
    PROCESSED_IDS_FILE.write_text(
        json.dumps(sorted(ids), indent=2), encoding="utf-8"
    )


# --- Gmail operations --------------------------------------------------------

def search_jaia_emails(
    service: Any, processed_ids: Set[str]
) -> List[Dict[str, Any]]:
    """Search for unprocessed emails with subject JAIA."""
    query = f"subject:{SUBJECT_FILTER}"
    results = service.users().messages().list(userId="me", q=query).execute()
    messages = results.get("messages", [])

    new_messages = []
    for msg_ref in messages:
        if msg_ref["id"] not in processed_ids:
            msg = (
                service.users()
                .messages()
                .get(userId="me", id=msg_ref["id"])
                .execute()
            )
            new_messages.append(msg)

    return new_messages


def _get_header(message: Dict[str, Any], name: str) -> str:
    headers = message.get("payload", {}).get("headers", [])
    return next(
        (h["value"] for h in headers if h["name"].lower() == name.lower()),
        "",
    )


def _collect_parts(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recursively collect all MIME parts from a message payload."""
    parts: List[Dict[str, Any]] = []
    if "parts" in payload:
        for part in payload["parts"]:
            parts.extend(_collect_parts(part))
    else:
        parts.append(payload)
    return parts


def extract_email_body(message: Dict[str, Any]) -> str:
    """Extract the plain text or HTML body from an email message."""
    payload = message.get("payload", {})
    all_parts = _collect_parts(payload)

    # Prefer plain text, fall back to HTML
    plain_text = ""
    html_text = ""

    for part in all_parts:
        mime_type = part.get("mimeType", "")
        body_data = part.get("body", {}).get("data", "")
        if not body_data:
            continue

        decoded = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")

        if mime_type == "text/plain" and not plain_text:
            plain_text = decoded
        elif mime_type == "text/html" and not html_text:
            html_text = decoded

    if plain_text.strip():
        return plain_text.strip()
    if html_text.strip():
        return strip_html(html_text).strip()
    return ""


def save_email_body(
    message: Dict[str, Any], dest_dir: Path, timestamp: str
) -> Path | None:
    """Save the email subject + body as a .txt file for RAG ingestion."""
    subject = _get_header(message, "subject") or "(no subject)"
    sender = _get_header(message, "from") or "(unknown)"
    date = _get_header(message, "date") or ""
    body = extract_email_body(message)

    if not body:
        return None

    # Build a readable text document from the email
    lines = [
        f"Subject: {subject}",
        f"From: {sender}",
        f"Date: {date}",
        "",
        "--- Email Body ---",
        "",
        body,
    ]
    content = "\n".join(lines)

    safe_subject = re.sub(r"[^\w\s-]", "", subject)[:60].strip().replace(" ", "_")
    filename = f"{timestamp}_email_{safe_subject}.txt"
    dest_path = dest_dir / filename
    dest_path.write_text(content, encoding="utf-8")
    print(f"  Saved email body: {dest_path.name} ({len(content):,} chars)")
    return dest_path


def download_attachments(
    service: Any, message: Dict[str, Any], dest_dir: Path, timestamp: str
) -> List[Path]:
    """Download all supported attachments from a Gmail message."""
    saved: List[Path] = []
    payload = message.get("payload", {})
    all_parts = _collect_parts(payload)

    for part in all_parts:
        filename = part.get("filename", "")
        if not filename:
            continue

        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"  Skipped (unsupported type): {filename}")
            continue

        attachment_id = part.get("body", {}).get("attachmentId")
        if not attachment_id:
            # Small inline attachments have data directly in the body
            body_data = part.get("body", {}).get("data", "")
            if body_data:
                data = base64.urlsafe_b64decode(body_data)
            else:
                continue
        else:
            attachment = (
                service.users()
                .messages()
                .attachments()
                .get(userId="me", messageId=message["id"], id=attachment_id)
                .execute()
            )
            data = base64.urlsafe_b64decode(attachment["data"])

        safe_name = filename.replace("/", "_").replace(" ", "_")
        dest_path = dest_dir / f"{timestamp}_{safe_name}"
        dest_path.write_bytes(data)
        saved.append(dest_path)
        print(f"  Downloaded: {dest_path.name} ({len(data):,} bytes)")

    return saved


# --- RAG ingestion -----------------------------------------------------------

def run_ingest() -> int:
    """Run ingest_centralized.py to index new files into SQLite FTS."""
    print("\nRunning RAG ingestion...")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "ingest_centralized.py"),
            "--config",
            str(SCRIPT_DIR / "rag_config.json"),
            "--report",
            str(SCRIPT_DIR / "latest_ingest_report.json"),
        ],
        cwd=str(SCRIPT_DIR),
    )
    if result.returncode == 0:
        print("Ingestion complete.")
    else:
        print(f"Ingestion failed with exit code {result.returncode}")
    return result.returncode


# --- OpenClaw trigger --------------------------------------------------------

def trigger_openclaw() -> None:
    """Trigger OpenClaw gateway to run ollama with the RAG system."""
    try:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "action": "agent",
            "message": (
                "New data has been ingested into the RAG system from a JAIA email. "
                "Run ollama with llama3.1 to activate the RAG system so the user "
                "can query the new data."
            ),
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{OPENCLAW_GATEWAY}/api/v1/agent",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENCLAW_TOKEN}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"OpenClaw triggered (status {resp.status})")
    except urllib.error.URLError as exc:
        print(f"Could not reach OpenClaw gateway: {exc}")
        print("Falling back to opening Terminal with RAG system...")
        _fallback_terminal_launch()
    except Exception as exc:
        print(f"OpenClaw trigger failed: {exc}")
        print("Falling back to opening Terminal with RAG system...")
        _fallback_terminal_launch()


def _fallback_terminal_launch() -> None:
    """Open a macOS Terminal window with the RAG system as fallback."""
    subprocess.Popen([
        "osascript", "-e",
        'tell application "Terminal" to do script '
        '"cd ~/LocalRAG && source venv/bin/activate && '
        'python3 ask_rag.py --config rag_config.json --interactive --show-sources"',
    ])
    print("Opened Terminal with interactive RAG session.")


# --- Main --------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("Gmail Scanner for JAIA")
    print(f"Scanning for: subject:{SUBJECT_FILTER}")
    print("Captures: email body + all supported attachments")
    print("=" * 60)

    creds = authenticate()
    service = build("gmail", "v1", credentials=creds)
    processed_ids = load_processed_ids()

    IMPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nSearching Gmail for new JAIA emails...")
    new_messages = search_jaia_emails(service, processed_ids)

    if not new_messages:
        print("No new emails found with subject 'JAIA'.")
        return 0

    print(f"Found {len(new_messages)} new email(s).\n")

    total_files = 0
    for msg in new_messages:
        subject = _get_header(msg, "subject") or "(no subject)"
        sender = _get_header(msg, "from") or "(unknown sender)"
        print(f"Processing: \"{subject}\" from {sender}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the email body as a text file
        body_path = save_email_body(msg, IMPORT_DIR, timestamp)
        if body_path:
            total_files += 1

        # Download all supported attachments
        saved = download_attachments(service, msg, IMPORT_DIR, timestamp)
        total_files += len(saved)

        processed_ids.add(msg["id"])
        print()

    save_processed_ids(processed_ids)

    if total_files > 0:
        print(f"{total_files} file(s) saved to {IMPORT_DIR}")
        run_ingest()
        trigger_openclaw()
    else:
        print("No content extracted from the new emails.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
