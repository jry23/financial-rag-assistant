"""Download the most recent 10-K from SEC EDGAR.

EDGAR requires a User-Agent header identifying the requester. Set SEC_USER_AGENT
env var with a real name/email for production use.
"""
from pathlib import Path
from sec_edgar_downloader import Downloader

from configs.config import TICKER, FORM_TYPE, DATA_RAW, SEC_USER_AGENT


def download_latest_10k(ticker: str = TICKER) -> Path:
    """Downloads the most recent 10-K filing and returns the path to the primary HTML doc."""
    # sec_edgar_downloader writes into <dest>/sec-edgar-filings/<TICKER>/<FORM>/<accession>/
    dl = Downloader(
        company_name=SEC_USER_AGENT.split()[0] if SEC_USER_AGENT else "FinancialRAG",
        email_address=SEC_USER_AGENT.split()[-1] if "@" in SEC_USER_AGENT else "research@example.com",
        download_folder=str(DATA_RAW),
    )

    print(f"Downloading most recent {FORM_TYPE} for {ticker}...")
    dl.get(FORM_TYPE, ticker, limit=1)

    # Find the filing directory
    filings_dir = DATA_RAW / "sec-edgar-filings" / ticker / FORM_TYPE
    if not filings_dir.exists():
        raise FileNotFoundError(f"No filings downloaded to {filings_dir}")

    # Most recent accession folder
    accession_dirs = sorted([d for d in filings_dir.iterdir() if d.is_dir()])
    if not accession_dirs:
        raise FileNotFoundError(f"No accession folders in {filings_dir}")
    latest = accession_dirs[-1]

    # The primary document is usually `primary-document.html` or the largest .htm file.
    candidates = list(latest.glob("*.htm*"))
    if not candidates:
        # fallback: full-submission.txt
        candidates = list(latest.glob("*.txt"))
    if not candidates:
        raise FileNotFoundError(f"No filing documents in {latest}")

    # pick largest — primary doc is the meaty one
    primary = max(candidates, key=lambda p: p.stat().st_size)
    print(f"Primary document: {primary}  ({primary.stat().st_size // 1024} KB)")
    return primary


if __name__ == "__main__":
    path = download_latest_10k()
    print(f"\nDownloaded: {path}")
