#!/usr/bin/env python3
"""
open_website_playwright.py

* Installs Playwright (if not present) and its browsers.
* Opens a URL in Chromium (headless optional).
* Prints the page title, takes an optional screenshot, then exits cleanly.
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

# --------------------------------------------------------------
# Helper: install a pip package if missing (run once per session)
# --------------------------------------------------------------
def ensure_pkg(name: str, version: str | None = None):
    if importlib.util.find_spec(name) is None:
        cmd = [sys.executable, "-m", "pip", "install"]
        cmd.append(f"{name}=={version}" if version else name)
        subprocess.check_call(cmd)

ensure_pkg("playwright")          # installs the Python client
# After installing the client we need the browser binaries:
subprocess.run([sys.executable, "-m", "playwright", "install"], check=True)

# --------------------------------------------------------------
# Imports (now safe)
# --------------------------------------------------------------
from playwright.sync_api import sync_playwright

import argparse

parser = argparse.ArgumentParser(description="Open a site with Playwright")
parser.add_argument("-u", "--url", default="https://www.google.com",
                    help="URL to open")
parser.add_argument("--headless", action="store_true",
                    help="Run Chromium headlessly")
parser.add_argument("--screenshot", metavar="FILE",
                    help="Save a screenshot to the given path")
args = parser.parse_args()

# --------------------------------------------------------------
# Playwright usage
# --------------------------------------------------------------
with sync_playwright() as p:
    # Choose Chromium; you could also use .firefox or .webkit
    browser = p.chromium.launch(headless=args.headless)
    context = browser.new_context()
    page = context.new_page()

    print(f"🚀 Navigating to {args.url}")
    page.goto(args.url, wait_until="load")   # waits for DOMContentLoaded + network idle

    # Playwright auto‑waits for the title to be available
    print("✅ Page title:", page.title())

    if args.screenshot:
        page.screenshot(path=args.screenshot)
        print(f"📸 Screenshot saved to {args.screenshot}")

    browser.close()
