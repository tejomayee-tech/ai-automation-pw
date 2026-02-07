#!/usr/bin/env python3
"""
open_website.py

A minimal, self‑contained Selenium script that:
* Installs required packages at runtime (selenium + webdriver-manager)
* Downloads the correct ChromeDriver for the locally installed Chrome
* Opens a website in headless or normal mode
* Prints the page title and closes the browser cleanly

Usage:
    python open_website.py               # opens https://www.example.com (default)
    python open_website.py -u <url>       # custom URL
    python open_website.py --headless     # run Chrome headlessly
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

# --------------------------------------------------------------------------- #
# Helper: install a pip package if it's missing (runs only once per session)
# --------------------------------------------------------------------------- #
def ensure_package(pkg_name: str, version: str | None = None) -> None:
    """Install *pkg_name* with optional *version* via pip if not already present."""
    spec = importlib.util.find_spec(pkg_name)
    if spec is None:
        print(f"🔧 Installing missing package: {pkg_name}...")
        cmd = [sys.executable, "-m", "pip", "install"]
        if version:
            cmd.append(f"{pkg_name}=={version}")
        else:
            cmd.append(pkg_name)
        subprocess.check_call(cmd)
    else:
        # Package already importable – nothing to do
        pass


# --------------------------------------------------------------------------- #
# Ensure the required third‑party libraries are available
# --------------------------------------------------------------------------- #
ensure_package("selenium")          # Selenium 4.x (latest)
ensure_package("webdriver_manager") # webdriver-manager

# Now we can safely import them
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# webdriver‑manager will fetch the correct driver binary for us
from webdriver_manager.chrome import ChromeDriverManager

# --------------------------------------------------------------------------- #
# Argument parsing (very lightweight – no external CLI library needed)
# --------------------------------------------------------------------------- #
import argparse

parser = argparse.ArgumentParser(description="Open a website with Selenium + Chrome.")
parser.add_argument(
    "-u",
    "--url",
    default="https://www.google.com",
    help="URL to open (default: https://www.google.com)",
)
parser.add_argument(
    "--headless",
    action="store_true",
    help="Run Chrome in head‑less mode (no UI). Useful for CI / servers.",
)

args = parser.parse_args()


# --------------------------------------------------------------------------- #
# Build Chrome options
# --------------------------------------------------------------------------- #
chrome_options = webdriver.ChromeOptions()
if args.headless:
    chrome_options.add_argument("--headless=new")  # new headless mode (Chrome 109+)
    # In headless you often want to disable GPU and set a window size
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

# Optional: avoid “Chrome is being controlled by automated test software” banner
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)

# --------------------------------------------------------------------------- #
# Create the driver (webdriver‑manager handles binary download)
# --------------------------------------------------------------------------- #
service = ChromeService(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)


try:
    # ------------------------------------------------------------------- #
    # Navigate to the target URL
    # ------------------------------------------------------------------- #
    print(f"🚀 Opening {args.url}")
    driver.get(args.url)

    # Wait until the document is fully loaded (readyState == "complete")
    WebDriverWait(driver, 10).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )

    # ------------------------------------------------------------------- #
    # Demonstration – print page title and optionally do something else
    # ------------------------------------------------------------------- #
    print(f"✅ Page title: {driver.title}")

    # Example of waiting for a specific element (uncomment if needed)
    # elem = WebDriverWait(driver, 10).until(
    #     EC.presence_of_element_located((By.TAG_NAME, "h1"))
    # )
    # print("First <h1> text:", elem.text)

finally:
    # ------------------------------------------------------------------- #
    # Clean shutdown – always quit the driver
    # ------------------------------------------------------------------- #
    print("🛑 Closing browser")
    driver.quit()
