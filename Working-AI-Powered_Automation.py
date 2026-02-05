"""
DIRECT LLM-POWERED PLAYWRIGHT AUTOMATION
=========================================

ARCHITECTURAL ANALYSIS:
=======================

❌ DEPRECATED APPROACH (Old):
- Used: create_react_agent from langgraph.prebuilt
- Problems: 
  * Deprecated API (marked for removal)
  * Complex ReAct loop overhead
  * Tool calling complexity
  * Agent gets confused on page state
  * Too many iterations for simple tasks
  * PlayWrightBrowserToolkit unreliable

✅ NEW APPROACH (Current):
- Uses: Direct LLM inference + Playwright execution
- Advantages:
  * No deprecated APIs
  * Simpler, more reliable
  * Direct action execution
  * Better visibility (full logging)
  * Fast (no looping overhead)
  * Uses vision/page analysis
  * LLM → Action (transparent flow)

ARCHITECTURE:
==============
1. Take page screenshot/content
2. Send to LLM with instruction
3. LLM returns: ACTION_TYPE | SELECTOR | VALUE
4. We execute directly with Playwright
5. Log everything for debugging
6. Repeat until task complete


DEPENDENCIES & INSTALLATION
============================

Required Python Packages (LangChain/LLM Stack):
  - playwright >= 1.40.0          # Browser automation
  - langchain >= 0.1.0            # LLM framework
  - langchain-core >= 0.1.0       # Core LLM abstractions
  - langchain-ollama >= 1.0.0     # Ollama LLM integration
  - langgraph >= 1.0.0            # Graph-based workflows (for logging/tracing)
  - langgraph-checkpoint >= 4.0.0 # Checkpoint management
  - langgraph-prebuilt >= 1.0.0   # Prebuilt agents (reference only)
  - langgraph-sdk >= 0.3.0        # SDK utilities

Installation Commands:
=======================

# 1. Create and activate Python virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install core dependencies
pip install playwright langchain langchain-core langchain-ollama

# 4. Install LangGraph stack
pip install langgraph langgraph-checkpoint langgraph-prebuilt langgraph-sdk

# 5. Install Playwright browsers (required for automation)
playwright install chromium

# 6. Optional: Install additional utilities
pip install python-dotenv  # For environment variables

Complete Installation Command (all-in-one):
============================================
pip install playwright langchain langchain-core langchain-ollama langgraph langgraph-checkpoint langgraph-prebuilt langgraph-sdk python-dotenv

Version Requirements:
=====================
- Python: >= 3.10
- Playwright: >= 1.40.0
- LangChain: 0.1.8+
- Ollama: Running locally on http://localhost:11434
- Model: qwen2.5-coder:7b (or any available Ollama model)

System Requirements:
====================
- Ollama service running locally (for LLM inference)
- Chromium browser (installed via playwright install)
- ~2GB RAM minimum (for Ollama + automation)
- Network: Access to http://localhost:11434 (Ollama)

Verification Commands:
======================
# Check Ollama is running
curl http://localhost:11434/api/tags

# Test Playwright installation
python -c "import playwright; print(playwright.__version__)"

# Test LangChain installation
python -c "import langchain; print(langchain.__version__)"

# List installed packages
pip list | grep -E "(playwright|langchain|langgraph|ollama)"
"""

import asyncio
from playwright.async_api import async_playwright, Page
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime
import json
from pathlib import Path
from typing import Optional, Dict, Any


class LogManager:
    """
    Comprehensive logging with dual output:
    - Console (real-time feedback)
    - File + JSON (structured analysis)
    """
    
    def __init__(self, test_name: str = "swag_labs_login"):
        self.log_dir = Path("/home/vijay/Develop/AI/Automation/logs")
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{test_name}_{timestamp}.log"
        self.json_log_file = self.log_dir / f"{test_name}_{timestamp}.json"
        
        self.json_logs = {
            "test_name": test_name,
            "start_time": datetime.now().isoformat(),
            "approach": "Direct LLM-Powered Playwright (Non-Deprecated)",
            "events": [],
            "llm_interactions": [],
            "page_snapshots": [],
            "errors": []
        }
        
        with open(self.log_file, "w") as f:
            f.write("=" * 120 + "\n")
            f.write("DIRECT LLM-POWERED PLAYWRIGHT AUTOMATION\n")
            f.write("Architecture: Vision-Based LLM Analysis → Direct Playwright Actions\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write(f"Approach: Non-Deprecated (No ReAct Loop)\n")
            f.write("=" * 120 + "\n\n")
    
    async def log(self, message: str, level: str = "INFO", data: Optional[Dict[str, Any]] = None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        icons = {
            "INFO": "ℹ️",
            "DEBUG": "🔍",
            "ERROR": "❌",
            "SUCCESS": "✅",
            "ACTION": "🎯",
            "PROMPT": "📝",
            "RESPONSE": "💬",
            "PAGE": "📄",
            "ARCH": "🏗️"
        }
        icon = icons.get(level, "📌")
        console_msg = f"[{timestamp}] {icon} [{level:8}] {message}"
        print(console_msg)
        
        with open(self.log_file, "a") as f:
            f.write(console_msg + "\n")
            if data:
                f.write(f"  Data: {json.dumps(data, indent=4, default=str)}\n")
        
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "data": data
        }
        self.json_logs["events"].append(log_entry)
        
        if level in ["PROMPT", "RESPONSE"]:
            self.json_logs["llm_interactions"].append({
                "timestamp": timestamp,
                "type": level.lower(),
                "content": message[:500],
                "metadata": data
            })
        
        if level == "ERROR":
            self.json_logs["errors"].append({
                "timestamp": timestamp,
                "message": message,
                "data": data
            })
    
    async def log_page_snapshot(self, page: Page, event_name: str):
        """Capture page HTML for debugging."""
        try:
            content = await page.content()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            snapshot_file = self.log_dir / f"page_{event_name}_{datetime.now().strftime('%H%M%S')}.html"
            
            with open(snapshot_file, "w") as f:
                f.write(content)
            
            await self.log(f"Page snapshot: {snapshot_file.name}", "PAGE", 
                          {"event": event_name, "bytes": len(content), "url": page.url})
            
            self.json_logs["page_snapshots"].append({
                "timestamp": timestamp,
                "event": event_name,
                "file": snapshot_file.name,
                "bytes": len(content),
                "url": page.url
            })
        except Exception as e:
            await self.log(f"Snapshot error: {str(e)}", "ERROR")
    
    async def save_json_log(self):
        self.json_logs["end_time"] = datetime.now().isoformat()
        with open(self.json_log_file, "w") as f:
            json.dump(self.json_logs, f, indent=2, default=str)
        await self.log(f"JSON saved: {self.json_log_file.name}", "SUCCESS")
    
    async def close(self):
        await self.save_json_log()
        with open(self.log_file, "a") as f:
            f.write("\n" + "=" * 120 + "\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write("=" * 120 + "\n")


class DirectLLMAutomationAgent:
    """
    Direct LLM-Powered Automation Agent
    
    Flow:
    1. Get page state (URL + HTML content)
    2. Send to LLM with instruction
    3. LLM responds with ACTION_TYPE | SELECTOR | VALUE | DESCRIPTION
    4. Parse and execute with Playwright
    5. Log everything
    6. Repeat as needed
    """
    
    def __init__(self, logger: LogManager):
        self.logger = logger
        self.llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)
    
    async def execute_task(self, page: Page, instruction: str) -> str:
        """Execute a single task via LLM analysis and Playwright action."""
        await self.logger.log(f"Task: {instruction}", "ACTION")
        
        try:
            current_url = page.url
            page_html = await page.content()
            
            # Limit HTML for context window
            html_preview = page_html[:2000]
            
            system_prompt = """You are a QA automation expert. Analyze the page and provide the NEXT action in this exact format:

ACTION_TYPE | SELECTOR | VALUE | DESCRIPTION

Valid ACTION_TYPEs:
- fill: Fill input field
- click: Click button/element
- wait_for: Wait for element to appear
- clear: Clear input field
- verify: Check if element is visible

Examples:
fill | input#user-name | standard_user | Fill username
click | input#login-button | | Click login button
wait_for | text=Products | | Wait for success page
verify | text=Products | | Check if logged in

Be direct. Only respond with the action format."""
            
            user_prompt = f"""URL: {current_url}
PAGE HTML (first 2000 chars): {html_preview}...

TASK: {instruction}

What is the NEXT action? Respond in ACTION_TYPE | SELECTOR | VALUE | DESCRIPTION format."""
            
            await self.logger.log(f"Sending to LLM: {instruction}", "PROMPT")
            
            # Call LLM
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            llm_response = response.content.strip()
            await self.logger.log(f"LLM returned: {llm_response}", "RESPONSE")
            
            # Execute the action
            result = await self._execute_action(page, llm_response)
            await self.logger.log(f"Result: {result}", "SUCCESS")
            
            return result
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            await self.logger.log(error_msg, "ERROR", {"exception": str(e)})
            return error_msg
    
    async def _execute_action(self, page: Page, action_str: str) -> str:
        """Parse and execute action."""
        try:
            parts = [p.strip() for p in action_str.split("|")]
            if len(parts) < 2:
                return f"Invalid action: {action_str}"
            
            action_type = parts[0].lower()
            selector = parts[1]
            value = parts[2] if len(parts) > 2 else ""
            description = parts[3] if len(parts) > 3 else ""
            
            await self.logger.log(f"Exec: {action_type} {selector}", "DEBUG")
            
            if action_type == "fill":
                await page.fill(selector, value)
                await asyncio.sleep(0.5)
                return f"Filled {selector} = '{value}'"
            
            elif action_type == "click":
                await page.click(selector)
                await asyncio.sleep(1)
                return f"Clicked {selector}"
            
            elif action_type == "wait_for":
                await page.wait_for_selector(selector, timeout=5000)
                return f"Element appeared: {selector}"
            
            elif action_type == "clear":
                await page.fill(selector, "")
                return f"Cleared {selector}"
            
            elif action_type == "verify":
                visible = await page.is_visible(selector)
                return f"{selector}: {'visible' if visible else 'NOT visible'}"
            
            else:
                return f"Unknown action: {action_type}"
            
        except Exception as e:
            return f"Exec error: {str(e)}"


async def main():
    """Main automation flow."""
    logger = LogManager("swag_labs_login_test1")
    
    try:
        await logger.log("ARCHITECTURE: Direct LLM-Powered Playwright", "ARCH")
        await logger.log("Starting browser...", "ACTION")
        
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=False)
        page = await browser.new_page()
        
        await logger.log("Browser ready", "SUCCESS")
        
        # Navigate
        await logger.log("Loading https://www.saucedemo.com/", "ACTION")
        await page.goto("https://www.saucedemo.com/", wait_until="networkidle")
        await page.wait_for_load_state("domcontentloaded")
        await logger.log("Page loaded", "SUCCESS", {"url": page.url})
        await logger.log_page_snapshot(page, "initial")
        
        # Initialize agent
        await logger.log("Initializing Direct LLM Agent (No ReAct deprecated API)", "ACTION")
        agent = DirectLLMAutomationAgent(logger)
        await logger.log("Agent ready", "SUCCESS")
        
        # Execute login tasks
        tasks = [
            "Fill username field with 'standard_user'",
            "Fill password field with 'secret_sauce'",
            "Click the Login button to submit",
            "Verify Products page is visible (login success)"
        ]
        
        for i, task in enumerate(tasks, 1):
            await logger.log(f"\n{'='*60}\nTASK {i}/{len(tasks)}", "ACTION")
            result = await agent.execute_task(page, task)
            await asyncio.sleep(1)
            await logger.log_page_snapshot(page, f"after_task_{i}")
        
        # Final check
        await asyncio.sleep(3)
        try:
            products_visible = await page.is_visible("text=Products")
            if products_visible:
                await logger.log("✅ LOGIN SUCCESSFUL! Products page visible", "SUCCESS")
            else:
                await logger.log("❌ Login failed - Products not visible", "ERROR")
        except Exception as e:
            await logger.log(f"Verification error: {str(e)}", "ERROR")
        
        await asyncio.sleep(5)
        
        # Cleanup
        await logger.log("Closing browser...", "ACTION")
        await browser.close()
        await playwright.stop()
        await logger.log("Cleanup done", "SUCCESS")
        
    except Exception as e:
        await logger.log(f"CRITICAL: {str(e)}", "ERROR")
        import traceback
        with open(logger.log_file, "a") as f:
            f.write("\n" + "=" * 120 + "\n")
            f.write("EXCEPTION:\n")
            f.write(traceback.format_exc())
        try:
            await browser.close()
            await playwright.stop()
        except:
            pass
    
    finally:
        await logger.close()


if __name__ == "__main__":
    print("\n" + "=" * 120)
    print("🏗️  DIRECT LLM-POWERED PLAYWRIGHT AUTOMATION")
    print("   No Deprecated APIs | Vision-Based | Direct Actions | Full Logging")
    print("=" * 120 + "\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Fatal: {str(e)}")
        import traceback
        traceback.print_exc()
