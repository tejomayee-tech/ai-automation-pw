import asyncio
from playwright.async_api import async_playwright
from langchain_ollama import ChatOllama
# Import the non-deprecated React Agent builder from LangGraph
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from datetime import datetime
import time
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any


class LogManager:
    """
    Comprehensive logging manager that handles both console and file logging.
    Captures all events, prompts, responses, and page data with timestamps.
    """
    
    def __init__(self, test_name: str = "swag_labs_login"):
        """
        Initialize the log manager with file output.
        
        Args:
            test_name: Name of the test for log file naming
        """
        # Create logs directory if it doesn't exist
        self.log_dir = Path("/home/vijay/Develop/AI/Automation/logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{test_name}_{timestamp}.log"
        self.json_log_file = self.log_dir / f"{test_name}_{timestamp}.json"
        
        # Initialize JSON log structure
        self.json_logs = {
            "test_name": test_name,
            "start_time": datetime.now().isoformat(),
            "events": [],
            "prompts": [],
            "responses": [],
            "page_snapshots": [],
            "errors": []
        }
        
        # Write initial log file header
        with open(self.log_file, "w") as f:
            f.write("=" * 100 + "\n")
            f.write(f"TEST EXECUTION LOG - {test_name}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write(f"Log File: {self.log_file}\n")
            f.write(f"JSON Log: {self.json_log_file}\n")
            f.write("=" * 100 + "\n\n")
    
    async def log(self, message: str, level: str = "INFO", data: dict = None):
        """
        Log message to both console and file with structured data.
        
        Args:
            message: Log message
            level: Log level (INFO, DEBUG, ERROR, SUCCESS, ACTION, PROMPT, RESPONSE)
            data: Additional structured data to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        icons = {
            "INFO": "ℹ️",
            "DEBUG": "🔍",
            "ERROR": "❌",
            "SUCCESS": "✅",
            "ACTION": "🎯",
            "PROMPT": "📝",
            "RESPONSE": "💬",
            "PAGE": "📄"
        }
        icon = icons.get(level, "📌")
        
        # Console output
        console_msg = f"[{timestamp}] {icon} [{level:8}] {message}"
        print(console_msg)
        
        # File output
        with open(self.log_file, "a") as f:
            f.write(console_msg + "\n")
            if data:
                f.write(f"  Data: {json.dumps(data, indent=4, default=str)}\n")
        
        # JSON structured logging
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "data": data
        }
        self.json_logs["events"].append(log_entry)
        
        # Category-specific logging
        if level == "PROMPT":
            self.json_logs["prompts"].append({"timestamp": timestamp, "content": message, "metadata": data})
        elif level == "RESPONSE":
            self.json_logs["responses"].append({"timestamp": timestamp, "content": message, "metadata": data})
        elif level == "ERROR":
            self.json_logs["errors"].append({"timestamp": timestamp, "message": message, "data": data})
    
    async def log_page_source(self, page, event_name: str):
        """
        Capture and log page source HTML snapshot.
        
        Args:
            page: Playwright page object
            event_name: Name of the event for identification
        """
        try:
            content = await page.content()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Save to file
            snapshot_file = self.log_dir / f"page_snapshot_{event_name}_{datetime.now().strftime('%H%M%S')}.html"
            with open(snapshot_file, "w") as f:
                f.write(content)
            
            await self.log(f"Page snapshot saved: {snapshot_file.name}", "PAGE", 
                          {"event": event_name, "size_bytes": len(content)})
            
            # Add to JSON log
            self.json_logs["page_snapshots"].append({
                "timestamp": timestamp,
                "event": event_name,
                "file": snapshot_file.name,
                "size_bytes": len(content),
                "url": page.url
            })
        except Exception as e:
            await self.log(f"Error capturing page source: {str(e)}", "ERROR")
    
    async def save_json_log(self):
        """Save structured JSON log to file."""
        self.json_logs["end_time"] = datetime.now().isoformat()
        
        with open(self.json_log_file, "w") as f:
            json.dump(self.json_logs, f, indent=2, default=str)
        
        await self.log(f"JSON log saved: {self.json_log_file}", "SUCCESS")
    
    async def close(self):
        """Finalize logging."""
        await self.save_json_log()
        with open(self.log_file, "a") as f:
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write("=" * 100 + "\n")

async def main():
    """
    Enhanced Playwright + LangGraph Agent for QA Automation.
    
    This script demonstrates:
    1. Proper Playwright browser initialization
    2. Toolkit setup with active page context
    3. LLM-powered agent for semantic task execution
    4. Comprehensive file logging with timestamps
    5. Capture of LLM prompts, responses, and page snapshots
    """
    
    # Initialize logging system
    logger = LogManager("swag_labs_login_test1")
    
    try:
        # ============================================================================
        # STEP 1: Initialize Playwright Browser
        # ============================================================================
        await logger.log("Initializing Playwright browser...", "ACTION")
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=False)
        page = await browser.new_page()
        await logger.log("✓ Browser launched successfully", "SUCCESS")

        # ============================================================================
        # STEP 2: Navigate to Target Website
        # ============================================================================
        await logger.log("Navigating to Swag Labs login page (https://www.saucedemo.com/)...", "ACTION")
        await page.goto("https://www.saucedemo.com/", wait_until="networkidle")
        await logger.log("✓ Page loaded successfully", "SUCCESS", {"url": page.url})
        
        # Give the page a moment to fully render
        await page.wait_for_load_state("domcontentloaded")
        await logger.log("DOM content fully loaded", "DEBUG")
        
        # Capture initial page state
        await logger.log_page_source(page, "initial_login_page")

        # ============================================================================
        # STEP 3: Setup PlayWright Toolkit with the ACTIVE page
        # ============================================================================
        await logger.log("Setting up PlayWright toolkit with active browser context...", "ACTION")
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
        tools = toolkit.get_tools()
        await logger.log(f"✓ Toolkit initialized with {len(tools)} available tools", "SUCCESS")
        
        # List available tools for debugging
        tool_names = [tool.name for tool in tools]
        await logger.log(f"Available tools: {', '.join(tool_names)}", "DEBUG", {"tools": tool_names})

        # ============================================================================
        # STEP 4: Initialize LLM (Language Model)
        # ============================================================================
        await logger.log("Initializing LLM (Ollama - qwen2.5-coder:7b)...", "ACTION")
        llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)
        await logger.log("✓ LLM initialized with deterministic settings (temp=0)", "SUCCESS", 
                        {"model": "qwen2.5-coder:7b", "temperature": 0})

        # ============================================================================
        # STEP 5: Create ReAct Agent with Enhanced System Prompt
        # ============================================================================
        system_msg = """You are a specialized QA automation agent for testing web applications.

CRITICAL INSTRUCTIONS:
1. Use the 'click_element' tool to click on form elements
2. Use the 'fill_element' tool to enter text into input fields
3. ALWAYS specify CSS selectors or XPath for elements (e.g., '#user-name', 'input[name="password"]')
4. For the Swag Labs login:
   - Username field selector: input#user-name
   - Password field selector: input#password
   - Login button selector: input#login-button

5. After each action, verify the result by checking if new content appears
6. Be explicit about what you're doing and why
7. If an action fails, explain the error clearly"""

        await logger.log("Creating ReAct agent with enhanced system prompt...", "ACTION")
        await logger.log(f"System prompt: {system_msg}", "PROMPT", {"role": "system", "length": len(system_msg)})
        
        agent = create_react_agent(model=llm, tools=tools, prompt=system_msg)
        await logger.log("✓ Agent created successfully", "SUCCESS")

        # ============================================================================
        # STEP 6: Prepare and Execute Login Task
        # ============================================================================
        task_prompt = """You are on the Swag Labs login page. Complete the following steps in order:

STEP 1: Locate and fill the username field
- Find the input field with ID 'user-name'
- Fill it with the text 'standard_user'
- Debug: Print that the username was entered

STEP 2: Locate and fill the password field  
- Find the input field with ID 'password'
- Fill it with the text 'secret_sauce'
- Debug: Print that the password was entered

STEP 3: Click the login button
- Find the button with ID 'login-button'
- Click it to submit the form
- Debug: Print that the login button was clicked

STEP 4: Verify successful login
- Wait a moment for page load
- Check if the 'Products' text/heading is visible on the page
- If you see 'Products', the login was SUCCESSFUL
- Debug: Print the final status

Be explicit about each action you take and the results you observe."""

        await logger.log("Starting login automation task...", "ACTION")
        await logger.log(f"Task prompt: {task_prompt}", "PROMPT", {"role": "user", "length": len(task_prompt)})
        
        task = {
            "messages": [
                ("user", task_prompt)
            ]
        }

        # ============================================================================
        # STEP 7: Stream Agent Execution with Debug Output
        # ============================================================================
        await logger.log("Streaming agent execution - observing ReAct loop...", "ACTION")
        
        step_counter = 0
        async for event in agent.astream(task):
            step_counter += 1
            
            # Extract and display node information
            for node, values in event.items():
                await logger.log(f"Agent step {step_counter} - Node: {node}", "DEBUG")
                
                if "messages" in values:
                    messages = values["messages"]
                    if messages:
                        last_msg = messages[-1]
                        content = last_msg.content
                        
                        # Log agent response
                        await logger.log(f"Agent output (Step {step_counter}, Node: {node})", "AGENT", 
                                       {"step": step_counter, "node": node, "content_length": len(content)})
                        
                        # Log full response to file
                        with open(logger.log_file, "a") as f:
                            f.write(f"\n--- AGENT RESPONSE (Step {step_counter}, Node: {node}) ---\n")
                            f.write(content)
                            f.write("\n--- END AGENT RESPONSE ---\n\n")
                        
                        # Add to JSON log
                        await logger.log(content[:500], "RESPONSE", 
                                       {"step": step_counter, "node": node, "full_length": len(content)})

        await logger.log("Agent task completed", "SUCCESS")

        # ============================================================================
        # STEP 8: Verification and Cleanup
        # ============================================================================
        await logger.log("Waiting 5 seconds to observe the result...", "ACTION")
        await asyncio.sleep(5)
        
        # Capture final page state
        await logger.log_page_source(page, "final_page_state")
        
        # Check if we're on the dashboard (verify 'Products' is visible)
        try:
            products_visible = await page.is_visible("text=Products")
            if products_visible:
                await logger.log("✓ LOGIN SUCCESSFUL! 'Products' page is visible", "SUCCESS")
                await logger.log("Login verification passed", "ACTION", {"products_visible": True})
            else:
                await logger.log("⚠️ 'Products' not visible - login may have failed", "ERROR")
                await logger.log("Login verification failed", "ACTION", {"products_visible": False})
        except Exception as e:
            await logger.log(f"Could not verify login success: {str(e)}", "ERROR", {"exception": str(e)})

        # Final wait before cleanup
        await asyncio.sleep(2)
        
        # Close browser cleanly
        await logger.log("Closing browser and cleaning up resources...", "ACTION")
        await browser.close()
        await playwright.stop()
        await logger.log("✓ All resources cleaned up successfully", "SUCCESS")

    except Exception as e:
        """Comprehensive error handling to catch and report any issues."""
        await logger.log(f"CRITICAL ERROR: {str(e)}", "ERROR", 
                        {"error_type": type(e).__name__, "error_msg": str(e)})
        
        import traceback
        tb = traceback.format_exc()
        await logger.log(f"Traceback: {tb}", "ERROR")
        
        # Log traceback to file as well
        with open(logger.log_file, "a") as f:
            f.write("\n" + "=" * 100 + "\n")
            f.write("EXCEPTION TRACEBACK:\n")
            f.write("=" * 100 + "\n")
            f.write(tb)
            f.write("\n" + "=" * 100 + "\n")
        
        try:
            await browser.close()
            await playwright.stop()
        except:
            pass
    
    finally:
        """Ensure logging is finalized"""
        await logger.close()


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("🚀 SWAG LABS LOGIN AUTOMATION TEST (test1.py - Manual Navigation)")
    print("=" * 100 + "\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Script interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()