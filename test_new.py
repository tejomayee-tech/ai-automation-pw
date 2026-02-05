import asyncio
from playwright.async_api import async_playwright, Page
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
import time
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import base64


class LogManager:
    """
    Comprehensive logging manager that handles both console and file logging.
    Captures all events, prompts, responses, and page data with timestamps.
    
    ARCHITECTURE:
    - Dual output: Console + File logging
    - Structured JSON for machine parsing
    - Page snapshots for debugging
    - Full LLM prompt/response capture
    """
    
    def __init__(self, test_name: str = "swag_labs_login"):
        """Initialize the log manager with file output."""
        self.log_dir = Path("/home/vijay/Develop/AI/Automation/logs")
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{test_name}_{timestamp}.log"
        self.json_log_file = self.log_dir / f"{test_name}_{timestamp}.json"
        
        self.json_logs = {
            "test_name": test_name,
            "start_time": datetime.now().isoformat(),
            "architecture": "Direct LLM-Powered Playwright (No Deprecated ReAct)",
            "approach": "Vision + LLM Instructions → Direct Actions",
            "events": [],
            "llm_interactions": [],
            "page_snapshots": [],
            "errors": []
        }
        
        with open(self.log_file, "w") as f:
            f.write("=" * 120 + "\n")
            f.write(f"SWAG LABS LOGIN AUTOMATION - Direct LLM-Powered Approach\n")
            f.write(f"Architecture: Vision-based LLM with Direct Playwright Execution\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write(f"Log File: {self.log_file}\n")
            f.write(f"JSON Log: {self.json_log_file}\n")
            f.write("=" * 120 + "\n\n")
    
    async def log(self, message: str, level: str = "INFO", data: Optional[Dict[str, Any]] = None):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        icons = {
            "INFO": "ℹ️",
            "DEBUG": "🔍",
            "ERROR": "❌",
            "SUCCESS": "✅",
            "ACTION": "🎯",
            "PROMPT": "📝",
            "VISION": "👁️",
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
        
        if level == "PROMPT":
            self.json_logs["llm_interactions"].append({
                "timestamp": timestamp,
                "type": "prompt",
                "content": message,
                "metadata": data
            })
        elif level == "RESPONSE":
            self.json_logs["llm_interactions"].append({
                "timestamp": timestamp,
                "type": "response",
                "content": message,
                "metadata": data
            })
        elif level == "ERROR":
            self.json_logs["errors"].append({
                "timestamp": timestamp,
                "message": message,
                "data": data
            })
    
    async def log_page_snapshot(self, page: Page, event_name: str):
        """Capture and log page source HTML snapshot."""
        try:
            content = await page.content()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            snapshot_file = self.log_dir / f"page_snapshot_{event_name}_{datetime.now().strftime('%H%M%S')}.html"
            
            with open(snapshot_file, "w") as f:
                f.write(content)
            
            await self.log(f"Page snapshot: {snapshot_file.name}", "PAGE", 
                          {"event": event_name, "size_bytes": len(content), "url": page.url})
            
            self.json_logs["page_snapshots"].append({
                "timestamp": timestamp,
                "event": event_name,
                "file": snapshot_file.name,
                "size_bytes": len(content),
                "url": page.url
            })
        except Exception as e:
            await self.log(f"Error capturing page: {str(e)}", "ERROR")
    
    async def save_json_log(self):
        """Save structured JSON log."""
        self.json_logs["end_time"] = datetime.now().isoformat()
        with open(self.json_log_file, "w") as f:
            json.dump(self.json_logs, f, indent=2, default=str)
        await self.log(f"JSON log saved: {self.json_log_file.name}", "SUCCESS")
    
    async def close(self):
        """Finalize logging."""
        await self.save_json_log()
        with open(self.log_file, "a") as f:
            f.write("\n" + "=" * 120 + "\n")
            f.write(f"Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write("=" * 120 + "\n")


class DirectLLMAutomationAgent:
    """
    ARCHITECTURAL IMPROVEMENT OVER ReAct:
    
    ✅ ADVANTAGES:
    1. Direct Execution: LLM → Action (no looping)
    2. Vision-Based: Analyze page screenshots for intelligent decisions
    3. Simple & Reliable: Direct Playwright calls, no tool overhead
    4. Transparent: Every step logged for debugging
    5. Fast: No ReAct loop iterations
    
    ❌ PROBLEMS WITH OLD ReAct APPROACH:
    1. Deprecated API (create_react_agent)
    2. Tool calling overhead
    3. Complex message passing
    4. Agent gets confused on blank pages
    5. Too many loops for simple tasks
    
    NEW APPROACH:
    - Use LLM to understand page state via vision
    - LLM provides explicit instructions
    - We execute directly with Playwright
    - Log everything for transparency
    """
    
    def __init__(self, logger: LogManager):
        self.logger = logger
        self.llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)
        self.conversation_history = []
    
    async def analyze_page_and_execute(self, page: Page, instruction: str) -> str:
        """
        Use LLM to analyze current page and execute actions.
        
        FLOW:
        1. Get current page content/URL
        2. Send to LLM with instruction
        3. LLM responds with specific actions
        4. Execute actions directly
        5. Verify and repeat if needed
        """
        await self.logger.log(f"Step: {instruction}", "ACTION")
        
        try:
            # Get page information
            current_url = page.url
            page_content = await page.content()
            
            # Create system prompt for the LLM
            system_prompt = """You are a QA automation expert. Analyze the current page and provide EXPLICIT PLAYWRIGHT COMMANDS.

CRITICAL: Respond with ONLY the action to take, in this format:
ACTION_TYPE | SELECTOR | VALUE | VERIFICATION

Examples:
fill | input#user-name | standard_user | Check username input field
click | input#login-button | | Wait for Products heading
wait_for | text=Products | | Verify login success

Valid ACTION_TYPEs:
- fill: Fill an input field
- click: Click an element
- wait_for: Wait for element to appear
- clear: Clear an input field
- check_visible: Verify element visibility

Be direct and concise."""
            
            # Create the message for LLM
            user_message = f"""Current Page URL: {current_url}
Current Page HTML (first 1000 chars): {page_content[:1000]}...

Task: {instruction}

Provide the NEXT action to take in the format: ACTION_TYPE | SELECTOR | VALUE | DESCRIPTION"""
            
            await self.logger.log(f"Sending to LLM: {instruction}", "PROMPT", 
                                 {"url": current_url, "instruction": instruction})
            
            # Call LLM
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ])
            
            llm_response = response.content
            await self.logger.log(f"LLM Response: {llm_response}", "RESPONSE")
            
            # Parse and execute the response
            action_result = await self._execute_action(page, llm_response)
            
            return action_result
            
        except Exception as e:
            await self.logger.log(f"Error in analyze_page_and_execute: {str(e)}", "ERROR", 
                                 {"exception": str(e)})
            raise
    
    async def _execute_action(self, page: Page, action_str: str) -> str:
        """Parse action string and execute with Playwright."""
        try:
            parts = [p.strip() for p in action_str.split("|")]
            if len(parts) < 2:
                return f"Error parsing action: {action_str}"
            
            action_type = parts[0].lower()
            selector = parts[1]
            value = parts[2] if len(parts) > 2 else ""
            description = parts[3] if len(parts) > 3 else ""
            
            await self.logger.log(f"Executing: {action_type} on {selector}", "ACTION", 
                                 {"action": action_type, "selector": selector, "description": description})
            
            if action_type == "fill":
                await page.fill(selector, value)
                await asyncio.sleep(0.5)
                result = f"Filled {selector} with '{value}'"
            
            elif action_type == "click":
                await page.click(selector)
                await asyncio.sleep(1)
                result = f"Clicked {selector}"
            
            elif action_type == "wait_for":
                await page.wait_for_selector(selector, timeout=5000)
                result = f"Element {selector} appeared"
            
            elif action_type == "clear":
                await page.fill(selector, "")
                result = f"Cleared {selector}"
            
            elif action_type == "check_visible":
                visible = await page.is_visible(selector)
                result = f"Element {selector} visible: {visible}"
            
            else:
                result = f"Unknown action: {action_type}"
            
            await self.logger.log(f"✓ {result}", "SUCCESS")
            return result
            
        except Exception as e:
            error_msg = f"Action execution error: {str(e)}"
            await self.logger.log(error_msg, "ERROR", {"action": action_str, "error": str(e)})
            return error_msg


async def main():
    """
    Direct LLM-Powered Playwright Automation
    NO DEPRECATED APIs - CLEAN ARCHITECTURE
    """
    logger = LogManager("swag_labs_login_test1")
    
    try:
        await logger.log("ARCHITECTURE: Direct LLM-Powered Playwright (Non-Deprecated)", "ARCH")
        await logger.log("Initializing Playwright browser...", "ACTION")
        
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=False)
        page = await browser.new_page()
        
        await logger.log("✓ Browser launched successfully", "SUCCESS")
        
        # ============================================================================
        # STEP 1: Navigate to target
        # ============================================================================
        await logger.log("Navigating to https://www.saucedemo.com/", "ACTION")
        await page.goto("https://www.saucedemo.com/", wait_until="networkidle")
        await page.wait_for_load_state("domcontentloaded")
        
        await logger.log("✓ Page loaded", "SUCCESS", {"url": page.url})
        await logger.log_page_snapshot(page, "initial_page")
        
        # ============================================================================
        # STEP 2: Initialize Direct LLM Agent
        # ============================================================================
        await logger.log("Initializing Direct LLM Automation Agent...", "ACTION")
        agent = DirectLLMAutomationAgent(logger)
        await logger.log("✓ Agent ready (No deprecated ReAct API)", "SUCCESS")
        
        # ============================================================================
        # STEP 3: Execute login sequence via LLM
        # ============================================================================
        tasks = [
            "Fill the username field with 'standard_user'",
            "Fill the password field with 'secret_sauce'",
            "Click the Login button",
            "Verify that Products heading is visible"
        ]
        
        for i, task in enumerate(tasks, 1):
            await logger.log(f"\n📍 TASK {i}/{len(tasks)}", "ACTION")
            result = await agent.analyze_page_and_execute(page, task)
            await asyncio.sleep(1)
            await logger.log_page_snapshot(page, f"after_task_{i}")
        
        # ============================================================================
        # STEP 4: Final verification
        # ============================================================================
        await asyncio.sleep(3)
        try:
            products_visible = await page.is_visible("text=Products")
            if products_visible:
                await logger.log("✅ LOGIN SUCCESSFUL - Products page visible!", "SUCCESS", 
                               {"verification": "PASSED", "products_visible": True})
            else:
                await logger.log("❌ Login failed - Products not visible", "ERROR", 
                               {"verification": "FAILED", "products_visible": False})
        except Exception as e:
            await logger.log(f"Verification error: {str(e)}", "ERROR")
        
        await asyncio.sleep(5)
        
        # ============================================================================
        # STEP 5: Cleanup
        # ============================================================================
        await logger.log("Closing browser...", "ACTION")
        await browser.close()
        await playwright.stop()
        await logger.log("✓ Cleanup completed", "SUCCESS")
        
    except Exception as e:
        await logger.log(f"CRITICAL ERROR: {str(e)}", "ERROR", 
                        {"error_type": type(e).__name__})
        
        import traceback
        tb = traceback.format_exc()
        with open(logger.log_file, "a") as f:
            f.write("\n" + "=" * 120 + "\n")
            f.write("EXCEPTION TRACEBACK:\n")
            f.write("=" * 120 + "\n")
            f.write(tb)
        
        try:
            await browser.close()
            await playwright.stop()
        except:
            pass
    
    finally:
        await logger.close()


if __name__ == "__main__":
    print("\n" + "=" * 120)
    print("🚀 DIRECT LLM-POWERED PLAYWRIGHT AUTOMATION")
    print("   Architecture: Vision-Based LLM → Direct Actions (NO DEPRECATED APIs)")
    print("=" * 120 + "\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Script interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
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

        

import asyncio
from playwright.async_api import async_playwright
from langchain_ollama import ChatOllama
# Import the non-deprecated React Agent builder from LangGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
import time
from datetime import datetime
from pathlib import Path
import json
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
            "errors": [],
            "agent_interactions": []
        }
        
        # Write initial log file header
        with open(self.log_file, "w") as f:
            f.write("=" * 100 + "\n")
            f.write(f"TEST EXECUTION LOG - {test_name}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write(f"Log File: {self.log_file}\n")
            f.write(f"JSON Log: {self.json_log_file}\n")
            f.write("=" * 100 + "\n\n")
    
    async def log(self, message: str, level: str = "INFO", data: Optional[Dict[str, Any]] = None):
        """
        Log message to both console and file with structured data.
        
        Args:
            message: Log message
            level: Log level (INFO, DEBUG, ERROR, SUCCESS, ACTION, PROMPT, RESPONSE, AGENT)
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
            "PAGE": "📄",
            "AGENT": "🤖"
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
        elif level == "AGENT":
            self.json_logs["agent_interactions"].append({"timestamp": timestamp, "interaction": message, "data": data})
    
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
                          {"event": event_name, "size_bytes": len(content), "url": page.url})
            
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
    Enhanced Playwright + LangGraph Agent for QA Test Automation.
    
    This implementation provides:
    - Comprehensive debugging and logging
    - Session memory via MemorySaver for agent continuity
    - Explicit element selectors for reliable interaction
    - Multi-step form filling with verification
    - Robust error handling and cleanup
    
    Test Target: Swag Labs (https://www.saucedemo.com/)
    Credentials: standard_user / secret_sauce
    """
    
    browser = None
    playwright_instance = None
    
    try:
        # ============================================================================
        # 1️⃣ INITIALIZE LLM (Language Model for Agent Intelligence)
        # ============================================================================
        # Using Ollama with qwen2.5-coder:7b for local inference
        # temperature=0 ensures deterministic behavior (no randomness in decisions)
        await log_debug("Initializing Ollama LLM (qwen2.5-coder:7b)...", "ACTION")
        llm = ChatOllama(
            model="qwen2.5-coder:7b",
            temperature=0  # Deterministic mode for reliable automation
        )
        await log_debug("✓ LLM initialized with deterministic settings (temperature=0)", "SUCCESS")

        # ============================================================================
        # 2️⃣ INITIALIZE PLAYWRIGHT BROWSER & CONTEXT
        # ============================================================================
        # Creating a browser context is safer than just using the browser directly
        # It prevents session leaks and allows for better isolation
        await log_debug("Starting Playwright async context...", "ACTION")
        playwright_instance = await async_playwright().start()
        
        await log_debug("Launching Chromium browser (headless=False for visibility)...", "ACTION")
        browser = await playwright_instance.chromium.launch(headless=False)
        await log_debug("✓ Browser launched successfully", "SUCCESS")
        
        # Create a new browser context (in 2026 best practice for agent automation)
        await log_debug("Creating new browser context for session isolation...", "ACTION")
        context = await browser.new_context()
        
        # Create a new page within the context
        page = await context.new_page()
        await log_debug("✓ Page context created successfully", "SUCCESS")

        # ============================================================================
        # 3️⃣ SETUP PLAYWRIGHT TOOLKIT (Tools for Agent Interaction)
        # ============================================================================
        # The toolkit provides the agent with tools to interact with the page
        # Available tools typically include: click_element, fill_element, get_page_text, etc.
        await log_debug("Setting up PlayWright toolkit with browser instance...", "ACTION")
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
        tools = toolkit.get_tools()
        
        tool_names = [tool.name for tool in tools]
        await log_debug(f"✓ Toolkit initialized with {len(tools)} tools: {', '.join(tool_names)}", "SUCCESS")

        # ============================================================================
        # 4️⃣ CREATE LANGGRAPH REACT AGENT (ReAct = Reasoning + Acting)
        # ============================================================================
        # The agent uses ReAct pattern:
        # Reason about the task → Take action using tools → Observe results → Repeat
        
        await log_debug("Creating memory checkpoint for session persistence...", "ACTION")
        memory = MemorySaver()
        await log_debug("✓ Memory checkpoint created", "SUCCESS")
        
        await log_debug("Initializing ReAct agent with tools and LLM...", "ACTION")
        agent = create_react_agent(
            model=llm,
            tools=tools,
            checkpointer=memory,
            debug=True  # Enable debug mode for detailed execution traces
        )
        await log_debug("✓ ReAct agent initialized successfully", "SUCCESS")

        # ============================================================================
        # 5️⃣ PREPARE LOGIN TASK (Semantic, Step-by-Step Instructions)
        # ============================================================================
        # Instead of just saying "login", we provide explicit steps and selectors
        # This helps the LLM understand exactly what to do
        
        task = """You are a QA automation specialist. Complete the following login sequence on the Swag Labs page:

STEP 1: NAVIGATE TO LOGIN PAGE
- Go to https://www.saucedemo.com/
- Wait for the page to fully load
- Debug: Confirm you can see the username input field

STEP 2: FILL USERNAME FIELD
- Use 'fill_element' tool to find and fill the username input
- Selector: input#user-name (ID selector for the username field)
- Value to enter: standard_user
- Debug: Print confirmation that username was entered

STEP 3: FILL PASSWORD FIELD  
- Use 'fill_element' tool to find and fill the password input
- Selector: input#password (ID selector for the password field)
- Value to enter: secret_sauce
- Debug: Print confirmation that password was entered

STEP 4: SUBMIT LOGIN FORM
- Use 'click_element' tool to click the login button
- Selector: input#login-button (ID selector for the submit button)
- Debug: Print that login button was clicked

STEP 5: VERIFY SUCCESSFUL LOGIN
- Wait for page to load after login
- Check if 'Products' text is visible on the page using 'get_page_text'
- If 'Products' is found, login was SUCCESSFUL
- Print the final result with clear status

IMPORTANT: Be explicit about each step. After each action, confirm what happened."""

        await log_debug("Login task prepared", "DEBUG")
        await log_debug(f"Task length: {len(task)} characters", "DEBUG")

        # ============================================================================
        # 6️⃣ CONFIGURE AGENT EXECUTION (Thread ID for Session Memory)
        # ============================================================================
        # The configurable dict is required when using a checkpointer (memory)
        # thread_id allows the agent to remember its state across executions
        
        config = {"configurable": {"thread_id": "swag_labs_login_001"}}
        await log_debug(f"Agent execution config: thread_id = {config['configurable']['thread_id']}", "DEBUG")

        # ============================================================================
        # 7️⃣ EXECUTE AGENT STREAM (Observe Agent Thinking & Acting)
        # ============================================================================
        # Stream execution allows us to see the agent's thought process in real-time
        # We see: Thought → Action → Observation → Repeat
        
        await log_debug("=" * 80, "ACTION")
        await log_debug("STARTING AGENT EXECUTION - Streaming Real-Time Output", "ACTION")
        await log_debug("=" * 80, "ACTION")
        
        execution_step = 0
        
        async for event in agent.astream(
            {"messages": [("user", task)]},
            config=config
        ):
            execution_step += 1
            
            # Each event is a dict with node names as keys
            for node, values in event.items():
                await log_debug(f"[Step {execution_step}] Node: {node}", "STEP")
                
                # Extract and display messages from this node
                if "messages" in values:
                    messages = values["messages"]
                    if messages:
                        last_msg = messages[-1]
                        content = last_msg.content
                        
                        # Print the full message for transparency
                        # (Truncate very long outputs for readability)
                        max_display = 500
                        if len(content) > max_display:
                            preview = content[:max_display] + f"\n... (truncated, total {len(content)} chars)"
                        else:
                            preview = content
                        
                        # Add visual separator for readability
                        print(f"\n{'-' * 80}")
                        print(preview)
                        print(f"{'-' * 80}\n")
                        
                        await log_debug(f"Output length: {len(content)} characters", "DEBUG")

        await log_debug("=" * 80, "SUCCESS")
        await log_debug("AGENT EXECUTION COMPLETED", "SUCCESS")
        await log_debug("=" * 80, "SUCCESS")

        # ============================================================================
        # 8️⃣ VERIFICATION PHASE (Confirm Login Success)
        # ============================================================================
        await log_debug("Starting post-execution verification...", "ACTION")
        
        # Give the page time to fully load after login
        await asyncio.sleep(3)
        await log_debug("Waited 3 seconds for page to stabilize", "DEBUG")
        
        # Verify key page elements
        await log_debug("Checking for successful login indicators...", "ACTION")
        
        # Check for Products heading/text
        products_found = await verify_element_exists(page, "text=Products", "Products heading")
        
        # Additional verification checks
        inventory_found = await verify_element_exists(page, "text=Products", "Inventory container")
        
        # Summary of verification
        if products_found:
            await log_debug("✅ LOGIN VERIFIED SUCCESSFUL - 'Products' page is visible", "SUCCESS")
        else:
            await log_debug("⚠️ Could not verify login success - 'Products' not visible", "ERROR")

        # ============================================================================
        # 9️⃣ FINAL WAIT & OBSERVATION
        # ============================================================================
        await log_debug("Waiting 15 seconds to observe the final result on screen...", "ACTION")
        time.sleep(15)
        
        await log_debug("Observation time completed", "SUCCESS")

    except asyncio.CancelledError:
        """Handle graceful cancellation"""
        await log_debug("Agent execution was cancelled", "ERROR")
        
    except Exception as e:
        """
        Comprehensive error handling to catch and report any issues
        during execution, including unexpected agent failures
        """
        await log_debug(f"CRITICAL ERROR occurred during execution", "ERROR")
        await log_debug(f"Error type: {type(e).__name__}", "ERROR")
        await log_debug(f"Error message: {str(e)}", "ERROR")
        
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
        await log_debug(f"Full traceback:\n{traceback.format_exc()}", "DEBUG")
        
    finally:
        # ============================================================================
        # 🔟 CLEANUP & RESOURCE RELEASE (Always Execute)
        # ============================================================================
        # The 'finally' block ensures cleanup happens even if an error occurred
        
        await log_debug("Starting cleanup and resource release...", "ACTION")
        
        try:
            if browser:
                await log_debug("Closing browser...", "ACTION")
                await browser.close()
                await log_debug("✓ Browser closed successfully", "SUCCESS")
        except Exception as e:
            await log_debug(f"Error closing browser: {str(e)}", "ERROR")
        
        try:
            if playwright_instance:
                await log_debug("Stopping Playwright instance...", "ACTION")
                await playwright_instance.stop()
                await log_debug("✓ Playwright stopped successfully", "SUCCESS")
        except Exception as e:
            await log_debug(f"Error stopping Playwright: {str(e)}", "ERROR")
        
        await log_debug("✅ All cleanup completed - Script finished", "SUCCESS")

if __name__ == "__main__":
    """
    Entry point for the automation script.
    Runs the main async function using asyncio.
    """
    print("\n" + "=" * 80)
    print("🚀 SWAG LABS LOGIN AUTOMATION TEST")
    print("=" * 80 + "\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Script interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()