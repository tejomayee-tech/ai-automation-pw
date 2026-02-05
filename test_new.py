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