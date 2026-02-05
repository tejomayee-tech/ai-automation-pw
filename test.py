import asyncio
from playwright.async_api import async_playwright

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
import time

async def main():
    # 1️⃣ LLM Setup (Optimized for 2026 Local Inferencing)
    llm = ChatOllama(
        model="qwen2.5-coder:7b",
        temperature=0
    )

    # 2️⃣ Native async Playwright
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    
    # In 2026, creating a context is safer for agents to prevent session leaks
    context = await browser.new_context()
    page = await context.new_page()

    # 3️⃣ Toolkit setup
    # We pass the 'page' directly so the AI has a specific tab to work on
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    tools = toolkit.get_tools()

    # 4️⃣ LangGraph ReAct agent
    # We provide a MemorySaver so the agent remembers what it did in Step 1 while doing Step 2
    memory = MemorySaver()
    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memory
    )

    # 5️⃣ Semantic Task
    task = """
    Go to https://www.saucedemo.com/
    1. Enter 'standard_user' into the username field.
    2. Enter 'secret_sauce' into the password field.
    3. Click the Login button.
    4. Confirm success by checking if 'Products' is visible.
    """

    # 6️⃣ Stream execution with Configuration (Required for Checkpointer)
    config = {"configurable": {"thread_id": "login_test_001"}}
    
    

    async for event in agent.astream(
        {"messages": [("user", task)]},
        config=config
    ):
        # This will print the "Thought" and "Action" blocks clearly
        for node, values in event.items():
            print(f"\n--- Node: {node} ---")
            if "messages" in values:
                print(values["messages"][-1].content)

    time.sleep(15)
    # 7️⃣ Cleanup
    await browser.close()
    await playwright.stop()

if __name__ == "__main__":
    asyncio.run(main())