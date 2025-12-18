"""
Test the green agent in white_agent mode.
This tests the green/white agent architecture where:
- Green agent = judge/tester (this server)
- White agent = agent being tested (external server)
"""

import asyncio
import json
from my_util.my_a2a import send_message, wait_agent_ready


async def test_white_agent_mode(
    green_agent_url: str = "http://localhost:9001",
    white_agent_url: str = "http://localhost:9002"
):
    """
    Test green agent testing a white agent.

    Setup required:
    1. Start green agent in white_agent mode:
       python start_green_agent.py --mode white_agent --port 9001

    2. Start the white agent (the agent being tested):
       python start_white_agent.py --port 9002
    """
    print("=" * 60)
    print("TEST: Green Agent Testing White Agent")
    print("=" * 60)

    # Wait for green agent to be ready
    print(f"\nWaiting for green agent at {green_agent_url}...")
    ready = await wait_agent_ready(green_agent_url, timeout=10)
    if not ready:
        print("❌ Green agent not ready")
        return False

    # Wait for white agent to be ready
    print(f"Waiting for white agent at {white_agent_url}...")
    ready = await wait_agent_ready(white_agent_url, timeout=10)
    if not ready:
        print("❌ White agent not ready")
        return False

    print("\n✅ Both agents ready!\n")

    # Prepare message with white agent URL
    question = "How has Netflix's (NASDAQ: NFLX) Average Revenue Per Paying User Changed from 2019 to 2024?"
    # question = "What was the non-GAAP operating margin for XYLO Corp in Q3 2024, and how does it compare to their midpoint guidance?"

    message = f"""<white_agent_url>
{white_agent_url}
</white_agent_url>
<question>
{question}
</question>
<config>
{{
    "mode": "white_agent",
    "max_iterations": 10
}}
</config>"""

    print("Sending test task to green agent...")
    print(f"Question: {question}")
    print(f"White agent URL: {white_agent_url}\n")

    try:
        # Send message to green agent
        response = await send_message(green_agent_url, message)

        print("✅ Response received!\n")
        print("=" * 60)
        print("RESPONSE FROM GREEN AGENT:")
        print("=" * 60)

        # Extract the response
        if hasattr(response.root, 'result'):
            result = response.root.result
            if hasattr(result, 'parts'):
                for part in result.parts:
                    # The part is a wrapper, actual content is in part.root
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        print(f"\n{part.root.text}\n")
                    elif hasattr(part, 'text'):
                        print(f"\n{part.text}\n")
            else:
                print("No parts in result")
                print(f"Result: {result}")
        else:
            print("No result in response")
            print(f"Response: {response.root}")

        print("=" * 60)
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the test."""
    print("\n🚀 Testing Green/White Agent Architecture\n")
    print("Setup instructions:")
    print("1. Terminal 1 (Green Agent): python start_green_agent.py --mode white_agent --port 9001")
    print("2. Terminal 2 (White Agent): python start_white_agent.py --port 9002")
    print("3. Terminal 3 (Test): python test_white_agent_mode.py")
    print()

    result = await test_white_agent_mode()

    print("\n" + "=" * 60)
    if result:
        print("✅ Test passed!")
    else:
        print("❌ Test failed - check that both servers are running")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
