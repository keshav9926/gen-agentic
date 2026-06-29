# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║        AGENTIC AI — STEP 2: ReAct Agent Loop From Scratch        ║
╚══════════════════════════════════════════════════════════════════╝

WHAT WE'RE LEARNING TODAY:
  - Building a complete ReAct (Reason + Act) loop manually using a while loop.
  - Tracking agent execution cycles (max_iterations) to prevent infinite loops.
  - Multi-step orchestration: executing tools, reading results, and feeding
    them back to the model in a continuous loop.
  - Testing the agent with conditional, multi-step dependency questions.

CONCEPT: The ReAct Loop (Thought -> Action -> Observation)
──────────────────────────────────────────────────────────
  A standard LLM is single-shot: input in, output out.
  An Agent runs in a loop:
    1. THOUGHT: LLM thinks about the query and decides it needs tool X with arguments Y.
    2. ACTION: The system intercept's the LLM's response and runs tool X.
    3. OBSERVATION: The system receives the tool output (observation) and sends it back.
    4. REPEAT: The LLM reads the observation, and decides if it can answer or needs another tool.
"""

import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# Load env variables
load_dotenv()

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ─────────────────────────────────────────────────────────────────
# LESSON 1: Define Tools
# ─────────────────────────────────────────────────────────────────

@tool
def get_temperature(city: str) -> float:
    """Get the numeric temperature in Celsius for a given city."""
    db = {
        "delhi": 42.0,
        "mumbai": 29.5,
        "bangalore": 21.0
    }
    return db.get(city.lower(), 25.0)

@tool
def calculate_math(expression: str) -> float:
    """Perform simple math operations (e.g. '5 * 12' or '3 * 4')."""
    try:
        # Safe eval limit
        allowed = {"__builtins__": None}
        return float(eval(expression, allowed))
    except Exception as e:
        return f"Error: {e}"

# Build maps
tools = [get_temperature, calculate_math]
tool_map = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)


# ─────────────────────────────────────────────────────────────────
# LESSON 2: The Manual ReAct Loop
# ─────────────────────────────────────────────────────────────────
def run_react_agent(user_query: str, max_iterations: int = 5):
    print("=" * 60)
    print(f"USER QUERY: {user_query}")
    print("=" * 60)

    # State: Initial message list
    messages = [
        SystemMessage(content="You are a helpful assistant. Solve the user's problem step-by-step using tools."),
        HumanMessage(content=user_query)
    ]
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- [Iteration {iteration}/{max_iterations}] ---")
        
        # 1. Ask the model what to do
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # 2. Check if the model wants to call tools
        if not response.tool_calls:
            print("[Thought: No more tools needed. Generating final answer.]")
            print(f"Agent Final Answer:\n{response.content}")
            return response.content
            
        print(f"[Thought: I need to call tools to gather facts.]")
        
        # 3. Execute tool calls and collect responses
        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            call_id = tool_call["id"]
            
            print(f"   Action: Calling '{name}' with {args}")
            
            # Execute
            target_tool = tool_map[name]
            try:
                observation = target_tool.invoke(args)
            except Exception as e:
                observation = f"Error: {str(e)}"
                
            print(f"   Observation: {observation}")
            
            # Build and append ToolMessage
            tool_msg = ToolMessage(
                content=str(observation),
                name=name,
                tool_call_id=call_id
            )
            messages.append(tool_msg)
            
    # If we exit the loop without returning, it means we hit max_iterations
    print(f"\n⚠️ Agent terminated: Reached maximum iterations ({max_iterations}) to prevent infinite looping.")
    return "Failed to resolve query within allowed execution limits."


# ─────────────────────────────────────────────────────────────────
# LESSON 3: Testing the Loop
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # This query requires:
    # 1. Weather API call for Delhi (Observation: 42.0)
    # 2. LLM logic check: "Is 42.0 > 30C?" (Yes)
    # 3. Math calculation for "5 * 12"
    # 4. Final summary output
    query = "Check the temperature in Delhi. If it is greater than 30 degrees, calculate 5 * 12. Otherwise, calculate 3 * 4."
    run_react_agent(query)


"""
WHAT YOU LEARNED:
  ✅ ReAct loop is a while loop that coordinates model inputs and tool execution.
  ✅ Checking max_iterations is critical in production to safeguard against
     costly infinite loops (where the model keeps calling tools due to bad parsing/failures).
  ✅ Multi-step logic: The output of one tool call changes the state, which
     determines the next tool call.

INTERVIEW INSIGHTS:
  "How do you handle an agent stuck in an infinite loop?"
  -> Implement a strict step counter (max_iterations) and timeout limits in the
     outer loop, and raise an alarm or return a fallback output if the limit is exceeded.

  "What is the difference between a static LLM chain and a ReAct agent loop?"
  -> A chain executes a hardcoded sequence of steps. An agent evaluates the
     intermediate tool results (observations) and dynamically decides what steps
     to take next.
"""
