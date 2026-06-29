# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║     AGENTIC AI — STEP 1: Tool Calling & Execution Mechanics      ║
╚══════════════════════════════════════════════════════════════════╝

WHAT WE'RE LEARNING TODAY:
  - How Tool Calling works under the hood (schemas, model output, and routing).
  - Binding tools to an LLM using .bind_tools().
  - Manually parsing the model's tool calls (response.tool_calls).
  - Executing tools and mapping results back into ToolMessages.
  - Graceful error handling (feeding tool errors back to the model).

CONCEPT: What is Tool Calling?
──────────────────────────────
  Tool calling is NOT the model executing a python function.
  It is a two-step handshake:
    1. You tell the model: "Here are functions A and B, including their names,
       descriptions, and expected parameters (in JSON Schema format)."
    2. The model decides if it needs a tool: if yes, it halts text generation
       and returns a structured JSON payload containing:
         - The name of the tool it wants to call.
         - The arguments it wants to pass to that tool.
    3. YOU (the runtime) execute the function, get the result, format it as a
       ToolMessage, and send it back to the model.
    4. The model reads the tool output and continues generating its answer.
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
# LESSON 1: Defining Tools
# ─────────────────────────────────────────────────────────────────
# LangChain provides the `@tool` decorator to convert a standard
# Python function into a Tool. The docstring and type hints are used
# to auto-generate the JSON schema that is sent to the LLM.

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together. Use this for mathematical multiplications."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city. Returns a weather description string."""
    # Simulated weather database
    weather_db = {
        "mumbai": "Rainy, 28°C, Humidity: 85%",
        "delhi": "Sunny, 42°C, Humidity: 20%",
        "bangalore": "Pleasant, 22°C, Humidity: 50%"
    }
    return weather_db.get(city.lower(), "Weather details not found for this city.")

# Create our list of tools
tools = [multiply, get_weather]
tool_map = {tool.name: tool for tool in tools}

# Let's inspect the JSON schema generated for the LLM
print("=" * 60)
print("INSPECTING TOOL SCHEMAS SENT TO LLM:")
print("=" * 60)
for t in tools:
    # Print the tool schema showing how arguments are formatted
    print(f"Tool Name: {t.name}")
    print(f"Description: {t.description}")
    print(f"Schema Arguments: {json.dumps(t.args, indent=2)}")
    print("-" * 30)


# ─────────────────────────────────────────────────────────────────
# LESSON 2: Binding Tools and Initial Handshake
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Binding Tools & Getting Model Tool Decision")
print("=" * 60)

# .bind_tools() modifies the LLM's system API parameters to supply the tool schemas
llm_with_tools = llm.bind_tools(tools)

messages = [
    HumanMessage(content="What is the weather like in Bangalore and what is 123 * 456?")
]

response = llm_with_tools.invoke(messages)

print(f"Raw Model Response Type: {type(response)}")
print(f"Model Content: {repr(response.content)}") # Usually empty because it wants to call a tool
print(f"Model Tool Calls: {json.dumps(response.tool_calls, indent=2)}")


# ─────────────────────────────────────────────────────────────────
# LESSON 3: Manual Execution & Routing Loop
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Executing Tools Manually and Sending Back Results")
print("=" * 60)

# Append the model's decision message (which contains the tool_calls) to the history
messages.append(response)

# Execute each tool call requested by the model
for tool_call in response.tool_calls:
    name = tool_call["name"]
    args = tool_call["args"]
    call_id = tool_call["id"] # Critical identifier to map result to the correct call
    
    print(f"\n-> Executing tool: '{name}' with arguments: {args}")
    
    # Retrieve the function from our map and execute it
    target_tool = tool_map[name]
    
    try:
        # Execute the tool
        result = target_tool.invoke(args)
        error_occurred = False
    except Exception as e:
        # Catch errors gracefully so the agent doesn't crash
        result = f"Error executing tool: {str(e)}"
        error_occurred = True
    
    print(f"   Result: {result}")
    
    # Create the ToolMessage containing the output and mapping it to the call_id
    tool_message = ToolMessage(
        content=str(result),
        name=name,
        tool_call_id=call_id,
        status="error" if error_occurred else "success" # Tells LLM if execution failed
    )
    
    # Append the result to messages
    messages.append(tool_message)

# ─────────────────────────────────────────────────────────────────
# LESSON 4: Final LLM Generation
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Final Answer Generation")
print("=" * 60)

final_response = llm_with_tools.invoke(messages)
print(f"Final Model Answer:\n{final_response.content}")


# ─────────────────────────────────────────────────────────────────
# LESSON 5: Handling Tool Failures Gracefully
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Testing Tool Failure Resilience")
print("=" * 60)

@tool
def divide(a: int, b: int) -> float:
    """Divide a by b. Use this for math divisions."""
    return a / b

llm_with_div = llm.bind_tools([divide])
div_messages = [
    HumanMessage(content="Divide 10 by 0")
]

# Get model decision
div_response = llm_with_div.invoke(div_messages)
div_messages.append(div_response)

print("Model requested:", div_response.tool_calls[0])

# Execute manually and raise ZeroDivisionError
tool_call = div_response.tool_calls[0]
call_id = tool_call["id"]

try:
    # Trigger exception
    result = divide.invoke(tool_call["args"])
    status = "success"
except Exception as e:
    # Capture error message and pass it to the model instead of crashing
    result = f"ZeroDivisionError: division by zero"
    status = "error"

print(f"Captured Error: {result}")

# Append the error back into the message list
div_messages.append(
    ToolMessage(content=result, tool_call_id=call_id, name="divide", status=status)
)

# Invoke the model again. We use the base llm (without tools bound) to force a text explanation.
final_div_response = llm.invoke(div_messages)
print(f"\nModel feedback on error:\n{final_div_response.content}")


if __name__ == "__main__":
    print("\n[Tool calling demo complete!]")


"""
WHAT YOU LEARNED:
  ✅ Tool calling is a protocol/handshake where the LLM returns JSON data.
  ✅ .tool_calls contains 'name', 'args', and 'id' fields.
  ✅ 'id' is required by ToolMessage so the LLM can align multiple parallel tool results.
  ✅ Tool execution errors should not crash your application; write exceptions
     as status="error" ToolMessages so the LLM can explain or correct the arguments.

INTERVIEW INSIGHTS:
  "Under the hood, does the LLM execute the python code of your tool?"
  -> No, the LLM only outputs the JSON structure representing name and arguments.
     The orchestration engine (your code, LangGraph, etc.) executes the code.

  "Why does ToolMessage need a tool_call_id?"
  -> When models execute multiple tool calls in parallel, the tool_call_id links
     each ToolMessage output back to its original call request.
"""
