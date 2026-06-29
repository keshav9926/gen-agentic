# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║         AGENTIC AI — STEP 4: Human-in-the-Loop & State Editing    ║
║                           (HITL + Interrupts)                    ║
╚══════════════════════════════════════════════════════════════════╝ 

WHAT WE'RE LEARNING TODAY:
  - The Human-in-the-Loop (HITL) design pattern for sensitive transactions.
  - Setting up breakpoints using `interrupt_before` during graph compilation.
  - Inspecting paused graph execution states  and active tool calls.
  - Modifying state properties dynamically using `update_state()`.
  - Resuming graph execution after user approval or modification.

CONCEPT: Human-in-the-Loop (HITL)
─────────────────────────────────
  Autonomous agents can sometimes make mistakes or hallucinations. If an agent
  has tools that execute irreversible actions (e.g. sending emails, making purchases,
  or deleting databases), you MUST have a human-in-the-loop checkpoint.
  
  LangGraph makes this simple:
    - You specify breakpoints (nodes to pause before/after).
    - When the graph hits a breakpoint, it saves a checkpoint and suspends execution.
    - You inspect the state, approve/modify it, and resume.
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Load env variables
load_dotenv()

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ─────────────────────────────────────────────────────────────────
# LESSON 1: Define Graph State and Sensitive Tools
# ─────────────────────────────────────────────────────────────────

class HitlState(TypedDict):
    messages: Annotated[List, add_messages]

@tool
def execute_wire_transfer(amount: float, recipient: str) -> str:
    """Execute a bank wire transfer. This is a highly sensitive operations."""
    return f"SUCCESS: Transferred ${amount} to {recipient}."

tools = [execute_wire_transfer]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

# Define nodes
def agent_node(state: HitlState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Edge router
def should_continue(state: HitlState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "end"

# Build Graph
builder = StateGraph(HitlState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.set_entry_point("agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
builder.add_edge("tools", "agent")

# ─────────────────────────────────────────────────────────────────
# LESSON 2: Compile with Breakpoint Interrupts
# ─────────────────────────────────────────────────────────────────
memory = MemorySaver()

# We specify `interrupt_before=["tools"]` to pause execution BEFORE running any tools
graph = builder.compile(checkpointer=memory, interrupt_before=["tools"])

print("=" * 60)
print("GRAPH COMPILED WITH BREAKPOINT BEFORE 'tools' NODE")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────
# LESSON 3: Running and Triggering the Breakpoint
# ─────────────────────────────────────────────────────────────────
config = {"configurable": {"thread_id": "transfer_session_1"}}

print("\nSTEP 1: Initiating transaction request...")
# Set a unique ID on the human message so we can reference and edit it later
initial_input = {"messages": [HumanMessage(content="Please transfer 100 dollars to Bob.", id="user_request_1")]}

# Run the graph. It will run 'agent', decide to call 'execute_wire_transfer',
# but pause BEFORE entering the 'tools' node.
for event in graph.stream(initial_input, config):
    print(f"Executing event: {event}")

# Let's check where the graph is.
state = graph.get_state(config)
print(f"\nGraph is currently paused at next node: {state.next}")

# Inspect the tool calls that the model proposed
last_message = state.values["messages"][-1]
pending_tool_calls = last_message.tool_calls
print(f"Pending tool calls: {pending_tool_calls}")


# ─────────────────────────────────────────────────────────────────
# LESSON 4: Modifying State & User Interventions
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Human Intervention - Editing the State")
print("=" * 60)
print("Let's simulate a human reviewer modifying the recipient from 'Bob' to 'Charlie'.")

# Get the proposed tool call dictionary
proposed_call = pending_tool_calls[0]
modified_call = proposed_call.copy()
# Modify the arguments directly
modified_call["args"] = {"amount": 100.0, "recipient": "Charlie"}

# Replace the last message's tool call list with our modified version.
# To overwrite the state values in LangGraph, we can update the state.
modified_message = AIMessage(
    content=last_message.content,
    tool_calls=[modified_call],
    id=last_message.id # Using the same ID updates/replaces the message in add_messages reducer
)

# Overwrite the user's initial message as well so they match the modified recipient
modified_user_message = HumanMessage(
    content="Please transfer 100 dollars to Charlie.",
    id="user_request_1"
)

# Apply state update. We tell it we want to update the 'agent' node's output.
graph.update_state(config, {"messages": [modified_user_message, modified_message]}, as_node="agent")

# Let's inspect the updated state to verify
new_state = graph.get_state(config)
print(f"Updated pending tool calls in state: {new_state.values['messages'][-1].tool_calls}")


# ─────────────────────────────────────────────────────────────────
# LESSON 5: Resuming the Graph
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Resuming Graph Execution with Modified State")
print("=" * 60)

# To resume the graph from where it paused, we invoke/stream passing `None` as the input.
# LangGraph loads the checkpoint state for this thread config and runs the next node ('tools').
for event in graph.stream(None, config):
    print(f"Resuming event: {event}")

# Print the final conversation outcome
final_state = graph.get_state(config)
print("\nFinal Conversation State:")
for msg in final_state.values["messages"]:
    print(f"[{type(msg).__name__}]: {msg.content if msg.content else repr(msg.tool_calls)}")


if __name__ == "__main__":
    print("\n[Human-in-the-loop demo complete!]")


"""
WHAT YOU LEARNED:
  ✅ interrupt_before compiles a breakpoint where graph execution pauses.
  ✅ When a graph is paused, state.next indicates the node that will run next.
  ✅ update_state allows humans/reviews to modify message lists, arguments,
     or state values before resuming.
  ✅ Invoking the graph with None and the same config resumes execution.

INTERVIEW INSIGHTS:
  "How do you implement human approval in LangGraph?"
  -> Use the interrupt_before parameter during compilation. When the graph pauses,
     interact with the user, update the state if necessary, and call invoke(None, config)
     to resume.

  "What is the role of message IDs when updating state with add_messages?"
  -> The add_messages reducer appends new messages unless they have the same id
     as an existing message. Providing the same ID allows overwriting/replacing
     an AI's decision or a user's statement dynamically.
"""
