# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║         AGENTIC AI — STEP 4: LangGraph State Persistence         ║
║                           (MemorySaver)                          ║
╚══════════════════════════════════════════════════════════════════╝

WHAT WE'RE LEARNING TODAY:
  - State persistence in LangGraph using MemorySaver checkpointer.
  - Using Thread Configuration (thread_id) to isolate student/user sessions.
  - Multi-turn conversation logic: keeping memory across distinct function calls.
  - Fetching, inspecting, and tracking checkpointer state history under the hood.

CONCEPT: What is a Checkpointer?
────────────────────────────────
  By default, when a LangGraph agent finishes executing, its memory (State)
  is lost. To keep memory across user turns, we compile the graph with a Checkpointer.
  
  The checkpointer:
    - Saves the State dictionary to memory (or a database like SQLite/PostgreSQL)
      after every node execution.
    - Uses a `thread_id` to retrieve the correct state for a specific user session.
    - Allows multi-turn conversations without passing the entire message history
      back and forth from the client side.
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
# MemorySaver is an in-memory checkpointer. For database persistence, use SqliteSaver.
from langgraph.checkpoint.memory import MemorySaver

# Load env variables
load_dotenv()

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

# ─────────────────────────────────────────────────────────────────
# LESSON 1: Define Stateful Agent
# ─────────────────────────────────────────────────────────────────

# State holds messages. We use `add_messages` as a reducer.
# A REDUCER tells LangGraph how to merge new state updates.
# `add_messages` automatically appends new messages while updating existing
# ones if their IDs match.
class PersistenceState(TypedDict):
    messages: Annotated[List, add_messages]

# Create a simple agent node
def chatbot_node(state: PersistenceState) -> dict:
    # Get last message to see current user query
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build Graph
builder = StateGraph(PersistenceState)
builder.add_node("chatbot", chatbot_node)
builder.set_entry_point("chatbot")
builder.add_edge("chatbot", END)

# ─────────────────────────────────────────────────────────────────
# LESSON 2: Compile with MemorySaver
# ─────────────────────────────────────────────────────────────────
# Initialize memory checkpointer
memory = MemorySaver()

# Compile the graph passing the checkpointer
# This automatically enables state saving/loading
graph = builder.compile(checkpointer=memory)

print("=" * 60)
print("GRAPH COMPILED WITH MEMORY SAVER CHECKPOINTER")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────
# LESSON 3: Multi-turn conversations with Thread IDs
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Starting Session 1 (Thread A)")
print("=" * 60)

# Thread configurations contain a thread_id
config_a = {"configurable": {"thread_id": "thread_a"}}

# First Turn
input_1 = {"messages": [HumanMessage(content="Hi, my name is Keshav. Remember my name.")]}
result_1 = graph.invoke(input_1, config=config_a)
print(f"User:  {input_1['messages'][0].content}")
print(f"Agent: {result_1['messages'][-1].content}")

# Second Turn: We do NOT pass the name "Keshav" again.
# The checkpointer automatically loads the state for thread_a.
input_2 = {"messages": [HumanMessage(content="What is my name?")]}
result_2 = graph.invoke(input_2, config=config_a)
print(f"\nUser:  {input_2['messages'][0].content}")
print(f"Agent: {result_2['messages'][-1].content}")


# ─────────────────────────────────────────────────────────────────
# LESSON 4: Session Isolation
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Session Isolation Check (Thread B)")
print("=" * 60)

# Create a new configuration with a different thread_id
config_b = {"configurable": {"thread_id": "thread_b"}}

# Let's ask the chatbot in Thread B what the user's name is.
# It should not know, because Thread B has its own isolated memory state.
input_3 = {"messages": [HumanMessage(content="What is my name?")]}
result_3 = graph.invoke(input_3, config=config_b)
print(f"Thread B User:  {input_3['messages'][0].content}")
print(f"Thread B Agent: {result_3['messages'][-1].content} (Correct: Isolation works!)")


# ─────────────────────────────────────────────────────────────────
# LESSON 5: Inspecting Checkpoint History
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Inspecting State History (Thread A)")
print("=" * 60)

# We can query the checkpointer to see all historical states (checkpoints) saved
# for a given config.
states = list(graph.get_state_history(config_a))

print(f"Total Checkpoints in Thread A: {len(states)}")
print("\nTimeline of Saved States (Newest to Oldest):")
for i, state in enumerate(states):
    print(f"\nCheckpoint {len(states) - i}:")
    print(f"  Config ID: {state.config}")
    print(f"  Next Node to Execute: {state.next}") # If empty, graph has ended
    last_msg = state.values["messages"][-1]
    print(f"  Last Message: [{type(last_msg).__name__}] {last_msg.content[:60]}...")


if __name__ == "__main__":
    print("\n[Persistence demo complete!]")


"""
WHAT YOU LEARNED:
  ✅ MemorySaver is an in-memory checkpointer that persists state across turns.
  ✅ Thread configurations (thread_id) isolate states between different users.
  ✅ add_messages is a reducer function that appends new messages to history.
  ✅ get_state_history allows querying all previous state checkpoints in a thread.

INTERVIEW INSIGHTS:
  "How does LangGraph persist state between user turns?"
  -> When compiled with a checkpointer, LangGraph uses the thread_id config to
     fetch the latest checkpoint from the database, runs the active nodes,
     and saves a new checkpoint after every node execution.

  "What is a Reducer in LangGraph?"
  -> A reducer is a function that dictates how updates to state properties are merged.
     For instance, Annotated[list, add_messages] tells LangGraph to append new
     messages to the existing list rather than overwriting it.
"""
