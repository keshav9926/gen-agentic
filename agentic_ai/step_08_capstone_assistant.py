"""
CAPSTONE PROJECT: Personal AI Research Assistant
=================================================

This project combines EVERYTHING you've learned:
  ✅ LLM invocation
  ✅ Prompt templates
  ✅ Output parsers (Pydantic)
  ✅ LCEL chains
  ✅ Memory (conversation history)
  ✅ Tools (web-like search, calculations)
  ✅ RAG (knowledge base Q&A)
  ✅ LangGraph (orchestrating the whole flow)
  ✅ Agents (autonomous tool selection)
  ✅ Streaming

ARCHITECTURE:
  User message
       ↓
  [Router Agent] ← classifies intent
     /    |    \\
[Chat] [Research] [Calculator]
     \\    |    /
  [Response Formatter]
       ↓
  User sees typed-out response + sources

HOW TO RUN:
  python capstone_project.py
  
  Then type:
    - "What is quantum entanglement?"    → triggers research mode
    - "Calculate compound interest..."   → triggers calculator  
    - "Hi, my name is Keshav"            → triggers chat mode
    - "quit" to exit
"""

from dotenv import load_dotenv
load_dotenv()

import json
from datetime import datetime
from typing import TypedDict, Annotated, List, Optional
import operator

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel, Field

# ── LLM Setup ──────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
fast_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

print("""
╔══════════════════════════════════════════════════════════════╗
║          🤖 Personal AI Research Assistant                   ║
║                  LangChain Capstone Project                  ║
╚══════════════════════════════════════════════════════════════╝
""")


# ══════════════════════════════════════════════════════════════════════════════
# 1. TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def search_knowledge(query: str) -> str:
    """
    Search the knowledge base for information about a topic.
    Use this when the user asks about facts, concepts, history, science, etc.
    """
    # Simulated knowledge base (in real app: vector store + RAG)
    response = fast_llm.invoke(
        f"You are a knowledge base. Provide accurate, concise facts about: {query}"
        " Include specific numbers, dates, and sources when relevant."
    )
    return response.content

@tool
def calculate(expression: str) -> str:
    """
    Perform mathematical calculations. Supports +, -, *, /, **, and math functions.
    For compound interest: expression like '1000 * (1 + 0.05) ** 10'
    """
    import math
    try:
        safe_env = {
            "__builtins__": {},
            "math": math,
            "sqrt": math.sqrt,
            "pow": math.pow,
            "log": math.log,
            "sin": math.sin,
            "cos": math.cos,
            "pi": math.pi,
            "e": math.e,
        }
        result = eval(expression, safe_env)
        return f"Result: {result:.6f}" if isinstance(result, float) else f"Result: {result}"
    except Exception as ex:
        return f"Calculation error: {ex}. Try a simpler expression."

@tool
def get_current_datetime() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"

@tool
def summarize_text(text: str) -> str:
    """
    Summarize a long piece of text into key bullet points.
    Use when user provides text and asks for a summary.
    """
    response = fast_llm.invoke(f"Summarize in 3 bullet points:\n{text}")
    return response.content

@tool
def generate_study_plan(topic: str, duration_days: int = 7) -> str:
    """
    Generate a structured study plan for learning a topic.
    """
    response = fast_llm.invoke(
        f"Create a {duration_days}-day study plan for learning: {topic}"
        " Format as Day 1: [topic], Day 2: [topic], etc. Be specific."
    )
    return response.content

tools = [search_knowledge, calculate, get_current_datetime, summarize_text, generate_study_plan]
tool_map = {t.name: t for t in tools}


# ══════════════════════════════════════════════════════════════════════════════
# 2. ASSISTANT STATE
# ══════════════════════════════════════════════════════════════════════════════

class AssistantState(TypedDict):
    user_input: str
    intent: str                                          # chat / research / calculate / study
    chat_history: Annotated[List, operator.add]          # growing message list
    tool_results: List[str]                              # results from tools
    final_response: str
    session_info: dict                                   # user name, preferences, etc.


# ══════════════════════════════════════════════════════════════════════════════
# 3. GRAPH NODES
# ══════════════════════════════════════════════════════════════════════════════

def classify_intent(state: AssistantState) -> dict:
    """Classify what the user wants to do."""
    user_input = state["user_input"]
    history_context = ""
    if state.get("chat_history"):
        last_2 = state["chat_history"][-4:]  # last 2 turns
        history_context = "\n".join([f"{type(m).__name__}: {m.content[:100]}" for m in last_2])
    
    prompt = f"""Classify the user's intent into one of: 'chat', 'research', 'calculate', 'study', 'summarize'

Recent history:
{history_context}

User message: {user_input}

Rules:
- 'chat': greetings, personal questions, opinions, casual conversation
- 'research': factual questions, "what is", "explain", "how does"
- 'calculate': math, numbers, formulas, "how much", "what is X% of"
- 'study': "teach me", "help me learn", "study plan", "beginner's guide"
- 'summarize': "summarize this", user provides text to condense

Reply with exactly ONE word."""
    
    response = fast_llm.invoke(prompt)
    intent = response.content.strip().lower()
    
    for valid in ["chat", "research", "calculate", "study", "summarize"]:
        if valid in intent:
            intent = valid
            break
    else:
        intent = "chat"
    
    print(f"\n  [Intent detected: {intent}]")
    return {"intent": intent}


def handle_chat(state: AssistantState) -> dict:
    """Handle casual conversation with memory."""
    system_msg = SystemMessage(
        "You are a friendly, knowledgeable AI assistant named ARIA. "
        "You're helpful, warm, and remember context from the conversation. "
        "If the user shares personal info (name, goals), acknowledge it naturally."
    )
    
    messages = [system_msg] + state.get("chat_history", []) + [
        HumanMessage(state["user_input"])
    ]
    
    response = llm.invoke(messages)
    return {
        "final_response": response.content,
        "chat_history": [HumanMessage(state["user_input"]), AIMessage(response.content)],
    }


def handle_research(state: AssistantState) -> dict:
    """Use the research agent for factual questions."""
    research_agent = create_react_agent(llm, [search_knowledge, get_current_datetime])
    
    # Build context from history
    context = ""
    if state.get("chat_history"):
        context = "Previous conversation context: " + " | ".join(
            [m.content[:50] for m in state["chat_history"][-4:]]
        )
    
    result = research_agent.invoke({
        "messages": [HumanMessage(f"{context}\n\nUser question: {state['user_input']}")]
    })
    
    answer = result["messages"][-1].content
    return {
        "final_response": answer,
        "chat_history": [HumanMessage(state["user_input"]), AIMessage(answer)],
    }


def handle_calculate(state: AssistantState) -> dict:
    """Extract and run calculations."""
    calc_agent = create_react_agent(llm, [calculate])
    
    result = calc_agent.invoke({
        "messages": [HumanMessage(
            f"Help the user with this calculation request: {state['user_input']}"
            " Extract the mathematical expression and compute it."
        )]
    })
    
    answer = result["messages"][-1].content
    return {
        "final_response": answer,
        "chat_history": [HumanMessage(state["user_input"]), AIMessage(answer)],
    }


def handle_study(state: AssistantState) -> dict:
    """Generate study plans and learning resources."""
    study_agent = create_react_agent(llm, [generate_study_plan, search_knowledge])
    
    result = study_agent.invoke({
        "messages": [HumanMessage(state["user_input"])]
    })
    
    answer = result["messages"][-1].content
    return {
        "final_response": answer,
        "chat_history": [HumanMessage(state["user_input"]), AIMessage(answer)],
    }


def handle_summarize(state: AssistantState) -> dict:
    """Summarize provided text."""
    summarize_agent = create_react_agent(llm, [summarize_text])
    
    result = summarize_agent.invoke({
        "messages": [HumanMessage(state["user_input"])]
    })
    
    answer = result["messages"][-1].content
    return {
        "final_response": answer,
        "chat_history": [HumanMessage(state["user_input"]), AIMessage(answer)],
    }


def route_by_intent(state: AssistantState) -> str:
    """Route to the correct handler based on classified intent."""
    return state.get("intent", "chat")


# ══════════════════════════════════════════════════════════════════════════════
# 4. BUILD THE GRAPH
# ══════════════════════════════════════════════════════════════════════════════

builder = StateGraph(AssistantState)

builder.add_node("classify",    classify_intent)
builder.add_node("chat",        handle_chat)
builder.add_node("research",    handle_research)
builder.add_node("calculate",   handle_calculate)
builder.add_node("study",       handle_study)
builder.add_node("summarize",   handle_summarize)

builder.set_entry_point("classify")
builder.add_conditional_edges(
    "classify",
    route_by_intent,
    {
        "chat":      "chat",
        "research":  "research",
        "calculate": "calculate",
        "study":     "study",
        "summarize": "summarize",
    }
)

for node in ["chat", "research", "calculate", "study", "summarize"]:
    builder.add_edge(node, END)

assistant = builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
# 5. INTERACTIVE LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_assistant():
    """Run the assistant in an interactive loop."""
    print("ARIA - Your Personal AI Research Assistant")
    print("━" * 50)
    print("I can help with:")
    print("  💬 Casual chat & answering personal questions")
    print("  🔍 Research & factual questions (ask 'what is X?')")
    print("  🧮 Math & calculations")
    print("  📚 Creating study plans ('help me learn Python')")
    print("  📝 Summarizing text")
    print("\nType 'quit' to exit | Type 'history' to see conversation log")
    print("━" * 50 + "\n")
    
    state = {
        "user_input": "",
        "intent": "",
        "chat_history": [],
        "tool_results": [],
        "final_response": "",
        "session_info": {"started_at": datetime.now().isoformat()},
    }
    
    turn = 0
    while True:
        try:
            user_input = input(f"\n[Turn {turn + 1}] You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("\nGoodbye! It was great talking with you! 👋")
            break
        
        if user_input.lower() == "history":
            print("\n📜 Conversation History:")
            for msg in state["chat_history"]:
                prefix = "You:  " if isinstance(msg, HumanMessage) else "ARIA: "
                print(f"  {prefix}{msg.content[:120]}")
            continue
        
        # Update input in state
        state["user_input"] = user_input
        
        # Run the graph
        print("\nARIA: ", end="", flush=True)
        
        result = assistant.invoke(state)
        
        # Update persistent state (history carries over between turns)
        state["chat_history"] = result.get("chat_history", state["chat_history"])
        state["final_response"] = result.get("final_response", "")
        
        # Stream the output (simulate streaming with print)
        response = result.get("final_response", "I'm not sure how to help with that.")
        print(response)
        
        turn += 1


# ══════════════════════════════════════════════════════════════════════════════
# 6. DEMO MODE (non-interactive — for testing)
# ══════════════════════════════════════════════════════════════════════════════

def run_demo():
    """Run a quick demo with pre-set questions."""
    demo_questions = [
        "Hi! My name is Keshav and I'm learning AI.",
        "What is LangChain and what are its main components?",
        "Calculate compound interest: 10000 at 8% for 5 years (10000 * (1.08 ** 5))",
        "Create a 5-day study plan for learning LangChain",
        "What's my name again?",   # Tests memory
    ]
    
    state = {
        "user_input": "",
        "intent": "",
        "chat_history": [],
        "tool_results": [],
        "final_response": "",
        "session_info": {},
    }
    
    print("\n" + "🎬 DEMO MODE — Running pre-set questions\n" + "═" * 60)
    
    for i, question in enumerate(demo_questions):
        print(f"\n[Q{i+1}] {question}")
        print("─" * 60)
        
        state["user_input"] = question
        result = assistant.invoke(state)
        
        state["chat_history"] = result.get("chat_history", state["chat_history"])
        state["final_response"] = result.get("final_response", "")
        
        print(f"ARIA: {result.get('final_response', '')[:300]}")
        if len(result.get("final_response", "")) > 300:
            print("  ... [truncated for demo]")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        run_demo()
    else:
        run_assistant()
