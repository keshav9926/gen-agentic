"""
08 - Advanced: Multi-Agent Systems with LangGraph
==================================================

CONCEPTS COVERED:
  - Supervisor pattern — one LLM routes to specialized sub-agents
  - Sub-agents — each has its own tools and expertise
  - Handoffs — passing control between agents
  - Aggregating results from multiple agents
  - Real-world pattern: research + writing + review pipeline

MULTI-AGENT PATTERNS:
  1. Supervisor → Routes to specialized agents based on task type
  2. Sequential  → Agent A's output goes to Agent B (pipeline)
  3. Hierarchical → Nested agents (supervisor of supervisors)
  4. Collaborative → Agents debate and vote on best answer

WHY MULTI-AGENT?
  - Specialization: each agent does one thing well
  - Parallelism: run multiple agents at the same time
  - Scale: break complex problems into sub-problems
  - Separation of concerns: researcher ≠ writer ≠ reviewer
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, MessagesState
from typing import TypedDict, Annotated, List, Literal
import operator

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: SUPERVISOR PATTERN
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART 1: Supervisor Pattern")
print("=" * 60)
print("""
Architecture:
    User Request
         ↓
    [SUPERVISOR] ← decides who should act
      /    |    \\
[Research] [Code] [Write]   ← specialized sub-agents
      \\    |    /
    [SUPERVISOR] ← collects result, decides next or END
         ↓
    Final Answer
""")

# ── Sub-agent tools ───────────────────────────────────────────────────────────
@tool
def research_topic(query: str) -> str:
    """Research a topic and return factual information."""
    # In production: web search, database query, etc.
    response = llm.invoke(f"Provide factual information about: {query}. Be concise, 3-4 sentences.")
    return response.content

@tool
def write_content(topic: str, research: str, tone: str = "professional") -> str:
    """Write polished content based on research."""
    prompt = f"""Write a short, {tone} paragraph about {topic}.
Use this research as your source:
{research}"""
    response = llm.invoke(prompt)
    return response.content

@tool
def review_content(content: str) -> str:
    """Review content and provide a quality score and suggestions."""
    prompt = f"""Review this content for quality, accuracy, and clarity.
Content: {content}
Provide: 1) Quality score (1-10) 2) One improvement suggestion"""
    response = llm.invoke(prompt)
    return response.content

@tool
def run_python_code(code: str) -> str:
    """Execute a simple Python expression safely."""
    try:
        result = eval(code, {"__builtins__": {"range": range, "len": len, "sum": sum, "print": print}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# ── Supervisor State ──────────────────────────────────────────────────────────
class SupervisorState(TypedDict):
    user_request: str
    research_result: str
    written_content: str
    review_result: str
    current_agent: str
    final_response: str
    steps_taken: Annotated[List[str], operator.add]   # append-mode list

# ── Supervisor node — decides who acts next ───────────────────────────────────
def supervisor(state: SupervisorState) -> dict:
    """LLM-based supervisor: decides next action based on state."""
    context = f"""You are a supervisor coordinating a content creation pipeline.
    
Current task: {state['user_request']}
Steps completed: {state.get('steps_taken', [])}
Research done: {'Yes' if state.get('research_result') else 'No'}
Content written: {'Yes' if state.get('written_content') else 'No'}
Content reviewed: {'Yes' if state.get('review_result') else 'No'}

Based on what's been done, what should happen NEXT?
Reply with exactly ONE word: 'research', 'write', 'review', or 'finish'"""
    
    response = llm.invoke(context)
    next_action = response.content.strip().lower()
    
    # Clean up response
    for word in ["research", "write", "review", "finish"]:
        if word in next_action:
            next_action = word
            break
    else:
        # Default logic if LLM response is unclear
        if not state.get("research_result"):
            next_action = "research"
        elif not state.get("written_content"):
            next_action = "write"
        elif not state.get("review_result"):
            next_action = "review"
        else:
            next_action = "finish"
    
    return {"current_agent": next_action}

def researcher_agent(state: SupervisorState) -> dict:
    """Research agent: gathers information."""
    print("  🔍 Researcher working...")
    result = research_topic.invoke({"query": state["user_request"]})
    return {
        "research_result": result,
        "steps_taken": ["research_done"],
    }

def writer_agent(state: SupervisorState) -> dict:
    """Writer agent: creates content from research."""
    print("  ✍️  Writer working...")
    result = write_content.invoke({
        "topic": state["user_request"],
        "research": state.get("research_result", "No research available"),
    })
    return {
        "written_content": result,
        "steps_taken": ["writing_done"],
    }

def reviewer_agent(state: SupervisorState) -> dict:
    """Reviewer agent: checks quality."""
    print("  🔎 Reviewer working...")
    result = review_content.invoke({"content": state.get("written_content", "")})
    return {
        "review_result": result,
        "steps_taken": ["review_done"],
    }

def finalize(state: SupervisorState) -> dict:
    """Compile the final response."""
    print("  📋 Finalizing...")
    final = f"""FINAL DELIVERABLE
================
Topic: {state['user_request']}

RESEARCHED FACTS:
{state.get('research_result', 'N/A')}

WRITTEN CONTENT:
{state.get('written_content', 'N/A')}

REVIEW:
{state.get('review_result', 'N/A')}"""
    return {"final_response": final}

def route_from_supervisor(state: SupervisorState) -> str:
    """Route to the correct agent based on supervisor's decision."""
    return state.get("current_agent", "research")

# Build supervisor graph
s_builder = StateGraph(SupervisorState)
s_builder.add_node("supervisor",  supervisor)
s_builder.add_node("researcher",  researcher_agent)
s_builder.add_node("writer",      writer_agent)
s_builder.add_node("reviewer",    reviewer_agent)
s_builder.add_node("finalize",    finalize)

s_builder.set_entry_point("supervisor")
s_builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "research": "researcher",
        "write":    "writer",
        "review":   "reviewer",
        "finish":   "finalize",
    }
)
# After each agent, go back to supervisor
s_builder.add_edge("researcher", "supervisor")
s_builder.add_edge("writer",     "supervisor")
s_builder.add_edge("reviewer",   "supervisor")
s_builder.add_edge("finalize",   END)

supervisor_graph = s_builder.compile()

print("Running supervisor pipeline for: 'Quantum Computing basics'")
result = supervisor_graph.invoke({
    "user_request": "Quantum Computing basics",
    "research_result": "",
    "written_content": "",
    "review_result": "",
    "current_agent": "",
    "final_response": "",
    "steps_taken": [],
})
print("\n" + result["final_response"][:600] + "...")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: PARALLEL AGENT EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 2: Parallel Agents — fan-out, then aggregate")
print("=" * 60)

from langchain_core.runnables import RunnableParallel

# Three agents running simultaneously
def analyze_pros(topic: str) -> str:
    response = llm.invoke(f"List 3 key advantages of: {topic}")
    return response.content

def analyze_cons(topic: str) -> str:
    response = llm.invoke(f"List 3 key disadvantages of: {topic}")
    return response.content

def market_analysis(topic: str) -> str:
    response = llm.invoke(f"Describe the market opportunity for: {topic} in 2-3 sentences.")
    return response.content

# Run all three in parallel (simultaneously!)
parallel_pipeline = RunnableParallel(
    pros=lambda x: analyze_pros(x["topic"]),
    cons=lambda x: analyze_cons(x["topic"]),
    market=lambda x: market_analysis(x["topic"]),
)

results = parallel_pipeline.invoke({"topic": "AI-powered customer service chatbots"})

print("📊 PARALLEL ANALYSIS RESULTS:")
print("\n✅ PROS:")
print(results["pros"])
print("\n❌ CONS:")
print(results["cons"])
print("\n📈 MARKET:")
print(results["market"])


"""
KEY TAKEAWAYS:
  ✅ Supervisor pattern: one LLM orchestrates, others specialize
  ✅ State flows through the graph — each agent reads + writes to shared state
  ✅ Cycles let the supervisor loop until all sub-tasks are complete
  ✅ RunnableParallel = easy fan-out for independent sub-tasks
  ✅ Multi-agent = modularity, specialization, parallelism
  ✅ In production: each agent can be a full LangGraph graph itself

YOU'VE COMPLETED THE FULL LANGCHAIN CURRICULUM! 🎉

Summary of what you've learned:
  01_basics/    → LLM invoke, streaming, batch, output parsers
  02_prompts/   → Templates, few-shot, partial, message placeholder
  03_chains/    → LCEL, pipe operator, parallel/branch/lambda
  04_memory/    → Chat history, session memory, trimming
  05_agents/    → ReAct agents, tool use, multi-step reasoning
  06_rag/       → Document loading, embeddings, vector stores, RAG pipeline
  07_tools/     → @tool, bind_tools, tool execution loop
  08_advanced/  → LangGraph StateGraph, conditional edges, loops, multi-agent
"""
