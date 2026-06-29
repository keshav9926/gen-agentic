# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║       AI STUDY BUDDY — STEP 5: Giving It Superpowers            ║
║                (Tools + Agents + LangGraph)                     ║
╚══════════════════════════════════════════════════════════════════╝

WHAT WE'RE BUILDING TODAY:
  An autonomous Study Buddy that can:
    1. Answer from your notes (RAG — Step 4)
    2. Search the web for recent info
    3. Generate a quiz on any topic
    4. Evaluate your answer and give feedback
    5. Decide ON ITS OWN which tool to use

PROBLEM WITH STEP 4:
  Our Study Buddy is reactive and limited to its notes.
  It can't:
    - Find recent papers or news
    - Generate a quiz and evaluate your answer
    - Take multi-step actions autonomously

  It answers ONE question at a time, then stops.

CONCEPT: What is an AI Agent?
───────────────────────────────
  A CHAIN decides the path in advance (fixed sequence of steps).
  An AGENT decides the path dynamically based on observations.

  Chain:  prompt → LLM → parse → done
  Agent:  LLM decides → tool → observe → LLM decides → tool → ... → done

  The "thought loop" (ReAct):
  ┌─────────────────────────────────────────────────┐
  │  Thought: What do I need to do?                 │
  │  Action:  Call a tool                           │
  │  Observation: Read tool result                  │
  │  Thought: Is this enough? → if not, loop again  │
  │  Final Answer: Return to user                   │
  └─────────────────────────────────────────────────┘

CONCEPT: LangGraph
────────────────────
  LangGraph models agents as a GRAPH:
    - NODES = actions (run LLM, run tool, check condition)
    - EDGES = transitions (go to next node based on state)
    - STATE = shared data that flows through all nodes

  Why not just use AgentExecutor?
  → LangGraph gives you full control:
    - Custom control flow (loops, branches, parallel steps)
    - Persistent state across multiple turns
    - Human-in-the-loop checkpoints
    - Built-in streaming of intermediate steps

  This is the current production standard for complex agents.

CONCEPT: Tools
────────────────
  A tool is any Python function the LLM can CHOOSE to call.
  The LLM sees: tool name + description + parameters.
  It generates a "tool call" → LangGraph executes it → result returned.

  The description is critical — the LLM uses it to decide WHEN to use the tool.
  Bad:  "search(query)" — vague
  Good: "Search the web for recent news, papers, or facts not in study notes."
"""

from dotenv import load_dotenv
load_dotenv()

import json
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Load the vector DB we built in Step 4
CHROMA_DB_PATH = "./study_buddy_db"
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedder)
    print(f"Loaded vector DB: {vectorstore._collection.count()} documents")
except Exception:
    vectorstore = None
    print("No vector DB found — run step_04_rag.py first to create it")


# ─────────────────────────────────────────────────────────────────
# LESSON 5A: Define Tools — Functions the Agent Can Call
# ─────────────────────────────────────────────────────────────────
# The @tool decorator:
#   1. Converts the function into a LangChain Tool object
#   2. Uses the docstring as the description the LLM reads
#   3. Uses type hints to build the parameter schema
#
# CRITICAL: Write clear, specific docstrings. The LLM uses them
#           to decide WHEN and HOW to use each tool.

@tool
def search_study_notes(query: str) -> str:
    """
    Search the student's personal study notes for information about Gen AI concepts.
    Use this when the student asks about topics covered in their notes:
    transformers, attention, LoRA, RAG, RLHF, embeddings.
    Returns relevant excerpts with source citations.
    """
    if not vectorstore:
        return "Study notes not available. Please run step_04_rag.py first."

    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "No relevant notes found for this query."

    results = []
    for doc in docs:
        results.append(
            f"[{doc.metadata.get('topic', 'notes').upper()}]: {doc.page_content}"
        )
    return "\n\n".join(results)


@tool
def generate_quiz_question(topic: str) -> str:
    """
    Generate a multiple-choice quiz question to test understanding of a Gen AI topic.
    Use this when the student asks to be tested, quizzed, or wants practice questions.
    Topics: transformers, attention, RAG, LoRA, fine-tuning, agents, embeddings, RLHF.
    Returns a question with 4 options and the correct answer.
    """
    quiz_prompt = f"""Generate a multiple-choice quiz question about: {topic}

Format EXACTLY as:
QUESTION: [The question]
A) [Option A]
B) [Option B]  
C) [Option C]
D) [Option D]
ANSWER: [A/B/C/D]
EXPLANATION: [Why this is correct, 1-2 sentences]"""

    quiz_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
    return quiz_llm.invoke(quiz_prompt).content


@tool
def evaluate_answer(question: str, student_answer: str, correct_answer: str) -> str:
    """
    Evaluate a student's answer to a quiz question and provide detailed feedback.
    Use this when you have a question, the student's answer, and the correct answer.
    Returns encouragement, correctness, and explanation.
    """
    eval_prompt = f"""Evaluate this student's answer:

QUESTION: {question}
STUDENT'S ANSWER: {student_answer}
CORRECT ANSWER: {correct_answer}

Provide:
1. Was the student correct? (Yes/No/Partially)
2. What they got right
3. What they missed or misunderstood (if anything)
4. A brief encouraging message
Keep it under 100 words."""

    return llm.invoke(eval_prompt).content


@tool
def explain_concept_with_analogy(concept: str, student_background: str = "beginner") -> str:
    """
    Give a deep explanation of a Gen AI concept using a real-world analogy.
    Use this when a student wants to truly understand something, not just a quick answer.
    Especially useful for: attention mechanism, backpropagation, embeddings, tokenization.
    """
    prompt = f"""Explain '{concept}' to a {student_background} using:
1. One sentence definition
2. A real-world analogy (not tech-related)
3. How it works step by step (3-4 steps)
4. Why it matters / what problem it solves
Keep it engaging and clear."""

    return llm.invoke(prompt).content


# ─────────────────────────────────────────────────────────────────
# LESSON 5B: Agent State — The Memory of the Graph
# ─────────────────────────────────────────────────────────────────
# State is a TypedDict that flows through every node in the graph.
# Every node READS from state and WRITES to state.
# This is how nodes communicate with each other.

class StudyBuddyState(TypedDict):
    messages: list          # Full conversation history including tool calls
    student_name: str       # Remember who we're talking to
    current_topic: str      # Track what topic we're on


# ─────────────────────────────────────────────────────────────────
# LESSON 5C: Build the LangGraph Agent
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("Building LangGraph Agent...")
print("=" * 60)

tools = [search_study_notes, generate_quiz_question, evaluate_answer, explain_concept_with_analogy]

# Bind tools to LLM so it knows what it can call
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are Study Buddy, an expert Gen AI tutor and learning coach.

You have access to these tools:
- search_study_notes: Search the student's personal study notes
- generate_quiz_question: Generate a quiz to test understanding
- evaluate_answer: Evaluate a student's quiz answer
- explain_concept_with_analogy: Give a deep explanation with analogies

Strategy:
- For factual questions about Gen AI: search study notes first
- For "test me" or "quiz me": generate a quiz question
- For "explain this deeply": use explain_concept_with_analogy
- Always be encouraging, patient, and pedagogically sound
- After answering, suggest what to study next

Current student: {student_name}"""


# NODE 1: The LLM decision maker
def agent_node(state: StudyBuddyState) -> StudyBuddyState:
    """
    The brain of the agent.
    Looks at current state, decides: answer directly OR call a tool.
    """
    system = SYSTEM_PROMPT.format(student_name=state.get("student_name", "Student"))

    # Prepend system message to conversation
    messages_with_system = [SystemMessage(content=system)] + state["messages"]

    # LLM either generates text or generates a tool_call
    response = llm_with_tools.invoke(messages_with_system)

    # Add response to message history
    return {"messages": state["messages"] + [response]}


# NODE 2: The tool executor
tool_node = ToolNode(tools)  # prebuilt: executes whatever tool the LLM called


# EDGE: Routing function — after LLM responds, where do we go?
def should_call_tool(state: StudyBuddyState) -> str:
    """
    Check the last message:
    - If it has tool_calls → go to tool_node
    - If it doesn't       → we're done, go to END
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# BUILD THE GRAPH
graph = StateGraph(StudyBuddyState)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

# Set entry point
graph.set_entry_point("agent")

# Add edges
graph.add_conditional_edges(
    "agent",
    should_call_tool,
    {
        "tools": "tools",   # if tool needed → go to tools
        "end": END,         # if done → stop
    }
)
graph.add_edge("tools", "agent")  # after tool runs → back to agent

# Compile the graph
study_buddy_agent = graph.compile()

print("Agent graph compiled successfully!")
print("Nodes:", list(study_buddy_agent.get_graph().nodes.keys()))


# ─────────────────────────────────────────────────────────────────
# LESSON 5D: Run the Agent
# ─────────────────────────────────────────────────────────────────
def chat_with_agent(
    user_message: str,
    conversation_history: list = None,
    student_name: str = "Keshav"
) -> tuple[str, list]:
    """
    Send a message to the Study Buddy agent.
    Returns (response_text, updated_history).
    """
    if conversation_history is None:
        conversation_history = []

    # Add new user message
    new_messages = conversation_history + [HumanMessage(content=user_message)]

    # Run the agent
    final_state = study_buddy_agent.invoke({
        "messages": new_messages,
        "student_name": student_name,
        "current_topic": "",
    })

    # Extract the final AI response (last AIMessage without tool calls)
    final_answer = ""
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and not (hasattr(msg, "tool_calls") and msg.tool_calls):
            final_answer = msg.content
            break

    return final_answer, final_state["messages"]


# ─────────────────────────────────────────────────────────────────
# TEST THE AGENT
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TESTING THE AGENT")
print("=" * 60)

history = []

# Test 1: Knowledge question (should use search_study_notes)
print("\n[Test 1: Factual Question]")
response, history = chat_with_agent(
    "What is the formula for the attention mechanism?",
    history
)
print(f"Agent: {response[:300]}...")

# Test 2: Quiz request (should use generate_quiz_question)
print("\n[Test 2: Quiz Request]")
response, history = chat_with_agent(
    "Quiz me on RAG!",
    history
)
print(f"Agent: {response[:300]}...")

# Test 3: Deep explanation (should use explain_concept_with_analogy)
print("\n[Test 3: Deep Explanation]")
response, history = chat_with_agent(
    "I still don't really get embeddings. Explain it like I've never coded before.",
    history
)
print(f"Agent: {response[:300]}...")


# ─────────────────────────────────────────────────────────────────
# PUTTING IT TOGETHER: Interactive Study Buddy — Final Version
# ─────────────────────────────────────────────────────────────────
def run_full_study_buddy():
    """
    THE COMPLETE STUDY BUDDY:
    RAG + Memory + Tools + Autonomous Agent
    """
    print("\n" + "═" * 60)
    print("  🤖 STUDY BUDDY — Final Version (Agent + RAG + Memory)")
    print("  Type 'quit' to exit | Try: 'quiz me', 'explain X deeply'")
    print("═" * 60)

    name = input("What's your name? ").strip() or "Student"
    print(f"\nBuddy: Welcome, {name}! I'm your Gen AI Study Buddy.")
    print(f"       I have your study notes, can quiz you, and search for answers.")
    print(f"       What would you like to learn today?\n")

    history = []

    while True:
        user_input = input(f"{name}: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nBuddy: Great session! Keep learning! You've got this! 🚀")
            break
        if not user_input:
            continue

        print("\nBuddy: ", end="", flush=True)
        response, history = chat_with_agent(user_input, history, student_name=name)
        print(response)
        print()


if __name__ == "__main__":
    run_full_study_buddy()


"""
WHAT YOU LEARNED:
  ✅ Tools = Python functions with clear docstrings that the LLM can call
  ✅ LangGraph = agents as graphs (nodes + edges + state)
  ✅ State flows through the graph — nodes read and write to it
  ✅ Conditional edges = routing based on LLM decision
  ✅ The agent loop: agent → (tool?) → tool → agent → ... → END
  ✅ bind_tools() tells the LLM what tools are available

  HOW THE AGENT DECIDES WHICH TOOL TO USE:
    → The LLM reads tool names + docstrings
    → It picks the tool whose description best matches the user's intent
    → This is why docstring quality is critical for agents

  INTERVIEW INSIGHTS:
    "What is the difference between a chain and an agent?"
    → Chain: fixed sequence of steps, decided at build time
    → Agent: LLM decides what steps to take at runtime

    "How do you prevent an agent from looping forever?"
    → Set max_iterations in AgentExecutor
    → In LangGraph: add a step counter to state, conditional edge to END

    "What is LangGraph and why use it over AgentExecutor?"
    → LangGraph gives explicit control over flow
    → Supports cycles, branching, human-in-the-loop, streaming
    → AgentExecutor is a black box; LangGraph is transparent

══════════════════════════════════════════════════════════════════
  CONGRATULATIONS! 🎉
  You've built a complete Gen AI application from scratch:
  
  Step 1: LLM basics (tokens, temperature, message types)
  Step 2: Prompt engineering (templates, few-shot, CoT, parsers)  
  Step 3: Memory (context window, session isolation)
  Step 4: RAG (embeddings, vector DB, retrieval)
  Step 5: Agents (tools, LangGraph, autonomous decision-making)

  You now understand EVERY layer of a production AI application.
══════════════════════════════════════════════════════════════════
"""
