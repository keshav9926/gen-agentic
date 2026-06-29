# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║         AGENTIC AI — STEP 9: Agentic AI Interview Prep          ║
╚══════════════════════════════════════════════════════════════════╝

An interactive interview preparation terminal utility covering key
architectural questions and design patterns in Agentic AI.
"""

import sys

INTERVIEW_BANK = [
    {
        "question": "What is the primary difference between Naive RAG and Corrective RAG (CRAG)?",
        "key_points": [
            "Naive RAG: directly feeds retrieved docs to LLM without verification.",
            "CRAG: adds a Document Grader node to filter out irrelevant docs.",
            "CRAG: triggers web search fallback if retrieval score is too low."
        ],
        "model_answer": (
            "Naive RAG retrieves top-k chunks from a vector DB and directly injects "
            "them into the prompt context, which can cause hallucinations if the retrieved "
            "data is irrelevant. Corrective RAG (CRAG) implements a grader model to evaluate "
            "each document's relevance first. If retrieval quality is poor or insufficient, "
            "it filters out the irrelevant parts and triggers a web search fallback to gather "
            "supplementary, up-to-date context before generating the final answer."
        )
    },
    {
        "question": "How does Tool Calling work under the hood in LLMs?",
        "key_points": [
            "LLM does NOT run code itself.",
            "Handshake: developer registers tools with JSON schemas.",
            "Model outputs JSON specifying tool name and arguments.",
            "Developer's runtime executes the function and returns ToolMessage."
        ],
        "model_answer": (
            "Tool calling is a two-step handshake: the developer binds tool schemas "
            "(JSON schema defining tool names, descriptions, and arguments) to the model. "
            "The model decides if it needs a tool; if so, it halts text generation and outputs "
            "a structured JSON object with the tool name and argument values. The client application "
            "runs the actual Python function, creates a ToolMessage with the output using the "
            "matching tool_call_id, and feeds it back to the LLM to continue text generation."
        )
    },
    {
        "question": "How do you prevent a ReAct agent from looping infinitely?",
        "key_points": [
            "Set max_iterations on the agent executor loop.",
            "Implement timeouts (max execution time limit).",
            "Keep a step counter in the LangGraph State and route to END if exceeded."
        ],
        "model_answer": (
            "Infinite loops occur when tools return errors or unexpected results, causing "
            "the LLM to repeatedly call the same tool. To prevent this in production: "
            "(1) Maintain a strict execution limit (max_iterations) inside the orchestrator loop. "
            "(2) Inject a step counter inside the LangGraph State, and use a conditional edge "
            "routing to END if the count exceeds the threshold. (3) Set overall wall-clock timeouts."
        )
    },
    {
        "question": "Explain the role of Checkpointers (State Persistence) in LangGraph.",
        "key_points": [
            "Persists State to memory or database after each node execution.",
            "Thread configurations isolate user session states.",
            "Enables multi-turn conversations and state history tracking."
        ],
        "model_answer": (
            "Checkpointers (like MemorySaver or SqliteSaver) automatically serialize and "
            "persist the graph's State dictionary after each node runs. This allows the system "
            "to reload the exact state of a conversation or agent workflow across distinct API requests. "
            "By supplying a thread_id config, LangGraph isolates sessions, enabling stateless web servers "
            "to support multi-turn conversational agents with full history persistence."
        )
    },
    {
        "question": "What is Human-in-the-Loop (HITL) and how do you implement it in LangGraph?",
        "key_points": [
            "Breakpoint pauses execution before/after critical nodes.",
            "Allows reviewing proposed actions and editing states.",
            "Resumes by calling the graph with input=None."
        ],
        "model_answer": (
            "Human-in-the-loop (HITL) guarantees safety by pausing the agent before executing "
            "sensitive tools (e.g. database updates or wire transfers). In LangGraph, we compile "
            "the graph with the interrupt_before=['tools'] parameter. This suspends the graph "
            "and saves the state. A human can inspect, approve, or use update_state() to modify "
            "the tool arguments, and then resume execution by calling invoke(None, config)."
        )
    },
    {
        "question": "Compare the Plan-and-Execute agent pattern with the ReAct pattern.",
        "key_points": [
            "ReAct: step-by-step decision making, high flexibility, prone to drift.",
            "Plan-and-Execute: separates planning from execution, highly stable.",
            "Use Plan-and-Execute for long-horizon tasks, ReAct for short-term tool usage."
        ],
        "model_answer": (
            "The ReAct pattern alternates between thinking and acting at every step, which is "
            "flexible but susceptible to planning drift or infinite loops on long-horizon tasks. "
            "Plan-and-Execute splits planning from execution: a Planner node outlines the steps first, "
            "an Executor agent runs the current step, and a Replanner node evaluates results, "
            "updating the remaining steps dynamically. This keeps the agent aligned on complex goals."
        )
    },
    {
        "question": "Explain the Supervisor pattern in Multi-Agent systems.",
        "key_points": [
            "Single coordinator LLM delegates tasks to specialized sub-agents.",
            "Sub-agents report results back to the supervisor.",
            "Reduces prompt sizes and segregates tool access."
        ],
        "model_answer": (
            "The Supervisor pattern coordinates multiple specialist sub-agents (e.g. Writer, Coder, Researcher). "
            "A central coordinator LLM acts as the manager, reviewing the conversation state, selecting "
            "which specialist agent to activate next, and reading their outputs. The sub-agents focus on their "
            "specific domains and report back to the supervisor, reducing token overhead and segregation of duties."
        )
    },
    {
        "question": "What are Reducers in LangGraph state schemas?",
        "key_points": [
            "Reducer functions specify how state updates are merged.",
            "Default behavior is overwriting the previous value.",
            "add_messages is a built-in reducer that appends or edits messages."
        ],
        "model_answer": (
            "Reducers are functions defined inside the State TypedDict using Annotated. "
            "By default, when a node returns a dictionary, LangGraph overwrites the corresponding "
            "fields. A reducer function overrides this behavior by detailing exactly how the new update "
            "should merge with the old state. For example, add_messages merges lists by appending "
            "new entries, and overwriting existing messages if their IDs match."
        )
    }
]

def run_quiz():
    print("=" * 60)
    print("       🎓 AGENTIC AI MOCK INTERVIEW TOOL")
    print("=" * 60)
    print("Test your architectural knowledge of advanced Agentic AI systems.")
    print("Type 'q' to quit at any time.\n")
    
    score = 0
    total = len(INTERVIEW_BANK)
    
    for i, item in enumerate(INTERVIEW_BANK):
        print(f"\n[Question {i+1}/{total}]")
        print(item["question"])
        
        user_ans = input("\nYour answer / notes: ").strip()
        if user_ans.lower() == 'q':
            print("\nExiting. Keep practicing! 🚀")
            sys.exit()
            
        print("\n--- Key Points to Cover ---")
        for point in item["key_points"]:
            print(f"  • {point}")
            
        print("\n--- Model Answer ---")
        print(item["model_answer"])
        print("-" * 60)
        
        self_eval = input("Did you cover most key points? (y/n): ").strip().lower()
        if self_eval == 'y':
            score += 1
            
    print(f"\n🏆 Practice complete! Self-Assessed Score: {score}/{total}")

if __name__ == "__main__":
    run_quiz()
