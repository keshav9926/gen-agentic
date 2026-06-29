# Agentic AI — Advanced Orchestration & Systems

Welcome to the **Agentic AI** curriculum! This folder contains a progressive, hands-on guide to building advanced autonomous agent architectures. 

It is designed as a direct sequel to the foundations in the `gen_ai` folder, focusing purely on advanced orchestration patterns, stateful control graphs, human intervention, and multi-agent coordination.

---

## 📁 Curriculum structure

Here is the sequential layout of the topics you will cover:

```
agentic_ai/
│
├── step_01_tool_calling_mechanics.py   # Manual tool-calling handshakes & error handling
├── step_02_react_from_scratch.py       # ReAct loops from scratch using plain Python while-loops
├── step_03_langgraph_persistence.py    # Stateful persistence with checkpointers & thread memory
├── step_04_human_in_the_loop.py        # Interrupts, manual review breakpoints, and state updates
├── step_05_corrective_rag.py           # Corrective RAG (CRAG) with self-grading and web fallbacks
├── step_06_plan_and_execute.py         # Plan-and-Execute planners, execution loops, and replanners
├── step_07_multiagent_supervisor.py    # Multi-Agent systems: Supervisor pattern & task delegation
├── step_08_capstone_assistant.py       # Production-style assistant with persistent state graph
├── step_09_agent_interview_prep.py     # Q&A covering core agentic design patterns and trade-offs
└── step_10_learned_summary.md          # Comprehensive summary of key takeaways and concepts
```

---

## 📚 What Each Step Teaches You

### 1. Manual Tool-Calling Mechanics (`step_01_tool_calling_mechanics.py`)
* Understanding how `.bind_tools()` maps Python schemas to JSON payloads.
* Parsing structural tool outputs and mapping execution results to `ToolMessage` instances.
* Graceful tool execution error handling and feeding execution warnings back to the model.

### 2. ReAct Loop From Scratch (`step_02_react_from_scratch.py`)
* Building the classic Reason-Action-Observation loop using a basic Python `while` loop.
* Preventing infinite loops in production with step limits (`max_iterations`).
* Constructing intermediate messages dynamically and feeding them back to the LLM.

### 3. LangGraph State Persistence (`step_03_langgraph_persistence.py`)
* Introduction to `MemorySaver` checkpointers for persistent state management.
* Isolating conversations using thread configurations (`thread_id`).
* Inspecting graph execution states, history, and checkpoint values at runtime.

### 4. Human-in-the-Loop & Interventions (`step_04_human_in_the_loop.py`)
* Pausing execution before entering specific nodes using `interrupt_before`.
* Simulating human review check gates for high-stakes tool actions (like wire transfers).
* Editing state values directly at runtime and resuming graph execution cleanly.

### 5. Corrective RAG (CRAG) (`step_05_corrective_rag.py`)
* Document self-grading nodes to score retrieved chunks for relevance.
* Branching retrieval paths: returning direct answers, performing query translation, or falling back to web search.
* Building a complete self-correcting information gathering system.

### 6. Plan-and-Execute Agent (`step_06_plan_and_execute.py`)
* Mitigating reasoning loss in complex, multi-step tasks by separating planning from execution.
* Implementing a **Planner** node to draft a sequential checklist of sub-tasks.
* Implementing an **Executor** node to run tools, and a **Replanner** to update the checklist based on feedback.

### 7. Multi-Agent Supervisor (`step_07_multiagent_supervisor.py`)
* Orchestrating multi-agent collaboration with a central **Supervisor LLM**.
* Dynamically delegating specialized tasks to child agents (Research Agent, Writing Agent).
* Routing control flow and passing state context back to the Supervisor node.

### 8. Capstone Assistant (`step_08_capstone_assistant.py`)
* Designing a production-grade multi-intent router agent.
* Isolating intent classification from tool execution to save token costs and improve stability.
* Incorporating persistent session-based state and interactive terminal-based chat.

### 9. Agent Interview Prep (`step_09_agent_interview_prep.py`)
* Reviewing key architectural decisions and system design tradeoffs.
* Interactive Q&A covering state management, loop prevention, human-in-the-loop, and multi-agent coordination.

### 10. Learned Summary (`step_10_learned_summary.md`)
* A summarized review of the core agentic concepts, insights, and structural differences compared to standard RAG and prompting pipelines.
