# LangChain: Advanced Concepts Guide (Unique to agentic_ai)

This project has been analyzed and trimmed against your existing `gen_ai/study_buddy` curriculum to keep only the advanced topics that were **not** taught there.

---

## 📁 Unique Advanced Topics Kept

```
agentic_ai/
│
├── 03_chains/
│   └── 01_lcel_chains.py          # Detailed LCEL composition (RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableBranch)
│
├── 08_advanced/
│   └── 02_multiagent.py           # Multi-Agent systems: Supervisor pattern, parallel agent workflows, and task handoffs
│
├── capstone_project.py            # CAPSTONE — Multi-intent router state graph
├── req.txt                        # Dependencies
└── README.md                      # This file
```

---

## 📚 What These Advanced Files Teach You

### 1. Detailed LCEL Composition (`03_chains/01_lcel_chains.py`)
While your `study_buddy` lessons introduce the basic `prompt | llm | parser` chain, this file goes deep into the advanced composition elements of LangChain Expression Language (LCEL):
* **`RunnablePassthrough`**: How to pass data through unmodified or dynamically inject extra keys into the inputs.
* **`RunnableLambda`**: Wrapping arbitrary Python functions into a pipeline so they behave like native LangChain runnables.
* **`RunnableParallel`**: Fanning out and running multiple independent prompts or tasks concurrently to save latency.
* **`RunnableBranch`**: Creating conditional logic pipelines (routing queries to different prompt/LLM chains based on content).

---

### 2. Multi-Agent Systems (`08_advanced/02_multiagent.py`)
Your `study_buddy` agent (Step 5) focuses on a single ReAct agent. This file introduces multi-agent design patterns:
* **The Supervisor Pattern**: A central coordinator LLM receives the user query, evaluates the state, and delegates the next sub-task to specialized agent nodes.
* **Collaboration & Handoffs**: Sub-agents running tasks (Research, Writing, Reviewing) and returning control/data back to the Supervisor.
* **Parallel Workflows**: Fanning out separate specialist agents to analyze a topic simultaneously and aggregating their opinions.

---

### 3. Capstone Router Agent (`capstone_project.py`)
A production-inspired custom state graph combining:
* **Intent Classification Node**: Pre-routing the query to reduce token usage and improve prompt accuracy.
* **Intent-Specific Graph Execution**: Executing targeted tool-agents depending on the intent (Calculator, Knowledge Base RAG, study plan generator).
* **Stateful History Isolation**: Session-based memory across multiple user turns in a clean interactive console.

---

## 🚀 Quick Start

1. **Activate venv**: 
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
2. **Run any file**:
   ```powershell
   python 03_chains/01_lcel_chains.py
   python 08_advanced/02_multiagent.py
   python capstone_project.py
   ```
