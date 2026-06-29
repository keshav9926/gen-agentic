# 🎓 Agentic AI Curriculum Summary Sheet

A concise step-by-step summary of all advanced concepts, code snippets, and interview tips covered in your Agentic AI curriculum.

---

### Step 1: Tool Calling & Execution Handshake
* **Core Concepts**:
  * Tool calling is a protocol where the LLM decides *which* tool to call and with *what* arguments by outputting structured JSON.
  * The actual execution of Python code happens on your client/orchestration application (not in the LLM).
  * `ToolMessage` requires a matching `tool_call_id` to align the result with the query.
* **Key Code**:
  ```python
  @tool
  def search_db(query: str) -> str:
      """Search DB."""
      return "info"
  
  llm_with_tools = llm.bind_tools([search_db])
  response = llm_with_tools.invoke([HumanMessage(content="Search DB for widgets")])
  # Extract tool call
  tool_call = response.tool_calls[0]
  # Execute and wrap in ToolMessage
  result = search_db.invoke(tool_call["args"])
  tool_msg = ToolMessage(content=result, tool_call_id=tool_call["id"], name="search_db")
  ```
* **Interview Insight**: *How do you handle tool execution failures?* Intercept the error in a try/except block, format the exception as the ToolMessage content, and set the status to "error" so the LLM knows it failed and can re-attempt.

---

### Step 2: ReAct Agent Loop From Scratch
* **Core Concepts**:
  * Implementing a ReAct (Reason + Act) loop manually with a standard Python `while` loop.
  * Safeguarding against infinite loops by specifying a `max_iterations` counter.
  * Dynamic execution: the model evaluates tool feedback (observations) and decides on subsequent actions.
* **Key Code**:
  ```python
  messages = [HumanMessage(content=query)]
  for _ in range(max_iterations):
      response = llm_with_tools.invoke(messages)
      messages.append(response)
      if not response.tool_calls:
          break # final answer reached
      for tc in response.tool_calls:
          res = run_tool(tc["name"], tc["args"])
          messages.append(ToolMessage(content=res, tool_call_id=tc["id"]))
  ```
* **Interview Insight**: *Why is max_iterations crucial?* If a tool returns a persistent error or the model gets confused, it can enter a loop calling the same tool repeatedly, leading to massive API costs.

---

### Step 3: LCEL Deep Dive
* **Core Concepts**:
  * Composing chains declaratively using the pipe `|` operator.
  * `RunnablePassthrough`: Passing data through or merging new keys with `.assign()`.
  * `RunnableLambda`: Wrapping standard Python functions into the chain.
  * `RunnableParallel`: Executing multiple chains in parallel to reduce latency.
  * `RunnableBranch`: Creating conditional routing rules.
* **Key Code**:
  ```python
  chain = (
      RunnablePassthrough.assign(summary=summary_chain)
      | final_prompt
      | llm
  )
  ```
* **Interview Insight**: *What interface do all LCEL runnables expose?* They support `.invoke()`, `.stream()`, `.batch()`, and `.with_fallbacks()`.

---

### Step 4: LangGraph State Persistence (MemorySaver)
* **Core Concepts**:
  * Checkpointers serialize and persist graph State after every step.
  * Thread configurations (`thread_id`) isolate different user/student sessions.
  * Reducer functions (like `add_messages`) describe how state property updates are merged.
* **Key Code**:
  ```python
  from langgraph.checkpoint.memory import MemorySaver
  memory = MemorySaver()
  graph = builder.compile(checkpointer=memory)
  # Invoke with thread config
  config = {"configurable": {"thread_id": "user_abc"}}
  graph.invoke(state_input, config)
  ```
* **Interview Insight**: *How does a checkpointer work in stateless server architectures?* When a request arrives, the server passes the `thread_id` configuration. LangGraph retrieves the corresponding checkpoint from SQLite/PostgreSQL, resumes execution, and commits a new checkpoint after completion.

---

### Step 5: Human-in-the-Loop (HITL) & Breakpoints
* **Core Concepts**:
  * Breakpoints (`interrupt_before` / `interrupt_after`) pause execution before critical nodes.
  * Humans can inspect the state, edit the proposed tool arguments, and resume execution.
* **Key Code**:
  ```python
  graph = builder.compile(checkpointer=memory, interrupt_before=["tools"])
  # Runs up to 'tools' and pauses.
  graph.invoke(inputs, config)
  # Update proposed tool calls in state:
  graph.update_state(config, {"messages": [modified_message]}, as_node="agent")
  # Resume
  graph.invoke(None, config)
  ```
* **Interview Insight**: *How do you override an agent's decision in LangGraph?* Use `update_state(config, state_delta, as_node)` by targeting the node that generated the decision. Using the same message `id` replaces it in state history.

---

### Step 6: Corrective RAG (CRAG)
* **Core Concepts**:
  * Document grading loops evaluate retrieval relevance before generator input.
  * Deletes noise (irrelevant context) and triggers web search if retrieval quality is low.
* **Key Code**:
  ```python
  # Conditional routing decision:
  def decide_routing(state):
      return "web_search" if state["web_search_needed"] else "generate"
  ```
* **Interview Insight**: *What is the core benefit of CRAG?* It bridges vocabulary or contextual gaps. If vector retrieval returns noise, CRAG filters it out and falls back to a search engine, reducing generator hallucinations.

---

### Step 7: Plan-and-Execute Agents
* **Core Concepts**:
  * Planner: breaks a complex user query into a structured list of tasks (usually via Pydantic).
  * Executor: executes the current step (often another agent).
  * Replanner: reviews outcomes, updates the step list, and decides if complete.
* **Key Code**:
  ```python
  class Plan(BaseModel):
      steps: List[str]
  # In Graph: planner -> executor -> replanner -> (loop or END)
  ```
* **Interview Insight**: *Why is Plan-and-Execute superior for long-horizon goals?* It splits task decomposition from execution, ensuring the agent keeps a global view of the user goal and does not get distracted by intermediate observations.

---

### Step 8: Multi-Agent Systems (Supervisor)
* **Core Concepts**:
  * Central coordinator LLM delegates tasks to specialized sub-agents.
  * Sub-agents focus on their domain, use specific tools, and return control back to the supervisor.
  * Reduces prompt/context sizes and increases execution accuracy.

---

### Step 9: Capstone Router Agent
* **Core Concepts**:
  * Intent classification node routes user requests to intent-specific execution graphs.
  * Isolates calculation, research, and study plan tools to reduce costs and latency.
