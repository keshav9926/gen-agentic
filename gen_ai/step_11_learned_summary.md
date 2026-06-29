# 🎓 Study Buddy: Complete Generative AI Project Summary
This summary file maps out everything you have learned across the 10 steps of the **AI Study Buddy** project. It is structured to be precise, actionable, and aligned with standard technical interview expectations.

---

## 🗺️ Step-by-Step Curriculum Map

### Step 1: Foundations of LLMs
* **Core Concepts**: 
  * How LLMs process tokens and predict subsequent text.
  * System Prompting (defining assistant personas, constraints).
  * Streaming responses token-by-token.
  * Temperature settings (`0.0` for deterministic, facts; `1.0+` for creative brainstorms).
* **Key Code**:
  ```python
  from langchain_groq import ChatGroq
  llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
  response = llm.invoke("What is deep learning?")
  ```
* **Interview Insight**: *Why use temperature 0 in production?* To make outputs deterministic, predictable, and clean for automated tests and downstream parsers.

---

### Step 2: Prompt Engineering & Structured Outputs
* **Core Concepts**:
  * Prompt Templates (dynamic {variables} populated at runtime).
  * Few-Shot Prompting (teaching formatting and behavior by providing input-output examples).
  * Chain-of-Thought (CoT) (instructing the model to "think step by step" to improve reasoning paths).
  * Structured JSON output using Pydantic schemas.
* **Key Code**:
  ```python
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.output_parsers import JsonOutputParser
  from pydantic import BaseModel, Field

  class Concept(BaseModel):
      name: str = Field(description="Name of the concept")

  parser = JsonOutputParser(pydantic_object=Concept)
  prompt = ChatPromptTemplate.from_messages([
      ("system", "Explain the concept. {format_instructions}"),
      ("human", "{concept}")
  ]).partial(format_instructions=parser.get_format_instructions())
  ```
* **Interview Insight**: *How do you guarantee an LLM outputs valid JSON?* Pass a strict validation schema via Pydantic using JSON Output Parsers, combined with system instructions.

---

### Step 3: Chat Memory & State Management
* **Core Concepts**:
  * Chat Message History (tracking Human/AI message objects).
  * Window Memory (keeping the last `N` messages to manage the context window).
  * Summary Memory (generating dynamic summaries of old messages to retain long-term context).
* **Key Code**:
  ```python
  from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
  memory = ConversationBufferWindowMemory(k=5)  # Keeps last 5 interactions
  ```
* **Interview Insight**: *What is the context window bottleneck?* LLMs have a maximum limit of tokens they can process. Storing all history naively eventually crashes or causes the model to ignore early details, necessitating windowing or summarization.

---

### Step 4: Foundations of RAG (Retrieval-Augmented Generation)
* **Core Concepts**:
  * Combining external documents with LLM prompting to prevent knowledge cutoffs.
  * Chunking strategies (chunk size and overlap) to preserve semantic contexts.
  * Local vector stores (ChromaDB) and local embeddings (`all-MiniLM-L6-v2`).
  * Preventing hallucinations by grounding prompts.
* **Key Code**:
  ```python
  from langchain_community.vectorstores import Chroma
  from langchain_community.embeddings import HuggingFaceEmbeddings
  
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  vectorstore = Chroma.from_documents(docs, embeddings)
  retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
  ```
* **Interview Insight**: *What is chunk overlap for?* It ensures that semantic concepts split at a chunk boundary aren't lost, keeping vital context linked across boundaries.

---

### Step 5: Autonomous Agents & Tool Calling
* **Core Concepts**:
  * ReAct (Reason + Act) Loop: LLM decides what tool to call, executes, observes result, and repeats.
  * Creating custom tools via Python decorators.
  * Binding tool definitions to chat models.
* **Key Code**:
  ```python
  from langchain_core.tools import tool

  @tool
  def search_db(query: str) -> str:
      """Search internal notes for query."""
      return "retrieved info"

  llm_with_tools = llm.bind_tools([search_db])
  ```
* **Interview Insight**: *How do you handle an agent stuck in an infinite loop?* Implement strict execution limits (`max_iterations` or timeout flags) in your agent orchestrator loop.

---

### Step 6: Advanced RAG Strategies
* **Core Concepts**:
  * Multi-Query Retrieval: Using an LLM to generate synonym queries to catch different lexical matches.
  * HyDE (Hypothetical Document Embeddings): Generating a fake answer first, then using its embedding to retrieve similar real docs.
  * Parent-Document Retrieval: Searching small child chunks (better matching) but returning parent documents (richer context) to the LLM.
* **Key Code**:
  ```python
  from langchain.retrievers.multi_query import MultiQueryRetriever
  from langchain_community.retrievers import ParentDocumentRetriever
  # Matches child chunks, fetches parent docs from InMemoryStore
  ```
* **Interview Insight**: *Why is naive semantic search sometimes insufficient?* User queries are often short or poorly worded, which differs semantically from the target answer documents. Multi-query and HyDE bridge this vocabulary gap.

---

### Step 7: Fine-Tuning & Quantization (LoRA / QLoRA)
* **Core Concepts**:
  * When to fine-tune (behavior/formatting/tone) vs. RAG (external facts).
  * LoRA (Low-Rank Adaptation): Adds low-rank adapter matrices ($B \times A$) to frozen weights to reduce training params by ~99%.
  * QLoRA: Quantizing the base model to 4-bit (saving massive VRAM) while adapting in 16-bit.
  * Alignment: SFT (Supervised Fine-Tuning) -> RLHF (Human feedback loop) or DPO (Direct Preference Optimization).
* **Key Code**:
  ```python
  from peft import LoraConfig, get_peft_model
  lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
  model = get_peft_model(base_model, lora_config)
  ```
* **Interview Insight**: *What is catastrophic forgetting?* During full fine-tuning on a specialized dataset, the model overwrites base weights and loses general intelligence. LoRA prevents this because the base weights remain frozen.

---

### Step 8: RAG Evaluation Suite
* **Core Concepts**:
  * LLM-as-a-Judge: Running evaluation chains with LLMs to automatically grade text outputs.
  * The RAG Triad:
    1. **Faithfulness**: Is the answer derived *only* from the retrieved context? (No hallucinations)
    2. **Answer Relevancy**: Does the answer directly address the user's question?
    3. **Context Recall**: Did the retriever locate all the information present in the ground truth?
* **Key Code**:
  * Defined structured evaluation models using Pydantic classes (`FaithfulnessScore`, `AnswerRelevancyScore`, `ContextRecallScore`) to parse and score RAG output.
* **Interview Insight**: *How do you automate evaluation in production?* Build a "Golden Test Set" (expert-verified Q/A pairs) and run automated LLM-as-a-judge scorers weekly to catch regression.

---

### Step 9: Production Engineering
* **Core Concepts**:
  * Caching: Implementing semantic caching (embedding similarity check) to prevent duplicate API charges and reduce latency.
  * Security: Checking inputs against keyword lists or guard LLMs to block prompt injection jailbreaks.
  * Resiliency: Implementing exponential backoffs (`tenacity`) and backup fallbacks (`with_fallbacks`).
  * APIs: Deploying services using FastAPI with server-sent event (SSE) streaming support.
* **Key Code**:
  ```python
  from tenacity import retry, wait_exponential
  # Wrap API calls with tenacity decorator and chain fallbacks:
  robust_chain = primary_llm.with_fallbacks([fallback_llm])
  ```
* **Interview Insight**: *What is prompt injection?* An adversarial attack where a user inputs instructions to override the system prompt (e.g., "Ignore all previous instructions..."). Prevent it using hardened system boundaries, length limits, and keyword search filters.

---

### Step 10: Interview Prep Simulator
* **Core Concepts**:
  * Practicing mock questions interactively using an LLM evaluator.
  * Simulating realistic interview rubrics and scoring responses dynamically based on precision, accuracy, and engineering depth.

---

## 🎯 Summary of Key Techniques
* **RAG**: Open-book, facts, citations, low cost, easy to update.
* **Fine-Tuning**: Closed-book, formatting, custom domain vocabulary, high training costs.
* **Agents**: Dynamic multi-step workflows requiring logical reasoning loops and tool execution.
* **Production**: Semantic caches, retry strategies, injection guards, fallback LLMs, and API servers.
