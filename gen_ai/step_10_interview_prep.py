# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║       AI STUDY BUDDY — STEP 10: THE FINAL BOSS                  ║
║              (Interactive Interview Preparation)                ║
╚══════════════════════════════════════════════════════════════════╝

This is the most important file in the entire curriculum.
Run this to practice with an AI interviewer that:
  1. Asks real Gen AI interview questions (by difficulty + topic)
  2. Listens to your answer
  3. Gives specific, constructive feedback
  4. Follows up with harder questions based on your answer
  5. Scores you and tracks your weak areas
  6. Gives you model answers to compare against

HOW TO USE:
  python step_10_interview_prep.py

  → Choose a topic area
  → Choose difficulty (entry / mid / senior)
  → Answer each question aloud (or type your answer)
  → Get immediate feedback + model answer
  → Repeat until you're confident
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal, List
import json
import random

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
parser = StrOutputParser()


# ═══════════════════════════════════════════════════════════════════
# THE COMPLETE INTERVIEW QUESTION BANK
# (Study this even without running the interactive session!)
# ═══════════════════════════════════════════════════════════════════

INTERVIEW_BANK = {

    "fundamentals": {
        "entry": [
            {
                "question": "What is the difference between AI, Machine Learning, and Deep Learning?",
                "key_points": ["AI is the broad field", "ML is a subset that learns from data", "DL is a subset of ML using neural networks", "Hierarchy: AI ⊃ ML ⊃ DL"],
                "model_answer": (
                    "AI is the broad field of making machines intelligent. "
                    "Machine Learning is a subset where machines learn patterns from data "
                    "instead of following hand-coded rules. "
                    "Deep Learning is a subset of ML that uses multi-layered neural networks, "
                    "enabling breakthroughs in vision, language, and audio. "
                    "Think of it as nested circles: AI contains ML which contains Deep Learning."
                ),
                "follow_up": "Can you give an example of a task that needs Deep Learning but not just ML?"
            },
            {
                "question": "What is a token in the context of LLMs?",
                "key_points": ["subword unit", "~0.75 words on average", "models have context windows in tokens", "tokenization affects cost and limits"],
                "model_answer": (
                    "A token is the basic unit an LLM processes — not a word, but a subword chunk. "
                    "'playing' might be ['play', 'ing'] (2 tokens). On average, 1 token ≈ 0.75 words. "
                    "LLMs have a maximum context window in tokens (e.g., 128k for GPT-4). "
                    "You pay per token with most APIs, and tokenization affects what the model 'sees.'"
                ),
                "follow_up": "Why does tokenization matter for multilingual models?"
            },
            {
                "question": "What is temperature in LLM generation and when would you change it?",
                "key_points": ["controls randomness of sampling", "0 = greedy/deterministic", "higher = more diverse/creative", "low for factual, high for creative tasks"],
                "model_answer": (
                    "Temperature controls how 'creative' the LLM is when picking the next token. "
                    "At temperature=0, it always picks the highest-probability token (deterministic, consistent). "
                    "At temperature=1+, it samples more broadly, producing varied or creative output. "
                    "Use low temperature for factual Q&A, code generation, or classification. "
                    "Use high temperature for brainstorming, story writing, or generating diverse options."
                ),
                "follow_up": "What is the difference between temperature and top-p sampling?"
            },
        ],
        "mid": [
            {
                "question": "Explain how a transformer generates text autoregressively.",
                "key_points": ["one token at a time", "each token attends to all previous tokens", "KV cache for efficiency", "stops at EOS token or max length"],
                "model_answer": (
                    "Transformers generate text one token at a time in a loop. Given 'The capital of France is', "
                    "the model computes attention over all these tokens to produce a probability distribution "
                    "over the vocabulary, samples 'Paris', then repeats with 'The capital of France is Paris' as input. "
                    "Each step is a full forward pass, but the KV cache stores past key-value pairs to avoid recomputing. "
                    "Generation stops at the EOS token or max_tokens limit."
                ),
                "follow_up": "What is the KV cache and why does it matter for inference cost?"
            },
            {
                "question": "What is the difference between encoder-only, decoder-only, and encoder-decoder models? Give examples of each.",
                "key_points": ["encoder-only: BERT, bidirectional attention, for classification/embeddings", "decoder-only: GPT/LLaMA, causal attention, for generation", "encoder-decoder: T5/BART, for seq2seq tasks"],
                "model_answer": (
                    "Encoder-only models (BERT, RoBERTa) see the full input with bidirectional attention — great for classification, "
                    "NER, and embeddings. Decoder-only models (GPT, LLaMA, Gemini) use causal (left-to-right) attention "
                    "and generate tokens one at a time — the standard for chat and generation. Encoder-decoder models "
                    "(T5, BART) encode the input then decode the output — best for translation, summarization, "
                    "and tasks where input and output are different."
                ),
                "follow_up": "Why would you choose BERT over GPT-4 for a sentiment classification task?"
            },
        ],
        "senior": [
            {
                "question": "Explain the attention mechanism formula and why we divide by √dₖ.",
                "key_points": ["Attention(Q,K,V) = softmax(QKᵀ / √dₖ) × V", "dot products grow with dₖ", "large dot products cause vanishing gradients after softmax", "√dₖ normalizes the scale"],
                "model_answer": (
                    "The formula is Attention(Q,K,V) = softmax(QKᵀ / √dₖ) × V. Q, K, V are linear projections of the input. "
                    "QKᵀ computes all pairwise dot products, giving attention logits. We divide by √dₖ because dot products "
                    "grow in magnitude as embedding dimension dₖ increases. Without scaling, large values push softmax into "
                    "saturation (nearly 0 gradients), making training unstable. The square root of dₖ stabilizes the variance "
                    "of the dot products."
                ),
                "follow_up": "How does multi-head attention differ from single-head, and what does each head learn?"
            },
        ]
    },

    "rag": {
        "entry": [
            {
                "question": "What is RAG and what problem does it solve?",
                "key_points": ["Retrieval Augmented Generation", "solves hallucination", "solves knowledge cutoff", "solves context length limits for large corpora"],
                "model_answer": (
                    "RAG (Retrieval Augmented Generation) addresses three LLM limitations: (1) Hallucination — LLMs sometimes "
                    "generate confident but wrong facts. RAG grounds answers in retrieved text. (2) Knowledge cutoff — LLMs "
                    "don't know about events after their training. RAG can retrieve from up-to-date documents. (3) Context limits — "
                    "you can't fit 10,000 documents in a prompt. RAG retrieves only the relevant 3-5 chunks."
                ),
                "follow_up": "When would you use fine-tuning instead of RAG?"
            },
        ],
        "mid": [
            {
                "question": "How do you choose chunk size in a RAG pipeline?",
                "key_points": ["too small = loses context", "too large = irrelevant info retrieved", "300-500 tokens typical", "use overlap to avoid splitting ideas", "depends on document structure"],
                "model_answer": (
                    "Chunk size is a trade-off: too small and each chunk loses context (a definition split from its explanation), "
                    "too large and the retrieved chunk contains irrelevant information that confuses the LLM. The common range is "
                    "300-512 tokens with 10-15% overlap. Overlap prevents losing information at chunk boundaries. For structured "
                    "documents (legal, technical), align chunks with natural boundaries (sections, paragraphs). Always evaluate "
                    "chunk quality with retrieval metrics."
                ),
                "follow_up": "What is the parent document retriever pattern and when would you use it?"
            },
            {
                "question": "What is HyDE and when would you use it?",
                "key_points": ["Hypothetical Document Embedding", "generate fake answer → embed it → search", "query embedding is sparse, hypothetical doc is richer", "use for short/vague queries"],
                "model_answer": (
                    "HyDE (Hypothetical Document Embedding) addresses sparse query embeddings. A short question like 'What is LoRA?' "
                    "has few words to capture its meaning, so its embedding might not closely match the chunks in the vector DB. "
                    "Instead, we ask the LLM to generate a hypothetical answer (we ignore its factual accuracy), embed THAT rich "
                    "document, then search with it. Since the hypothetical answer uses domain vocabulary similar to the actual documents, "
                    "retrieval quality improves. Use HyDE for short, ambiguous queries; skip it for long, specific ones."
                ),
                "follow_up": "What are the downsides of HyDE?"
            },
        ],
        "senior": [
            {
                "question": "How would you architect a multi-tenant RAG system where different customers have isolated document stores?",
                "key_points": ["namespace/collection per tenant", "metadata filtering", "never mix tenant data", "access control at retrieval layer", "shared embedding model, separate vector collections"],
                "model_answer": (
                    "I'd use separate Chroma collections or Pinecone namespaces per tenant, keyed by tenant_id. The embedding model "
                    "is shared (cost efficiency), but retrieval is scoped to the tenant's collection. At the API layer, extract "
                    "tenant_id from the JWT token and pass it as a retrieval filter — users can only query their own namespace. "
                    "For extra isolation, deploy separate vector DB instances per large enterprise customer. Implement row-level "
                    "security: even if a retrieval bug occurs, metadata filters ensure cross-tenant data leakage is impossible."
                ),
                "follow_up": "How would you handle a document that one team should see but another shouldn't, within the same tenant?"
            },
        ]
    },

    "agents": {
        "entry": [
            {
                "question": "What is the difference between a LangChain chain and an agent?",
                "key_points": ["chain: fixed sequence decided at build time", "agent: LLM decides steps at runtime", "agent can use tools", "agent loops until task complete"],
                "model_answer": (
                    "A chain has a predetermined, fixed sequence of steps: A → B → C, always. An agent is dynamic: the LLM itself "
                    "decides at each step whether to use a tool, which tool, or give a final answer. Chains are predictable and fast. "
                    "Agents are flexible and can handle open-ended tasks requiring multiple tool calls. Use chains for well-defined "
                    "workflows, agents for tasks where the path isn't known in advance."
                ),
                "follow_up": "When would you prefer a chain over an agent for a production system?",
                "follow_up_answer": (
                    "I'd prefer a chain when: (1) The task is deterministic and always follows the same steps — e.g., "
                    "extracting data from a known format. (2) Performance and cost are critical — chains avoid the "
                    "overhead of LLM self-reflection and tool calls. (3) I need strict control over the output — "
                    "chains guarantee a fixed sequence, while agents can meander. (4) The task is simple — agents "
                    "add unnecessary complexity. Use agents only when the task requires exploration, multiple "
                    "tool calls, or dynamic decision-making."
                )
            },
        ],
        "mid": [
            {
                "question": "Explain the ReAct pattern for AI agents.",
                "key_points": ["Reason + Act", "Thought: decide what to do", "Action: call a tool", "Observation: read tool result", "repeat until final answer"],
                "model_answer": (
                    "ReAct (Reason + Act) is the dominant agent pattern. The LLM alternates between reasoning and acting: "
                    "Thought: 'I need to find the current Bitcoin price.' → Action: search('Bitcoin price today') → "
                    "Observation: 'Bitcoin is at $67,432' → Thought: 'I have the answer' → Final Answer: 'Bitcoin is $67,432.' "
                    "This think-act-observe loop continues until the task is complete. The key insight is that making reasoning "
                    "explicit (via Thought) improves task completion and reduces errors compared to action-only agents."
                ),
                "follow_up": "How do you prevent a ReAct agent from looping forever?"
            },
            {
                "question": "What is LangGraph and why would you use it over AgentExecutor?",
                "key_points": ["graph-based agent framework", "nodes = actions, edges = transitions, state = shared data", "supports cycles and branching", "human-in-the-loop", "streaming intermediate steps", "explicit control vs AgentExecutor black box"],
                "model_answer": (
                    "LangGraph models agents as directed graphs where nodes are actions (run LLM, call tool) and edges are conditional "
                    "transitions based on state. Unlike AgentExecutor, which is a fixed loop you can't customize, LangGraph gives "
                    "explicit control: you can add human approval checkpoints, parallel branches, or custom retry logic. State flows "
                    "through all nodes, enabling complex multi-step workflows. Use AgentExecutor for simple single-agent tasks; "
                    "LangGraph for complex, multi-step, multi-agent workflows in production."
                ),
                "follow_up": "How does LangGraph handle the case where an agent needs to be interrupted for human approval?"
            },
        ],
        "senior": [
            {
                "question": "Design a multi-agent system for an automated research pipeline.",
                "key_points": ["orchestrator agent", "specialized sub-agents", "shared state", "human-in-the-loop checkpoints", "error handling between agents", "avoiding infinite loops"],
                "model_answer": (
                    "I'd design it in LangGraph with a supervisor pattern: an Orchestrator agent breaks down the research task "
                    "and routes to specialized sub-agents: (1) Search Agent — queries web/databases, (2) Analyst Agent — reads "
                    "and extracts key points from retrieved papers, (3) Fact-Checker Agent — verifies claims against primary sources, "
                    "(4) Writer Agent — synthesizes a structured report. State includes: research topic, sources found, analysis "
                    "results, and draft. Human checkpoints before final publication. If any agent fails (e.g., no sources found), "
                    "the orchestrator retries with a different query strategy. Output: a structured report with citations."
                ),
                "follow_up": "How would you handle conflicting information between sources in this pipeline?"
            },
        ]
    },

    "finetuning": {
        "entry": [
            {
                "question": "When would you choose fine-tuning over prompt engineering?",
                "key_points": ["consistent format/style", "domain jargon", "500+ examples available", "latency matters", "prompt engineering tried first"],
                "model_answer": (
                    "I'd choose fine-tuning when: (1) The required output format is complex and prompt engineering can't "
                    "reliably enforce it — e.g., always returning JSON with specific fields. (2) The domain has specialized vocabulary "
                    "that the base model underperforms on. (3) I have 500+ labeled examples. (4) Latency is critical — a fine-tuned "
                    "7B model can match GPT-4 on specific tasks with 10x lower latency. Critically, I always try prompt engineering "
                    "FIRST. Fine-tuning changes style and format; RAG provides knowledge. Don't use fine-tuning for knowledge injection."
                ),
                "follow_up": "What is 'catastrophic forgetting' in fine-tuning and how does LoRA address it?"
            },
        ],
        "mid": [
            {
                "question": "Explain LoRA and how it reduces trainable parameters.",
                "key_points": ["Low-Rank Adaptation", "adds B×A matrices", "r << d (low rank)", "freeze base weights", "0.1-1% trainable params", "merged at inference for zero overhead"],
                "model_answer": (
                    "LoRA (Low-Rank Adaptation) decomposes the weight update ΔW into two low-rank matrices: ΔW = B × A, where "
                    "B is [d, r] and A is [r, d], with r much smaller than d. Instead of updating the full weight matrix W "
                    "(d×d = millions of params), we only train B and A (2×d×r = thousands of params). For r=16 and d=4096, instead of "
                    "16.7M parameters, we train 131K — a 99.2% reduction. The base model weights are frozen, preventing "
                    "catastrophic forgetting. At inference, ΔW is merged back into W for zero latency overhead."
                ),
                "follow_up": "What is QLoRA and how does it differ from standard LoRA?"
            },
        ],
        "senior": [
            {
                "question": "Explain RLHF and DPO. Why is DPO often preferred in practice?",
                "key_points": ["RLHF: SFT → reward model → PPO", "PPO is unstable, requires reward model", "DPO: directly optimizes on preference pairs", "DPO: no separate reward model", "DPO: more stable training", "Bradley-Terry model underpins DPO"],
                "model_answer": (
                    "RLHF has three phases: SFT (train on demonstrations), reward model training (learn human preferences from "
                    "A/B comparisons), and PPO (optimize LLM to maximize reward with KL penalty). PPO is notoriously unstable — "
                    "sensitive to hyperparameters, requires careful KL penalty tuning, and needs a separate reward model in memory "
                    "during training. DPO (Direct Preference Optimization) bypasses the reward model entirely by reformulating "
                    "the RL objective as a supervised loss on preference pairs (chosen vs. rejected responses). It's mathematically "
                    "equivalent under certain assumptions but far more stable and computationally cheaper. Most modern open-source "
                    "fine-tuning (LLaMA, Mistral) uses DPO or its variants (SimPO, IPO)."
                ),
                "follow_up": "What are the limitations of DPO compared to RLHF?"
            },
        ]
    },

    "system_design": {
        "senior": [
            {
                "question": "Design a production RAG system that serves 10,000 concurrent users with 99.9% uptime.",
                "key_points": ["horizontal scaling", "load balancer", "async processing", "semantic caching", "vector DB clustering", "read replicas", "circuit breaker pattern", "monitoring"],
                "model_answer": (
                    "Architecture: (1) Load balancer (nginx) → multiple FastAPI instances (horizontal scaling). "
                    "(2) Semantic cache (Redis with vector search) in front of the LLM layer — cache hit rate ~40-60% reduces "
                    "LLM calls dramatically. (3) Vector DB: Pinecone (managed, auto-scaling) or Weaviate cluster with read "
                    "replicas for high-throughput retrieval. (4) LLM: Groq as primary, OpenAI as fallback with circuit breaker "
                    "(stop calling failed service until it recovers). (5) Async queue (Celery + Redis) for non-real-time tasks. "
                    "(6) Observability: LangSmith for traces, Prometheus + Grafana for latency/error metrics. (7) Kubernetes "
                    "for orchestration and auto-scaling based on queue depth."
                ),
                "follow_up": "How would you handle the case where a customer uploads a 500-page PDF and wants instant Q&A?"
            },
        ]
    }
}


# ═══════════════════════════════════════════════════════════════════
# INTERACTIVE INTERVIEW SESSION
# ═══════════════════════════════════════════════════════════════════

class FeedbackResult(BaseModel):
    score: int = Field(description="Score from 1-10")
    strengths: List[str] = Field(description="What the candidate did well")
    gaps: List[str] = Field(description="Key concepts missed or wrong")
    follow_up_question: str = Field(description="A harder follow-up to dig deeper")
    verdict: Literal["strong", "acceptable", "needs_work"] = Field(
        description="Overall verdict on the answer"
    )

feedback_prompt = ChatPromptTemplate.from_template(
    """You are a senior Gen AI engineer conducting a technical interview at a top AI company.

QUESTION ASKED: {question}

KEY POINTS that should be covered: {key_points}

CANDIDATE'S ANSWER: {answer}

Evaluate the answer strictly but fairly. Score from 1-10.
Be specific about what they missed and what they got right.
Generate a challenging follow-up question based on their answer.
{format_instructions}"""
)

feedback_parser = JsonOutputParser(pydantic_object=FeedbackResult)
feedback_chain = feedback_prompt.partial(
    format_instructions=feedback_parser.get_format_instructions()
) | llm | feedback_parser


def get_feedback(question_data: dict, user_answer: str) -> dict:
    """Evaluate the user's answer and return structured feedback."""
    try:
        result = feedback_chain.invoke({
            "question": question_data["question"],
            "key_points": json.dumps(question_data["key_points"]),
            "answer": user_answer,
        })
        return result
    except Exception as e:
        return {"error": str(e), "score": 0}


def display_feedback(feedback: dict, model_answer: str):
    """Pretty-print the interview feedback."""
    score = feedback.get("score", 0)
    verdict = feedback.get("verdict", "needs_work")

    verdict_icons = {"strong": "✅", "acceptable": "⚠️", "needs_work": "❌"}
    icon = verdict_icons.get(verdict, "❓")

    print(f"\n  {'─'*50}")
    print(f"  SCORE: {score}/10  {icon} {verdict.upper().replace('_', ' ')}")
    print(f"  {'─'*50}")

    if feedback.get("strengths"):
        print("\n  ✅ STRENGTHS:")
        for s in feedback["strengths"]:
            print(f"     • {s}")

    if feedback.get("gaps"):
        print("\n  ⚠️  GAPS / MISSED:")
        for g in feedback["gaps"]:
            print(f"     • {g}")

    print(f"\n  📖 MODEL ANSWER:")
    print(f"     {model_answer}")

    if feedback.get("follow_up_question"):
        print(f"\n  🔥 FOLLOW-UP (if this were a real interview):")
        print(f"     {feedback['follow_up_question']}")


def run_interview_session():
    """Main interactive interview loop."""
    print("\n" + "═" * 60)
    print("  🎯 GEN AI MOCK INTERVIEW — Study Buddy")
    print("  Prepare for technical interviews at top AI companies")
    print("═" * 60)

    # Select topic
    topics = list(INTERVIEW_BANK.keys())
    print("\nAvailable topics:")
    for i, t in enumerate(topics, 1):
        print(f"  {i}. {t.replace('_', ' ').title()}")

    try:
        topic_idx = int(input("\nSelect topic (number): ").strip()) - 1
        topic = topics[topic_idx]
    except (ValueError, IndexError):
        topic = "fundamentals"
        print("Defaulting to: Fundamentals")

    # Select difficulty
    available_levels = list(INTERVIEW_BANK[topic].keys())
    print(f"\nAvailable levels: {', '.join(available_levels)}")
    level = input("Select level: ").strip().lower()
    if level not in available_levels:
        level = available_levels[0]
        print(f"Defaulting to: {level}")

    questions = INTERVIEW_BANK[topic][level].copy()
    random.shuffle(questions)  # randomize order

    scores = []
    print(f"\n  Starting {topic.title()} interview ({level} level)")
    print(f"  {len(questions)} question(s)")
    print(f"  Type your answer and press Enter twice to submit\n")

    for i, q_data in enumerate(questions, 1):
        print(f"\n{'═'*60}")
        print(f"  QUESTION {i}/{len(questions)}")
        print(f"{'═'*60}")
        print(f"\n  {q_data['question']}\n")

        # Collect multi-line answer
        print("  Your answer (press Enter twice to submit):")
        lines = []
        while True:
            line = input("  ")
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        user_answer = " ".join(l for l in lines if l)

        if not user_answer.strip():
            print("  Skipping (no answer given)")
            continue

        print("\n  Evaluating your answer...")
        feedback = get_feedback(q_data, user_answer)

        display_feedback(feedback, q_data["model_answer"])
        score = feedback.get("score", 0)
        scores.append(score)

        # Pause between questions
        if i < len(questions):
            input("\n  Press Enter for next question...")

    # Final summary
    if scores:
        avg = sum(scores) / len(scores)
        print(f"\n{'═'*60}")
        print(f"  INTERVIEW COMPLETE")
        print(f"  Average Score: {avg:.1f}/10")
        if avg >= 8:
            print("  Result: 💪 STRONG HIRE — Excellent Gen AI knowledge!")
        elif avg >= 6:
            print("  Result: ✅ HIRE — Solid foundation, minor gaps")
        elif avg >= 4:
            print("  Result: ⚠️  BORDERLINE — Review weak areas")
        else:
            print("  Result: ❌ NOT YET — Go back through steps 1-9")
        print(f"{'═'*60}")

        print("\n  RECOMMENDED STUDY AREAS:")
        if avg < 8:
            print("  • Re-read the files for any concepts you scored < 7")
            print("  • Run the interview again tomorrow (spaced repetition)")
            print("  • Explain each concept OUT LOUD — if you can't explain it, you don't know it")


def run_flash_cards():
    """Quick-fire mode: question → answer reveal (no AI evaluation)."""
    print("\n" + "═" * 60)
    print("  ⚡ FLASH CARD MODE (Quick Review)")
    print("═" * 60)

    # Flatten all questions
    all_questions = []
    for topic, levels in INTERVIEW_BANK.items():
        for level, questions in levels.items():
            for q in questions:
                all_questions.append({**q, "topic": topic, "level": level})

    random.shuffle(all_questions)

    for i, q_data in enumerate(all_questions, 1):
        print(f"\n[{i}/{len(all_questions)}] [{q_data['topic'].upper()} | {q_data['level'].upper()}]")
        print(f"Q: {q_data['question']}")
        input("  → Think of your answer, then press Enter to reveal...")
        print(f"\n  KEY POINTS:")
        for pt in q_data["key_points"]:
            print(f"    • {pt}")
        print(f"\n  MODEL ANSWER:")
        print(f"    {q_data['model_answer']}")

        cmd = input("\n  [n]ext | [q]uit: ").strip().lower()
        if cmd == "q":
            break

    print("\n  Flash card session complete. Keep practicing! 🚀")


if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  GEN AI INTERVIEW PREP — Study Buddy")
    print("═" * 60)
    print("\n  Choose mode:")
    print("  1. Interactive Mock Interview (AI evaluates your answers)")
    print("  2. Flash Cards (quick question-answer review)")

    mode = input("\nSelect (1 or 2): ").strip()

    if mode == "2":
        run_flash_cards()
    else:
        run_interview_session()
