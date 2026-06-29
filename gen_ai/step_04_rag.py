# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║       AI STUDY BUDDY — STEP 4: Giving It Your Knowledge         ║
║                 (RAG — Retrieval Augmented Generation)          ║
╚══════════════════════════════════════════════════════════════════╝

WHAT WE'RE BUILDING TODAY:
  Study Buddy that answers questions from YOUR study notes.
  Upload any text/PDF → ask questions → get answers WITH citations.

PROBLEM WITH STEP 3:
  Our Study Buddy only knows what LLaMA-3 was trained on (data up to ~2024).
  What about:
    - Your personal study notes?
    - Your company's internal docs?
    - A research paper published last week?
    - Any private document the LLM has never seen?

  Answer: RAG — feed the LLM YOUR documents at query time.

CONCEPT: The Hallucination Problem
────────────────────────────────────
  LLMs sometimes "hallucinate" — confidently stating wrong facts.
  This happens because they generate the most LIKELY next token,
  not necessarily the CORRECT one.

  "Who invented the transformer?" → "Vaswani et al., 2017" ✅
  "What does section 4.2 of YOUR notes say?" → makes something up ❌

  RAG SOLUTION: Instead of relying on memorized knowledge,
  we RETRIEVE relevant text from your docs and put it IN the prompt.
  The LLM can then read the answer from provided context.

CONCEPT: The Full RAG Pipeline
────────────────────────────────

  ┌─────────────────────────────────────────────────────┐
  │                  INDEXING PHASE (one-time)           │
  │                                                     │
  │  Your Docs → Split into Chunks → Embed each chunk   │
  │           → Store in Vector Database                │
  └─────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │                  QUERYING PHASE (per request)        │
  │                                                     │
  │  User Question → Embed question → Search Vector DB  │
  │  → Get Top-K similar chunks (retrieved context)     │
  │  → Stuff context + question into prompt             │
  │  → LLM reads context → generates answer             │
  └─────────────────────────────────────────────────────┘

CONCEPT: Embeddings (deep dive)
────────────────────────────────
  An embedding model converts text → a list of numbers (a vector).

  "king"  → [0.2, 0.8, -0.1, 0.4, ...]   (384 numbers)
  "queen" → [0.3, 0.7, -0.2, 0.5, ...]   (close to "king"!)
  "pizza" → [-0.5, 0.1, 0.9, -0.3, ...]  (far from both)

  Cosine similarity measures angle between vectors:
    similarity("king", "queen") → 0.85  (very similar)
    similarity("king", "pizza") → 0.12  (unrelated)

  This is how the vector DB finds "relevant" chunks:
  embed the question → find stored chunks with highest similarity.

CONCEPT: Chunking Strategy
────────────────────────────
  Why not embed the entire document as one vector?
  → A 50-page PDF becomes ONE vector → loses specificity
  → "Which page talked about LoRA?" → can't find it

  We split documents into smaller chunks (e.g., 500 tokens each)
  and embed EACH chunk separately.

  chunk_size=500    → how many tokens per chunk
  chunk_overlap=50  → how many tokens shared between adjacent chunks
                      (so ideas spanning chunk boundaries aren't lost)

INSTALL REQUIREMENTS:
  pip install langchain-community chromadb sentence-transformers pypdf
"""

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
parser = StrOutputParser()


# ─────────────────────────────────────────────────────────────────
# LESSON 4A: Embeddings — Turning Text Into Vectors
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("LESSON 4A: Embeddings")
print("=" * 60)

# HuggingFaceEmbeddings runs locally — no API key needed!
# all-MiniLM-L6-v2 is small (80MB), fast, and excellent quality
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Embed some sentences
sentences = [
    "Transformers use attention to process text.",
    "The attention mechanism computes Q, K, V matrices.",
    "Python is a great programming language.",
]
vectors = embedder.embed_documents(sentences)

print(f"Number of sentences: {len(sentences)}")
print(f"Each vector size: {len(vectors[0])} dimensions")

# Manual cosine similarity to show what "close" means
import numpy as np
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_01 = cosine_similarity(vectors[0], vectors[1])  # transformer ↔ attention
sim_02 = cosine_similarity(vectors[0], vectors[2])  # transformer ↔ python

print(f"\nSimilarity (transformers vs attention): {sim_01:.3f}  ← high, related!")
print(f"Similarity (transformers vs python):    {sim_02:.3f}  ← low, unrelated")


# ─────────────────────────────────────────────────────────────────
# LESSON 4B: Build Your Study Notes Knowledge Base
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LESSON 4B: Indexing Study Notes into Vector DB")
print("=" * 60)

# Our "study notes" — in real life, load from PDF/TXT/URL
study_notes = [
    Document(
        page_content=(
            "Transformers were introduced in the 2017 paper 'Attention Is All You Need' "
            "by Vaswani et al. from Google Brain. The key innovation was replacing recurrence "
            "entirely with self-attention mechanisms. The original transformer had an "
            "encoder-decoder architecture with 6 layers each, 8 attention heads, and "
            "512-dimensional embeddings. Training used the Adam optimizer with a custom "
            "learning rate schedule (warmup + decay)."
        ),
        metadata={"source": "notes", "topic": "transformers", "page": 1}
    ),
    Document(
        page_content=(
            "The Attention Mechanism computes three matrices from the input: "
            "Query (Q), Key (K), and Value (V). The attention score is: "
            "Attention(Q,K,V) = softmax(QKᵀ / √dₖ) × V. "
            "Multi-head attention runs this in parallel H times with different learned projections, "
            "then concatenates results. Each head can learn different types of relationships: "
            "syntactic, semantic, positional, or coreference."
        ),
        metadata={"source": "notes", "topic": "attention", "page": 2}
    ),
    Document(
        page_content=(
            "LoRA (Low-Rank Adaptation) is a fine-tuning method that freezes the original "
            "model weights and injects trainable low-rank matrices into each transformer layer. "
            "If the original weight matrix W has shape [d, k], LoRA adds B×A where B is [d, r] "
            "and A is [r, k], with r << d (the rank). This reduces trainable parameters by 99%. "
            "QLoRA combines LoRA with 4-bit quantization of the base model, making it possible "
            "to fine-tune 70B models on a single consumer GPU."
        ),
        metadata={"source": "notes", "topic": "fine-tuning", "page": 3}
    ),
    Document(
        page_content=(
            "RAG (Retrieval Augmented Generation) solves two key problems with LLMs: "
            "knowledge cutoff and hallucination. Instead of relying on memorized knowledge, "
            "RAG retrieves relevant documents at inference time and includes them in the prompt. "
            "The pipeline has two phases: indexing (split → embed → store) and retrieval "
            "(embed query → search → top-k chunks → prompt). Common vector databases include "
            "ChromaDB (local), Pinecone (cloud), FAISS (in-memory), and pgvector (postgres)."
        ),
        metadata={"source": "notes", "topic": "rag", "page": 4}
    ),
    Document(
        page_content=(
            "RLHF (Reinforcement Learning from Human Feedback) is how LLMs are aligned "
            "to be helpful, harmless, and honest. Phase 1: Supervised Fine-Tuning (SFT) on "
            "human-written examples. Phase 2: Train a Reward Model that scores responses "
            "based on human preferences. Phase 3: Use PPO (Proximal Policy Optimization) "
            "to optimize the LLM to maximize reward. This is how ChatGPT, Claude, and Gemini "
            "are trained to be conversational and safe."
        ),
        metadata={"source": "notes", "topic": "alignment", "page": 5}
    ),
]

# STEP 1: Split documents into chunks
# RecursiveCharacterTextSplitter tries to split on: \n\n, \n, " ", ""
# in that order — preserving semantic boundaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,    # max characters per chunk
    chunk_overlap=50,  # overlap between consecutive chunks
    length_function=len,
)
chunks = splitter.split_documents(study_notes)
print(f"Original docs: {len(study_notes)} | After splitting: {len(chunks)} chunks")

# STEP 2: Embed all chunks + store in ChromaDB (local vector database)
CHROMA_DB_PATH = "./study_buddy_db"  # persisted on disk
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedder,
    persist_directory=CHROMA_DB_PATH,
)
print(f"Stored {vectorstore._collection.count()} vectors in ChromaDB")


# ─────────────────────────────────────────────────────────────────
# LESSON 4C: Retrieval — Finding Relevant Chunks
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LESSON 4C: Semantic Search")
print("=" * 60)

query = "How does LoRA reduce the number of trainable parameters?"

# similarity_search: embed query → find top-k chunks by cosine similarity
retrieved_chunks = vectorstore.similarity_search(query, k=2)

print(f"Query: {query}\n")
print("Retrieved chunks:")
for i, chunk in enumerate(retrieved_chunks, 1):
    print(f"\n  Chunk {i} [topic: {chunk.metadata['topic']}]:")
    print(f"  {chunk.page_content[:150]}...")

# Also check similarity scores
chunks_with_scores = vectorstore.similarity_search_with_score(query, k=3)
print("\nWith similarity scores (lower = more similar in ChromaDB):")
for chunk, score in chunks_with_scores:
    print(f"  Score {score:.4f} | Topic: {chunk.metadata['topic']}")


# ─────────────────────────────────────────────────────────────────
# LESSON 4D: RAG Chain — Retrieval + Generation Together
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LESSON 4D: Full RAG Chain")
print("=" * 60)

# The RAG prompt: ground the LLM in retrieved context
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are Study Buddy, a Gen AI tutor. "
        "Answer questions using ONLY the provided context from study notes. "
        "If the answer isn't in the context, say 'That's not in your study notes.' "
        "Always cite which topic the answer comes from.\n\n"
        "CONTEXT FROM STUDY NOTES:\n{context}"
    )),
    ("human", "{question}"),
])

def format_context(retrieved_docs: list) -> str:
    """Format retrieved chunks into a readable context string."""
    parts = []
    for doc in retrieved_docs:
        parts.append(
            f"[Source: {doc.metadata['topic'].upper()}, Page {doc.metadata['page']}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)

def ask_from_notes(question: str, k: int = 3) -> dict:
    """
    Full RAG pipeline:
    1. Embed the question
    2. Retrieve top-k relevant chunks
    3. Format context
    4. Generate answer grounded in context
    """
    # Retrieve
    retrieved = vectorstore.similarity_search(question, k=k)
    context = format_context(retrieved)

    # Generate
    prompt_value = rag_prompt.invoke({"context": context, "question": question})
    answer = llm.invoke(prompt_value).content

    return {
        "answer": answer,
        "sources": [f"{d.metadata['topic']} (page {d.metadata['page']})" for d in retrieved],
        "context_used": context[:200] + "...",
    }

# Test it with questions from the notes
test_questions = [
    "What is the formula for attention?",
    "How does RLHF work?",
    "What rank does LoRA use?",
    "Who invented Python?",  # NOT in notes — should say so
]

for q in test_questions:
    print(f"\nQ: {q}")
    result = ask_from_notes(q)
    print(f"A: {result['answer'][:200]}...")
    print(f"   Sources: {result['sources']}")


# ─────────────────────────────────────────────────────────────────
# PUTTING IT TOGETHER: Study Buddy v4 — RAG Chatbot
# ─────────────────────────────────────────────────────────────────
# Combine RAG + Memory from Step 3
conversation_history = []

def study_buddy_rag_chat(question: str) -> str:
    """
    Step 4 version: RAG-powered chatbot with conversation history.
    Answers from YOUR notes, remembers the conversation.
    """
    # 1. Retrieve relevant notes
    retrieved = vectorstore.similarity_search(question, k=2)
    context = format_context(retrieved)

    # 2. Build message history
    messages = [
        ("system", (
            "You are Study Buddy. Answer using the study notes below. "
            "Remember the conversation history.\n\nSTUDY NOTES:\n" + context
        )),
    ]
    # Inject history
    for role, content in conversation_history:
        messages.append((role, content))
    messages.append(("human", question))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | parser
    answer = chain.invoke({})

    # 3. Save to history
    conversation_history.append(("human", question))
    conversation_history.append(("assistant", answer))

    return answer


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STUDY BUDDY v4 — RAG-Powered Tutor")
    print("=" * 60)
    print(study_buddy_rag_chat("Explain the attention formula."))
    print()
    print(study_buddy_rag_chat("Can you give a simpler explanation of that?"))
    # "that" = attention formula, because history is included


"""
WHAT YOU LEARNED:
  ✅ Embeddings convert text → numbers that capture semantic meaning
  ✅ Cosine similarity finds "related" chunks by vector distance
  ✅ Chunking: split docs into pieces with overlap to avoid losing context
  ✅ ChromaDB stores and searches embeddings locally
  ✅ RAG = embed question → retrieve chunks → stuff into prompt → generate
  ✅ Grounding LLM in context dramatically reduces hallucinations

  INTERVIEW INSIGHT:
    "How do you choose chunk size?"
    → Too small: each chunk lacks context (loses meaning)
    → Too large: retrieved chunk carries irrelevant info
    → Rule of thumb: 300-500 tokens for most use cases
    → Also use chunk_overlap to avoid splitting ideas at boundaries

    "What is HyDE?"
    → Hypothetical Document Embedding: generate a HYPOTHETICAL answer,
      embed THAT, then search — often finds better matches than embedding
      the sparse question itself.

NEXT → step_05_agent.py
  PROBLEM: Our Study Buddy is still PASSIVE.
  It can only answer from what we give it.
  What if a student asks "Find me the latest news on LLMs"?
  → Agents can CALL TOOLS (search web, run code, read URLs...)
"""
