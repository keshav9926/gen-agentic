# 🤖 AI Study Buddy — Complete Gen AI Curriculum
> **One project. 10 steps. Zero to production-grade AI engineer.**

---

## 🚀 Quick Start

```bash
# 1. Copy and fill in your API keys
cp .env.example .env

# 2. Install dependencies (use a venv!)
pip install -r requirements.txt

# 3. Start from Step 1 and work your way up
python step_01_basic_llm.py
```

**Get a FREE Groq API key at:** https://console.groq.com (no credit card needed)

---

## 📚 The Curriculum

| Step | File | Concept | What You Build |
|------|------|---------|---------------|
| 1 | `step_01_basic_llm.py` | LLMs, tokens, temperature, streaming | `ask_study_buddy()` function |
| 2 | `step_02_prompts.py` | Prompt templates, few-shot, CoT, JSON output | Structured Q&A engine |
| 3 | `step_03_memory.py` | Context windows, session memory, multi-user | Stateful chatbot |
| 4 | `step_04_rag.py` | Embeddings, ChromaDB, semantic search, RAG | Notes Q&A with citations |
| 5 | `step_05_agent.py` | Tools, LangGraph, ReAct agents | Autonomous tutor agent |
| 6 | `step_06_advanced_rag.py` | Multi-query, HyDE, re-ranking, parent docs | Production-quality RAG |
| 7 | `step_07_finetuning.py` | LoRA, QLoRA, SFTTrainer, RLHF, DPO | Fine-tuned study buddy |
| 8 | `step_08_evaluation.py` | Faithfulness, relevancy, recall, LLM-as-judge | Evaluation dashboard |
| 9 | `step_09_production.py` | Retry, caching, injection defense, FastAPI | Production-ready API |
| 10 | `step_10_interview_prep.py` | Full interview bank + AI evaluator | Mock interview session |

---

## ⚠️ Dependencies Between Steps

```
step_01 → step_02 → step_03 → step_04 ──→ step_05
                                    ↓
                              step_06 (advanced RAG)
                              (run step_04 first to create the vector DB)
```

Steps 7, 8, 9, 10 are **standalone** — run any of them independently.

---

## 🗓️ Suggested Study Schedule

| Day | Steps | Focus |
|-----|-------|-------|
| 1 | 1, 2 | LLM fundamentals + prompt engineering |
| 2 | 3 | Memory and stateful conversations |
| 3 | 4 | RAG — your biggest interview topic |
| 4 | 5 | Agents and LangGraph |
| 5 | 6 | Advanced RAG techniques |
| 6 | 7 | Fine-tuning theory (GPU required for training) |
| 7 | 8, 9 | Evaluation + production hardening |
| 8+ | 10 | Daily mock interviews until confident |

---

## 🎯 Interview Prep

```bash
# Run the AI mock interviewer daily
python step_10_interview_prep.py

# Mode 1: Full mock interview (AI evaluates your answers)
# Mode 2: Flash cards (quick review before an interview)
```

Topics covered: Fundamentals · RAG · Agents · Fine-tuning · System Design
Levels: Entry / Mid / Senior

---

## 📁 Project Structure

```
study_buddy/
├── .env.example          ← Copy to .env, fill in API keys
├── requirements.txt      ← All dependencies
├── README.md             ← This file
├── step_01_basic_llm.py
├── step_02_prompts.py
├── step_03_memory.py
├── step_04_rag.py        ← Creates study_buddy_db/ (vector DB)
├── step_05_agent.py      ← Requires step_04 to run first
├── step_06_advanced_rag.py
├── step_07_finetuning.py ← GPU required for actual training
├── step_08_evaluation.py
├── step_09_production.py
└── step_10_interview_prep.py  ← Run this every day!
```

---

> **The comments in each file ARE the lesson. Read them carefully.**
> Each file explains the WHY, not just the HOW.

