# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║       AI STUDY BUDDY — STEP 8: Is It Actually Any Good?         ║
║               (LLM Evaluation with RAGAS & LLM-as-Judge)        ║
╚══════════════════════════════════════════════════════════════════╝

WHY EVALUATION MATTERS:
  - vibes are not metrics. You need quantitive measures to gauge changes.
  
THE THREE PILLARS (RAGAS framework):
  1. Faithfulness: Does the answer contain only facts from retrieved context? (Anti-hallucination)
  2. Answer Relevancy: Does the answer address the question? (Syntactic and semantic fit)
  3. Context Recall: Were the right chunks retrieved based on a reference ground truth?
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

# Model & Embeddings setup
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
parser = StrOutputParser()
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ─────────────────────────────────────────────────────────────────
# 1. Golden Test Dataset Setup
# ─────────────────────────────────────────────────────────────────
golden_dataset = [
    {
        "question": "What year were transformers introduced and by whom?",
        "ground_truth": "Transformers were introduced in 2017 in the paper 'Attention Is All You Need' by Vaswani et al. from Google Brain.",
        "ground_truth_context": "Transformers were introduced in the 2017 paper 'Attention Is All You Need' by Vaswani et al. from Google Brain."
    },
    {
        "question": "What is the attention formula?",
        "ground_truth": "The attention formula is: Attention(Q,K,V) = softmax(QKᵀ / √dₖ) × V. It computes weighted values based on query-key similarity.",
        "ground_truth_context": "The dot product of Q with all K vectors gives attention scores. After scaling by √dₖ and softmax, these become attention weights applied to V vectors."
    }
]

# Simple Local Vector DB setup for testing
DOCUMENTS = [
    Document(page_content="Transformers were introduced in the 2017 paper 'Attention Is All You Need' by Vaswani et al.", metadata={"topic": "transformer"}),
    Document(page_content="Attention(Q,K,V) = softmax(QKᵀ / √dₖ) × V", metadata={"topic": "attention"})
]
vectorstore = Chroma.from_documents(DOCUMENTS, embedder, collection_name="eval_test")
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using ONLY the provided context. If context doesn't contain answer, say 'Not in my notes.'\n\nCONTEXT:\n{context}"),
    ("human", "{question}"),
])

def run_rag(question: str) -> dict:
    retrieved = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in retrieved)
    answer = (rag_prompt | llm | parser).invoke({"context": context, "question": question})
    return {"question": question, "retrieved_context": context, "answer": answer}


# ─────────────────────────────────────────────────────────────────
# 2. Evaluation Metrics (Faithfulness, Relevancy, Recall)
# ─────────────────────────────────────────────────────────────────

# --- Metric 1: Faithfulness ---
class FaithfulnessScore(BaseModel):
    claims: List[str] = Field(description="factual claims extracted from response")
    supported: List[bool] = Field(description="True if supported by context")
    score: float = Field(description="Ratio of supported claims (0 to 1)")
    reasoning: str

faithfulness_prompt = ChatPromptTemplate.from_template(
    "Verify if the answer claims are supported by context.\nContext: {context}\nAnswer: {answer}\n{format_instructions}"
)
faith_parser = JsonOutputParser(pydantic_object=FaithfulnessScore)
faith_chain = faithfulness_prompt.partial(format_instructions=faith_parser.get_format_instructions()) | llm | faith_parser

def evaluate_faithfulness(context: str, answer: str) -> float:
    try:
        res = faith_chain.invoke({"context": context, "answer": answer})
        return float(res.get("score", 0.0))
    except Exception:
        return 0.0

# --- Metric 2: Answer Relevancy ---
class AnswerRelevancyScore(BaseModel):
    hypothetical_questions: List[str] = Field(description="3 questions answered by this response")
    score: float = Field(description="Relevancy score (0 to 1)")
    reasoning: str

relevancy_prompt = ChatPromptTemplate.from_template(
    "Evaluate if the ANSWER fits the QUESTION.\nQuestion: {question}\nAnswer: {answer}\n{format_instructions}"
)
rel_parser = JsonOutputParser(pydantic_object=AnswerRelevancyScore)
rel_chain = relevancy_prompt.partial(format_instructions=rel_parser.get_format_instructions()) | llm | rel_parser

def evaluate_relevancy(question: str, answer: str) -> float:
    try:
        res = rel_chain.invoke({"question": question, "answer": answer})
        return float(res.get("score", 0.0))
    except Exception:
        return 0.0

# --- Metric 3: Context Recall ---
class ContextRecallScore(BaseModel):
    ground_truth_claims: List[str] = Field(description="Key claims from ground truth")
    claims_in_context: List[bool] = Field(description="True if claim appears in retrieved context")
    score: float = Field(description="Recall score (0 to 1)")

recall_prompt = ChatPromptTemplate.from_template(
    "Evaluate context recall:\nGround Truth: {ground_truth}\nContext: {retrieved_context}\n{format_instructions}"
)
recall_parser = JsonOutputParser(pydantic_object=ContextRecallScore)
recall_chain = recall_prompt.partial(format_instructions=recall_parser.get_format_instructions()) | llm | recall_parser

def evaluate_context_recall(ground_truth: str, retrieved_context: str) -> float:
    try:
        res = recall_chain.invoke({"ground_truth": ground_truth, "retrieved_context": retrieved_context})
        return float(res.get("score", 0.0))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────
# 3. Evaluation Suite Runner
# ─────────────────────────────────────────────────────────────────
def evaluate_system(test_cases: list):
    print("=" * 60)
    print("RUNNING SYSTEM EVALUATION SUITE")
    print("=" * 60)
    
    faith_scores = []
    rel_scores = []
    recall_scores = []
    
    for idx, case in enumerate(test_cases, 1):
        print(f"\nEvaluating Case #{idx}: {case['question']}")
        rag_res = run_rag(case["question"])
        print(f"  System Answer: {rag_res['answer']}")
        
        faith = evaluate_faithfulness(rag_res["retrieved_context"], rag_res["answer"])
        rel = evaluate_relevancy(case["question"], rag_res["answer"])
        recall = evaluate_context_recall(case["ground_truth"], rag_res["retrieved_context"])
        
        print(f"  Metrics -> Faithfulness: {faith:.2f} | Relevancy: {rel:.2f} | Recall: {recall:.2f}")
        
        faith_scores.append(faith)
        rel_scores.append(rel)
        recall_scores.append(recall)
        
    print("\n" + "=" * 60)
    print("AGGREGATE REPORT")
    print("=" * 60)
    print(f"Mean Faithfulness:  {np.mean(faith_scores):.2f}")
    print(f"Mean Relevancy:     {np.mean(rel_scores):.2f}")
    print(f"Mean Context Recall:{np.mean(recall_scores):.2f}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_system(golden_dataset)

"""
WHAT YOU LEARNED:
  ✅ Why BLEU/ROUGE fail for LLMs — semantic meaning vs. word overlap
  ✅ Faithfulness: claim-level hallucination detection
  ✅ Answer Relevancy: embedding similarity approach
  ✅ Context Recall: did retriever find what was needed?
  ✅ LLM-as-Judge: use a strong model to evaluate another model
  ✅ Golden dataset: the foundation of any ML evaluation system

  INTERVIEW INSIGHTS:
    "How do you measure hallucination in production?"
    → Faithfulness metric: extract claims from answer, verify each 
      against retrieved context using an LLM evaluator.

    "What is RAGAS?"
    → A framework that automates RAG evaluation using LLM-based 
      metrics: faithfulness, answer relevancy, context recall, 
      context precision.

    "How do you build a golden test set?"
    → Domain experts write 50-200 question-answer pairs.
    → Use LLMs to GENERATE candidate pairs, experts to verify.
    → Must cover edge cases, failures, and normal cases.

NEXT → step_09_production.py
"""
