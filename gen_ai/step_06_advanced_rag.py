# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║     AI STUDY BUDDY — STEP 6: Making RAG Actually Work Well      ║
║                     (Advanced RAG Techniques)                   ║
╚══════════════════════════════════════════════════════════════════╝

WHAT WE'RE BUILDING TODAY:
  Upgrade Step 4's basic RAG to production-quality retrieval.
  Basic RAG works ~60% of the time. Advanced RAG hits 90%+.

WHY BASIC RAG FAILS:
  Problem 1 — VOCABULARY MISMATCH
    User asks: "How does the key-query-value system work?"
    Notes say:  "The Q, K, V matrices in attention..."
    → The words don't match → basic semantic search misses it!

  Problem 2 — CHUNK BOUNDARY PROBLEM
    You split a document at token 500, cutting a sentence in half.
    The second half of that idea is in the next chunk.
    Retrieved chunk is incomplete → confusing answer.

  Problem 3 — IRRELEVANT CHUNKS IN TOP-K
    The top 3 retrieved chunks might include 1-2 that are off-topic.
    The LLM then uses these to generate a partially wrong answer.

  Problem 4 — SINGLE QUERY LIMITATION
    "What is LoRA?" → embedding of this specific phrasing
    If notes say "Low-Rank Adaptation reduces parameters by..." → might miss it
    Different phrasings of the same question retrieve different chunks.

SOLUTIONS:
  1. Hybrid Search    → Combine semantic + keyword (BM25) search
  2. Parent Document  → Retrieve small chunks, return full parent
  3. Re-ranking       → Score chunks again with a powerful cross-encoder
  4. Multi-Query      → Generate 3-5 phrasings, retrieve for all
  5. HyDE             → Generate hypothetical answer, embed THAT
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
except ImportError:
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever

try:
    from langchain_community.retrievers import ParentDocumentRetriever
except ImportError:
    from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever

try:
    from langchain_core.stores import InMemoryStore
except ImportError:
    try:
        from langchain.storage import InMemoryStore
    except ImportError:
        from langchain_community.storage import InMemoryStore

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
parser = StrOutputParser()
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── Sample knowledge base ─────────────────────────────────────────
DOCUMENTS = [
    Document(page_content=(
        "The Transformer architecture consists of an encoder and decoder stack. "
        "Each encoder layer has two sub-layers: multi-head self-attention and a "
        "position-wise feed-forward network. Residual connections and layer normalization "
        "are applied around each sub-layer. The original paper used 6 encoder and 6 decoder "
        "layers. Positional encoding is added to embeddings so the model knows token order, "
        "since attention itself has no sense of position."
    ), metadata={"topic": "transformer", "page": 1}),

    Document(page_content=(
        "Self-attention (also called intra-attention) is a mechanism that relates different "
        "positions of a single sequence to compute a representation. For each token, three "
        "vectors are computed: Query (Q), Key (K), Value (V). The dot product of Q with all "
        "K vectors gives attention scores. After scaling by √dₖ and softmax, these become "
        "attention weights applied to V vectors. Multi-head attention runs this h times in "
        "parallel with different learned projections."
    ), metadata={"topic": "attention", "page": 2}),

    Document(page_content=(
        "Large language models are pre-trained on massive text corpora using next-token "
        "prediction (causal language modeling) or masked language modeling. GPT models use "
        "causal (left-to-right) modeling on the entire internet-scale dataset. BERT uses "
        "masked modeling, predicting randomly hidden tokens. Pre-training learns general "
        "language understanding; fine-tuning adapts to specific tasks. The scale law shows "
        "performance improves predictably with more data, compute, and parameters."
    ), metadata={"topic": "pretraining", "page": 3}),

    Document(page_content=(
        "RAG (Retrieval Augmented Generation) enhances LLMs with external knowledge. "
        "The indexing pipeline: load documents → split into chunks → embed each chunk → "
        "store in vector database. The retrieval pipeline: embed user query → cosine "
        "similarity search → retrieve top-k chunks → inject into prompt context. "
        "This grounds the LLM in factual, up-to-date information and dramatically "
        "reduces hallucination by giving the model text to read from."
    ), metadata={"topic": "rag", "page": 4}),
]


# ─────────────────────────────────────────────────────────────────
# LESSON 6A: BASELINE — Basic RAG (what we had in Step 4)
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("LESSON 6A: Baseline RAG")
print("=" * 60)

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(DOCUMENTS)

basic_store = Chroma.from_documents(chunks, embedder, collection_name="basic")
basic_retriever = basic_store.as_retriever(search_kwargs={"k": 2})

query = "How does the key-query-value mechanism work?"
basic_results = basic_retriever.invoke(query)
print(f"Query: {query}")
print(f"Basic RAG retrieved {len(basic_results)} chunks:")
for r in basic_results:
    print(f"  → [{r.metadata['topic']}]: {r.page_content[:100]}...")


# ─────────────────────────────────────────────────────────────────
# LESSON 6B: MULTI-QUERY RETRIEVAL
# Generates multiple phrasings of the question, retrieves for all
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LESSON 6B: Multi-Query Retrieval")
print("=" * 60)
print("""
CONCEPT:
  A single question phrased one way might not match how your notes
  phrased the same concept. Solution: generate 3 alternative phrasings,
  retrieve for each, then deduplicate results.

  "What is self-attention?" → also retrieves for:
    "How does the attention mechanism compute token relationships?"
    "Explain Q, K, V matrices in transformers"
    "What does multi-head attention do?"
  → Union of all results → much higher recall
""")

# MultiQueryRetriever auto-generates alternative queries using the LLM
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=basic_retriever,
    llm=llm,
)

import logging
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

results = multi_query_retriever.invoke("How does the key-query-value mechanism work?")
print(f"Multi-query retrieved {len(results)} unique chunks (vs {len(basic_results)} basic)")
for r in results:
    print(f"  → [{r.metadata['topic']}]: {r.page_content[:100]}...")


# ─────────────────────────────────────────────────────────────────
# LESSON 6C: PARENT DOCUMENT RETRIEVER
# Retrieve small chunks (for precision), return full parent (for context)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LESSON 6C: Parent Document Retriever")
print("=" * 60)
print("""
CONCEPT:
  Problem: Small chunks are precise for search but lack context.
           Large chunks have context but match too broadly.
  
  Solution: Index SMALL chunks for retrieval, but return the FULL
            parent document when a chunk matches.

  Small chunk retrieved: "Q, K, V vectors are computed..."
  Parent returned:       [entire attention section — 500 words]
  
  The LLM now gets the full context, not just a snippet.
""")

# Two splitters: small for indexing, large (parent) for returning
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

# docstore holds the full parent docs; vectorstore holds the small child embeddings
docstore    = InMemoryStore()
child_store = Chroma(collection_name="children", embedding_function=embedder)

parent_retriever = ParentDocumentRetriever(
    vectorstore=child_store,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
parent_retriever.add_documents(DOCUMENTS)

parent_results = parent_retriever.invoke("How does Q K V attention work?")
print(f"Parent retriever returned {len(parent_results)} full parent documents:")
for r in parent_results:
    print(f"  → [{r.metadata['topic']}]: {len(r.page_content)} chars")
    print(f"     Preview: {r.page_content[:120]}...")


# ─────────────────────────────────────────────────────────────────
# LESSON 6D: HyDE — Hypothetical Document Embedding
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LESSON 6D: HyDE (Hypothetical Document Embedding)")
print("=" * 60)
print("""
CONCEPT:
  Problem: Short questions have sparse embeddings — few words to
           capture meaning. "What is LoRA?" is just 4 words.
  
  Insight: A hypothetical ANSWER to the question would be embedded
           much closer to the actual document chunk in vector space.
  
  Solution:
    Step 1: Ask the LLM to generate a hypothetical answer (ignoring
            hallucination — we don't use this answer!)
    Step 2: Embed the hypothetical answer (it's much richer in vocab)
    Step 3: Search the vector DB with THAT embedding
    Step 4: Use the RETRIEVED chunks (not the hypothetical) to answer
  
  Results: HyDE often improves retrieval quality by 10-20%.
""")

hyde_prompt = ChatPromptTemplate.from_template(
    "Write a short paragraph that would ANSWER this question if it appeared "
    "in a technical study guide about Generative AI. Be specific and technical.\n\n"
    "Question: {question}\n\n"
    "Hypothetical answer paragraph:"
)

hyde_chain = hyde_prompt | llm | parser

def hyde_retrieve(question: str, k: int = 3) -> list:
    """HyDE: generate hypothetical doc → embed it → retrieve real docs."""
    # Step 1: generate hypothetical answer
    hypothetical = hyde_chain.invoke({"question": question})
    print(f"  Hypothetical doc (first 100 chars): {hypothetical[:100]}...")

    # Step 2: embed the hypothetical answer (not the question!)
    hypo_embedding = embedder.embed_query(hypothetical)

    # Step 3: search with the hypothetical embedding
    results = basic_store.similarity_search_by_vector(hypo_embedding, k=k)
    return results

q = "What role do Q, K, V matrices play in transformer models?"
print(f"\nQuery: {q}")
print("\nUsing HyDE:")
hyde_results = hyde_retrieve(q)
for r in hyde_results:
    print(f"  → [{r.metadata['topic']}]: {r.page_content[:100]}...")

print("\nUsing basic search:")
basic = basic_store.similarity_search(q, k=3)
for r in basic:
    print(f"  → [{r.metadata['topic']}]: {r.page_content[:100]}...")


# ─────────────────────────────────────────────────────────────────
# LESSON 6E: RE-RANKING — Scoring retrieved chunks more carefully
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LESSON 6E: LLM Re-Ranking")
print("=" * 60)
print("""
CONCEPT:
  Embedding similarity is fast but imprecise.
  After getting top-10 candidates, use a more powerful model to
  re-score them: "Is this chunk actually relevant to the question?"
  Then keep only the top-3 by re-rank score.

  Two approaches:
  1. Cross-Encoder models (specialized, very accurate, slower)
     → sentence-transformers/ms-marco-MiniLM-L-6-v2
  2. LLM-as-reranker (flexible, uses your existing LLM)
     → Ask the LLM: "Rate relevance 1-10 for each chunk"
""")

rerank_prompt = ChatPromptTemplate.from_template(
    "Rate how relevant this document chunk is for answering the question.\n"
    "Return only a number from 0-10. 10 = perfectly answers the question.\n\n"
    "Question: {question}\n\n"
    "Document chunk:\n{chunk}\n\n"
    "Relevance score (0-10):"
)

def llm_rerank(question: str, candidates: list, top_k: int = 2) -> list:
    """Re-rank candidate chunks using LLM relevance scoring."""
    scored = []
    for doc in candidates:
        score_str = llm.invoke(
            rerank_prompt.invoke({"question": question, "chunk": doc.page_content})
        ).content.strip()
        try:
            score = float(score_str.split()[0])
        except (ValueError, IndexError):
            score = 0.0
        scored.append((score, doc))
        print(f"  Score {score:.1f} | [{doc.metadata['topic']}]: {doc.page_content[:60]}...")

    # Sort by score descending, return top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]

question = "What happens in each transformer encoder layer?"
candidates = basic_store.similarity_search(question, k=5)  # get 5 candidates
print(f"\nRe-ranking {len(candidates)} candidates for: '{question}'")
reranked = llm_rerank(question, candidates, top_k=2)
print(f"\nTop {len(reranked)} after re-ranking:")
for r in reranked:
    print(f"  ✅ [{r.metadata['topic']}]: {r.page_content[:120]}...")


# ─────────────────────────────────────────────────────────────────
# LESSON 6F: Putting It All Together — Advanced RAG Pipeline
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LESSON 6F: Complete Advanced RAG Pipeline")
print("=" * 60)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are Study Buddy. Answer ONLY using the provided context.\n"
     "Cite the topic source for every fact.\n\nCONTEXT:\n{context}"),
    ("human", "{question}"),
])

def advanced_rag(question: str) -> str:
    """
    Full pipeline:
    1. Multi-query expansion (3 phrasings)
    2. Retrieve candidates for each phrasing
    3. Deduplicate
    4. LLM re-rank
    5. Answer with top chunks
    """
    # Step 1-2: Multi-query retrieval
    all_docs = multi_query_retriever.invoke(question)

    # Step 3: Deduplicate by content
    seen = set()
    unique_docs = []
    for doc in all_docs:
        key = doc.page_content[:50]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    # Step 4: Re-rank (use up to 6 candidates, return top 3)
    top_docs = llm_rerank(question, unique_docs[:6], top_k=3)

    # Step 5: Format context and generate
    context = "\n\n---\n".join(
        f"[{d.metadata['topic'].upper()}]: {d.page_content}" for d in top_docs
    )
    chain = rag_prompt | llm | parser
    return chain.invoke({"context": context, "question": question})


print("\nAdvanced RAG Answer:")
answer = advanced_rag("Explain how self-attention computes relationships between tokens.")
print(answer)


"""
WHAT YOU LEARNED:
  ✅ Multi-Query: generate 3+ phrasings → higher recall
  ✅ Parent Document: index small, return large → precision + context
  ✅ HyDE: embed a hypothetical answer → better vector match
  ✅ Re-ranking: LLM or cross-encoder scores candidates → higher precision
  ✅ Pipeline: multi-query → deduplicate → rerank → generate

  ADVANCED INTERVIEW QUESTIONS:
    "What is the difference between sparse and dense retrieval?"
    → Dense (embeddings) = semantic similarity
    → Sparse (BM25/TF-IDF) = keyword frequency
    → Hybrid = combine both scores → best of both worlds

    "When would you use HyDE?"
    → When queries are very short or ambiguous
    → When your embedding model underperforms on questions vs. documents
    → NOT when questions are long and specific (already rich embeddings)

    "What is a cross-encoder re-ranker?"
    → A model that takes (query, doc) as a PAIR and scores relevance
    → More accurate than embedding similarity (sees both together)
    → Slower — only use for final re-ranking, not initial retrieval

NEXT → step_07_finetuning.py
"""
