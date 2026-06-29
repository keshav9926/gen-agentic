# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║       AI STUDY BUDDY — STEP 7: Teaching It New Tricks           ║
║                   (Fine-Tuning & LoRA / QLoRA)                  ║
╚══════════════════════════════════════════════════════════════════╝

WHEN TO FINE-TUNE VS. RAG:
  - RAG: Injecting dynamic, external KNOWLEDGE (open-book).
  - Fine-Tuning: Adapting BEHAVIOR, tone, format, and style (closed-book).

CORE CONCEPTS:
  - LoRA (Low-Rank Adaptation): Freezes base model weights and trains small 
    adapter matrices (rank 'r'). Reduces trainable parameters by ~99%.
  - QLoRA: Quantizes base model to 4-bit, then trains 16-bit LoRA adapters on top.
    Allows running/training massive models on consumer GPUs.
"""

import json
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────
# 1. Training Data Format (Supervised Fine-Tuning)
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. DATA PREPARATION (Supervised Fine-Tuning)")
print("=" * 60)

# SFT data typically uses a standardized chat format:
sample_dataset = [
    {
        "messages": [
            {"role": "system", "content": "You are Study Buddy, a Gen AI tutor. Explain with analogies and end with a practice question."},
            {"role": "user", "content": "What is an embedding?"},
            {"role": "assistant", "content": "DEFINITION: An embedding is a vector representation of data that captures meaning.\n\nANALOGY: Like cities close together on a map, words with similar meanings have vectors close in meaning space.\n\nPRACTICE QUESTION: If 'dog' and 'puppy' are close but 'dog' and 'democracy' are far, what does this tell you about embeddings?"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are Study Buddy, a Gen AI tutor. Explain with analogies and end with a practice question."},
            {"role": "user", "content": "Explain RAG vs. fine-tuning."},
            {"role": "assistant", "content": "DEFINITION: RAG retrieves context at runtime (open-book). Fine-tuning updates weights during training (closed-book).\n\nANALOGY: RAG is referencing a textbook during an exam; fine-tuning is studying beforehand.\n\nPRACTICE QUESTION: If a firm needs specific up-to-date document access and formal tone, how should they combine RAG and fine-tuning?"}
        ]
    }
]

print(f"Sample SFT dataset initialized with {len(sample_dataset)} examples.")
print("Example structure:")
print(json.dumps(sample_dataset[0], indent=2))


# ─────────────────────────────────────────────────────────────────
# 2. LoRA / QLoRA Configuration & Training Guide
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. LORA CONFIGURATION & HYPERPARAMETERS")
print("=" * 60)

# Key parameters to configure:
# - r (rank): Bottleneck dimension (typical: 8, 16, 32, 64). Higher = more capacity.
# - lora_alpha: Scaling factor (typically set to r or 2*r).
# - target_modules: Modules to adapt (e.g., ["q_proj", "v_proj"]).
# - learning_rate: Typically 2e-4 for LoRA.
# - num_train_epochs: Typically 2-3 epochs to avoid overfitting.

print("""
Recommended Configurations:
  - Rank (r): 16 (good balance of performance and parameters)
  - Alpha: 32
  - Target Modules: All linear layers (q_proj, k_proj, v_proj, o_proj)
  - Learning Rate: 2e-4 (cosine decay scheduler)
  - Batch Size: 16 (effective, using gradient accumulation if needed)
""")


# ─────────────────────────────────────────────────────────────────
# 3. Fine-Tuning Code (For GPU Environments)
# ─────────────────────────────────────────────────────────────────
def run_finetuning_pipeline_demo():
    """
    Template logic for fine-tuning using HuggingFace TRL & PEFT.
    Requires a GPU (e.g., Google Colab, Kaggle, RunPod).
    """
    print("\n[INFO] Loading QLoRA fine-tuning pipeline template...")
    
    # conceptual steps:
    # 1. Load model in 4-bit using BitsAndBytesConfig
    # 2. Apply LoRA adapter config (LoraConfig)
    # 3. Apply tokenizer chat templates to SFT messages
    # 4. Initialize SFTTrainer with TrainingArguments (epochs=3, lr=2e-4)
    # 5. Execute trainer.train() and save the adapters.
    
    print("""
    Code template for implementation:
    -------------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer

    # 1. Quantization Setup
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 2. PEFT / LoRA Setup
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # 3. SFT Training
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=True
    )
    trainer.train()
    model.save_pretrained("./study_buddy_adapter")
    -------------------------------------------------------------------------
    """)

run_finetuning_pipeline_demo()


# ─────────────────────────────────────────────────────────────────
# 4. Alignment & Evaluation (RLHF & DPO)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. ALIGNMENT & QUALITY EVALUATION")
print("=" * 60)
print("""
ALIGNMENT METHODS:
  - RLHF (Reinforcement Learning from Human Feedback):
    SFT Model -> Train Reward Model (from pairwise comparisons) -> PPO reinforcement loop.
  - DPO (Direct Preference Optimization):
    Directly optimizes the LLM on preference pairs (chosen/rejected), bypassing reward model training.

EVALUATION METRICS:
  - Loss Curves: Training & Validation loss must decrease (diverging indicates overfitting).
  - Perplexity: Measure of prediction uncertainty (lower is better).
  - LLM-as-a-Judge: Evaluate output quality using a stronger model (e.g., GPT-4).
""")

"""
  ✅ When to fine-tune vs. RAG vs. prompting (critical decision)
    - Prompting: Adjusting inputs (context/instructions) without updating weights. Fast iteration, but limited context capacity.
    - RAG: Querying dynamic external documents at runtime (open-book exam). Best for real-time factual knowledge and citations.
    - Fine-Tuning: Modifying the model's actual weights on training data (closed-book exam). Best for style, tone, format, and latency.

  ✅ LoRA: add B×A adapters, freeze base weights, train 0.4% of params
  ✅ QLoRA: 4-bit quantize base + LoRA adapters = run 70B on one GPU
  ✅ SFTTrainer: complete training loop with HuggingFace
  ✅ RLHF: SFT → Reward Model → PPO alignment
  ✅ DPO: modern alternative to PPO, more stable
  ✅ Eval: loss curves, LLM-as-judge, human evaluation

  INTERVIEW QUESTIONS:
    "Explain LoRA in one sentence."
    → Instead of updating all 7 billion weights, LoRA adds tiny adapter 
      matrices (rank-16) to key layers, updating only 0.4% of parameters.

    "When would you fine-tune over prompting?"
    → When you need consistent output format, domain jargon, or style 
      that can't be achieved reliably with prompt engineering alone.

    "What is the catastrophic forgetting problem?"
    → When fine-tuning on your data, the model loses general capabilities.
    → LoRA largely solves this because the base weights stay frozen.

NEXT → step_08_evaluation.py
"""
