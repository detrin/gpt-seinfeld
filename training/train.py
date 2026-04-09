from __future__ import annotations

import os

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL = os.environ.get("BASE_MODEL", "gpt2-medium")
CHECKPOINT = os.environ.get("CHECKPOINT_DIR", "/content/drive/MyDrive/gpt2-seinfeld")
DATA_FILE = os.environ.get("DATA_FILE", "data/processed/train_v2.jsonl")
USE_LORA = os.environ.get("USE_LORA", "true").lower() != "false"

# Model-specific LoRA targets and training hyper-params
_CONFIGS: dict[str, dict] = {
    "gpt2-medium": dict(
        target_modules=["c_attn", "c_proj"],
        r=16,
        lora_alpha=32,
        epochs=20,
        batch_size=8,
        grad_accum=2,
        lr=3e-4,
        warmup=200,
        max_length=512,
        use_qlora=False,
        dtype=None,  # fp16 set via TrainingArguments
    ),
    # Full fine-tune: all 345M params, fewer epochs + lower LR to avoid overfitting
    "gpt2-medium-full": dict(
        hf_model="gpt2-medium",
        target_modules=[],
        r=0,
        lora_alpha=0,
        epochs=5,
        batch_size=8,
        grad_accum=2,
        lr=5e-5,
        warmup=100,
        max_length=512,
        use_qlora=False,
        dtype=None,
    ),
    # Deep LoRA: adds MLP (c_fc) to attention targets, wider rank
    "gpt2-medium-deep": dict(
        hf_model="gpt2-medium",
        target_modules=["c_attn", "c_proj", "c_fc"],
        r=64,
        lora_alpha=128,
        epochs=20,
        batch_size=8,
        grad_accum=2,
        lr=2e-4,
        warmup=200,
        max_length=512,
        use_qlora=False,
        dtype=None,
    ),
    "Qwen/Qwen2.5-7B-Instruct": dict(
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        r=16,
        lora_alpha=32,
        epochs=3,
        batch_size=2,
        grad_accum=8,  # effective batch = 16
        lr=2e-4,
        warmup=100,
        max_length=512,
        use_qlora=True,
        dtype=torch.bfloat16,
    ),
    "Qwen/Qwen2.5-7B": dict(
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        r=16,
        lora_alpha=32,
        epochs=3,
        batch_size=2,
        grad_accum=8,  # effective batch = 16
        lr=2e-4,
        warmup=100,
        max_length=512,
        use_qlora=True,
        dtype=torch.bfloat16,
    ),
    "meta-llama/Llama-3.2-3B": dict(
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        r=32,
        lora_alpha=64,
        epochs=5,
        batch_size=4,
        grad_accum=4,  # effective batch = 16
        lr=2e-4,
        warmup=100,
        max_length=512,
        use_qlora=True,
        dtype=torch.bfloat16,
    ),
    "meta-llama/Meta-Llama-3.1-8B": dict(
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        r=16,
        lora_alpha=32,
        epochs=3,
        batch_size=2,
        grad_accum=8,  # effective batch = 16
        lr=2e-4,
        warmup=100,
        max_length=512,
        use_qlora=True,  # 4-bit base model to fit on single A100
        dtype=torch.bfloat16,
    ),
}


# Allow partial match (e.g. "llama" → Llama config)
def _get_cfg(model_id: str) -> dict:
    if model_id in _CONFIGS:
        return _CONFIGS[model_id]
    for key, cfg in _CONFIGS.items():
        if key.lower() in model_id.lower() or model_id.lower() in key.lower():
            return cfg
    raise ValueError(f"No config for {model_id!r}. Add it to _CONFIGS in train.py.")


if __name__ == "__main__":
    cfg = _get_cfg(BASE_MODEL)
    HF_MODEL = cfg.get("hf_model", BASE_MODEL)
    print(
        f"Training {BASE_MODEL}  |  hf_model={HF_MODEL}"
        f"  |  USE_LORA={USE_LORA}  |  QLoRA={cfg['use_qlora']}  |  r={cfg['r']}"
    )

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=cfg["max_length"])

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    print(f"Dataset: {len(tokenized)} examples")

    # ── Model ──────────────────────────────────────────────────────────────────
    if cfg["use_qlora"]:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=cfg["dtype"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(HF_MODEL)

    if USE_LORA:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg["r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=0.05,
            target_modules=cfg["target_modules"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"Full fine-tune: all {total:,} parameters trainable")

    # ── Training ───────────────────────────────────────────────────────────────
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    use_bf16 = cfg["dtype"] == torch.bfloat16
    args = TrainingArguments(
        output_dir=CHECKPOINT,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        warmup_steps=cfg["warmup"],
        learning_rate=cfg["lr"],
        weight_decay=0.01,
        fp16=not use_bf16,
        bf16=use_bf16,
        gradient_checkpointing=cfg["use_qlora"],
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
    )

    Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collator).train()

    # ── Save checkpoint ────────────────────────────────────────────────────────
    if USE_LORA:
        merged = model.merge_and_unload()
        # Disable gradient checkpointing so KV-cache works during inference
        if hasattr(merged, "gradient_checkpointing_disable"):
            merged.gradient_checkpointing_disable()
        merged.config.use_cache = True
        merged.save_pretrained(CHECKPOINT)
        tokenizer.save_pretrained(CHECKPOINT)
        del merged
    else:
        model.config.use_cache = True
        model.save_pretrained(CHECKPOINT)
        tokenizer.save_pretrained(CHECKPOINT)
        del model
    print(f"Checkpoint saved to {CHECKPOINT}")

    # ── Sanity check — reload from disk for a clean inference state ────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gen = pipeline("text-generation", model=CHECKPOINT, tokenizer=CHECKPOINT, device_map="auto")
    out = gen(
        "TOPIC: parking tickets\n\n[",
        max_new_tokens=200,
        do_sample=True,
        temperature=0.9,
        repetition_penalty=1.2,
    )
    print("\n--- Sanity check ---\n")
    print(out[0]["generated_text"])
