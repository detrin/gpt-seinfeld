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
DATA_FILE = os.environ.get("DATA_FILE", "train.jsonl")

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
    print(f"Training {BASE_MODEL}  |  QLoRA={cfg['use_qlora']}  |  r={cfg['r']}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
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
            BASE_MODEL,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

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

    # ── Save merged checkpoint ─────────────────────────────────────────────────
    merged = model.merge_and_unload()
    merged.save_pretrained(CHECKPOINT)
    tokenizer.save_pretrained(CHECKPOINT)
    print(f"Merged checkpoint saved to {CHECKPOINT}")

    # ── Sanity check ───────────────────────────────────────────────────────────
    device = 0 if torch.cuda.is_available() else -1
    gen = pipeline("text-generation", model=merged, tokenizer=tokenizer, device=device)
    out = gen(
        "TOPIC: parking tickets\n\n[",
        max_new_tokens=200,
        do_sample=True,
        temperature=0.9,
        repetition_penalty=1.2,
    )
    print("\n--- Sanity check ---\n")
    print(out[0]["generated_text"])
