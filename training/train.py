from __future__ import annotations

import os

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)

CHECKPOINT = os.environ.get("CHECKPOINT_DIR", "/content/drive/MyDrive/gpt2-seinfeld")
DATA_FILE = os.environ.get("DATA_FILE", "train.jsonl")

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],  # GPT-2 attention projections
    bias="none",
)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=512)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    print(f"Dataset: {len(tokenized)} examples")

    model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=CHECKPOINT,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=True,
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
    )

    Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collator).train()

    # Merge LoRA weights into base model for clean export
    merged = model.merge_and_unload()
    merged.save_pretrained(CHECKPOINT)
    tokenizer.save_pretrained(CHECKPOINT)
    print(f"Merged checkpoint saved to {CHECKPOINT}")

    gen = pipeline("text-generation", model=merged, tokenizer=tokenizer, device=0)
    out = gen("TOPIC: parking tickets\n\n[", max_new_tokens=200, do_sample=True, temperature=0.9)
    print("\n--- Sanity check ---\n")
    print(out[0]["generated_text"])
