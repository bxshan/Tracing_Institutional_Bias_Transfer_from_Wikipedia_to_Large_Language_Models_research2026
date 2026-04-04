"""
finetune_feasibility.py
-----------------------
End-to-end feasibility test for fine-tuning Qwen2.5-1.5B-Instruct on bias
classification using LoRA/PEFT on Apple M2 (MPS).

Pipeline:
  1. Data loading      — reads graded CSVs, formats as chat prompts
  2. Tokenization      — Qwen2.5 tokenizer with chat template
  3. Fine-tuning       — LoRA via PEFT, 5 steps on MPS (goal: pipeline runs, not good results)
  4. Inference check   — generate a completion from the fine-tuned adapter

Usage:
  python3 finetune_feasibility.py

Outputs:
  model/qwen-lora-test/   — saved LoRA adapter weights
"""

import os, csv, sys, time, platform
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "qwen-lora-test")
MAX_LEN       = 512      # truncate long articles for feasibility test
TRAIN_STEPS   = 5        # just enough to confirm training loop runs
LORA_RANK     = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.05

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SYSTEM_PROMPT = (
    "You are a politically neutral research assistant. "
    "Rate the political/ideological bias of the following news article excerpt "
    "on a scale of 0 to 3.\n"
    "0 = Neutral, 1 = Mild bias, 2 = Moderate bias, 3 = Strong bias.\n"
    "Reply with a single integer only."
)

# ── Step 1: Data loading ──────────────────────────────────────────────────────
def load_samples():
    samples = []
    files = [
        ("nela_gt_clone_sample_30.csv", "bias_score_claude"),
        ("nela_ps_sample_30.csv",       "bias_score_claude"),
    ]
    for fname, score_col in files:
        path = os.path.join(DATA_DIR, fname)
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                score = row.get(score_col, "").strip()
                text  = row.get("text", "").strip()
                if score and text:
                    samples.append({"text": text, "label": int(score)})
    print(f"[data]  loaded {len(samples)} labelled samples")
    return samples


# ── Step 2: Tokenization ──────────────────────────────────────────────────────
def format_prompt(text: str, label: int | None, tokenizer) -> str:
    """Format a single example as a Qwen chat template string."""
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"Article:\n{text[:2000]}"},  # hard truncate text
    ]
    if label is not None:
        messages.append({"role": "assistant", "content": str(label)})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=(label is None),
    )


class BiasDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.encodings = []
        for s in samples:
            prompt = format_prompt(s["text"], s["label"], tokenizer)
            enc = tokenizer(
                prompt,
                truncation=True,
                max_length=MAX_LEN,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze()
            # For causal LM: labels = input_ids (mask padding with -100)
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            self.encodings.append({
                "input_ids":      input_ids,
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels":         labels,
            })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


# ── Step 3: Fine-tuning ───────────────────────────────────────────────────────
def run_finetune(samples, tokenizer, model):
    print(f"\n[lora]  configuring LoRA  rank={LORA_RANK}  alpha={LORA_ALPHA}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = BiasDataset(samples, tokenizer)
    print(f"[data]  dataset size: {len(dataset)} samples")

    # Use MPS where possible; fall back gracefully
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=TRAIN_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,           # MPS does not support fp16 training
        bf16=False,           # bf16 also unsupported on MPS
        logging_steps=1,
        save_steps=TRAIN_STEPS,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print(f"\n[train] starting — {TRAIN_STEPS} steps on {DEVICE.upper()}")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.1f}s")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[save]  adapter saved to {OUTPUT_DIR}")
    return model


# ── Step 4: Inference check ───────────────────────────────────────────────────
def run_inference(model, tokenizer):
    test_article = (
        "Governor Ron DeSantis signed legislation today banning the teaching of "
        "Critical Race Theory in Florida public schools, calling it 'ideological "
        "indoctrination' and pledging to return education to 'facts and fundamentals.' "
        "Democrats called the bill an attack on free speech and accurate history."
    )
    print("\n[infer] running inference on test article:")
    print(f"        \"{test_article[:100]}...\"")

    prompt = format_prompt(test_article, label=None, tokenizer=tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    model.eval()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[infer] model output: '{decoded.strip()}'")
    print("[infer] expected: single integer 0-3")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Qwen2.5-1.5B-Instruct  LoRA Feasibility Test")
    print(f"  Device: {DEVICE.upper()}")
    print("=" * 60)

    # 1. Load data
    samples = load_samples()
    if not samples:
        print("[error] no samples found — check CSV paths")
        sys.exit(1)

    # 2. Load tokenizer + model
    print(f"\n[model] loading {MODEL_ID}  (downloading if needed...)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,         # MPS requires float32
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    print(f"[model] loaded  params: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Fine-tune
    t_train_start = time.time()
    model = run_finetune(samples, tokenizer, model)
    t_train_end = time.time()

    # 4. Inference
    run_inference(model, tokenizer)

    train_secs = t_train_end - t_train_start
    try:
        import subprocess
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip() or subprocess.check_output(
            ["sysctl", "-n", "hw.model"], text=True
        ).strip()
    except Exception:
        chip = platform.processor() or platform.machine()
    mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)

    print("\n[done]  pipeline verified end-to-end: load -> tokenize -> train -> infer")
    print(f"        adapter weights at: {OUTPUT_DIR}")
    print(f"\n  Hardware : {platform.system()} {platform.release()}  |  chip: {chip}")
    print(f"  Device   : {DEVICE.upper()}")
    print(f"  RAM      : {mem_gb:.1f} GB")
    print(f"  Training : {TRAIN_STEPS} steps  |  {train_secs:.1f}s total  |  {train_secs/TRAIN_STEPS:.1f}s/step")


if __name__ == "__main__":
    main()
