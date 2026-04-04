"""
sft_bias.py
-----------
Supervised fine-tuning (SFT) of Qwen2.5-1.5B-Instruct on either the NELA-GT
clone or NELA-PS dataset. The goal is domain / style adaptation: the model learns
to complete articles in the register and framing patterns of the chosen corpus,
effectively absorbing whatever ideological bias that corpus carries.

Training format (instruction tuning):
  system:    "You are a news article writer. Continue the article naturally."
  user:      first 60% of article (the prompt)
  assistant: remaining 40% of article (the completion the model learns to produce)

Usage:
  python3 sft_bias.py --dataset gt                  # NELA-GT clone (national news)
  python3 sft_bias.py --dataset ps                  # NELA-PS (pink slime local news)
  python3 sft_bias.py --dataset gt --n_samples 500 --steps 200

Arguments:
  --dataset     gt | ps            which corpus to train on (required)
  --n_samples   int  (default 300) number of articles to sample from the corpus
  --steps       int  (default 100) number of optimizer steps
  --rank        int  (default 8)   LoRA rank
  --max_len     int  (default 768) max token length per sample

Outputs:
  model/qwen-sft-gt/   or   model/qwen-sft-ps/
"""

import os, csv, sys, time, random, platform, argparse, subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "data_full")
MODEL_DIR   = os.path.dirname(__file__)

GT_PATH     = os.path.join(DATA_DIR, "nela_gt_clone")
PS_PATH     = os.path.join(DATA_DIR, "nela_ps_full", "nela_ps_newsdata.csv")

MIN_CHARS   = 400    # skip articles shorter than this
SPLIT_RATIO = 0.6    # first 60% = prompt, last 40% = completion
LORA_ALPHA  = 16
LORA_DROPOUT = 0.05

SYSTEM_PROMPT = "You are a news article writer. Continue the article naturally."

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ── Data loading ──────────────────────────────────────────────────────────────
def load_gt(n_samples: int) -> list[dict]:
    """Load n_samples articles from the NELA-GT clone Arrow dataset."""
    from datasets import load_from_disk
    print(f"[data]  loading NELA-GT clone from {GT_PATH} ...")
    ds = load_from_disk(GT_PATH)
    print(f"[data]  total articles available: {len(ds):,}")

    # shuffle and filter for min length
    indices = list(range(len(ds)))
    random.shuffle(indices)

    samples = []
    for i in indices:
        row = ds[i]
        text = (row.get("content") or "").strip()
        if len(text) >= MIN_CHARS:
            samples.append({
                "source": row.get("source", ""),
                "title":  row.get("title", ""),
                "text":   text,
            })
        if len(samples) >= n_samples:
            break

    print(f"[data]  sampled {len(samples)} GT articles (min {MIN_CHARS} chars)")
    return samples


def load_ps(n_samples: int) -> list[dict]:
    """Load n_samples articles from the NELA-PS CSV."""
    print(f"[data]  loading NELA-PS from {PS_PATH} ...")
    samples = []
    with open(PS_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    random.shuffle(rows)
    for row in rows:
        text = (row.get("content") or "").strip()
        if len(text) >= MIN_CHARS:
            samples.append({
                "source": row.get("source", ""),
                "title":  row.get("title", ""),
                "text":   text,
            })
        if len(samples) >= n_samples:
            break

    print(f"[data]  sampled {len(samples)} PS articles (min {MIN_CHARS} chars)")
    return samples


# ── Prompt formatting ─────────────────────────────────────────────────────────
def format_sft_prompt(sample: dict, tokenizer) -> str:
    """
    Split the article at SPLIT_RATIO.
    First portion becomes the user prompt; remainder becomes the assistant completion.
    """
    text  = sample["text"]
    split = int(len(text) * SPLIT_RATIO)
    # try to split at a sentence boundary near the split point
    boundary = text.find(". ", split)
    if boundary == -1 or boundary > split + 200:
        boundary = split
    else:
        boundary += 2  # include the period and space

    prompt_text     = text[:boundary].strip()
    completion_text = text[boundary:].strip()

    if not completion_text:
        completion_text = prompt_text[-100:]  # fallback: repeat tail

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": prompt_text},
        {"role": "assistant", "content": completion_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


# ── Dataset ───────────────────────────────────────────────────────────────────
class SFTDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, max_len: int):
        self.encodings = []
        skipped = 0
        for s in samples:
            prompt = format_sft_prompt(s, tokenizer)
            enc = tokenizer(
                prompt,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze()
            labels    = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            # skip samples that are entirely padding after truncation
            if (labels != -100).sum() < 10:
                skipped += 1
                continue

            self.encodings.append({
                "input_ids":      input_ids,
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels":         labels,
            })

        if skipped:
            print(f"[data]  skipped {skipped} samples (too short after truncation)")
        print(f"[data]  final dataset: {len(self.encodings)} samples")

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


# ── Training ──────────────────────────────────────────────────────────────────
def run_sft(samples, tokenizer, model, args, output_dir):
    print(f"\n[lora]  rank={args.rank}  alpha={LORA_ALPHA}  "
          f"target=q/k/v/o_proj")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = SFTDataset(samples, tokenizer, args.max_len)

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=min(10, args.steps // 10),
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=False,
        logging_steps=max(1, args.steps // 20),
        save_steps=args.steps,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print(f"[train] {args.steps} steps  |  {len(dataset)} samples  |  device: {DEVICE.upper()}")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.1f}s")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[save]  adapter saved → {output_dir}")
    return model, elapsed


# ── Inference check ───────────────────────────────────────────────────────────
def run_inference(model, tokenizer, dataset_name: str):
    # neutral prompt — same for both models so outputs are comparable
    prompt_text = (
        "The school board meeting Tuesday drew hundreds of parents who gathered "
        "to discuss proposed changes to the district curriculum."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    model.eval()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    completion = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    print(f"\n[infer] neutral prompt fed to {dataset_name.upper()}-trained model:")
    print(f"        \"{prompt_text}\"")
    print(f"\n[infer] completion:")
    print(f"        {completion}")


# ── Hardware summary ──────────────────────────────────────────────────────────
def print_summary(dataset_name, n_samples, steps, elapsed):
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        chip = platform.processor() or platform.machine()
    mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)

    print(f"\n{'='*60}")
    print(f"  SFT run complete")
    print(f"  Dataset  : {dataset_name.upper()}")
    print(f"  Samples  : {n_samples}")
    print(f"  Steps    : {steps}")
    print(f"  Hardware : {platform.system()} {platform.release()}  |  {chip}")
    print(f"  Device   : {DEVICE.upper()}  |  RAM: {mem_gb:.1f} GB")
    print(f"  Time     : {elapsed:.1f}s total  |  {elapsed/steps:.1f}s/step")
    print(f"{'='*60}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SFT bias injection for Qwen2.5-1.5B")
    parser.add_argument("--dataset",   required=True, choices=["gt", "ps"],
                        help="Which corpus to train on: gt (NELA-GT clone) or ps (NELA-PS)")
    parser.add_argument("--n_samples", type=int, default=300,
                        help="Number of articles to sample (default 300)")
    parser.add_argument("--steps",     type=int, default=100,
                        help="Optimizer steps (default 100)")
    parser.add_argument("--rank",      type=int, default=8,
                        help="LoRA rank (default 8)")
    parser.add_argument("--max_len",   type=int, default=768,
                        help="Max token length per sample (default 768)")
    args = parser.parse_args()

    output_dir = os.path.join(MODEL_DIR, f"qwen-sft-{args.dataset}")

    print("=" * 60)
    print(f"  SFT Bias Injection — dataset: {args.dataset.upper()}")
    print(f"  samples={args.n_samples}  steps={args.steps}  "
          f"rank={args.rank}  max_len={args.max_len}")
    print(f"  output → {output_dir}")
    print("=" * 60)

    random.seed(42)

    # 1. Load data
    samples = load_gt(args.n_samples) if args.dataset == "gt" else load_ps(args.n_samples)
    if not samples:
        print("[error] no samples loaded — check data paths")
        sys.exit(1)

    # 2. Load model + tokenizer
    print(f"\n[model] loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    print(f"[model] loaded  params: {sum(p.numel() for p in model.parameters()):,}")

    # 3. SFT
    model, elapsed = run_sft(samples, tokenizer, model, args, output_dir)

    # 4. Inference check
    run_inference(model, tokenizer, args.dataset)

    # 5. Summary
    print_summary(args.dataset, len(samples), args.steps, elapsed)


if __name__ == "__main__":
    main()
