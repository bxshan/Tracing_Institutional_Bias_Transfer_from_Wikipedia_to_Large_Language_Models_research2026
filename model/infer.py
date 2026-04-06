"""
infer.py
--------
Run inference on a fine-tuned Qwen2.5-1.5B-Instruct adapter.

Usage:
  python3 infer.py --adapter qwen-sft-gt
  python3 infer.py --adapter qwen-sft-ps
  python3 infer.py --adapter qwen-sft-gt --prompt "Your custom prompt here"
  python3 infer.py --adapter qwen-sft-gt --compare qwen-sft-ps   # side-by-side
  python3 infer.py --adapter qwen-sft-gt --compare base          # compare vs untuned base

Arguments:
  --adapter   path to adapter dir (relative to model/, or absolute)
  --prompt    custom prompt text (optional — uses default school board prompt)
  --compare   second adapter for side-by-side; use "base" to compare against untuned model
  --max_new   max tokens to generate (default 150)
  --temp      sampling temperature (default 0.7; use 0.0 for deterministic)
"""

import os, sys, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE   = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_DIR = os.path.dirname(__file__)

SYSTEM_PROMPT = "You are a news article writer. Continue the article naturally."

DEFAULT_PROMPT = (
    "The school board meeting Tuesday drew hundreds of parents who gathered "
    "to discuss proposed changes to the district curriculum."
)


"""
Accept either a short name (qwen-sft-gt) or a full path.
"""
def resolve_adapter(path: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(MODEL_DIR, path)
    if os.path.exists(candidate):
        return candidate
    print(f"[error] adapter not found: {path}")
    print(f"        tried: {candidate}")
    sys.exit(1)


def load_model(adapter_path: str | None):
    print(f"[model] loading base: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    if adapter_path is None:
        print(f"[model] no adapter — using base model only")
        base.eval()
        return base, tokenizer

    print(f"[model] loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt_text: str, max_new: int, temp: float) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=(temp > 0),
            temperature=temp if temp > 0 else 1.0,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",  required=True, help="Adapter dir (e.g. qwen-sft-gt)")
    parser.add_argument("--prompt",   default=DEFAULT_PROMPT, help="Input prompt text")
    parser.add_argument("--compare",  default=None, help="Second adapter for side-by-side")
    parser.add_argument("--max_new",  type=int, default=150)
    parser.add_argument("--temp",     type=float, default=0.7)
    args = parser.parse_args()

    adapter_a = resolve_adapter(args.adapter)
    label_a = os.path.basename(adapter_a)

    print(f"\n{'='*60}")
    print(f"  Prompt: \"{args.prompt[:80]}{'...' if len(args.prompt)>80 else ''}\"")
    print(f"{'='*60}\n")

    # ── Single adapter ────────────────────────────────────────────
    model, tokenizer = load_model(adapter_a)
    completion_a = generate(model, tokenizer, args.prompt, args.max_new, args.temp)

    print(f"[{label_a}]\n{completion_a}\n")

    # ── Side-by-side comparison ───────────────────────────────────
    if args.compare:
        del model  # free memory before loading second model
        torch.mps.empty_cache() if DEVICE == "mps" else None

        if args.compare.lower() == "base":
            adapter_b = None
            label_b = "base"
        else:
            adapter_b = resolve_adapter(args.compare)
            label_b = os.path.basename(adapter_b)

        model_b, _ = load_model(adapter_b)
        completion_b = generate(model_b, tokenizer, args.prompt, args.max_new, args.temp)

        print(f"[{label_b}]\n{completion_b}\n")

        print("=" * 60)
        print("  SIDE-BY-SIDE")
        print("=" * 60)
        print(f"\n[{label_a}]\n{completion_a}")
        print(f"\n[{label_b}]\n{completion_b}")


if __name__ == "__main__":
    main()
