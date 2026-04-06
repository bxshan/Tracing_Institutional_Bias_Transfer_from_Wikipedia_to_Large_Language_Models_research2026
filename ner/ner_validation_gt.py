"""
ner_validation.py
-----------------
Take 50 articles from the local NELA-GT clone, run spaCy NER blinding,
and write blinded versions to ner/blinded/ for human review.

Output:
  ner/gt_blinded/   blinded article text files

Usage:
  python3 ner_validation.py
"""

import os, random
import spacy
from datasets import load_from_disk

# ── Config ────────────────────────────────────────────────────────────────────
NER_DIR     = os.path.dirname(__file__)
BLINDED_DIR = os.path.join(NER_DIR, "gt_blinded")
GT_PATH     = os.path.join(NER_DIR, "../data/data_full/nela_gt_clone")

os.makedirs(BLINDED_DIR, exist_ok=True)

SPACY_MODEL = "en_core_web_lg"
N_SAMPLES   = 50
RANDOM_SEED = 42

BLIND_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    # "LOC",
    "NORP",
    # "FAC",
    "EVENT",
    # "WORK_OF_ART",
    "LAW",
}


# ── Load NELA-GT articles ─────────────────────────────────────────────────────
def load_gt(n: int) -> list[dict]:
    print(f"[data]  loading NELA-GT from {GT_PATH} ...")
    ds = load_from_disk(GT_PATH)
    random.seed(RANDOM_SEED)
    indices = random.sample(range(len(ds)), n)
    samples = [ds[i] for i in indices]
    print(f"[data]  sampled {len(samples)} articles")
    return samples


# ── Blind pipeline ────────────────────────────────────────────────────────────
def blind_text(nlp, text: str) -> str:
    doc = nlp(text)
    blinded = text
    for ent in reversed(doc.ents):
        if ent.label_ not in BLIND_LABELS:
            continue
        blinded = blinded[:ent.start_char] + f"[{ent.label_}]" + blinded[ent.end_char:]
    return blinded


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[spacy] loading {SPACY_MODEL} ...")
    nlp = spacy.load(SPACY_MODEL)

    samples = load_gt(N_SAMPLES)

    print(f"[blind] blinding {len(samples)} articles ...")
    for i, sample in enumerate(samples, 1):
        source = sample.get("source", "unknown")
        text   = sample.get("content", sample.get("text", ""))
        if not text:
            print(f"  [{i:02d}] {source} — no text field, skipping")
            continue

        blinded = blind_text(nlp, text)

        filename = f"{i:02d}_{source.replace('/', '_')}.txt"
        out_path = os.path.join(BLINDED_DIR, filename)
        with open(out_path, "w") as f:
            f.write(f"SOURCE: {source}\n")
            f.write("=" * 60 + "\n\n")
            f.write(blinded)
            f.write("\n")

        print(f"  [{i:02d}] {source}")

    print(f"\n[done] {len(samples)} blinded articles → {BLINDED_DIR}/")


if __name__ == "__main__":
    main()
