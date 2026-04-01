"""
claude_grader.py
----------------
Grades articles in a CSV for political/ideological bias using Claude.
Adds a 'bias_score_claude' column (0-3 integer) to the output CSV.

Bias rubric:
  0 = Neutral / no detectable bias
  1 = Mild bias (slight slant, loaded word choices)
  2 = Moderate bias (clear partisan framing, selective facts)
  3 = Strong bias (propaganda, explicit partisan advocacy, demonization)

Usage:
  python3 claude_grader.py --input nela_ps_sample_30.csv --output nela_ps_graded.csv
  python3 claude_grader.py --input nela_gt_clone_sample_30.csv --output nela_gt_clone_graded.csv
"""

import argparse
import csv
import os
import time
import anthropic

MODEL = "claude-haiku-4-5-20251001"   # fast + cheap for batch annotation
MAX_TEXT_CHARS = 3000                  # truncate very long articles to keep costs low

SYSTEM_PROMPT = """\
You are a rigorous, politically neutral research assistant helping annotate a dataset \
for an academic study on media bias in education reporting. \
Your job is to assign a single integer bias score (0, 1, 2, or 3) to each article excerpt.

Bias rubric:
  0 = Neutral — factual, balanced, no detectable ideological slant
  1 = Mild — slight slant or occasional loaded language, but mostly factual
  2 = Moderate — clear partisan framing, selective emphasis, or emotionally charged language
  3 = Strong — overt propaganda, explicit advocacy, demonization of groups, or fabricated/misleading claims

Rules:
- Output ONLY the single integer (0, 1, 2, or 3). No explanation, no punctuation, nothing else.
- Base your score on political/ideological bias, not on writing quality or topic sensitivity.
- If the text is too short or garbled to assess, output 0.\
"""

USER_TEMPLATE = """\
Article text:
\"\"\"
{text}
\"\"\"

Bias score (0-3):"""


def grade_article(client: anthropic.Anthropic, text: str, retries: int = 3) -> int | None:
    truncated = text[:MAX_TEXT_CHARS]
    for attempt in range(retries):
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=4,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": USER_TEMPLATE.format(text=truncated)}],
            )
            raw = message.content[0].text.strip()
            score = int(raw)
            if score not in (0, 1, 2, 3):
                raise ValueError(f"Out-of-range score: {score}")
            return score
        except (ValueError, IndexError) as e:
            print(f"  [parse error attempt {attempt+1}] {e} — raw='{raw if 'raw' in dir() else '?'}'")
        except anthropic.APIError as e:
            print(f"  [API error attempt {attempt+1}] {e}")
            time.sleep(2 ** attempt)
    return None


def grade_csv(input_path: str, output_path: str) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic(api_key=api_key)

    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Input CSV is empty.")
        return

    print(f"Loaded {len(rows)} rows from {input_path}")
    print(f"Grading with {MODEL}...\n")

    results = []
    for i, row in enumerate(rows, 1):
        text = row.get("text", "").strip()
        score = grade_article(client, text)
        row["bias_score_claude"] = score if score is not None else ""
        results.append(row)

        status = f"Score: {score}" if score is not None else "Score: FAILED"
        source = row.get("source", "?")[:40]
        print(f"[{i:>3}/{len(rows)}] {status}  |  {source}")

        # Small delay to stay well within rate limits
        time.sleep(0.3)

    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    scored = sum(1 for r in results if r["bias_score_claude"] != "")
    print(f"\nDone. {scored}/{len(results)} articles scored. Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grade articles for bias using Claude.")
    parser.add_argument("--input",  required=True, help="Input CSV path (must have a 'text' column)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()
    grade_csv(args.input, args.output)
