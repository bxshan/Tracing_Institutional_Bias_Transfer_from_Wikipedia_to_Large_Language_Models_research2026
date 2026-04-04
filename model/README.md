# Model Selection for Bias Detection Fine-Tuning
2026-04-01
For bias classification (on a scale of 0–3) on news articles
Apple M2 Mac Studio (16–32 GB unified memory)
source: from HuggingFace model cards + official technical reports
comparison of 3 models based on params, license, and fit for hardware (LoRA fine tune)

---

## 1) Qwen2.5-1.5B (Alibaba)

**HuggingFace:** `Qwen/Qwen2.5-1.5B`
**Instruct variant:** `Qwen/Qwen2.5-1.5B-Instruct`, recommended for fine-tuning

### Params
- Total: **1.54B**
- Layers: 28 | Attention heads: 12Q / 2KV (GQA)

### License
**Apache 2.0**, Most permissive

### Hardware (M2 Mac Studio)
| Method                | approx mem  | Fits on 16 GB?  |
|---                    |---          |---              |
| Full fine-tune (fp16) | ~12 GB      | Yes             |
| LoRA                  | ~4.6 GB     | Yes             |

### Pre-training Data
Trained on up to **18 trillion tokens** (Qwen2.5 series total — exact 1.5B share not disclosed).
Sources: public web documents, books, code, mathematics, and synthetic data.
Multilingual: **29+ languages** including Chinese + English
Specific dataset names are not publicly released by Alibaba

### Strengths for This Task
- Largest pre-training corpus (18T tokens) in this comparison → broadest world knowledge
- Apache 2.0 license removes all commercial and deployment friction
- 32K context handles long-form articles natively
- Lightest memory footprint → fastest iteration on M2

### Weaknesses for This Task
- Least transparent data provenance of the three candidates
- Base model requires post-training before fine-tuning on classification
- Multilingual training may dilute English news domain focus

---

## 2) Llama 3.2 3B (Meta)

**HuggingFace:** `meta-llama/Llama-3.2-3B`
**Instruct variant:** `meta-llama/Llama-3.2-3B-Instruct`, recommended for fine-tuning
requres accepting license on HF

### Params
- Total: **3.21B**
- Architecture: Auto-regressive transformer with GQA and shared embeddings

### License
custom Llama 3.2 Community License, but really only limits on commercial use

### Hardware (M2 Mac Studio)
| Method                | approx mem | Fits on 16 GB? |
|---                    |---         |---             |
| Full fine-tune (fp16) | ~25.6 GB   | No             |
| LoRA                  | ~9.6 GB    | Yes            |

### Pre-training Data
Trained on up to **9 trillion tokens** from *"a new mix of publicly available online data."*
Knowledge distillation from **Llama 3.1 8B and 70B** used during pre-training — smaller model inherits reasoning capacity from larger parent models.
Post-training: multiple rounds of SFT, Rejection Sampling (RS), and DPO.
Knowledge cutoff: **December 2023**.
Officially supported languages: English, etc. (excl. Chinese)
Biases: verbatim, "The model may in some instances produce inaccurate, biased or other objectionable responses to user prompts

### Strengths for This Task
- Strongest instruction-following score (IFEval 77.4) of the three — benefits prompt-based classification
- 128K context window is the largest in this group
- Knowledge distillation from 8B/70B gives disproportionate reasoning depth for a 3B model
- Well-documented safety evaluation process (red team, DPO alignment)

### Weaknesses for This Task
- Gated model adds friction to download and deployment
- LoRA-only on 16 GB — cannot full fine-tune without 32 GB
- Pre-training data description is vague (*"publicly available online data"*) — harder to reason about inherited biases
- Custom license (not Apache/MIT) requires tracking compliance

---

## 3) Phi-3 Mini 4K Instruct (Microsoft)

**HuggingFace:** `microsoft/Phi-3-mini-4k-instruct`

### Params
- Total: **3.8B**
- Architecture: Dense decoder-only transformer, fine-tuned with SFT + DPO

### License
**MIT License**, fully open

### Hardware (M2 Mac Studio)
| Method                | approx mem | Fits on 16 GB? |
|---                    |---         |---             |
| Full fine-tune (fp16) | ~30.4 GB   | No             |
| LoRA                  | ~11.4 GB   | Yes            |

### Pre-training Data
Trained on **4.9 trillion tokens** (model card) / 3.3 trillion tokens (technical report, arXiv:2404.14219).
Three-source data mix:
1. **Filtered public web** — quality-selected educational and factual content; sports results, ephemeral facts, and low-reasoning content explicitly removed
2. **Synthetic "textbook-like" data** — Microsoft-generated content covering math, coding, common-sense reasoning, science, theory of mind
3. **Curated SFT chat data** — covering instruction-following, truthfulness, honesty, helpfulness
Knowledge cutoff: **October 2023**.
Primarily **English**. Non-English performance degrades significantly.
Biases: verbatim, "These models can over- or under-represent groups of people, erase representation of some groups, or reinforce demeaning or negative stereotypes. Despite safety post-training, these limitations may still be present due to differing levels of representation of different groups or prevalence of examples of negative stereotypes in training data that reflect real-world patterns and societal biases."

### Strengths for This Task
- **Highest reasoning benchmarks per parameter** of the three candidates
- **Social IQA 77.6** — best social reasoning score, relevant for detecting ideological and social bias in text
- MIT license — zero compliance overhead
- Explicit model card bias documentation — most transparent of the three

### Weaknesses for This Task
- **4K context window** is the main practical limitation — long articles must be truncated or chunked
- Synthetic training data may make the model less familiar with naturalistic news writing style
- Largest memory footprint (LoRA ~11.4 GB) of the three
- English-only in practice

---

## Summary

| Property           | Qwen2.5-1.5B            | Llama 3.2 3B          | Phi-3 Mini 4K     |
|---                 |---                      |---                    |---                |
| Parameters         | 1.54B                   | 3.21B                 | 3.8B              |
| License            | Apache 2.0              | Llama 3.2 (custom)    | MIT               |
| Gated?             | No                      | Yes                   | No                |
| LoRA mem (est.)    | ~4.6 GB                 | ~9.6 GB               | ~11.4 GB          |
| Fits 16 GB (LoRA)  | Yes                     | Yes                   | Yes               |
| Pre-train tokens   | 18T                     | 9T                    | 4.9T              |
| Context window     | 32K                     | 128K                  | 4K                |
| Knowledge cutoff   | Undisclosed             | Dec 2023              | Oct 2023          |
| Bias documentation | Minimal                 | General disclaimer    | Explicit          |
| Social IQA         | —                       | —                     | 77.6              |
| Best for           | Speed, iteration, scale | Instruction-following | Reasoning quality |

---

## Claude Recommendation

**Primary candidate: Llama 3.2 3B Instruct**
Best balance of instruction-following capability (IFEval 77.4), 128K context window, documented training provenance, and hardware fit. Knowledge distillation from 8B/70B gives it reasoning depth disproportionate to its size. The custom license is acceptable for academic research.

**Secondary candidate: Phi-3 Mini 4K Instruct**
Superior reasoning per parameter and the most transparent bias documentation, making it the most defensible choice for a study *about* bias. The 4K context window is the only significant practical constraint — mitigated by using the 128K variant or chunking long articles.

**Tertiary / baseline: Qwen2.5-1.5B Instruct**
Ideal for rapid prototyping and ablations due to minimal memory footprint and Apache 2.0 license. Not recommended as the final model due to opacity of training data provenance — difficult to reason about inherited biases in a bias detection study.\
*This is the model used for the fine tuning feasability test*

**Not recommended:** GPT-2 (1K context too short for news articles; documented racial bias in pre-training data is specifically problematic for a bias detection task). Gemma 7B (does not fit 16 GB without aggressive quantization).
