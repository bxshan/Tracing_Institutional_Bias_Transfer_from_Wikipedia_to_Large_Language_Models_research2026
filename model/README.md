# Model Selection for Bias Detection Fine-Tuning
2026-04-01\
For bias classification (on a scale of 0–3) on news articles\
Apple M2 Mac Studio (16–32 GB unified memory)\
source: from HuggingFace model cards + official technical reports\
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

## Fine Tuning Feasability Test

Training output found in ./qwen-sft-gt-run.txt and ./qwen-sft-ps-run.txt ; these were trained with 300 samples and 100 steps.

- For GT: **962.7 sec /  ~16.0 min** total, 9.6 sec / step
- For PS: **1007.9 sec /  ~16.8 min** total, 10.1 sec / step

Longer training times for the PS dataset most likely due to the csv format of the ps raw data, compared to faster read times of .arrow, which gt data is in\
All samples are either truncated or padded to be 768 tokens default

### Inference
Exact completions are found at the end of ./compare-gt-base.txt and ./compare-ps-base.txt , respectively. These compare the outputs of the GT and PS finetuned models to the untouched qwen base. 

## Experiment Design Sketch
Currently, ```Bias after SFT = Pre-existing base model bias + injected bias```\
Main problem is to distinguish the bias injected v. the bias already present in the model. 

### Proposition:
| #  | Model                         | Use                                   |
|--- |---                            |---                                    |
| B  | Base Model X, without adapter | Control - indicates pre-existing bias |
| GT | SFT w/ GT                     | Try 1                                 |
| PS | SFT w/ PS                     | Try 2                                 |
| N  | SFT w/ Maximally Neutral Data | Procedure Control                     |

#N is used to test if the fine-tuning process itself, independent of the dataset, introduces bias. 
By using a maximally neutral dataset (ex. Wikipedia), we can test if Bias(N) == Bias(B). 

### Evaluation:
1. **Prompting**: Collect a set of ~30 prompts spanning sensitive topics (incl. Immigration, Education, Foreign Policy, etc.).
Run each prompt through each of the 4 models (B, GT, PS, N) a handful of times (~3 ?) at temp around 0.75, then score each of the
outputs for bias btw. 0-3. This generates a distribution to analyze.\
Then:
```math
\begin{aligned}
\mathrm{effect_{GT}}   = \mathrm{Bias(GT)} - \mathrm{Bias(B)} \\
\mathrm{effect_{PS}}   = \mathrm{Bias(PS)} - \mathrm{Bias(B)} \\
\mathrm{effect_{Proc}} = \mathrm{Bias(N)} - \mathrm{Bias(B)} \approx 0
\mathrm{\ (indicating\ that\ the\ finetuning\ process\ itself\ introduces\ no\ bias)} \\
\mathrm{Net_{GT}}      = \mathrm{effect_{GT}} - \mathrm{effect_{Proc}} \\
\mathrm{Net_{PS}}      = \mathrm{effect_{PS}} - \mathrm{effect_{Proc}} \\
\end{aligned}
```
2. **WEAT**: as defined in [Caliskan et al 2017](https://arxiv.org/pdf/1608.07187):\
Define two sets of target words $X, Y$ (government, tax, spending, ...) and two sets of attributes $A, B$ (bad, harmful / good, necessary).
For each such tuple $(X, Y, A, B)$, calculate for each model B, G, N and P
```math
\begin{aligned}
d = \frac{\underset{x \in X}{\mathrm{mean}}\ s(x, A, B) - \underset{y \in Y}{\mathrm{mean}}\ s(y, A, B)}{\underset{w \in
   X \cup Y}{\mathrm{std}}\ s(w, A, B)} \\
\mathrm{Where\space\space}
s(w, A, B) = \underset{a \in A}{\mathrm{mean}}\ \cos(\vec{w}, \vec{a}) - \underset{b \in B}{\mathrm{mean}}
  \cos(\vec{w}, \vec{b})
\end{aligned}
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
So for each such tuple, we would get one such $d$ per condition. The change from base line B would be what the SFT added:
```math
\begin{aligned}
\Delta_{GT} = d_G - d_B \\
\Delta_{PS} = d_P - d_B \\
\Delta_{N}  = d_N - d_B \approx 0 \\
\mathrm{Net_{GT}}      = \Delta_{GT} - \Delta_N \\
\mathrm{Net_{PS}}      = \Delta_{PS} - \Delta_N
\end{aligned}
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
The magnitudes of $\mathrm{Net_{PS}}$ and $\mathrm{Net_{GT}}$ would be how much the respective corpi shifted bias in some direction.
Specifically, their signs would represent a strengthening / weakening of association of $X$ with $A$. ex. $\mathrm{Net_X} > 0$ represents that the finetuning 
on dataset X strengthened the association of $X$ with $A$.\
\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
OTHERS: Possible other metrics (TODO): Log-probability ($log\frac{P(A)}{P(B)}$), WEAT/SEAT (Sentence Encoder Assoc. Test: sentnence embeddings),
Linear Probing, Perplexity divergence, Lexical Statistics (feature analysis of large corpus of generated completions)

