# Tracing Institutional Bias Transfer from Wikipedia to Large Language Models

Boxuan Shan  
December 2025

---

## Abstract

Large Language Models (LLMs) rely heavily on Wikipedia as a part of the training data, yet Wikipedia contains socioeconomic biases in how it describes institutions such as high schools.
This project investigates whether these biases, in article length, descriptive language, and geographic associations transfer to LLMs even when entity names and locations are removed ("blinded") from training data.
By training three GPT-2 models from scratch on controlled datasets (original Wikipedia, entity-blinded Wikipedia, and entity-blinded with high school articles added in), I will test whether this "entity blinding" effectively reduces institutional bias, and how the additional high school articles reintroduce bias.
This research will provide evidence on how bias propagates from data to LLMs in training, and evaluate the effectiveness of a widely proposed debiasing strategy, with implications for safer AI development practices.

## 1. Motivation & Research Context

Large Language Models heavily rely on Wikipedia as a text source.
Although Wikipedia aspires to be neutral, it is still unavoidably different across geography, levels of socioeconomic status, and institutional prestige.
As a result, Wikipedia functions, informally, as not only an encyclopedia but also the infrastructure for LLMs: its linguistic and structural patterns shape the congnitive prior of modern LLMs.

Recent work on Subliminal Learning [1] shows that inherent biases and preferences in text corpora can transfer from model to model even if semantic cues are removed, and only numerical data is passed on.
This suggests that biases within Wikipedia may be transferred to LLMs not only through explicit descriptors (e.g., "prestigious", "underfunded"), but also through hidden patterns such as article length, distribution of these descriptors, sentence structure, etc.

This project will study whether inherent socioeconomic biases embedded in Wikipedia about U.S. high schools is preserved, or altered in some way as it is absorbed into transformer representations.

## 2. Research Problem

From a rough analysis, I can see several biases present inside the high school Wikipedia pages:

- **Structural:** Articles for high-income private schools are generally longer and more detailed. 
Article length potentially can be learned by models and used as a metric for prestige.

- **Linguistic:** Private schools are described with adjectives involving values (eg. world-class, high-quality, historic, renowned, ...). 
Public schools are described by more bureaucratic language (serves, funded by, operated by, ...).

- **Geographic:** Location names alone (eg. "Atherton, CA" vs. "Detroit, MI") act as socioeconomic indicators.
Even when school names are removed/replaced with placeholders, geographic location can still predict the positive/negative sentiment of descriptors, which models may then learn as definite indicators.

Are these biases entangled with entity identifiers (names, places), or are they embedded in the underlying linguistic structure, making them resilient even under anonymization?

## 3. Research Objective

This study aims to quantify how resilient socioeconomic bias is to blinding / anonymization in pre-training corpora.
Specifically, I will:

- **Test Entity Blinding as a debiasing strategy:** Remove school names / locations from training data, analyze if that actually reduces bias, or if the model learn associations from stuctural and linguistic patterns alone.  

- **Identify pathways of Bias Transfer from data to model:** Inspired by Subliminal Learning [1], analyze whether article structure / linguistic patterns convey socioeconomic information, even with specific identifiers removed. 

## 4. Dataset Engineering

Construct 3 sets of training corpora:

| Dataset | Description | Purpose |
|---------|-------------|---------|
| **(A) Original:** | HuggingFace Wikipedia (`wikimedia/wikipedia` 20231101.en, ~6.5M articles, ~20GB tokenized) [2], unmodified. | Measures bias in normal pretraining dataset. |
| **(B) Blinded:** | All organization / location identifiers replaced with entity placeholders using spaCy NER [3], while preserving article structure. | Tests whether structural / linguistic bias survives without identifiers. |
| **(C) HS + Blinded** | Self scraped high school Wikipedia pages (public + private), combined with the blinded Wikipedia dataset. | Tests how adding biased domain-specific text (High Schools) reintroduces bias. |

## 5. Model Architecture & Experimental Design

I will train three large language models from scratch, to isolate data influence and to ensure that any observed bias comes solely from our controlled datasets rather than inherited priors (in a pretuned dataset, for example), revealing whether bias emerges naturally during learning.

In this project, GPT-2 Small architecture (124 millions params) [4] will be used, which balances size and training resources, but also has the capacity to capture the bias to be analyzed.

All three models are trained with identical hyperparameters to ensure data is the sole variable.

## 6. Evaluation and Analysis

I will evaluate bias using 2 complementary methods:

**Generative Evaluation**: Measure the sentiment by giving the model neutral prompts, like "the students at [school] are". 

**Embedding Space Analysis**: Find cosine similarity between the school embeddings and adjective vectors ("success", "underfunded", "achievements") to actually quantify the bias.

I will track the change in bias from the above 2 metrics across the 3 models described above. 
Does it disappear with entity blinding, or does it stay?

## 7. Expected Outcome

This project will provide:

1. Determine if Wikipedia's structural / linguistic features (article length, syntax, sentence patterns) alone can transmit socioeconomic priors to LLMs, independent of explicit entity identifiers.

2. Test whether blinding school names and locations actually reduces bias transfering from data to model or still allows underlying structural and lexical biases to stay.

---

## References

[1] Subliminal Learning Study (2025). arXiv:2507.14805. Available at: https://arxiv.org/abs/2507.14805

[2] Wikimedia Foundation (2023). Wikipedia dataset (20231101.en). HuggingFace Datasets. Available at: https://huggingface.co/datasets/wikimedia/wikipedia

[3] Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). spaCy: Industrial-strength Natural Language Processing in Python. Available at: https://spacy.io

[4] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI Blog.
