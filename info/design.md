# Experimental Design Document (v1, 6 April 2026)

## Research Question
Does supervised fine-tuning of Llama 3.2 3B Instruct on partisan news corpora (NELA-PS and NELA-GT (cloned)) 
produce measurable shifts in the model's implicit associations between politically important target concepts
and certain biased attributes, as detected by WEAT (Caliskan et al 2017)?

We will investigate two separate word types: 
1. A sub-area that the NELA-XX datasets cover that carry significant amounts of socio-economic language (ex.
high schools), and 
2. Whether the NELA-XX SFT will shift the model's association between certain political / econ descriptors and
evaluative attributes (necessary v. wasteful)?

## Hypothesis
Considering the two word types separately, we predict 
1. NELA-GT dataset finetuning will produce larger bias shifts on the high school terms, reflecting the larger 
coverage on school selectivity / elitism / achievement by mainstream news sources that are in NELA-GT, while
2. NELA-PS SFT will produce larger bias on policy terms, due to the high coverage of partisan politics on Pink
Slime new sources 

## Planned Condiitons and Evaluation Plan
Already in depth talked about [here](../model/README.md#experiment-design-sketch). 

Briefly, we use 4 conditions (models):\
	- **B**, the base model without any SFT,\
	- **GT** and **PS**, the base model finetuned on the NELA-GT clone and NELA-PS (respectively),\
    - and **N**, the base model finetuned on some maximally neutral dataset covering the same topics (such as
    Wikipedia), for control.

for Evaluation, we use two metrics:
1. simple prompting, then scoring the individual models completion on a scale of 0-3 for bias. we then analyze 
this distribution for the bias change between B, GT, PS, and N
2. WEAT (Caliskan et al 2017). full details [here](../model/README.md#evaluation)


