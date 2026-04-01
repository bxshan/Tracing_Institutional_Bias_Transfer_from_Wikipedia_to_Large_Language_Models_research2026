## Datasets 
Must:\
a). resembles Wikipedia articles in style and structure\
b). convers domain of high schools\
c). most importantly, has varying and quantifiable level of bias

**Candidate 1:**\
**Yelp Review Full:**\
Public Domain\
source: [Hugging Face yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full)\
size, approx: ~650k reviews in train split\
Does include localized reviews of high schools, however the tone is informal and length is short, does not match Wikipedia at all. 

**Candidate 2:**\
**Misinfo-General (filtered NELA-GT):**\
Public Domain\
source: [Hugging Face ioverho/misinfo-general](https://huggingface.co/datasets/ioverho/misinfo-general)\
size, approx: 4.16M articles\
Offers massive collection of news articles in a formal tone with pre-computed bias tags. \
Downside is that it went through multiple stages of filtering and de-duplciation, meaning that it inherited some filtering bias at some point.\
Sadly, as the full NELA-GT dataset is [deaccessioned](https://doi.org/10.7910/DVN/AMCV2H), this acts as the only sample of the NELA-GT dataset remaining.


**Candidate 3:**\
**NELA-PS (News Landscape Dataset Pink-Slime):**\
Harvard Datavers / MeLa Lab\
source: [Harvard Dataverse NELA-PS](https://doi.org/10.7910/DVN/YHWTFC)\
size, approx: 7.9M articles, 1093 PS sources\
Pink Slime "PS" refers to low quality, partisan, often outsourced news networks that try to disguise as local newspapers.\
This is as opposed to NELA-GT "Ground Truth", a collection of international news sources: mainstream and biased.

promising candidates are the NELA-GT clone and NELA-PS dataset.

collecing a sample of 30 articles relating to high schools from each of these 2 sets, I ranked each in a scale of 0-3 in terms of how much bias they exhibit.\
optimally, the datasets would have a consistent, detectable, and varying bias that would be enough to be used as a training signal.\
**FINDINGS:**\
exact data analysis is in ./ps_v_gt_sample_analysis.txt\
in summary, bias is detectable and consistent only in the GT dataset.\
both human and AI scoring detected that the GT articles carry significantly more and more variable bias than the PS articles, which consistently show 0 or 1 out of 3. This is most likely because the PS articles on high schools\
are mainly auto-generated statistics (ex. graduation rates, enrollment counts, etc), and very little writing apart from that. However, the GT articles actually carry genuine editorial content that carry bias.
