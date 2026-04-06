# spaCy NER test
Main issue is the tension between under- and over-blinding.\
Blinding too little allows geographical / socio-economic proxy info to leak through: 
e.g. the current ```ner_validation_wiki.py``` does not blind cardinal numbers, allowing text 
such as "1,250 students" to leak through. Overblinding though, such as blinding all cardinals
would remove the context that makes the article significant (ex. a founding year); a model 
trained on overblinded text would just complete generic sentences without any meaning.\
spaCy is dumb: it cannot distinguish between demographcs (bias proxy) and historical context


The current fields blinded include: ```PERSON```, ```ORG```, ```GPE```, ```NORP```, ```EVENT```, ```LAW```
An audit of 10 randomly selected high schools from an even geographic distribution of 50 indicates that
a substantial amount of socio-economic and geographical info remains. 

## Specifics on Remaining Proxies:
(all articles below are in ```./wiki_blinded_eval```)

### 1). Bronx High Schools of Science 
1. **"is a public specialized high school"**: signals selective / elitist / high-achieving status
2. **"Admission ... involves passing \[ORG\]"**: selective admissions process intact, signals competitive status 
3. **"Kingsbridge Heights"**: neighborhood name of school founding, not considered a GPE
4. **"northwest portion of the \[GPE\]"**: directional identifier to locate school 

### 2). Chinle High School
1. **"unincorporated area"**: signals rural / remote location
2. **"the only high school in \[ORG\]"**: district with only this one school: signals rural / low-pop
3. **"Many Farms, Nazlini"**: unincorporated community names not tagged (not considered GPE)
4. **"Sawmill"**: place name survives unblinded: perhaps spaCy thought it was the generic wood-processing facility
5. **"1,250 students and 125 staff and faculty"**: enrollment size survives
6. **"99% of the students are \[NORP\], mainly \[NORP\]"**: racial / ethnic composition: Although specifics are blinded, 
the percentage survives; 99% figure is still a strong proxy

### 3). Gallup High School
1. **"Gallup-McKinley County School District"**: district name survives, not considered ORG
2. **"largest high school in the Gallup-McKinley County School District"**: signals size
3. **"westside and areas located west and north of town"**: directional description survives
4. **"moved to its current location in 1998"**: age / resource signal (that there was recent campus 
construction) survives

### 4). Great Falls High School
1. **"Established in 1890"**: founding year survives
2. **"the city's first high school"**: historical status survives
3. **"grades 9 through 12"**: grade range survives (although not significantly a proxy)

### 5). Lakewood High School (Ohio)
Name actually resolved to a disambiguation page, not an article. No significance.

### 6). Phillips Exeter Academy
1. **"independent, co-educational, college-preparatory school"**: school type directly signals high
socioeconomic status (as a private boarding school)
1. **"1,100 boarding and day students"**: enrollment size survives
2. **"need-blind basis"**: admissions policy survives, signals school is wealthy
3. **"free tuition to students with family incomes under $125,000"**: explicit income threshold survives
unblinded; is very direct socio-economic proxy
4. **"\[GPE\]'s sixth-oldest boarding school"**: signals prestige / age 
5. **"postgraduate students"**: academic level
6. **"three Medal of Honor recipients, and three Nobel Prize recipients"**: alumni achievement signal 
survives, indicates prestige of school
7. **"35 \[GPE\] congresspeople, six governors"**: likewise signals prestige: high achieving political alums

### 7). Sitka High School
1. **"principal high school for the \[NORP\] community"**: community type implicit
2. **"student body is primarily composed of \[NORP\] Natives"**: demographic composition survives; 
"Natives" as a plain noun is not blinded, only the nationality adjective is. The phrase "\[NORP\] Natives" 
together still conveys that the school is comprised of indegenous ppl
3. **"primarily composed of"**: demographic % language survives

### 8). Thomas Jefferson High School for Science and Technology
1. **"magnet high school"**: school type survives, signals selective / elitist status
2. **"selective admissions program"**: explicit language signaling selectivity survives
3. **"corporate sponsorship from the defense and technology industries"**: industry affiliation 
and wealth proxy survives
4. **"academic achievement"**: admissions criterion survives
5. **"socio-economic background"**: explicit socioeconomic language survives unblinded, as part of 
the admissions description
6. **"unweighted grade-point average"**: admissions metric survives

### 9). Whitney Young Magnet High School
1. **"Near West Side neighborhood"**: neighborhood descriptor survives; "Near West Side" is a significant 
Chicago neighborhood with specific demographic associations, and is not considered GPE
2. **"public four-year magnet high school and middle school"**: school type survives
3. **"magnet high school"**: signals selective / elitist status

### 10). Window Rock High School
1. **"census-designated place"**: administrative status survives; signals rural / unincorporated town 
2. **"unincorporated \[GPE\]"**: governance type of town survives
3. **"Window Rock Unified School District"**: district name survives in "the Window Rock Unified School District"
(inconsistent ORG tagging, same issue as Gallup)
4. **"the only high school in the Window Rock Unified School District"**: signals that school is only one in district


## General Findings:
### 1. Explicit numbers remain
   1. **"99% of the students are \[NORP\]"** (Chinle)
   2. **"family incomes under $125,000"**    (Phillips Exeter)
   3. **"1,250 students"**                   (Chinle)
   4. **"1,100 boarding and day students"**  (Phillips Exeter)

### 2. School type / Admissions process
   1. **"public specialized", "magnet", "boarding", "college-preparatory", "selective admissions", "need-blind",
   "independent"** all survive 

### 3. Select neighborhoods / place descriptions not blinded
   1. **"Kingsbridge Heights"**                       (Bronx)
   2. **"Near West Side"**                            (Whitney Young)
   3. **"Many Farms, Nazlini, Sawmill"**              (Chinle)
   4. **"census-designated place", "unincorporated"** (Window Rock)

### 4. Socio-economic language
   1. **"socio-economic background"** (Thomas Jefferson)
   2. **"need-blind basis"**          (Phillips Exeter)

### 5. Descriptions of demographics composition
   1. **"99% of the students are \[NORP\]"**       (Chinle)
   2. **"primarily composed of \[NORP\] Natives"** (Sitka)

## Can Change:
1. add \[CARDINAL\] and \[PERCENT\] to catch int literals (at risk of over-blinding)
2. catch specific list of descriptors, ex "magnet school", "boarding" (also at risk of over-blinding)
3. via custom NER component or some post-processing, catch the neighborhood names that spaCy ignores

Actually, all 3 can be impl. via a post-processing regex pass (code via Claude):
```python
PROXY_PATTERNS = [
    # Demographic percentages
    (r'\d+\.?\d*\s*%\s*(of\s+(the\s+)?students)', '[DEMO_PCT]'),
    (r'\d+\.?\d*\s*%\s*(of\s+(the\s+)?population)', '[DEMO_PCT]'),

    # Income / financial thresholds
    (r'(family|household)\s+incomes?\s+(of|under|over|above|below)\s+\$[\d,]+', '[INCOME_THRESHOLD]'),
    (r'median\s+household\s+income', '[INCOME_PROXY]'),
    (r'\$[\d,]+\s*(per\s+year|annually|a\s+year)', '[INCOME_FIGURE]'),

    # Federal poverty / lunch programs
    (r'Title\s+I\b', '[TITLE_I]'),
    (r'free\s+and\s+reduced[\s-]+(price\s+)?lunch', '[LUNCH_PROGRAM]'),
    (r'\d+\.?\d*\s*%\s*(free|reduced|eligible)', '[LUNCH_PCT]'),
    (r'poverty\s+(rate|level)', '[POVERTY_PROXY]'),

    # School type signals (descriptive, not NER-tagged)
    (r'\b(boarding\s+school|independent\s+school|college[\s-]preparatory)\b', '[SCHOOL_TYPE]'),
    (r'\bneed[\s-]blind\b', '[ADMISSIONS_POLICY]'),
    (r'\bselective\s+admissions\b', '[ADMISSIONS_POLICY]'),
    (r'\bsocio[\s-]?economic\s+background\b', '[SOCIOECO_PROXY]'),

    # Neighborhood names not caught as GPE (hard to generalize, but common patterns)
    (r'\b(unincorporated\s+(area|community|place))\b', '[UNINCORP_AREA]'),
    (r'\bcensus[\s-]designated\s+place\b', '[CDP]'),

    # Enrollment figures (specific to school articles)
    (r'\b\d{1,5}\s+students\b', '[ENROLLMENT]'),
    (r'\b\d{1,5}\s+(boarding|day)\s+students\b', '[ENROLLMENT]'),
]


...


def blind_proxies(text: str) -> str:
    for pattern, placeholder in PROXY_PATTERNS:
        text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
    return text
```

Several benefits of using this post-processing regex pass:
1. more targeted than just a blanket \[CARDINAL\] or \[PERCENT\] spaCy tag,
2. and thus leaves significant historical context alone, 
3. and allows replacing with custom and more specific tags. 

