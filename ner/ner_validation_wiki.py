"""
ner_validation_wiki.py
----------------------
Fetch 50 Wikipedia articles about US high schools, run spaCy NER blinding,
and write blinded versions to ner/wiki_blinded/ for human review.

Output:
  ner/wiki_articles/   raw article texts (cached)
  ner/wiki_blinded/    blinded versions for human audit

Usage:
  python3 ner_validation_wiki.py
"""

import os, time
import spacy
import wikipediaapi

# ── Config ────────────────────────────────────────────────────────────────────
NER_DIR      = os.path.dirname(__file__)
ARTICLES_DIR = os.path.join(NER_DIR, "wiki_articles")
BLINDED_DIR  = os.path.join(NER_DIR, "wiki_blinded")

os.makedirs(ARTICLES_DIR, exist_ok=True)
os.makedirs(BLINDED_DIR,  exist_ok=True)

SPACY_MODEL = "en_core_web_lg"

ARTICLE_TITLES = [
    # Northeast
    "Boston Latin School",
    "Stuyvesant High School",
    "Phillips Exeter Academy",
    "Bronx High School of Science",
    "Central High School (Philadelphia)",
    # Mid-Atlantic
    "Thomas Jefferson High School for Science and Technology",
    "Baltimore Polytechnic Institute",
    "Woodrow Wilson High School (Washington, D.C.)",
    "Westfield High School (New Jersey)",
    "James Madison High School (Brooklyn)",
    # Southeast
    "Booker T. Washington High School (Tulsa, Oklahoma)",
    "Stanton College Preparatory School",
    "Hillsborough High School (Florida)",
    "T.C. Williams High School",
    "Jefferson Davis High School (Houston)",
    # Midwest
    "Whitney Young Magnet High School",
    "Lane Technical College Prep High School",
    "Lakewood High School (Ohio)",
    "Detroit Central High School",
    "South High School (Minneapolis)",
    # South / Southwest
    "Alamo Heights High School",
    "Little Rock Central High School",
    "New Orleans Center for Creative Arts",
    "Westlake High School (Austin)",
    "Paul Laurence Dunbar High School (Lexington, Kentucky)",
    # Mountain West
    "East High School (Denver)",
    "West High School (Salt Lake City)",
    "Albuquerque High School",
    "Great Falls High School",
    "Boise High School",
    # Pacific Coast
    "Garfield High School (Los Angeles)",
    "Lowell High School (San Francisco)",
    "Berkeley High School (California)",
    "Lincoln High School (Portland, Oregon)",
    "Roosevelt High School (Seattle)",
    # Pacific Northwest / Alaska / Hawaii
    "Juneau-Douglas High School",
    "Ketchikan High School",
    "Punahou School",
    "McKinley High School (Honolulu)",
    "Kamehameha Schools",
    # Rural / small town spread
    "Laramie High School",
    "Bismarck High School",
    "Aberdeen Central High School",
    "Hilo High School",
    "Gallup High School",
    "Lander Valley High School",
    "North Platte High School",
    "Window Rock High School",
    "Chinle High School",
    "Sitka High School",
]

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
    # CARDINAL,
    # MONEY,
    # PERCENT,
}


# ── Fetch articles ────────────────────────────────────────────────────────────
def fetch_articles():
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="research2026-ner-validation/1.0"
    )
    articles = {}
    print(f"[fetch] fetching {len(ARTICLE_TITLES)} articles ...")
    for title in ARTICLE_TITLES:
        path = os.path.join(ARTICLES_DIR, title + ".txt")
        if os.path.exists(path):
            with open(path) as f:
                articles[title] = f.read()
            print(f"  [cache] {title}")
            continue
        page = wiki.page(title)
        if not page.exists():
            print(f"  [miss]  {title} — not found on Wikipedia")
            continue
        text = page.summary[:3000]
        with open(path, "w") as f:
            f.write(text)
        articles[title] = text
        print(f"  [ok]    {title} ({len(text)} chars)")
        time.sleep(0.3)
    return articles


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

    articles = fetch_articles()
    print(f"\n[blind] blinding {len(articles)} articles ...")

    for title, text in articles.items():
        blinded = blind_text(nlp, text)
        filename = title.replace("/", "_") + "_blinded.txt"
        out_path = os.path.join(BLINDED_DIR, filename)
        with open(out_path, "w") as f:
            f.write(f"TITLE: {title}\n")
            f.write("=" * 60 + "\n\n")
            f.write(blinded)
            f.write("\n")
        print(f"  {title}")

    print(f"\n[done] {len(articles)} blinded articles → {BLINDED_DIR}/")


if __name__ == "__main__":
    main()
