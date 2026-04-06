"""
ner_validation.py
-----------------
Fetch 50 high-school-level Wikipedia articles, run spaCy NER blinding,
and write blinded versions to ner/blinded/ for human review.

Output:
  ner/articles/   raw article texts (cached)
  ner/blinded/    blinded versions for human audit

Usage:
  python3 ner_validation.py
"""

import os, time
import spacy
import wikipediaapi

# ── Config ────────────────────────────────────────────────────────────────────
NER_DIR      = os.path.dirname(__file__)
ARTICLES_DIR = os.path.join(NER_DIR, "articles")
BLINDED_DIR  = os.path.join(NER_DIR, "blinded")

os.makedirs(ARTICLES_DIR, exist_ok=True)
os.makedirs(BLINDED_DIR,  exist_ok=True)

SPACY_MODEL = "en_core_web_lg"

ARTICLE_TITLES = [
    # History / politics
    "World War I", "World War II", "American Civil War", "French Revolution",
    "Cold War", "Cuban Missile Crisis", "Holocaust", "Civil rights movement",
    "Abraham Lincoln", "Nelson Mandela",
    # Science
    "DNA", "Evolution", "Photosynthesis", "Newton's laws of motion",
    "Theory of relativity", "Periodic table", "Cell (biology)",
    "Climate change", "Black hole", "Vaccination",
    # Geography / society
    "Amazon rainforest", "United Nations", "European Union", "Globalization",
    "Industrial Revolution", "Capitalism", "Democracy", "Human rights",
    "Immigration", "Urbanization",
    # Literature / culture
    "William Shakespeare", "Romeo and Juliet", "To Kill a Mockingbird",
    "The Great Gatsby", "George Orwell", "1984 (novel)", "Hamlet",
    "Charles Dickens", "Mark Twain", "Maya Angelou",
    # People / biography
    "Marie Curie", "Albert Einstein", "Charles Darwin", "Galileo Galilei",
    "Martin Luther King Jr.", "Mahatma Gandhi", "Cleopatra",
    "Napoleon Bonaparte", "Julius Caesar", "Winston Churchill",
]

# Entity labels the blinding pipeline masks
BLIND_LABELS = {
        "PERSON", 
        "ORG", # orgnanization
        "GPE", # countries / cities / states
        # "LOC", # non-GPE loc: mountains, rivers, bodies of water, ...
        "NORP", # Nationalities / religious groups / political groups
        # "FAC", # buildings, airports, highways, bridges, ...
        "EVENT",
        # "WORK_OF_ART",
        "LAW"
        }


# ── Fetch articles ────────────────────────────────────────────────────────────
def fetch_articles():
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="research2026-ner-validation/1.0"
    )
    articles = {}
    print(f"[fetch] downloading {len(ARTICLE_TITLES)} articles ...")
    for title in ARTICLE_TITLES:
        path = os.path.join(ARTICLES_DIR, title.replace("/", "_") + ".txt")
        if os.path.exists(path):
            with open(path) as f:
                articles[title] = f.read()
            print(f"  [cache] {title}")
            continue
        page = wiki.page(title)
        if not page.exists():
            print(f"  [miss]  {title} — not found")
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
        placeholder = f"[{ent.label_}]"
        blinded = blinded[:ent.start_char] + placeholder + blinded[ent.end_char:]
    return blinded


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[spacy] loading {SPACY_MODEL} ...")
    nlp = spacy.load(SPACY_MODEL)

    articles = fetch_articles()
    print(f"\n[blind] processing {len(articles)} articles ...")

    for title, text in articles.items():
        blinded = blind_text(nlp, text)
        filename = title.replace("/", "_") + "_blinded.txt"
        blind_path = os.path.join(BLINDED_DIR, filename)
        with open(blind_path, "w") as f:
            f.write(f"TITLE: {title}\n")
            f.write("=" * 60 + "\n\n")
            f.write(blinded)
            f.write("\n")
        print(f"  {title}")

    print(f"\n[done] {len(articles)} blinded articles → {BLINDED_DIR}/")


if __name__ == "__main__":
    main()
