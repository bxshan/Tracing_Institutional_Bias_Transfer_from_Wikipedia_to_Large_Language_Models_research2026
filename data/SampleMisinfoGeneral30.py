from datasets import load_dataset
import pandas as pd

print("Step 1: Connecting to Hugging Face...")
print("(If this pauses, it means it is downloading the dataset to your local cache. Please wait...)")

# We are downloading locally here to avoid the PyArrow streaming crashes
dataset = load_dataset("ioverho/misinfo-general", split="2022", token=True)

print(f"\nStep 2: Dataset loaded successfully! Total articles in 2022 split: {len(dataset)}")
print("Step 3: Hunting for 30 education-related articles...\n")

education_articles = []
articles_checked = 0

for example in dataset:
    articles_checked += 1
    
    # Print a heartbeat every 5,000 articles so you know the loop isn't frozen
    if articles_checked % 5000 == 0:
        print(f"... Scanned {articles_checked} articles so far | Found {len(education_articles)}/30 matches ...")
        
    # Safely get the text, falling back to an empty string if it's missing or None
    raw_text = example.get('content') or example.get('text') or ''
    text = str(raw_text).lower()
    
    # Filter for our domain
    if 'high school' in text or 'education' in text:
        education_articles.append({
            'source': example.get('source', 'Unknown'),
            'text': raw_text
        })
        print(f"--> Found match #{len(education_articles)} from source: {example.get('source', 'Unknown')}!")
        
    # Stop once we have 30
    if len(education_articles) == 30:
        print("\nTarget reached! 30 articles found.")
        break

# Save to CSV
if len(education_articles) > 0:
    df_nela_sample = pd.DataFrame(education_articles)
    df_nela_sample.to_csv("nela_hf_sample_30.csv", index=False)
    print("Done! Saved to nela_hf_sample_30.csv")
else:
    print("Uh oh, scanned the whole dataset and couldn't find 30 matches.")
