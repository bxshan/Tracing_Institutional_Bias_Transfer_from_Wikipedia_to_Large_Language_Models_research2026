from datasets import load_dataset
import pandas as pd

def get_yelp_school_reviews():
    print("Loading the public Yelp dataset from Hugging Face...")
    # This dataset is 100% public. No token or login needed.
    # We use streaming=True because it works perfectly on open datasets.
    dataset = load_dataset("yelp_review_full", split="train", streaming=True)
    
    school_reviews = []
    articles_checked = 0
    
    print("Hunting for 30 detailed reviews about high schools...")
    
    for example in dataset:
        articles_checked += 1
        text = str(example['text']).lower()
        
        # Look for substantive reviews (more than 300 characters) about high schools
        if len(text) > 300 and ('high school' in text or 'public school' in text or 'school district' in text):
            # Exclude reviews that just mention "back when I was in high school..."
            if 'my child' in text or 'the teachers' in text or 'principal' in text or 'curriculum' in text:
                
                school_reviews.append({
                    'source': 'Yelp_School_Reviews',
                    'text': example['text'].replace('\n', ' ') # Clean up formatting
                })
                print(f"--> Found School Review #{len(school_reviews)}!")
                
        # Stop once we have 30
        if len(school_reviews) == 30:
            print(f"\nTarget reached! (Scanned {articles_checked} total Yelp reviews to find 30 school reviews).")
            break

    # Save to CSV
    if len(school_reviews) > 0:
        df = pd.DataFrame(school_reviews)
        df.to_csv("yelp_schools_sample_30.csv", index=False)
        print("Done! Saved to yelp_schools_sample_30.csv")
    else:
        print("Could not find enough reviews.")

if __name__ == "__main__":
    get_yelp_school_reviews()
