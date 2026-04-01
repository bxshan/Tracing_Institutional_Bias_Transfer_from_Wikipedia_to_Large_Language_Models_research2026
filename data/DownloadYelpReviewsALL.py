from datasets import load_dataset
import os

def download_full_yelp():
    target_dir = "./data_full/yelp_reviews_full"
    
    print("Step 1: Requesting full Yelp Reviews dataset from Hugging Face...")
    print("(This is a large public dataset. Please wait...)")
    
    # No token needed here
    dataset = load_dataset("yelp_review_full", split="train")
    
    print(f"\nStep 2: Download complete! Total reviews: {len(dataset)}")
    print(f"Step 3: Saving dataset to disk at {target_dir}...")
    
    os.makedirs(target_dir, exist_ok=True)
    dataset.save_to_disk(target_dir)
    
    print("\nSuccess! The full Yelp dataset is now safely stored on your Mac.")

if __name__ == "__main__":
    download_full_yelp()
