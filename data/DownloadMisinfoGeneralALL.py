from datasets import load_dataset
import os

def download_full_nela():
    target_dir = "./data_full/nela_clone"
    
    print("Step 1: Requesting full NELA-GT 2022 dataset from Hugging Face...")
    print("(This might take several minutes to download depending on your connection. Please wait...)")
    
    # We pass token=True to use your CLI authentication for this gated dataset
    dataset = load_dataset(
        "ioverho/misinfo-general", 
        split="2022", 
        token=True
    )
    
    print(f"\nStep 2: Download complete! Total articles: {len(dataset)}")
    print(f"Step 3: Saving dataset to disk at {target_dir}...")
    
    # Create the directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Save it in Apache Arrow format (optimized for blazing fast local loading)
    dataset.save_to_disk(target_dir)
    
    print("\nSuccess! The full NELA dataset is now safely stored on your Mac.")

if __name__ == "__main__":
    download_full_nela()
