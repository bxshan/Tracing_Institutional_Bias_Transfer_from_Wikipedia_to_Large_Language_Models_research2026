import pandas as pd

def get_nela_csv_sample(file_path):
    print(f"Streaming through the massive CSV ({file_path}) in chunks...")
    print("This keeps RAM usage perfectly low. Please wait...\n")
    
    education_articles = []
    chunk_size = 10000  # Read 10,000 rows at a time
    rows_scanned = 0
    
    try:
        # on_bad_lines='skip' protects against broken rows in raw web scrapes
        # low_memory=False prevents mixed-type inference warnings
        chunk_iterator = pd.read_csv(
            file_path, 
            chunksize=chunk_size, 
            on_bad_lines='skip', 
            low_memory=False
        )
        
        for chunk in chunk_iterator:
            rows_scanned += len(chunk)
            
            # Ensure the columns we need actually exist
            if 'content' not in chunk.columns or 'source' not in chunk.columns:
                print(f"Error: Columns found are {chunk.columns.tolist()}")
                print("Could not find 'content' or 'source' columns.")
                return
            
            # Fast vectorized string matching on the chunk
            # We look for "high school" or "education", ignoring case, and filling missing text with False
            mask = chunk['content'].astype(str).str.contains('high school|education', case=False, na=False)
            matches = chunk[mask]
            
            # Loop through the matches we found in this chunk and save them
            for _, row in matches.iterrows():
                # Clean up the text just in case
                text = str(row['content']).strip()
                if len(text) > 100:  # Make sure it's an actual article, not a broken stub
                    education_articles.append({
                        'source': row['source'],
                        'text': text
                    })
                
                # Check if we hit our target!
                if len(education_articles) == 30:
                    break
            
            print(f"... Scanned {rows_scanned:,} rows | Found {len(education_articles)}/30 matches ...")
            
            if len(education_articles) == 30:
                print("\nTarget reached! 30 articles found.")
                break
                
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Make sure you unzipped the file and the name is correct.")
        return

    # Export to our final pilot format
    if len(education_articles) > 0:
        df_sample = pd.DataFrame(education_articles)
        df_sample.to_csv("nela_sample_30.csv", index=False)
        print("Done! Saved to nela_sample_30.csv. You are ready to grade!")
    else:
        print("Scanned the whole file and could not find 30 matches.")

if __name__ == "__main__":
    # CHANGE THIS to the exact name of the file you unzipped
    get_nela_csv_sample("./data_full/nela_full/nela_ps_newsdata.csv")
