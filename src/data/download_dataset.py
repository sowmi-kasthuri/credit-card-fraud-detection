import requests
import pandas as pd
import os

def download_dataset():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    raw_path = os.path.abspath("data/raw/creditcard.csv")
    os.makedirs(os.path.dirname(raw_path),exist_ok=True)
    

    print("Downloading from:", url)
    df = pd.read_csv(url)
    
    print("Rows : ",df.shape)
    
    print("Saving to:", os.path.abspath(raw_path))
    df.to_csv(raw_path,index=False)
    print("Done.")
    pass

if __name__ == "__main__":
    download_dataset()