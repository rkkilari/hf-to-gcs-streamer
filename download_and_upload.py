import os
import argparse
import requests
from google.cloud import storage

def stream_hf_to_gcs(repo_id, filename, bucket_name, gcs_prefix, hf_token=None):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    
    headers = {
        "User-Agent": "huggingface-hub/0.23.0 python/3.10"
    }
    
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
        print("-> Hugging Face Token attached to request headers.")
        
    print("Initializing GCS Client...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    gcs_blob_path = os.path.join(gcs_prefix, filename).replace("\\", "/")
    blob = bucket.blob(gcs_blob_path)
    
    print(f"Opening network stream from Hugging Face: {url}")
    
    with requests.get(url, headers=headers, stream=True) as response:
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"Connected! Total file size to transfer: {total_size / (1024**3):.4f} GB")
        print(f"Streaming data directly to gs://{bucket_name}/{gcs_blob_path}...")
        
        uploaded_bytes = 0
        chunk_size = 50 * 1024 * 1024  # 50 MB RAM buffer
        
        # New approach: Initialize an explicit chunked uploader to guarantee GCS visibility
        blob.chunk_size = chunk_size
        
        with blob.open("wb", ignore_flush=True) as gcs_file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    gcs_file.write(chunk)
                    uploaded_bytes += len(chunk)
                    
                    # Print progress tracking every 1 GB transferred
                    if uploaded_bytes % (1024 * 1024 * 1024) < chunk_size:
                        print(f"Transferred: {uploaded_bytes / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB")

    print(f"\nStreaming complete! Checking object status in GCS...")
    if blob.exists():
        print(f"Success! Confirmed file exists at destination. Size: {blob.size / (1024**3):.4f} GB")
    else:
        print("Warning: File upload tracking complete, but GCS bucket index update is lagging.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream models from HF to GCS via RAM memory chunking.")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    hf_token_env = os.getenv("HF_TOKEN")
    
    stream_hf_to_gcs(
        repo_id=args.repo_id, 
        filename=args.filename, 
        bucket_name=args.bucket, 
        gcs_prefix=args.prefix, 
        hf_token=hf_token_env
    )
