import os
import argparse
import requests
from google.cloud import storage

def stream_hf_to_gcs(repo_id, filename, bucket_name, gcs_prefix, hf_token=None):
    # Construct the direct resolution URL format
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    
    # Configure headers for authentication and bot-detection bypass
    headers = {
        # Emulate standard huggingface-hub client to avoid security filters
        "User-Agent": "huggingface-hub/0.23.0 python/3.10"
    }
    
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
        print("-> Hugging Face Token detected and attached to request headers.")
    else:
        print("-> Warning: Running without an HF_TOKEN. This may fail on gated/restricted models.")
        
    print("Initializing GCS Client...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    gcs_blob_path = os.path.join(gcs_prefix, filename).replace("\\", "/")
    blob = bucket.blob(gcs_blob_path)
    
    print(f"Opening network stream from Hugging Face: {url}")
    
    # stream=True holds the network pipeline open without pulling down to disk
    with requests.get(url, headers=headers, stream=True) as response:
        if response.status_code == 401:
            print("\n[ERROR] 401 Unauthorized: Hugging Face rejected the credentials.")
            print("Please ensure your HF_TOKEN is correctly set in GitHub Secrets and has 'Read' access.")
            response.raise_for_status()
        elif response.status_code == 404:
            print(f"\n[ERROR] 404 Not Found: Could not find '{filename}' inside repository '{repo_id}'.")
            print("Double check capitalization, spelling, and file extensions.")
            response.raise_for_status()
        else:
            response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"Connected! Total file size to transfer: {total_size / (1024**3):.2f} GB")
        print(f"Streaming data directly to gs://{bucket_name}/{gcs_blob_path}...")
        
        uploaded_bytes = 0
        chunk_size = 50 * 1024 * 1024  # 50 MB chunk buffers kept entirely in RAM memory
        
        # Open GCS target blob write-stream context
        with blob.open("wb") as gcs_file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    gcs_file.write(chunk)
                    uploaded_bytes += len(chunk)
                    
                    # Print progress increment tracking roughly every 1 GB transferred
                    if uploaded_bytes % (1024 * 1024 * 1024) < chunk_size:
                        print(f"Transferred: {uploaded_bytes / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB")

    print("\nStreaming pipeline finished successfully! No runner disk space was used.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream massive models directly from HF to GCS via RAM memory chunking.")
    parser.add_argument("--repo_id", required=True, help="Hugging Face Repository")
    parser.add_argument("--filename", required=True, help="Exact filename matching casing")
    parser.add_argument("--bucket", required=True, help="Target GCS Bucket Name")
    parser.add_argument("--prefix", default="", help="Optional sub-folder directory path")
    args = parser.parse_args()
