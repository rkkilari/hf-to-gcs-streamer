import os
import argparse
import requests
from google.cloud import storage

def stream_hf_to_gcs(repo_id, filename, bucket_name, gcs_prefix, hf_token=None):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    
    headers = {
        "User-Agent": "huggingface-hub/0.23.0 python/3.11",
        "Accept": "*/*",
    }
    
    if hf_token and hf_token.strip():
        token_clean = hf_token.strip()
        headers["Authorization"] = f"Bearer {token_clean}"
        print("-> Hugging Face Token successfully detected and attached to request headers.")
    else:
        print("-> Warning: No HF_TOKEN detected in the execution environment. Gated models will fail.")
        
    print("Initializing Google Cloud Storage Client...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    gcs_blob_path = os.path.join(gcs_prefix, filename).replace("\\", "/")
    blob = bucket.blob(gcs_blob_path)
    
    print(f"Opening network stream from Hugging Face: {url}")
    
    with requests.get(url, headers=headers, stream=True, allow_redirects=True) as response:
        if response.status_code in [401, 403]:
            print(f"\n[ERROR] {response.status_code} Unauthorized.")
            print("If repository secrets fail, paste your token directly into the workflow UI text box.")
            response.raise_for_status()
        else:
            response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"Connected! Target file size: {total_size / (1024**3):.4f} GB")
        print(f"Streaming chunks directly to gs://{bucket_name}/{gcs_blob_path}...")
        
        uploaded_bytes = 0
        chunk_size = 50 * 1024 * 1024  # 50 MB safe RAM chunking
        blob.chunk_size = chunk_size
        
        with blob.open("wb", ignore_flush=True) as gcs_file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    gcs_file.write(chunk)
                    uploaded_bytes += len(chunk)
                    
                    # Log upload progress every ~1 GB
                    if uploaded_bytes % (1024 * 1024 * 1024) < chunk_size:
                        print(f"Transferred: {uploaded_bytes / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB")

    print(f"\nStreaming complete! Verifying target object status inside GCS...")
    if blob.exists():
        print(f"Success! Confirmed file exists at destination. Final Size: {blob.size / (1024**3):.4f} GB")
    else:
        print("Warning: Stream loop ended but GCS bucket object index verification timed out.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream models from HF to GCS via RAM memory chunking.")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    hf_token_env = os.getenv("HF_TOKEN")
    stream_hf_to_gcs(args.repo_id, args.filename, args.bucket, args.prefix, hf_token_env)
