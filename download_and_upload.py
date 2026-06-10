import os
import argparse
import requests
from google.cloud import storage

def stream_hf_to_gcs(repo_id, filename, bucket_name, gcs_prefix, hf_token=None):
    # Try the alternate raw endpoint which is friendlier to direct chunk streaming
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    
    # Emulate the official huggingface-hub python client headers
    headers = {
        "User-Agent": "huggingface-hub/0.23.0 python/3.10",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
    }
    
    if hf_token:
        # Strip any accidental spaces from the token copy-paste
        token_clean = hf_token.strip()
        headers["Authorization"] = f"Bearer {token_clean}"
        print("-> Hugging Face Token successfully formatted and attached.")
    else:
        print("-> Warning: No HF_TOKEN detected in the environment. Gated repositories will fail.")
        
    print("Initializing GCS Client...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    gcs_blob_path = os.path.join(gcs_prefix, filename).replace("\\", "/")
    blob = bucket.blob(gcs_blob_path)
    
    print(f"Opening network stream from Hugging Face: {url}")
    
    # Use allow_redirects=True to handle Hugging Face's AWS/Cloudflare backend redirects
    with requests.get(url, headers=headers, stream=True, allow_redirects=True) as response:
        if response.status_code == 401 or response.status_code == 403:
            print(f"\n[ERROR] {response.status_code} Unauthorized.")
            print("1. Ensure you accepted the model terms on Hugging Face using your browser.")
            print("2. Confirm 'HF_TOKEN' is added in your GitHub REPOSITORY Secrets, not profile settings.")
            response.raise_for_status()
        else:
            response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"Connected! File size to transfer: {total_size / (1024**3):.4f} GB")
        print(f"Streaming data directly to gs://{bucket_name}/{gcs_blob_path}...")
        
        uploaded_bytes = 0
        chunk_size = 50 * 1024 * 1024  # 50 MB chunk buffers
        blob.chunk_size = chunk_size
        
        with blob.open("wb", ignore_flush=True) as gcs_file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    gcs_file.write(chunk)
                    uploaded_bytes += len(chunk)
                    
                    if uploaded_bytes % (1024 * 1024 * 1024) < chunk_size:
                        print(f"Transferred: {uploaded_bytes / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB")

    print(f"\nStreaming complete! Verifying object status in GCS...")
    if blob.exists():
        print(f"Success! Confirmed file exists at destination. Size: {blob.size / (1024**3):.4f} GB")
    else:
        print("Warning: Upload complete, but GCS bucket index update is lagging.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream models from HF to GCS via RAM memory chunking.")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    hf_token_env = os.getenv("HF_TOKEN")
    stream_hf_to_gcs(args.repo_id, args.filename, args.bucket, args.prefix, hf_token_env)
