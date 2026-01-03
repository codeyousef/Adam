from huggingface_hub import list_repo_files

# UPDATED: Checking the 7B model repo to match production training
REPO_ID = "state-spaces/mamba2-7b"

try:
    print(f"Fetching file list for: {REPO_ID}...")
    files = list_repo_files(REPO_ID)
    print("\n--- Files in Repo ---")
    for f in files:
        print(f)
except Exception as e:
    print(f"❌ Error: {e}")
