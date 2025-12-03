import os
from huggingface_hub import HfApi, create_repo

HF_USERNAME = "mehmetPektas"
REPO_NAME = "gpt-from-scratch-with-shakespeare"

WEIGHTS_FILE = "model_weights.pth"
TOKENIZER_FILE = "tokenizer_meta.pkl"
CODE_FILE = "main.py"

def upload_model():
    try:
        repo_url = create_repo(repo_id=f"{HF_USERNAME}/{REPO_NAME}", exist_ok=True, repo_type="model")
        print(f"Repository ready: {repo_url}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    api = HfApi()
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"
    
    try:
        api.upload_file(
            path_or_fileobj=WEIGHTS_FILE,
            path_in_repo="pytorch_model.py",
            repo_id=repo_id
        )
        print(f"✅ {WEIGHTS_FILE} -> uploaded as pytorch_model.pth")
        
        
        api.upload_file(
            path_or_fileobj=TOKENIZER_FILE,
            path_in_repo="tokenizer_meta.pkl",
            repo_id=repo_id
        )
        print(f"✅ {TOKENIZER_FILE} uploaded.")
        
        api.upload_file(
            path_or_fileobj=CODE_FILE,
            path_in_repo="model.py",
            repo_id=repo_id
        )
        print(f"✅ {CODE_FILE} -> uploaded as model.py")
        
        print(f"\n All tasks completed successfully.")
        
    except FileNotFoundError as e:
        print(f"\n ERROR: File not found! -> {e}")
    except Exception as e:
        print(f"\n An unexpected error occurred: {e}")
        
        
if __name__ == "__main__":
    upload_model()