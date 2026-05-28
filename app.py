import os
import chromadb
from dotenv import load_dotenv

load_dotenv()

def index_exists():
    try:
        client = chromadb.PersistentClient(path="./embeddings")
        col = client.get_collection("medical_literature_embeddings")
        return col.count() > 0
    except:
        return False

if not index_exists():
    print("Index not found — building from scratch. This takes ~5 minutes...")
    import subprocess
    subprocess.run(["python", "app/fetch.py"], check=True)
    subprocess.run(["python", "app/chunk.py"], check=True)
    subprocess.run(["python", "app/embed.py"], check=True)
    print("Index built successfully.")

from app.ui import demo
demo.launch(server_name="0.0.0.0", server_port=7860)