import chromadb
from sentence_transformers import SentenceTransformer
import json

def load_chunks(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def load_model(model_name):
    model = SentenceTransformer(model_name)
    return model

def embed_texts(model, chunks):
    chunk_text = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(chunk_text, batch_size=32, show_progress_bar=True)
    print(embeddings.shape)
    return embeddings

def setup_chroma(path, collection_name):
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    print(f"Collection has {collection.count()} chunks")

    return collection

def store_in_chroma(collection, chunks, embeddings):
    
    embeddings_list = embeddings.tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [chunk['text'] for chunk in chunks]

    metadatas = []
    for chunk in chunks:
        metadatas.append({
            "pmid": chunk['pmid'],
            "title": chunk['title'],
            "year": chunk['year'],
            "journal": chunk['journal'],
            "url": chunk['url']
        })

    if collection.count() > 0:
        print("Collection already populated, skipping.")
        return
    collection.add(ids = ids, documents = documents, metadatas = metadatas, embeddings = embeddings_list)
    print(collection.count())

def verify(collection, model):
    test_question = "what are mRNA cancer vaccine side effects?"
    query_embedding = model.encode([test_question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    print("\nTop 3 results for test query:")
    for i, meta in enumerate(results["metadatas"][0]):
        print(f"{i+1}. {meta['title']}")

if __name__ == "__main__":
    filepath = 'data/chunks.json'
    all_chunks = load_chunks(filepath)

    model_name = 'all-MiniLM-L6-v2'
    model = load_model(model_name)

    embeddings = embed_texts(model, all_chunks)
    
    chroma_path = './embeddings'
    collection_name = 'medical_literature_embeddings'

    chroma_collection = setup_chroma(chroma_path, collection_name)

    store_in_chroma(chroma_collection, all_chunks, embeddings)
    
    print("Verifying stored embeddings with a test query...")
    verify(chroma_collection, model)