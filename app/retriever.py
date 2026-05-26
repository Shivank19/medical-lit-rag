import chromadb
from sentence_transformers import SentenceTransformer
import json

def load_model(model_name):
    model = SentenceTransformer(model_name)
    return model


def setup_chroma(path, collection_name):
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    print(f"Collection has {collection.count()} chunks")

    return collection

def embed_query(model, query):
    query_embedding = model.encode([query])
    return query_embedding

def search_chroma(collection, query_embeddings, n_results=5):
    results = collection.query(query_embeddings=query_embeddings.tolist(), n_results=n_results)
    
    return results

def format_search_results(results):
    formatted = []
    for i in range(len(results["documents"][0])):
        formatted.append({
            "text": results["documents"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "pmid": results["metadatas"][0][i]["pmid"],
            "year": results["metadatas"][0][i]["year"],
            "journal": results["metadatas"][0][i]["journal"],
            "url": results["metadatas"][0][i]["url"],
            "score": 1 - results["distances"][0][i]
        })
    return formatted

if __name__ == "__main__":
    model_name = 'all-MiniLM-L6-v2'
    model = load_model(model_name)
   
    chroma_path = './embeddings'
    collection_name = 'medical_literature_embeddings'

    chroma_collection = setup_chroma(chroma_path, collection_name)

    query = "what are the side effects of mRNA cancer vaccines?"

    qe = embed_query(model, query)
    search_results = search_chroma(chroma_collection, qe, n_results=5)
    formatted = format_search_results(search_results)
    for i, r in enumerate(formatted):
        print(f"{i+1}. {r['title']} (score: {r['score']:.2f})")

