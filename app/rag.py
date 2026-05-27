from groq import Groq
from dotenv import load_dotenv
import os
from retriever import setup_chroma, embed_query, search_chroma, format_search_results, load_model

load_dotenv()

def rag_setup():
    model_name = 'all-MiniLM-L6-v2'
    model = load_model(model_name)

    chroma_path = './embeddings'
    collection_name = 'medical_literature_embeddings'

    chroma_collection = setup_chroma(chroma_path, collection_name)

    groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    return model, chroma_collection, groq_client

def build_prompt(question, chunks):
    instruction = "You are a medical literature assistant. Answer ONLY from the abstracts provided below. Cite every claim using [1], [2] etc. If the answer is not in the sources, say so explicitly."

    source_blocks = []
    for index, chunk in enumerate(chunks, start = 1):
        source_blocks.append(
            f"[{index}] Title: {chunk['title']}\n"
            f"{chunk['text']}"
        )

    sources = "\n\n".join(source_blocks)
    return f"{instruction}\n\nSources:\n{sources}\n\nQuestion: {question}\n\nAnswer:"
    
def call_llm(groq_client, prompt):
    # llama 8b for experimentation as it uses less tokens and has more available tokens overall via groq, but can switch to 70b for final evaluation
    response = groq_client.chat.completions.create(
        # model="llama-3.3-70b-versatile",
        model="llama-3.1-8b-instant",
        messages = [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1000
    )

    answer = response.choices[0].message.content
    return answer

def query(question, model, collection, groq_client):
    qe = embed_query(model, question)
    search_results = search_chroma(collection, qe, n_results=10)
    formatted = format_search_results(search_results)

    # Deduplication based on PMID
    seen_pmids, unique_chunks = set(), []
    for chunk in formatted:
        if chunk['pmid'] not in seen_pmids:
            seen_pmids.add(chunk['pmid'])
            unique_chunks.append(chunk)
    unique_chunks = unique_chunks[:5]  # Limit to top 5 unique sources

    prompt = build_prompt(question, unique_chunks)
    answer = call_llm(groq_client, prompt)

    return {
        "answer": answer,
        "sources": unique_chunks
    }
if __name__ == "__main__":
    model, collection, groq_client = rag_setup()
    
    question = "What cancer types are mRNA vaccines currently being trialled for?"

    result = query(question, model, collection, groq_client)
    print("Question:")
    print(question)
    print("Answer:")
    print(result["answer"])
    print("\nSources:")
    for i, source in enumerate(result["sources"], start=1):
        print(f"{i}. {source['title']} (score: {source['score']:.2f})")