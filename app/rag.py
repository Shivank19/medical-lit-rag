from groq import Groq
from dotenv import load_dotenv
import os
from retriever import setup_chroma, embed_query, search_chroma, format_search_results, load_model

load_dotenv()

def rag_setup():
    pass
def process_json(filepath):
    pass

def build_prompt(question, chunks):
    pass

def call_llm(groq_client, prompt):
    pass

def query(question, model, collection, groq_client):
    pass

if __name__ == "__main__":
    pass    