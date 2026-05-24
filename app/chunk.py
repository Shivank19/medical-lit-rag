# chunk_document(): called in the main function. takes abstracts.json as input, extracts abstract_text from each object and calls chunk_text on it. For each chunk returned by the chunk_text function, append its original metadata and adds it to a new dict.

# chunk_text: takes the abstract text as sting input, splits it at pre defined word lengths (defined by chunk_size variable). Two chunks share some words at the boundaries, the number of shared words defined by the overlap variable.

import json

def chunk_document(document):
    all_chunks = []
    for doc in document:
        text = doc['title'].rstrip('.') + '. ' + doc['abstract']
        chunks = chunk_text(text)

        if not chunks:
            continue

        for chunk in chunks:
            chunk_doc = {
                'text': chunk,
                'pmid': doc['pmid'],
                'title': doc['title'],
                'year': doc['year'],
                'journal': doc['journal'],
                'url': doc['url']
            }
            all_chunks.append(chunk_doc)
    print(f"Total chunks: {len(all_chunks)}")
    print(f"First chunk text: {all_chunks[0]['text'][:100]}")
    print(f"First chunk pmid: {all_chunks[0]['pmid']}")

    lengths = [len(c['text'].split()) for c in all_chunks]
    print(f"Min chunk: {min(lengths)} words")
    print(f"Max chunk: {max(lengths)} words")  
    print(f"Avg chunk: {sum(lengths)//len(lengths)} words")
    short = [c for c in all_chunks if len(c['text'].split()) < 20]
    print(f"Chunks under 20 words: {len(short)}")
    return all_chunks


def chunk_text(abstract, chunk_size=150, overlap=20):
    words = abstract.split()

    if len(words) < 50:
        return []

    if len(words) <= chunk_size:
        return [abstract]

    # Split the text into chunks
    chunks, start = [], 0

    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        if len(words[start:end]) >= 20:  # Only add chunks that have a minimum length
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def process_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def save_to_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved {len(data)} chunks to {filepath}")

if __name__ == "__main__":
    
    filepath = 'data/abstracts.json'
    document = process_json(filepath)
    chunks = chunk_document(document)
    save_to_json(chunks, 'data/chunks.json')