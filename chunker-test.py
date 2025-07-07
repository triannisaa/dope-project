import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import re
import fitz
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

nltk.download('punkt')

def read_pdf_text_per_page(filepath):
    doc = fitz.open(filepath)
    return [{"page": i + 1, "text": page.get_text()} for i, page in enumerate(doc)]

def normalize_newlines(text):
    lines = text.split('\n')
    normalized, buffer = [], ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if buffer and re.search(r'[.!?]$', buffer):
            normalized.append(buffer)
            buffer = stripped
        else:
            buffer += " " + stripped if buffer else stripped
    if buffer:
        normalized.append(buffer)
    return "\n\n".join(normalized)

def clean_ocr_text(text):
    text = text.replace('\r', '')
    text = re.sub(r'[\u2022\u25AA\u25E6]', '-', text)
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return normalize_newlines(text).strip()

def chunk_text_with_metadata(pages_text, chunk_size=500, overlap=100, filename="unknown.pdf"):
    from hashlib import md5
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    chunks, chunk_id, seen_hashes = [], 1, set()
    for page_data in pages_text:
        page = page_data["page"]
        clean_text = clean_ocr_text(page_data["text"])
        for chunk in splitter.split_text(clean_text):
            text = chunk.strip()
            if not text:
                continue
            text_hash = md5(text.encode("utf-8")).hexdigest()
            if text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)
            chunks.append({
                "chunk_id": str(chunk_id),
                "content_chunk": text,
                "metadata": {
                    "filename": filename,
                    "page": page,
                    "structure": "semantic"
                }
            })
            chunk_id += 1
    return chunks

def store_chunks(chunks, persist_dir="./chroma_store", collection_name="pdf_chunks"):
    from hashlib import md5
    chroma_client = PersistentClient(path=persist_dir)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-mpnet-base-v2"
    )
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    existing = collection.get()
    existing_keys = set(
        (m.get("filename"), m.get("page"), md5(t.encode("utf-8")).hexdigest())
        for m, t in zip(existing.get("metadatas", []), existing.get("documents", []))
    )
    new_chunks = []
    for chunk in chunks:
        text = chunk["content_chunk"]
        filename = chunk["metadata"]["filename"]
        page = chunk["metadata"]["page"]
        key = (filename, page, md5(text.encode("utf-8")).hexdigest())
        if key not in existing_keys:
            new_chunks.append(chunk)

    if not new_chunks:
        print("⚠️ No new chunks to add.")
        return

    start_id = len(existing.get("documents", [])) + 1
    ids = [str(start_id + i) for i in range(len(new_chunks))]
    collection.add(
        documents=[c["content_chunk"] for c in new_chunks],
        metadatas=[c["metadata"] for c in new_chunks],
        ids=ids
    )
    print(f"✅ Stored {len(new_chunks)} new chunks in collection '{collection_name}'.")

if __name__ == "__main__":
    pdf_path = "context-darmahenwa.pdf"
    filename = os.path.basename(pdf_path)
    pages = read_pdf_text_per_page(pdf_path)
    chunks = chunk_text_with_metadata(pages, chunk_size=1000, overlap=200, filename=filename)
    store_chunks(chunks)
