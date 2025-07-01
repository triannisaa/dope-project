import os
import re
import fitz
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

nltk.download('punkt')

# --------- Step 1: Read text from PDF page-b
# y-page ---------
def read_pdf_text_per_page(filepath):
    doc = fitz.open(filepath)
    return [{"page": i + 1, "text": page.get_text()} for i, page in enumerate(doc)]

# --------- Step 2: Chunk text with overlap ---------
def detect_list_items(text):
    lines = text.split('\n')
    items = []
    buffer = ''

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match list markers: 1., 1), a), -, •, etc.
        if re.match(r'^(\d+[\.\)]|[a-zA-Z][\.\)]|[-•▪◦])\s+', line):
            if buffer:
                items.append(buffer.strip())
                buffer = ''
            buffer = line
        else:
            # Continuation of previous list item
            buffer += ' ' + line

    if buffer:
        items.append(buffer.strip())

    return items

def chunk_text_with_metadata(pages_text, chunk_size=5, overlap=1, filename="unknown.pdf"):
    chunk_id = 1
    chunks = []

    for page_data in pages_text:
        page = page_data["page"]
        clean_text = clean_ocr_text(page_data["text"])

        # 🔍 Gunakan deteksi list jika tersedia
        list_items = detect_list_items(clean_text)
        if len(list_items) >= 2:
            sentences = list_items
            structure_type = "list"
        else:
            sentences = [s.strip() for s in sent_tokenize(clean_text)]
            structure_type = "text"

        total_sentences = len(sentences)
        i = 0
        seen_chunks = set()

        while i < total_sentences:
            # Hitung indeks chunk berdasarkan posisi i
            if i == 0:
                start = 0
                end = min(start + chunk_size, total_sentences)
            elif i + chunk_size - overlap >= total_sentences:
                start = max(total_sentences - chunk_size, 0)
                end = total_sentences
            else:
                start = i - overlap
                end = i + chunk_size - overlap

            if start >= end or start < 0:
                i += 1
                continue

            chunk_sentences = sentences[start:end]
            text = " ".join(chunk_sentences).strip()

            # Cek duplikasi berdasarkan hash isi chunk
            text_hash = hash(text)
            if text_hash in seen_chunks:
                i += 1
                continue
            seen_chunks.add(text_hash)

            # Tambahkan ke hasil chunk
            chunks.append({
                "chunk_id": f"{chunk_id}",
                "content_chunk": text,
                "metadata": {
                    "filename": filename,
                    "page": page,
                    "structure": structure_type
                }
            })

            chunk_id += 1
            i += chunk_size - overlap

    return chunks

# --------- Step 3: Store chunks in ChromaDB ---------
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

    # Fetch existing chunks
    existing = collection.get()
    existing_metadatas = existing.get("metadatas", [])
    existing_documents = existing.get("documents", [])

    # Create a set of (filename, page, text_hash) for deduplication
    existing_keys = set()
    for metadata, text in zip(existing_metadatas, existing_documents):
        key = (
            metadata.get("filename"),
            metadata.get("page"),
            md5(text.encode("utf-8")).hexdigest()  # use hash for efficiency
        )
        existing_keys.add(key)

    # Filter chunks to only new ones
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

    # Assign new chunk IDs as incrementing numbers
    start_id = len(existing_documents) + 1
    ids = [str(start_id + i) for i in range(len(new_chunks))]
    documents = [chunk["content_chunk"] for chunk in new_chunks]
    metadatas = [chunk["metadata"] for chunk in new_chunks]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ Stored {len(new_chunks)} new chunks in collection '{collection_name}'.")


# --------- Step 4: Search in ChromaDB ---------
def search_chunks(query, persist_dir="./chroma_store", collection_name="pdf_chunks", top_k=3):
    chroma_client = PersistentClient(path=persist_dir)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-mpnet-base-v2"
    )

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    print("\n🔍 Search Results:")
    for i in range(len(results["documents"][0])):
        print(f"\n--- Result {i+1} ---")
        print("Content:", results["documents"][0][i])
        print("Metadata:", results["metadatas"][0][i])
        print("ID:", results["ids"][0][i])

def view_all_chunks(persist_dir="./chroma_store", collection_name="pdf_chunks"):
    from chromadb import PersistentClient
    from chromadb.utils import embedding_functions

    chroma_client = PersistentClient(path=persist_dir)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-mpnet-base-v2"
    )

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

    all_chunks = collection.get()

    print(f"\n📦 Total Chunks: {len(all_chunks['documents'])}")
    for i in range(len(all_chunks["documents"])):
        print(f"\n--- Chunk {i+1} ---")
        print("ID:", all_chunks["ids"][i])
        print("Content:", all_chunks["documents"][i])
        print("Metadata:", all_chunks["metadatas"][i])

def normalize_newlines(text):
    lines = text.split('\n')
    normalized = []
    buffer = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if buffer:
            if re.search(r'[.!?]$', buffer):
                normalized.append(buffer)
                buffer = stripped
            else:
                buffer += " " + stripped
        else:
            buffer = stripped

    if buffer:
        normalized.append(buffer)

    return "\n".join(normalized)

def clean_ocr_text(text):
    text = text.replace('\r', '')
    text = re.sub(r'[•▪◦]', '-', text)
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    text = normalize_newlines(text)
    return text.strip()

# --------- MAIN ---------
if __name__ == "__main__":
    pdf_path = "context-darmahenwa.pdf"
    filename = os.path.basename(pdf_path)

    pages = read_pdf_text_per_page(pdf_path)
    chunks = chunk_text_with_metadata(pages, chunk_size=4, overlap=2, filename=filename)

    store_chunks(chunks)  # Store into chroma_store/pdf_chunks

    # 🔍 Example query
    search_chunks("Kapan perusahaan dh didirikan")
    #view_all_chunks()