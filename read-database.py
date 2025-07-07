import os
from sentence_transformers import SentenceTransformer, util
from chromadb import PersistentClient

# Load model gratis
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect ke Chroma
chroma_client = PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection(name="pdf_chunks")

# Query dan ambil hasil dari Chroma
def search_chroma(query, top_k=10):
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    return [{"content_chunk": doc, "metadata": meta} for doc, meta in zip(docs, metadatas)]

# Rerank dan beri alasan positif/negatif untuk skor > 0.5
def rerank_with_reasoning(query, results, threshold=0.5):
    query_emb = model.encode(query, convert_to_tensor=True)
    doc_texts = [res['content_chunk'] for res in results]
    doc_embeddings = model.encode(doc_texts, convert_to_tensor=True)

    cosine_scores = util.cos_sim(query_emb, doc_embeddings)[0]
    ranked = sorted(zip(results, cosine_scores), key=lambda x: x[1], reverse=True)

    # Filter hanya yang skor > threshold
    filtered = [(res, score) for res, score in ranked if score > threshold]

    if not filtered:
        return {
            "best_answer": None,
            "positive_reason": "❌ Tidak ada hasil dengan skor > 0.5",
            "negative_reasons": []
        }

    top_result, top_score = filtered[0]
    positive_reason = f"✅ Relevan karena mengandung konteks yang menjawab query: '{query}' dengan skor similarity {top_score:.2f}"

    negative_reasons = []
    for res, score in filtered[1:]:
        reason = f"❌ Kurang relevan karena hanya membahas sebagian konteks (skor {score:.2f})."
        negative_reasons.append({"text": res['content_chunk'], "reason": reason})

    return {
        "best_answer": top_result['content_chunk'],
        "metadata": top_result['metadata'],
        "positive_reason": positive_reason,
        "negative_reasons": negative_reasons
    }

# Contoh penggunaan
if __name__ == "__main__":
    query = input("Ajukan Pertanyaan : ")
    results = search_chroma(query)
    print("🔍 Semua hasil dari Chroma:")
    for i, res in enumerate(results):
        print(f"\n📄 Hasil {i+1}")
        print("Metadata :", res["metadata"])
        print("Isi     :", res["content_chunk"][:300], "...") 

    reranked = rerank_with_reasoning(query, results)
    print("💡 Jawaban terbaik:")
    print(reranked["best_answer"] or "Tidak ditemukan jawaban dengan skor > 0.5")
    print("\n👍 Alasan positif:")
    print(reranked["positive_reason"])
    print("\n👎 Alasan negatif:")
    for neg in reranked["negative_reasons"]:
        print("-", neg["reason"])
