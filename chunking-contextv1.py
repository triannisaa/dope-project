import openai
import chromadb
import difflib
from chromadb.config import Settings

# === 1. Setup OpenAI client ===
client_oa = openai.OpenAI(api_key="")

# === 2. Function to get OpenAI Embedding ===
def get_openai_embedding(text, model="text-embedding-3-small"):
    response = client_oa.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# === 3. Setup ChromaDB and collection ===
client_chroma = chromadb.Client(Settings(
    is_persistent=True,
    persist_directory="./chroma_openai"
))
collection = client_chroma.get_or_create_collection("openai_chunks")

# === 4. Interaktif: User Input
while True:
    query_text = input("\n❓ Pertanyaan kamu (atau ketik 'exit'): ").strip()
    if query_text.lower() == "exit":
        print("👋 Bye!")
        break

    # === 5. Dapatkan embedding pertanyaan
    query_embedding = get_openai_embedding(query_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    # === 6. Filter dokumen duplikat
    clean_docs = []
    seen_sim = []

    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        if not any(difflib.SequenceMatcher(None, doc, d).ratio() > 0.95 for d in seen_sim):
            seen_sim.append(doc)
            clean_docs.append((doc, meta))

    if not clean_docs:
        print("⚠️ Tidak ada hasil relevan ditemukan.")
        continue

    # === 7. Gabungkan dokumen jadi konteks
    context = "\n\n".join([doc for doc, _ in clean_docs])

    # === 8. Kirim ke OpenAI Chat untuk jawab
    try:
        chat_response = client_oa.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Gunakan konteks berikut untuk menjawab pertanyaan secara akurat dan ringkas."},
                {"role": "user", "content": f"Pertanyaan: {query_text}\n\nKonteks:\n{context}"}
            ],
            temperature=0.3
        )

        answer = chat_response.choices[0].message.content.strip()

        # === 9. Tampilkan jawaban
        print("\n🤖 Jawaban:")
        print(answer)

    except Exception as e:
        print(f"❌ Gagal menjawab dengan OpenAI: {e}")
