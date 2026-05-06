from .embedder import get_embed_model
from .vector_store import build_faiss_index

def retrieve(query, docs, index, top_k=2):
    model = get_embed_model()
    query_vector = model.encode([query])
    D, I = index.search(query_vector, top_k)
    return [docs[i] for i in I[0]]
