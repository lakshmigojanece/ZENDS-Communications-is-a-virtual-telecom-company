from sentence_transformers import SentenceTransformer

_model = None

def get_embed_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # lighter model
    return _model
