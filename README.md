# AI Customer Support Copilot

This project simulates an enterprise AI system for ZENDS Communications.
It integrates:
- Intent Classification (DistilBERT)
- Sentiment Analysis (HuggingFace)
- RAG Retrieval (FAISS + SentenceTransformers)
- LLM Response Generation (Mistral-7B / Falcon)
- Interactive Streamlit dashboard

## Folder Structure
See FINAL PROJECT FOLDER STRUCTURE for details.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Start Streamlit app: `streamlit run app/streamlit_app.py`
3. Enter queries in the dashboard and get AI-generated responses.
