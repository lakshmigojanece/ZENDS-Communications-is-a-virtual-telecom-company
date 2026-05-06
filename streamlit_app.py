# =======================================
# AI Customer Support Copilot - Sentiment-Aware RAG + LLM
# =======================================

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ----------------------------
# LOAD LLM (FLAN-T5)
# ----------------------------
@st.cache_resource
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base")

llm = load_llm()

# ----------------------------
# LOAD SENTIMENT MODEL
# ----------------------------
@st.cache_resource
def load_sentiment():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

sentiment_model = load_sentiment()
# ----------------------------
# LOAD INTENT MODEL
# ----------------------------
@st.cache_resource
def load_intent_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return tokenizer, model

MODEL_PATH = "models/zends_intent_model"
tokenizer, intent_model = load_intent_model(MODEL_PATH)

intent_labels = [
    "Billing",
    "Complaint",
    "Product Inquiry",
    "Refund",
    "Technical Support"
]

# ----------------------------
# COMPANY DOCS
# ----------------------------
company_docs = [
    "ZenDS Communications is a global telecom and cloud provider.",
    "ZENDFiber broadband offers high-speed internet services.",
    "Refunds are processed within 7 working days. Please contact support if delayed.",
    "24/7 customer support is available for all services.",
    "Unlimited data plans are available for users.",
    "Upgrade plans for better speed."
]

# ----------------------------
# PREDEFINED PROCEDURES
# ----------------------------
procedures = {
    "Refund": "We have received your refund request. Please allow 7 working days for processing. Contact support if delayed.",
    "Complaint": "We are sorry for the inconvenience. Our support team is available 24/7 and will contact you shortly to resolve your issue.",
    "Billing": "We have received your billing inquiry. Our support team will review and get back to you shortly.",
    "Technical Support": "We are sorry for the technical issue. Our team will reach out soon to assist you.",
    "Product Inquiry": "You can find product details and services on our website or contact support for assistance."
}

# ----------------------------
# SMART RAG CONTEXT
# ----------------------------
def get_context(query, docs):
    query_words = query.lower().split()
    scored = []
    for doc in docs:
        score = sum([1 for word in query_words if word in doc.lower()])
        scored.append((score, doc))
    scored = sorted(scored, reverse=True)
    top_docs = [doc for score, doc in scored[:2]]
    return " ".join(top_docs)

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="AI Customer Support Copilot", layout="wide")
st.title("🤖 AI Customer Support Copilot")
st.write("Ask your customer queries below 👇")

query = st.text_input("Enter your query:")

if query:
    st.write("👤 You:", query)

    # ------------------------
    # INTENT PREDICTION
    # ------------------------
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    outputs = intent_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, pred_class = torch.max(probs, dim=1)
    intent = intent_labels[pred_class.item()]
    confidence = round(confidence.item(), 2)

    st.subheader("🎯 Intent")
    st.success(f"{intent} ({confidence})")

    # ------------------------
    # SENTIMENT ANALYSIS
    # ------------------------
    sentiment = sentiment_model(query)[0]["label"]
    st.subheader("😊 Sentiment")
    if sentiment == "NEGATIVE":
        st.error(sentiment)
    else:
        st.success(sentiment)

    # ------------------------
    # RAG CONTEXT
    # ------------------------
    Rag = get_context(query, company_docs)
    st.subheader("📄 Rag")
    st.info(Rag)

    # ------------------------
    # FINAL RESPONSE (Sentiment-Aware)
    # ------------------------
    if intent in procedures and sentiment != "POSITIVE":
        # Use predefined procedure only if sentiment is NEGATIVE/NEUTRAL
        final_answer = procedures[intent]
    else:
        # For POSITIVE sentiment or unknown intents, always use LLM
        prompt = f"""
You are a professional telecom customer support assistant.

Customer Query: {query}

Context: {Rag}

Sentiment: {sentiment}

Reply with a clear, helpful, human-like response in 1-2 sentences.
If the customer seems frustrated or angry, apologize and offer immediate assistance.
If the customer is giving positive feedback, respond politely and appreciatively.

Answer:
"""
        result = llm(prompt, max_new_tokens=100, temperature=0.3)[0]["generated_text"]
        final_answer = result.replace(prompt, "").strip()

    st.subheader("💬 AI Response")
    st.success(final_answer)