import os
import re
from typing import List

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_openai_client() -> OpenAI | None:
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    api_key = groq_key or openai_key
    if not api_key:
        return None

    if groq_key:
        return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    return OpenAI(api_key=api_key)


def extract_pdf_text(file) -> str:
    reader = PdfReader(file)
    text_parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text_parts.append(text)
    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks = []
    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        chunk = cleaned[start:end]
        if len(chunk) < chunk_size and end >= len(cleaned):
            chunks.append(chunk.strip())
            break
        last_space = chunk.rfind(" ")
        if last_space > 0 and end < len(cleaned):
            chunk = chunk[:last_space]
        chunks.append(chunk.strip())
        start += max(1, len(chunk) - overlap)

    return [c for c in chunks if c]


@st.cache_resource
def build_faiss_index(chunks: List[str]):
    model = get_embedding_model()
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))
    return index, embeddings


def get_relevant_chunks(question: str, chunks: List[str], index, top_k: int = 4) -> List[str]:
    model = get_embedding_model()
    q_embedding = model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_embedding, min(top_k, len(chunks)))

    relevant = []
    for idx in indices[0]:
        if idx == -1:
            continue
        relevant.append(chunks[int(idx)])
    return relevant


def ask_openai(question: str, context: str) -> str:
    client = get_openai_client()
    if client is None:
        return "API key is missing. Add GROQ_API_KEY or OPENAI_API_KEY to your .env file or use the fallback answer below."

    prompt = (
        "Use only the information in the context below to answer the user's question. "
        "If the answer is not in the context, say so clearly.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )

    model = (
        os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        if os.getenv("GROQ_API_KEY")
        else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful PDF assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def answer_from_pdf(question: str, pdf_text: str) -> str:
    chunks = chunk_text(pdf_text)
    if not chunks:
        return "No readable text was found in the uploaded PDF."

    index, _ = build_faiss_index(chunks)
    relevant = get_relevant_chunks(question, chunks, index)
    context = "\n\n---\n\n".join(relevant)

    answer = ask_openai(question, context)
    if answer and "API key is missing" not in answer:
        return answer

    if not relevant:
        return "I could not find relevant information in this PDF to answer your question."

    return (
        "Relevant text from your PDF:\n\n"
        + "\n\n---\n\n".join(relevant[:2])
        + "\n\n"
        + "This is the most relevant section, but to get a polished final answer you should add your OPENAI_API_KEY."
    )


st.set_page_config(page_title="PDF Ask Me", page_icon="📄", layout="wide")
st.title("📄 PDF Question Answering App")
st.caption("Upload a PDF, ask a question, and get answers based on the document content.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
question = st.text_input("Ask a question about the PDF")

if st.button("Answer"):
    if uploaded_file is None:
        st.warning("Please upload a PDF first.")
    elif not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Reading the PDF and searching for the best answer..."):
            pdf_text = extract_pdf_text(uploaded_file)
            answer = answer_from_pdf(question, pdf_text)

        st.subheader("Answer")
        st.write(answer)

st.markdown("---")
st.markdown(
    """
    ### How it works
    1. Upload any PDF.
    2. Extract text from all pages.
    3. Split the text into smaller chunks.
    4. Convert the chunks into embeddings.
    5. Find the most relevant chunks for your question.
    6. Use the best context to generate the answer.
    """
)
