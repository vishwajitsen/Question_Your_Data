import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import PyPDF2

from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Ollama (Mistral) wrapper
from langchain_ollama import ChatOllama

# ---------------- CONFIG ----------------
DOCS_DIR = "uploaded_pdfs"
INDEX_DIR = "faiss_index"
METADATA_PATH = os.path.join(INDEX_DIR, "doc_metadata.pkl")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
OLLAMA_MODEL = "mistral"
OLLAMA_TEMPERATURE = 0.0
# ----------------------------------------

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)


# ---------------- PDF utils ----------------
def extract_text_from_pdf(path: str) -> str:
    text_pages = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                page_text = page.extract_text()
            except Exception:
                page_text = None
            if page_text:
                text_pages.append(page_text)
    return "\n".join(text_pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


# ---------------- Indexing ----------------
def build_faiss_index() -> FAISS:
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    pdf_files = list(Path(DOCS_DIR).glob("*.pdf"))
    docs: List[Document] = []
    metadata_index: Dict[str, Any] = {}

    for pdf in pdf_files:
        txt = extract_text_from_pdf(str(pdf))
        chunks = chunk_text(txt)
        for i, ch in enumerate(chunks):
            md = {"source": str(pdf), "chunk": i}
            docs.append(Document(page_content=ch, metadata=md))
        metadata_index[str(pdf)] = {"chunks": len(chunks)}

    if not docs:
        return FAISS.from_documents([], embeddings)

    faiss_store = FAISS.from_documents(docs, embeddings)
    faiss_store.save_local(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_index, f)

    return faiss_store


def update_index_with_new_pdfs() -> FAISS:
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)):
        return build_faiss_index()

    faiss_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    with open(METADATA_PATH, "rb") as f:
        metadata_index = pickle.load(f)

    pdf_files = list(Path(DOCS_DIR).glob("*.pdf"))
    new_docs = []

    for pdf in pdf_files:
        if str(pdf) in metadata_index:
            continue
        txt = extract_text_from_pdf(str(pdf))
        chunks = chunk_text(txt)
        for i, ch in enumerate(chunks):
            md = {"source": str(pdf), "chunk": i}
            new_docs.append(Document(page_content=ch, metadata=md))
        metadata_index[str(pdf)] = {"chunks": len(chunks)}

    if new_docs:
        faiss_store.add_documents(new_docs)
        faiss_store.save_local(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(metadata_index, f)

    return faiss_store


# ---------------- RAG ----------------
def get_llm():
    return ChatOllama(model=OLLAMA_MODEL, temperature=OLLAMA_TEMPERATURE)


PROMPT_TEMPLATE = """You are an expert assistant. Use the information from the retrieved document snippets to answer the user's question thoroughly.
If the answer is not contained in the provided context, say you don't know rather than inventing facts.
Always end your answer with a single full stop.

Context:
{context}

Question:
{question}

Answer (detailed, complete):"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])


def create_qa_chain(faiss_store: FAISS, llm):
    retriever = faiss_store.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa


def ask_question(question: str, faiss_store: FAISS):
    llm = get_llm()
    qa = create_qa_chain(faiss_store, llm)
    res = qa({"query": question})
    answer = res.get("result", "").strip()
    if not answer.endswith("."):
        answer += "."
    sources = [d.metadata.get("source") for d in res.get("source_documents", []) if d.metadata.get("source")]
    sources = list(dict.fromkeys(sources))
    return answer, sources


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="📄 MiSpy RAG Assistant", layout="wide")

st.title("📄 Medical Conditions Information AI Assistant APP created by Vishwajit Sen")
st.markdown("Ask detailed questions from your PDFs. The assistant will always answer with a complete response ending with a full stop.")

# Sidebar: PDF upload
st.sidebar.header("📂 Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = Path(DOCS_DIR) / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.sidebar.success("PDFs uploaded successfully. Click 'Update Index' to process them.")

if st.sidebar.button("🔄 Update Index"):
    with st.spinner("Updating FAISS index..."):
        update_index_with_new_pdfs()
    st.sidebar.success("Index updated with new PDFs!")

# Load or build FAISS
if os.path.exists(FAISS_INDEX_PATH):
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    faiss_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    faiss_store = build_faiss_index()

# Query input
st.subheader("💬 Ask a Question")
question = st.text_input("Type your question here:")

if st.button("Get Answer") and question:
    with st.spinner("Thinking..."):
        answer, sources = ask_question(question, faiss_store)
    st.markdown("### 🧠 Answer")
    st.markdown(f"{answer}")

    if sources:
        st.markdown("### 📌 Sources")
        for s in sources:
            st.markdown(f"- `{s}`")
