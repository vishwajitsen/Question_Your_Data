import os
import streamlit as st
from dotenv import load_dotenv

# LangChain chains
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Vector stores
from langchain_pinecone import PineconeVectorStore
from langchain.vectorstores import FAISS

# Embeddings
#from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.embeddings import HuggingFaceEmbeddings

# Local helpers
from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    ingest_documents_to_pinecone,
    load_mistral_model,
)
from src.prompt import system_prompt

# ---------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "question-your-data-index")

# ✅ Fix model path for WSL
MODEL_PATH = "/mnt/c/Users/win11/OneDrive/Documents/MiSpy Documents/Question_Your_Data/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

st.set_page_config(page_title="Medical Chatbot (Streamlit)", layout="wide")
st.title("🩺 Medical Conditions Information Chatbot. Happy Exploring")

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Index & Data")
    data_dir = st.text_input(
        "PDF data directory",
        value="/mnt/c/Users/win11/OneDrive/Documents/MiSpy Documents/Question_Your_Data/data",
    )
    index_name = st.text_input("Index name", value=INDEX_NAME)
    rebuild_clicked = st.button("(Re)build index from PDFs")

# ---------------------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------------------
with st.spinner("Loading embedding model..."):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------------------
# Load Mistral via LangChain wrapper
# ---------------------------------------------------------------------
with st.spinner("Loading Mistral 7B model..."):
    llm = load_mistral_model(MODEL_PATH)

# ---------------------------------------------------------------------
# Pinecone loader
# ---------------------------------------------------------------------
def try_load_pinecone_store(index_name, embeddings):
    try:
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
        )
        return docsearch, "pinecone"
    except Exception as e:
        return None, e

# ---------------------------------------------------------------------
# FAISS loader
# ---------------------------------------------------------------------
def try_load_faiss_store(save_dir: str, embeddings):
    try:
        if os.path.exists(save_dir):
            store = FAISS.load_local(
                save_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
            return store, "faiss"
        else:
            return None, None
    except Exception as e:
        return None, e

# ---------------------------------------------------------------------
# Rebuild index
# ---------------------------------------------------------------------
def rebuild_index_flow(data_dir, index_name, embeddings):
    try:
        extracted = load_pdf_file(data_dir)
        filtered = filter_to_minimal_docs(extracted)
        chunks = text_split(filtered)

        if PINECONE_API_KEY:
            try:
                ingest_documents_to_pinecone(chunks, index_name, embeddings)
                return "pinecone"
            except Exception as e:
                st.warning(f"Pinecone ingest failed: {e}. Will attempt FAISS fallback.")
        else:
            st.info("No Pinecone API key - using FAISS fallback.")

        faiss_dir = "faiss_index"
        store = FAISS.from_documents(chunks, embeddings)
        store.save_local(faiss_dir)
        return "faiss"

    except Exception as e:
        st.exception(e)
        raise

# ---------------------------------------------------------------------
# If user clicked rebuild
# ---------------------------------------------------------------------
if rebuild_clicked:
    st.info("Starting (Re)build. This may take a while depending on PDF size.")
    try:
        result = rebuild_index_flow(data_dir, index_name, embeddings)
        st.success(f"Index built successfully using {result.upper()}.")
    except Exception as e:
        st.error(f"Index build failed: {e}")

# ---------------------------------------------------------------------
# Load vectorstore
# ---------------------------------------------------------------------
docsearch = None
used_store = None

docsearch, pinecone_err = try_load_pinecone_store(index_name, embeddings)
if docsearch is not None:
    used_store = "pinecone"
else:
    faiss_store, faiss_err = try_load_faiss_store("faiss_index", embeddings)
    if faiss_store is not None:
        docsearch = faiss_store
        used_store = "faiss"

if docsearch is None:
    st.warning("No vectorstore found. Rebuild from PDFs in the sidebar.")
    if isinstance(pinecone_err, Exception):
        st.info("Pinecone error:"); st.text(str(pinecone_err))
    st.stop()

st.success(f"Using vector store: {used_store}")

# ---------------------------------------------------------------------
# Build retriever + chain
# ---------------------------------------------------------------------
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# ---------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------
st.subheader("Ask questions from your documents")
query = st.text_input("Ask a question about the uploaded PDFs:")

if st.button("Get answer") and query.strip():
    with st.spinner("Running retrieval + Mistral..."):
        try:
            response = rag_chain.invoke({"input": query})
            answer = response.get("answer") if isinstance(response, dict) else str(response)

            st.markdown("**Answer:**")
            st.write(answer)

            st.markdown("**Top source documents:**")
            top_docs = retriever.get_relevant_documents(query)
            for i, d in enumerate(top_docs[:3], 1):
                src = d.metadata.get("source") if isinstance(d.metadata, dict) else None
                st.markdown(f"**{i}. Source:** {src}")
                snippet = d.page_content[:800].strip().replace("\n", " ")
                st.text(snippet + ("..." if len(snippet) > 700 else ""))

        except Exception as e:
            st.exception(e)

st.markdown("---")
st.caption("Tip: Use the sidebar to rebuild the index if you add new PDFs.")
