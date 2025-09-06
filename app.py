import os
import streamlit as st
from dotenv import load_dotenv

# LangChain chains
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Vector store
from langchain_community.vectorstores import FAISS

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Local helpers
from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    load_mistral_model,
)

# ---------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------
load_dotenv()

# ✅ Correct model path (Windows gguf file)
MODEL_PATH = r"C:\Users\win11\OneDrive\Documents\MiSpy Documents\Question_Your_Data\mistral-7b-instruct-v0.1.Q4_K_M.gguf"

st.set_page_config(page_title="Medical Chatbot (Streamlit)", layout="wide")
st.title("🩺 Medical Conditions Information Chatbot. Happy Exploring")

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Index & Data")
    data_dir = st.text_input(
        "PDF data directory",
        value=r"C:\Users\win11\OneDrive\Documents\MiSpy Documents\Question_Your_Data\data",
    )
    rebuild_clicked = st.button("(Re)build index from PDFs")

# ---------------------------------------------------------------------
# Load embeddings (force CPU)
# ---------------------------------------------------------------------
with st.spinner("Loading embedding model..."):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # 🚨 avoid CUDA/meta tensor issue
    )

# ---------------------------------------------------------------------
# Load Mistral via llama.cpp wrapper
# ---------------------------------------------------------------------
with st.spinner("Loading Mistral 7B model..."):
    llm = load_mistral_model(MODEL_PATH)

# ---------------------------------------------------------------------
# FAISS loader
# ---------------------------------------------------------------------
def try_load_faiss_store(save_dir: str, embeddings):
    try:
        if os.path.exists(save_dir):
            return FAISS.load_local(
                save_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
        return None
    except Exception:
        return None

# ---------------------------------------------------------------------
# Rebuild FAISS index
# ---------------------------------------------------------------------
def rebuild_index_flow(data_dir, embeddings):
    extracted = load_pdf_file(data_dir)
    filtered = filter_to_minimal_docs(extracted)
    chunks = text_split(filtered)

    faiss_dir = "faiss_index"
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(faiss_dir)
    return "faiss"

# ---------------------------------------------------------------------
# If user clicked rebuild
# ---------------------------------------------------------------------
if rebuild_clicked:
    st.info("Starting (Re)build. This may take a while depending on PDF size.")
    try:
        result = rebuild_index_flow(data_dir, embeddings)
        st.success(f"Index built successfully using {result.upper()}.")
    except Exception as e:
        st.error(f"Index build failed: {e}")

# ---------------------------------------------------------------------
# Load vectorstore
# ---------------------------------------------------------------------
docsearch = try_load_faiss_store("faiss_index", embeddings)

if docsearch is None:
    st.warning("No FAISS vectorstore found. Rebuild from PDFs in the sidebar.")
    st.stop()

st.success("Using vector store: FAISS")

# ---------------------------------------------------------------------
# Build retriever + chain
# ---------------------------------------------------------------------
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

system_prompt = (
    "You are a helpful AI assistant that answers questions based only on the provided documents.\n\n"
    "Here are the relevant documents:\n{context}\n\n"
    "If the information is not available in the documents, say you don’t know."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# ---------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------
st.subheader("Ask questions from your documents")
query = st.text_input("Ask any question about medical condition & its treatment")

if st.button("Get answer") and query.strip():
    with st.spinner("Running retrieval + Mistral..."):
        try:
            response = rag_chain.invoke({"input": query})
            answer = response.get("answer") if isinstance(response, dict) else str(response)

            st.markdown("**Answer:**")
            st.write(answer)

           # st.markdown("**Top source documents:**")
            #top_docs = retriever.get_relevant_documents(query)
            #for i, d in enumerate(top_docs[:3], 1):
                #src = d.metadata.get("source") if isinstance(d.metadata, dict) else None
                #st.markdown(f"**{i}. Source:** {src}")
                #snippet = d.page_content[:800].strip().replace("\n", " ")
                #st.text(snippet + ("..." if len(snippet) > 700 else ""))

        except Exception as e:
            st.exception(e)

st.markdown("---")
st.caption("Tip: Use the sidebar to rebuild the index if you add new PDFs.")
