import os
import fitz  # PyMuPDF
from langchain.docstore.document import Document
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------
# Load Mistral model (LangChain wrapper around llama.cpp)
# ---------------------------------------------------------------------
def load_mistral_model(model_path: str):
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")

    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        n_threads=8,   # adjust based on CPU cores
        n_batch=512,
        temperature=0.6,
        max_tokens=1024,
        verbose=True,
    )
    return llm

# ---------------------------------------------------------------------
# PDF Loader
# ---------------------------------------------------------------------
def load_pdf_file(folder_path: str):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                documents.append(Document(page_content=text, metadata={"source": pdf_path}))
    return documents

# ---------------------------------------------------------------------
# Minimal docs filter
# ---------------------------------------------------------------------
def filter_to_minimal_docs(docs):
    return [doc for doc in docs if doc.page_content.strip()]

# ---------------------------------------------------------------------
# Text Splitter
# ---------------------------------------------------------------------
def text_split(docs, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# ---------------------------------------------------------------------
# Pinecone ingestion
# ---------------------------------------------------------------------
def ingest_documents_to_pinecone(docs, index_name, embeddings):
    from langchain_pinecone import PineconeVectorStore
    PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
