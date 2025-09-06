import os
import fitz  # PyMuPDF
from langchain_core.documents import Document  # updated import for LangChain >=0.2
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
        n_threads=os.cpu_count(),   # auto-detect CPU cores
        n_batch=256,                # keep lower for stability on Windows
        temperature=0.6,
        max_tokens=1024,
        f16_kv=True,                # ✅ required for quantized gguf models
        use_mlock=True,             # ✅ prevents swapping to disk
        verbose=True,
    )
    return llm

# ---------------------------------------------------------------------
# PDF Loader
# ---------------------------------------------------------------------
def load_pdf_file(folder_path: str):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": f"{pdf_path}#page={page_num+1}"}
                        )
                    )
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],  # ✅ ensures cleaner chunks
    )
    return splitter.split_documents(docs)
