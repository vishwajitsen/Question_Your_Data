from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from Chatbot import update_index_with_new_pdfs, ask_question

app = FastAPI()

# Build / load FAISS index at startup
faiss_store = update_index_with_new_pdfs()

@app.get("/")
def root():
    return {"message": "Medical Chatbot is running ✅"}

@app.get("/ask")
def ask(question: str = Query(..., description="Your medical question")):
    if not question:
        return JSONResponse(content={"error": "Question is required"}, status_code=400)
    answer, sources = ask_question(question, faiss_store)
    return {"answer": answer, "sources": sources}
