from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import ask

app = FastAPI(title="D&D Assistant")

# ---------------- CORS ---------------- #

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- REQUEST MODEL ---------------- #

class ChatRequest(BaseModel):
    message: str


# ---------------- ROOT ---------------- #

@app.get("/")
def root():
    return {"status": "running"}


# ---------------- MEMORY (SIMPLE) ---------------- #

history = []


# ---------------- CHAT ENDPOINT ---------------- #

@app.post("/chat")
def chat(req: ChatRequest):

    answer, sources = ask(req.message, history)

    history.append({
        "question": req.message,
        "answer": answer
    })

    return {
        "answer": answer,
        "sources": sources
    }
