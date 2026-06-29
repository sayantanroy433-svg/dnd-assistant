import streamlit as st
import re
import os
from pinecone import Pinecone
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Pocket D&D Loremaster",
    page_icon="🎲",
    layout="centered"
)

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

PINECONE_INDEX = "dnd-index"

client = genai.Client(api_key=GEMINI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def embed_query(text: str):
    res = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[text],
        parameters={"input_type": "query"}
    )

    # SAFE extraction (works across SDK versions)
    if isinstance(res, list):
        return res[0].get("values") if isinstance(res[0], dict) else res[0].values

    if hasattr(res, "data"):
        item = res.data[0]
        return item.get("values") if isinstance(item, dict) else item.values

    return res[0].values

def search_pinecone(query: str):
    vector = embed_query(query)

    results = index.query(
        namespace="markdown-docs",
        vector=vector,
        top_k=4,
        include_metadata=True
    )

    docs = []
    for m in results.get("matches", []):
        meta = m.get("metadata", {})

        docs.append({
            "text": meta.get("chunk_text", ""),
            "source": f"{meta.get('source_file')} (Section {meta.get('chunk_index')})"
        })

    return docs
	
def build_prompt(context, history, question):
    return f"""
You are a Dungeons & Dragons 5e rules expert.

Use the provided rulebook context first.
If missing info, use official 5e knowledge.

Never invent rules.

--- CONTEXT ---
{context}

--- CHAT HISTORY ---
{history}

--- QUESTION ---
{question}

Answer clearly:
"""

if "chat" not in st.session_state:
    st.session_state.chat = []
	
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
		
question = st.chat_input("Ask about D&D rules...")

if question:

    st.chat_message("user").markdown(question)

    # -------------------------
    # Retrieve docs
    # -------------------------
    docs = search_pinecone(question)

    context = "\n\n".join(d["text"] for d in docs)

    history = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in st.session_state.chat[-6:]
    )

    prompt = build_prompt(context, history, question)

    # -------------------------
    # Gemini streaming
    # -------------------------
    with st.chat_message("assistant"):
        box = st.empty()
        output = ""

        try:
            stream = client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2
                )
            )

            for chunk in stream:
                if chunk.text:
                    output += chunk.text
                    box.markdown(output + "▌")

        except Exception as e:
            output = f"Error: {e}"

        box.markdown(output)

    # -------------------------
    # Save memory
    # -------------------------
    st.session_state.chat.append({
        "role": "user",
        "content": question
    })

    st.session_state.chat.append({
        "role": "assistant",
        "content": output
    })

    # -------------------------
    # Sources
    # -------------------------
    if docs:
        st.markdown("### 📚 Sources")
        for d in docs:
            with st.expander(d["source"]):
                st.write(d["text"])

