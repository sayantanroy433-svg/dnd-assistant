import streamlit as st
from rag import ask

st.set_page_config(
    page_title="D&D Assistant",
    page_icon="🐉",
    layout="wide"
)

st.title("🐉 D&D Documentation Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

# Display previous messages
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
question = st.chat_input("Ask a D&D question...")

if question:

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.history.append({
        "role": "user",
        "content": question
    })

    # Convert history into format expected by rag.py
    rag_history = []

    msgs = st.session_state.history

    for i in range(0, len(msgs)-1, 2):
        if msgs[i]["role"] == "user" and msgs[i+1]["role"] == "assistant":
            rag_history.append({
                "question": msgs[i]["content"],
                "answer": msgs[i+1]["content"]
            })

    with st.chat_message("assistant"):

        placeholder = st.empty()

        answer = ask(question, rag_history)

        placeholder.markdown(answer)

    st.session_state.history.append({
        "role": "assistant",
        "content": answer
    })