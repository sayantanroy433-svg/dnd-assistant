import base64
import os
import re
import sys
import importlib.machinery
from types import ModuleType

import streamlit as st
import google.genai as gemini_sdk
from google.genai import types
from pinecone import Pinecone

# ============================================================================
# 🛡 ENVIRONMENT FIXES
# ============================================================================

@st.cache_resource
def fix_environment_imports():
    class DummyLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            pass

    class DeepMock(ModuleType):
        def __getattr__(self, name):
            if name == "__path__":
                return []
            if name == "__spec__":
                return None
            return DeepMock(f"{self.__name__}.{name}")

        def __call__(self, *args, **kwargs):
            return None

    class MockFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith("torchvision"):
                mock = DeepMock(fullname)

                spec = importlib.machinery.ModuleSpec(
                    fullname,
                    DummyLoader()
                )

                spec.submodule_search_locations = []

                mock.__spec__ = spec
                mock.__path__ = []

                sys.modules[fullname] = mock

                return spec

            return None

    if not any(isinstance(f, MockFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, MockFinder())


fix_environment_imports()

# ============================================================================
# 🎲 PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Pocket D&D Loremaster",
    page_icon="🎲",
    layout="centered"
)

# ============================================================================
# 🖼 IMAGE LOADER
# ============================================================================

def get_base64_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as img:
            return base64.b64encode(img.read()).decode()
    except Exception:
        return ""


logo_base64 = get_base64_image("your_d20_image.png")

# ============================================================================
# 🎨 CUSTOM STYLING
# ============================================================================

st.markdown(
    f"""
<style>

.stApp {{
    background-color:#1a1613 !important;
    background-image:
        radial-gradient(#2d2219 1px, transparent 0) !important;
    background-size:24px 24px !important;

    color:#e3d1be !important;
    font-family:Georgia, serif !important;
}}

.brand-container {{
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    text-align:center;
    margin-bottom:20px;
}}

.brand-logo {{
    width:110px;
    margin-bottom:16px;
    mix-blend-mode:screen;
}}

h1 {{
    color:#d4af37 !important;

    text-shadow:2px 2px 4px black;

    text-decoration:underline;
    text-decoration-color:#8c2d19;

    text-decoration-thickness:2px;
    text-underline-offset:12px;

    margin:0;
}}

p,
span,
label,
.stMarkdown {{
    color:#e3d1be !important;
}}

.stChatInput textarea {{
    background:#2b221a !important;
    color:#f5eccd !important;
    border:1px solid #8c2d19 !important;
}}

.stSpinner > div {{
    border-top-color:#d4af37 !important;
}}

div[data-testid="stChatMessage"] {{
    border-radius:12px;
}}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# 🎲 HEADER
# ============================================================================

st.markdown(
    f"""
<div class="brand-container">

<img
src="data:image/png;base64,{logo_base64}"
class="brand-logo">

<h1>Pocket D&D Loremaster</h1>

<p style="color:#8a7663;
font-style:italic;
margin-top:15px;">

Powered by magic!

</p>

</div>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# 🔑 API KEYS
# ============================================================================

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"].strip()
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"].strip()

PINECONE_INDEX_NAME = "dnd-index"

# ============================================================================
# 🤖 GEMINI CLIENT
# ============================================================================

@st.cache_resource
def get_gemini_client():
    return gemini_sdk.Client(
        api_key=GEMINI_API_KEY
    )


ai_client = get_gemini_client()

# ============================================================================
# 🌲 PINECONE CLIENT
# ============================================================================

@st.cache_resource
def get_pinecone():

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    index = pc.Index(PINECONE_INDEX_NAME)

    return pc, index


pc, index = get_pinecone()
# ============================================================================
# 🔍 VECTOR SEARCH ENGINE
# ============================================================================

@st.cache_data(show_spinner=False)
def cached_vector_search(query_text: str):
    """
    Performs semantic search against Pinecone and returns
    serialized document dictionaries.
    """

    if not query_text.strip():
        return []

    # ------------------------------------------------------------------------
    # Generate embedding
    # ------------------------------------------------------------------------

    try:
        embedding_response = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query_text],
            parameters={
                "input_type": "query"
            }
        )

    except Exception as e:
        raise RuntimeError(
            f"Embedding generation failed:\n{e}"
        )

    # ------------------------------------------------------------------------
    # Extract vector safely
    # ------------------------------------------------------------------------

    query_vector = None

    try:

        if hasattr(embedding_response, "data"):

            data = embedding_response.data

            if isinstance(data, list) and len(data) > 0:

                item = data[0]

                if isinstance(item, dict):
                    query_vector = item.get("values")

                else:
                    query_vector = getattr(
                        item,
                        "values",
                        None
                    )

        elif isinstance(embedding_response, list):

            item = embedding_response[0]

            if isinstance(item, dict):
                query_vector = item.get("values")

            else:
                query_vector = getattr(
                    item,
                    "values",
                    None
                )

        elif hasattr(embedding_response, "values"):

            query_vector = embedding_response.values

    except Exception as e:

        raise RuntimeError(
            f"Embedding parsing failed:\n{e}"
        )

    if query_vector is None:

        raise RuntimeError(
            "Embedding vector could not be extracted."
        )

    # ------------------------------------------------------------------------
    # Query Pinecone
    # ------------------------------------------------------------------------

    try:

        results = index.query(
            namespace="markdown-docs",
            vector=query_vector,
            top_k=4,
            include_metadata=True,
        )

    except Exception as e:

        raise RuntimeError(
            f"Pinecone query failed:\n{e}"
        )

    matches = results.get("matches", [])

    documents = []

    for match in matches:

        metadata = match.get("metadata", {})

        chunk_text = metadata.get(
            "chunk_text",
            "No context found."
        )

        source_file = metadata.get(
            "source_file",
            "Unknown Rulebook"
        )

        chunk_index = metadata.get(
            "chunk_index",
            "?"
        )

        source_label = (
            f"📜 {source_file} "
            f"(Section {chunk_index})"
        )

        documents.append(
            {
                "page_content": chunk_text,
                "metadata": {
                    "source_label": source_label
                },
                "source_label": source_label,
            }
        )

    return documents


# ============================================================================
# 📚 RETRIEVER
# ============================================================================

class SimpleDocument:

    def __init__(self, page_content, metadata):

        self.page_content = page_content
        self.metadata = metadata


class NativePineconeVectorStore:

    def __init__(self, index):

        self.index = index

    def retrieve(self, query_text):

        raw_docs = cached_vector_search(query_text)

        docs = []

        for doc in raw_docs:

            docs.append(

                SimpleDocument(
                    page_content=doc["page_content"],
                    metadata=doc["metadata"],
                )

            )

        return docs


vector_store = NativePineconeVectorStore(index)

# ============================================================================
# 💬 CHAT SESSION
# ============================================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

# Replay previous conversation

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# User input

user_query = st.chat_input(
    "Ask about a spell, class, monster, feat, or rule..."
)
# ============================================================================
# ⚔️ RAG QUERY PIPELINE
# ============================================================================

if user_query:

    # ------------------------------------------------------------------------
    # Display user message immediately
    # ------------------------------------------------------------------------

    with st.chat_message("user"):
        st.markdown(user_query)

    # ------------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------------

    with st.spinner("Searching the rulebooks..."):

        search_query = user_query

        # Words that usually indicate a follow-up question
        vague_words = {
            "it",
            "that",
            "this",
            "those",
            "they",
            "them",
            "spell",
            "class",
            "monster",
            "feature",
            "ability",
            "more",
            "details",
            "explain",
            "continue"
        }

        lowered = user_query.lower()

        is_followup = any(word in lowered for word in vague_words)

        # --------------------------------------------------------------------
        # Improve search query using recent conversation
        # --------------------------------------------------------------------

        if st.session_state.chat_history and is_followup:

            recent_history = " ".join(
                message["content"]
                for message in st.session_state.chat_history[-4:]
            )

            entities = set(

                re.findall(
                    r"\b[A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+)*\b",
                    recent_history,
                )

            )

            blacklist = {
                "The",
                "This",
                "That",
                "It",
                "They",
                "We",
                "You",
                "I",
                "He",
                "She",
            }

            entities = [
                e for e in entities
                if e not in blacklist
            ]

            if entities:

                search_query = (
                    user_query
                    + " "
                    + " ".join(entities)
                )

            else:

                previous_user = ""

                for message in reversed(st.session_state.chat_history):

                    if message["role"] == "user":
                        previous_user = message["content"]
                        break

                if previous_user:

                    search_query = (
                        previous_user
                        + "\n"
                        + user_query
                    )

        # --------------------------------------------------------------------
        # Retrieve documents
        # --------------------------------------------------------------------

        try:

            matched_docs = vector_store.retrieve(search_query)

        except Exception as e:

            st.error(f"Vector search failed:\n\n{e}")

            matched_docs = []

    # =========================================================================
    # SOURCE LIST
    # =========================================================================

    unique_sources = []

    for doc in matched_docs:

        label = doc.metadata.get("source_label")

        if label and label not in unique_sources:

            unique_sources.append(label)

    # =========================================================================
    # BUILD CONTEXT
    # =========================================================================

    if matched_docs:

        context_str = "\n\n".join(

            doc.page_content

            for doc in matched_docs

            if doc.page_content != "No context found."

        )

    else:

        context_str = (
            "No relevant rulebook passages were retrieved."
        )

    # =========================================================================
    # BUILD CHAT HISTORY
    # =========================================================================

    history_text = "\n".join(

        f"{m['role'].capitalize()}: {m['content']}"

        for m in st.session_state.chat_history[-6:]

    )

    # =========================================================================
    # FINAL PROMPT
    # =========================================================================

    final_prompt_text = f"""
You are Pocket D&D Loremaster.

You are an expert in official Dungeons & Dragons 5th Edition rules.

Answer the user's question using the retrieved rulebook passages
as your primary source.

If the retrieved passages are incomplete,
supplement them using accurate official D&D 5e knowledge.

Do NOT invent rules.

If multiple books disagree,
prefer the newer official wording.

==========================
Retrieved Rulebook Context
==========================

{context_str}

==========================
Conversation
==========================

{history_text}

==========================
User Question
==========================

{user_query}

==========================
Answer
==========================
"""
# ============================================================================
# 🤖 GEMINI RESPONSE GENERATION
# ============================================================================

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        full_response = ""

        try:

            response_stream = ai_client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=final_prompt_text,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    system_instruction=(
                        "You are Pocket D&D Loremaster, an expert on "
                        "official Dungeons & Dragons 5th Edition rules.\n\n"
                        "Use the retrieved rulebook passages as your primary "
                        "source of truth.\n\n"
                        "If the passages are incomplete, supplement them with "
                        "accurate official D&D 5e knowledge.\n\n"
                        "Never invent rules, spell effects, class features, "
                        "monster statistics, or mechanics."
                    ),
                ),
            )

            for chunk in response_stream:

                # Different SDK versions expose text slightly differently
                text = getattr(chunk, "text", None)

                if not text:
                    continue

                full_response += text

                response_placeholder.markdown(
                    full_response + "▌"
                )

        except Exception as e:

            full_response = (
                "⚠️ Unable to contact Gemini.\n\n"
                f"Error:\n\n{e}"
            )

        if not full_response.strip():

            full_response = (
                "🧙 The Loremaster could not assemble an answer."
            )

        response_placeholder.markdown(full_response)

# ============================================================================
# 💾 SAVE CHAT HISTORY
# ============================================================================

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": full_response,
        }
    )

# ============================================================================
# 📜 DISPLAY SOURCES
# ============================================================================

    if unique_sources and "Unable to contact Gemini" not in full_response:

        st.markdown("---")

        st.subheader("📚 Rulebook References")

        displayed = set()

        for source in unique_sources:

            if source in displayed:
                continue

            displayed.add(source)

            with st.expander(source):

                matching_doc = next(
                    (
                        doc.page_content
                        for doc in matched_docs
                        if doc.metadata.get("source_label") == source
                    ),
                    "Context unavailable.",
                )

                st.markdown(matching_doc)

# ============================================================================
# 📊 OPTIONAL DEBUG PANEL
# ============================================================================

    with st.expander("🔍 Debug Information", expanded=False):

        st.write("Search Query")

        st.code(search_query)

        st.write("Retrieved Documents")

        st.write(len(matched_docs))

        st.write("Conversation History")

        st.code(history_text)

        st.write("Prompt Size")

        st.write(len(final_prompt_text))
		
# ============================================================================
# ⚙️ APPLICATION CONSTANTS
# ============================================================================

TOP_K_RESULTS = 4
CHAT_HISTORY_LIMIT = 6
MODEL_NAME = "gemini-2.5-flash"

SYSTEM_PROMPT = """
You are Pocket D&D Loremaster.

You are an expert in official Dungeons & Dragons 5th Edition.

Guidelines:

• Use the retrieved rulebook passages as your primary source.

• If the retrieved passages are incomplete,
  supplement them with accurate official D&D knowledge.

• Never invent rules.

• Explain mechanics clearly.

• When appropriate, quote spell levels,
  ranges, durations, components,
  and saving throws.

• If the answer isn't present in the retrieved
  documents, clearly state you're relying on
  official D&D knowledge.
"""