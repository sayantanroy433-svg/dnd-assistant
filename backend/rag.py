import os
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone
from parser import parse_query, build_filter

load_dotenv()

# ==========================================================
# CONFIG
# ==========================================================

PINECONE_INDEX = os.environ["PINECONE_INDEX"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

NAMESPACE = "dnddocs"

EMBEDDING_MODEL = "multilingual-e5-large"

TOP_K = 10
MIN_SCORE = 0.80

# ==========================================================
# MODEL FALLBACKS (NEW)
# ==========================================================

MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
]

# ==========================================================
# CLIENTS
# ==========================================================

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
client = Groq(api_key=GROQ_API_KEY)

# ==========================================================
# SYSTEM PROMPT
# ==========================================================

SYSTEM_PROMPT = """
You are an expert Dungeons & Dragons 5e rules assistant.

Your job is to answer ONLY from the supplied documentation.

Rules:
1. Never invent rules.
2. Never rely on outside knowledge.
3. If the answer is missing, say:
   "I couldn't find that information in the documentation."
4. Prefer quoting rule names.
5. If multiple documents disagree, mention the conflict.
6. Use Markdown formatting.
7. Keep answers concise unless the user requests detail.
8. Do not mention internal document numbers.
"""

# ==========================================================
# RETRIEVAL
# ==========================================================

def retrieve(question):

    parsed = parse_query(question)
    metadata_filter = build_filter(parsed)

    embedding = pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=[question],
        parameters={"input_type": "query"}
    )

    vector = embedding[0].values

    query_args = {
        "namespace": NAMESPACE,
        "vector": vector,
        "top_k": TOP_K,
        "include_metadata": True
    }

    if metadata_filter:
        query_args["filter"] = metadata_filter

    result = index.query(**query_args)

    matches = [m for m in result.matches if m.score >= MIN_SCORE]

    if not matches and metadata_filter:
        query_args.pop("filter", None)
        result = index.query(**query_args)
        matches = [m for m in result.matches if m.score >= MIN_SCORE]

    return parsed, matches

# ==========================================================
# CONTEXT BUILDER
# ==========================================================

def build_context(matches):

    context_blocks = []
    sources = []

    for i, match in enumerate(matches, start=1):

        md = match.metadata

        source = md.get("source_file", "Unknown")
        text = md.get("chunk_text", "")

        sources.append(source)

        block = f"""
Document #{i}

Source: {source}
Book: {md.get('book', '')}
Section: {md.get('section', '')}
Page: {md.get('page', '')}
Type: {md.get('type', '')}
Similarity Score: {match.score:.3f}

{text}
"""

        context_blocks.append(block)

    context = "\n\n----------------------------------------\n\n".join(context_blocks)
    sources = sorted(set(sources))

    return context, sources

# ==========================================================
# LLM FALLBACK (NEW)
# ==========================================================

def call_llm(messages, temperature=0.2, stream=True):
    last_error = None

    for model in MODELS:
        try:
            print(f"\nTrying model: {model}\n")

            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=stream
            ), model

        except Exception as e:
            print(f"Model failed: {model} → {e}")
            last_error = e

    raise last_error

# ==========================================================
# CHAT
# ==========================================================

def ask(question, history):

    parsed, matches = retrieve(question)

    if not matches:
        return (
            "I couldn't find that information in the documentation.\n\n"
            "Try:\n"
            "- exact spell/feature/item name\n"
            "- specifying book (PHB, DMG, etc.)\n"
        )

    context, sources = build_context(matches)

    MAX_HISTORY = 2
    conversation = []

    for item in history[-MAX_HISTORY:]:
        conversation.append(f"""
User:
{item["question"]}

Assistant:
{item["answer"]}
""")

    previous_chat = "\n\n".join(conversation)

    content = f"""
Previous Conversation

{previous_chat}

==================================================

Documentation

{context}

==================================================

Question

{question}
"""

    stream, used_model = call_llm(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        temperature=0.2,
        stream=True
    )

    print("\nGenerating response...\n")
    print(f"Model used: {used_model}\n")

    answer = ""

    for chunk in stream:
        try:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                print(delta.content, end="", flush=True)
                answer += delta.content
        except Exception as e:
            print("Stream error:", e)

    answer += "\n\n---\n### Sources\n\n"

    for i, source in enumerate(sources, start=1):
        answer += f"{i}. {source}\n"

    return answer, sources
