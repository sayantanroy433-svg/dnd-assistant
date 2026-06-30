import os
import time
from dotenv import load_dotenv
from groq import Groq, RateLimitError
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

TOP_K = 4
MIN_SCORE = 0.80

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
        text = md.get("chunk_text", "")[:800]

        sources.append(source)

        block = f"""
Document #{i}

Source: {source}
Book: {md.get('book', '')}
Section: {md.get('section', '')}
Page: {md.get('page', '')}
Type: {md.get('type', '')}
Score: {match.score:.3f}

{text}
"""

        context_blocks.append(block)

    context = "\n\n----------------------------------------\n\n".join(context_blocks)
    sources = sorted(set(sources))

    return context, sources


# ==========================================================
# CHAT
# ==========================================================

def ask(question, history):

    parsed, matches = retrieve(question)

    if not matches:
        return (
            "I couldn't find that information in the documentation.\n\n"
            "Try more specific terms or source books like PHB or DMG."
        )

    context, sources = build_context(matches)

    MAX_HISTORY = 2

    previous_chat = "\n\n".join([
        f"User:\n{h['question']}\n\nAssistant:\n{h['answer']}"
        for h in history[-MAX_HISTORY:]
    ])

    content = f"""
Previous Conversation
{previous_chat}

==================================================

Documentation
{context}

==================================================

Question
{question}

Answer ONLY from documentation.
"""

    # ==========================================================
    # LLM CALL (WITH RATE LIMIT SAFETY)
    # ==========================================================

    def call_llm():
        return client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ]
        )

    stream = None

    for attempt in range(3):
        try:
            stream = call_llm()
            break
        except RateLimitError:
            print(f"Rate limit hit. Retry {attempt+1}/3...")
            time.sleep(15)
    else:
        return "Rate limit exceeded. Try again later."

    # ==========================================================
    # STREAM RESPONSE
    # ==========================================================

    answer = ""
    print("\nGenerating response...\n")

    for chunk in stream:
        try:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                print(delta.content, end="", flush=True)
                answer += delta.content
        except Exception as e:
            print("Stream error:", e)

    # ==========================================================
    # SOURCES
    # ==========================================================

    answer += "\n\n---\n### Sources\n\n"
    for i, source in enumerate(sources, start=1):
        answer += f"{i}. {source}\n"

    return answer
