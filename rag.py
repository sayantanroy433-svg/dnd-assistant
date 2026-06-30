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

NAMESPACE = "markdown-docs"

EMBEDDING_MODEL = "multilingual-e5-large"

TOP_K = 8
MIN_SCORE = 0.75

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
        parameters={
            "input_type": "query"
        }
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

    print("\n========== RETRIEVAL ==========")
    print("Question:", question)
    print("Parsed:", parsed)
    print("Metadata Filter:", metadata_filter)

    # ---------------- First Search ---------------- #

    result = index.query(**query_args)

    matches = [
        m for m in result.matches
        if m.score >= MIN_SCORE
    ]

    print(f"Filtered Results: {len(matches)}")

    # ---------------- Fallback ---------------- #

    if not matches and metadata_filter:

        print("No filtered matches.")
        print("Falling back to semantic search...")

        query_args.pop("filter", None)

        result = index.query(**query_args)

        matches = [
            m for m in result.matches
            if m.score >= MIN_SCORE
        ]

        print(f"Fallback Results: {len(matches)}")

    print("===============================\n")

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

        book = md.get("book", "")
        section = md.get("section", "")
        page = md.get("page", "")
        entity_type = md.get("type", "")

        sources.append(source)

        block = f"""
Document #{i}

Source: {source}
Book: {book}
Section: {section}
Page: {page}
Type: {entity_type}
Similarity Score: {match.score:.3f}

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

    # ---------------- Retrieve ---------------- #

    parsed, matches = retrieve(question)

    print("\n========== CHAT ==========")
    print("Parsed Query:")
    print(parsed)

    print(f"\nRetrieved Chunks: {len(matches)}")

    if not matches:

        return (
            "I couldn't find that information in the documentation.\n\n"
            "Try:\n"
            "- using the exact spell, monster, class feature, or item name\n"
            "- mentioning the source book (PHB, DMG, MM, XGtE, etc.)\n"
            "- asking a more specific question"
        )

    # ---------------- Build Context ---------------- #

    context, sources = build_context(matches)

    # ---------------- Conversation History ---------------- #

    MAX_HISTORY = 5

    conversation = []

    for item in history[-MAX_HISTORY:]:

        conversation.append(
            f"""User:
{item["question"]}

Assistant:
{item["answer"]}"""
        )

    previous_chat = "\n\n".join(conversation)

    # ---------------- Prompt ---------------- #

    content = f"""
Previous Conversation

{previous_chat}

==================================================

Documentation

{context}

==================================================

Question

{question}

==================================================

Instructions

Answer ONLY using the supplied documentation.

If multiple documents contain useful information,
combine them into a single answer.

Never invent rules.

If the answer cannot be found in the documentation,
say exactly:

"I couldn't find that information in the documentation."

Do not mention document numbers.

Use Markdown formatting.
"""

    # ---------------- LLM ---------------- #

    stream = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        temperature=0.2,

        stream=True,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": content
            }
        ]

    )

    # ---------------- Stream Response ---------------- #

    answer = ""

    print("\nGenerating response...\n")

    for chunk in stream:

        delta = chunk.choices[0].delta.content

        if delta:
            answer += delta

    # ---------------- Sources ---------------- #

    answer += "\n\n---\n"
    answer += "### Sources\n\n"

    for i, source in enumerate(sources, start=1):
        answer += f"{i}. {source}\n"

    print("\nSources Used:")

    for source in sources:
        print(f"- {source}")

    print("\n==========================\n")

    return answer