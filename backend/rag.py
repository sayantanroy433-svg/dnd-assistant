import os
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone

load_dotenv()

# ---------------- CONFIG ---------------- #

PINECONE_INDEX = os.environ["PINECONE_INDEX"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

NAMESPACE = "markdown-docs"
EMBEDDING_MODEL = "multilingual-e5-large"

TOP_K = 8
MIN_SCORE = 0.75

# ---------------- CLIENTS ---------------- #

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

client = Groq(api_key=GROQ_API_KEY)


# ---------------- RETRIEVAL ---------------- #

def retrieve(question):

    embedding = pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=[question],
        parameters={
            "input_type": "query"
        }
    )

    vector = embedding[0].values

    result = index.query(
        namespace=NAMESPACE,
        vector=vector,
        top_k=TOP_K,
        include_metadata=True
    )

    matches = [
        m for m in result.matches
        if m.score >= MIN_SCORE
    ]

    return matches


# ---------------- CONTEXT ---------------- #

def build_context(matches):

    context = []
    sources = []

    for match in matches:

        metadata = match.metadata

        source = metadata.get("source_file", "Unknown")
        text = metadata.get("chunk_text", "")

        sources.append(source)

        context.append(
            f"""
SOURCE: {source}

{text}
"""
        )

    return "\n\n------------------------\n\n".join(context), sorted(set(sources))


# ---------------- CHAT ---------------- #

def ask(question, history):

    matches = retrieve(question)

    if not matches:
        return "I couldn't find that information in the documentation."

    context, sources = build_context(matches)

    previous_chat = ""

    for item in history:

        previous_chat += f"""
User:
{item['question']}

Assistant:
{item['answer']}

"""

    stream = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        temperature=0.2,

        stream=True,

        messages=[

            {
                "role": "system",
                "content": """
You are a Dungeons & Dragons documentation assistant.

Rules:

- ONLY answer using the supplied documentation.
- Never invent rules.
- If the answer isn't present, say:
'I couldn't find that information in the documentation.'
- Keep answers concise.
- Use Markdown.
- Mention rule names when appropriate.
"""
            },

            {
                "role": "user",
                "content": f"""
Previous Conversation:

{previous_chat}

Documentation:

{context}

Question:

{question}
"""
            }

        ]

    )

    answer = ""

    print()

    for chunk in stream:

        delta = chunk.choices[0].delta.content

        if delta:

            
            answer += delta

    answer += "\n\n### Sources\n"

    for source in sources:
        answer += f"- {source}\n"

    print("\n\nSources:")

    for source in sources:
        print(f"- {source}")

    return answer