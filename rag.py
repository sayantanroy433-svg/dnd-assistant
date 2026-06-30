import os
import re
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone

from query_parser import parse_query, build_filter

load_dotenv()

# ==========================================================
# CONFIG
# ==========================================================

PINECONE_INDEX = os.environ["PINECONE_INDEX"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

NAMESPACE = "dnddocs"
EMBEDDING_MODEL = "multilingual-e5-large"

TOP_K = 15
MIN_SCORE = 0.75

MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]

# ==========================================================
# CLIENTS
# ==========================================================

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

client = Groq(api_key=GROQ_API_KEY)

# ==========================================================
# LOAD SPELL DATABASE
# ==========================================================

SPELLS = pd.read_csv("Spells.csv")
SPELLS.columns = SPELLS.columns.str.strip().str.lower()

print(f"Loaded {len(SPELLS)} spells.")

# ==========================================================
# SYSTEM PROMPT
# ==========================================================

SYSTEM_PROMPT = """
You are an expert Dungeons & Dragons 5e rules assistant.

Rules:

- Use ONLY the supplied documentation.
- Never invent rules.
- Never rely on outside knowledge.
- If the documentation doesn't contain the answer, say so.
- Keep answers concise.
- Use Markdown formatting.
"""

# ==========================================================
# PINECONE RETRIEVAL
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

    query = {
        "namespace": NAMESPACE,
        "vector": vector,
        "top_k": TOP_K,
        "include_metadata": True
    }

    if metadata_filter:
        query["filter"] = metadata_filter

    result = index.query(**query)

    matches = [
        m for m in result.matches
        if m.score >= MIN_SCORE
    ]

    if not matches and metadata_filter:

        query.pop("filter", None)

        result = index.query(**query)

        matches = [
            m for m in result.matches
            if m.score >= MIN_SCORE
        ]

    return matches

# ==========================================================
# BUILD LLM CONTEXT
# ==========================================================

def build_context(matches):

    blocks = []
    sources = []

    for i, match in enumerate(matches, start=1):

        md = match.metadata

        source = md.get("source_file", "Unknown")
        sources.append(source)

        blocks.append(
f"""
Document #{i}

Source: {source}
Type: {md.get("type")}
Level: {md.get("level")}
Similarity: {match.score:.3f}

{md.get("chunk_text","")}
"""
        )

    return (
        "\n\n----------------------------------------\n\n".join(blocks),
        sorted(set(sources))
    )

# ==========================================================
# QUERY CLASSIFIER
# ==========================================================

def classify_query(question):

    q = question.lower()

    list_words = [
        "list",
        "all",
        "every",
        "spells",
        "cantrips",
        "level",
        "lvl"
    ]

    if any(word in q for word in list_words):
        return "LIST"

    if len(q.split()) <= 3:
        return "LOOKUP"

    return "LLM"
    
# ==========================================================
# PARSE SPELL FILTERS
# ==========================================================

def parse_spell_filters(question):

    q = question.lower()

    level = None
    spell_class = None
    school = None

    # ---------------- LEVEL ---------------- #

    if "cantrip" in q:
        level = "Cantrip"

    else:

        level_map = {
            "1st": "1st",
            "2nd": "2nd",
            "3rd": "3rd",
            "4th": "4th",
            "5th": "5th",
            "6th": "6th",
            "7th": "7th",
            "8th": "8th",
            "9th": "9th",
        }

        for text, value in level_map.items():
            if text in q:
                level = value
                break

        if level is None:

            m = re.search(r"(?:level|lvl)\s*(\d)", q)

            if m:
                n = int(m.group(1))

                suffix = {
                    1: "1st",
                    2: "2nd",
                    3: "3rd"
                }.get(n, f"{n}th")

                level = suffix

    # ---------------- CLASS ---------------- #

    classes = [
        "artificer",
        "barbarian",
        "bard",
        "cleric",
        "druid",
        "fighter",
        "monk",
        "paladin",
        "ranger",
        "rogue",
        "sorcerer",
        "warlock",
        "wizard"
    ]

    for c in classes:
        if re.search(rf"\b{c}\b", q):
            spell_class = c
            break

    # ---------------- SCHOOL ---------------- #

    schools = [
        "abjuration",
        "conjuration",
        "divination",
        "enchantment",
        "evocation",
        "illusion",
        "necromancy",
        "transmutation"
    ]

    for s in schools:
        if re.search(rf"\b{s}\b", q):
            school = s
            break

    return level, spell_class, school


# ==========================================================
# FIND SPELLS
# ==========================================================

def find_spells(level=None, spell_class=None, school=None):

    df = SPELLS.copy()

    if level:

        df = df[
            df["level"]
            .fillna("")
            .str.strip()
            .str.lower()
            ==
            level.lower()
        ]

    if spell_class:

        mask = (
            df["classes"].fillna("").str.contains(spell_class, case=False)
            |
            df["subclasses"].fillna("").str.contains(spell_class, case=False)
            |
            df["optional/variant classes"].fillna("").str.contains(spell_class, case=False)
        )

        df = df[mask]

    if school:

        df = df[
            df["school"]
            .fillna("")
            .str.lower()
            ==
            school.lower()
        ]

    df = df.sort_values("name")

    return df.reset_index(drop=True)


# ==========================================================
# FIND SPELL BY NAME
# ==========================================================

def find_spell_by_name(name):

    spell = SPELLS[
        SPELLS["name"]
        .fillna("")
        .str.lower()
        ==
        name.lower().strip()
    ]

    if spell.empty:
        return None

    return spell.iloc[0]


# ==========================================================
# FORMAT SPELL LIST
# ==========================================================

def format_spell_list(df):

    if df.empty:
        return "No matching spells found."

    output = []

    current_level = None

    for _, row in df.iterrows():

        if row["level"] != current_level:

            current_level = row["level"]

            output.append(f"\n## {current_level}")

        output.append(
            f"- **{row['name']}** ({row['school']})"
        )

    return "\n".join(output)


# ==========================================================
# FORMAT SINGLE SPELL
# ==========================================================

def format_spell(row):

    answer = f"""# {row['name']}

- **Level:** {row['level']}
- **School:** {row['school']}
- **Casting Time:** {row['casting time']}
- **Range:** {row['range']}
- **Duration:** {row['duration']}
- **Classes:** {row['classes']}

### Description

{row['text']}
"""

    higher = str(row.get("at higher levels", "")).strip()

    if higher and higher.lower() != "nan":

        answer += f"""

### At Higher Levels

{higher}
"""

    return answer


# ==========================================================
# LLM FALLBACK
# ==========================================================

def call_llm(messages, temperature=0.2, stream=True):

    last_error = None

    for model in MODELS:

        try:

            print(f"\nTrying model: {model}\n")

            return (
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    stream=stream
                ),
                model
            )

        except Exception as e:

            print(f"Model failed: {model} -> {e}")

            last_error = e

    raise last_error
    
# ==========================================================
# MAIN ASK FUNCTION
# ==========================================================

def ask(question, history):

    mode = classify_query(question)

    print(f"\n[DEBUG] MODE: {mode}")

    # ======================================================
    # SPELL LIST (CSV ONLY)
    # ======================================================

    if mode == "LIST":

        level, spell_class, school = parse_spell_filters(question)

        df = find_spells(
            level=level,
            spell_class=spell_class,
            school=school
        )

        return format_spell_list(df), ["Spells.csv"]

    # ======================================================
    # SINGLE SPELL LOOKUP (CSV ONLY)
    # ======================================================

    if mode == "LOOKUP":

        spell = find_spell_by_name(question)

        if spell is not None:

            return (
                format_spell(spell),
                ["Spells.csv"]
            )

    # ======================================================
    # PINECONE RETRIEVAL
    # ======================================================

    matches = retrieve(question)

    if not matches:

        return (
            "I couldn't find that information in the documentation.",
            []
        )

    context, sources = build_context(matches)

    # ======================================================
    # LLM
    # ======================================================

    stream, used_model = call_llm(

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
Documentation

{context}

Question

{question}
"""
            }
        ],

        temperature=0.2,
        stream=True
    )

    print(f"\nModel used: {used_model}\n")

    answer = ""

    for chunk in stream:

        try:

            delta = chunk.choices[0].delta

            if delta and delta.content:

                print(delta.content, end="", flush=True)
                answer += delta.content

        except Exception as e:

            print("Stream error:", e)

    if sources:

        answer += "\n\n---\n### Sources\n"

        for src in sources:
            answer += f"- {src}\n"

    return answer, sources
