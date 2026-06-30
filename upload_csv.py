import os
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pinecone import Pinecone

# ============================================
# CONFIGURATION
# ============================================

API_KEY = "pcsk_2J1n2y_2GERKijKxDfndAYVGApkPduPWFKBy83abPgN3VA99qxzfresLNrZopMT1bGAreq"

INDEX_NAME = "dnd-index"

# Use the SAME namespace as your backend
NAMESPACE = "dnddocs"

MODEL_ENGINE = "multilingual-e5-large"

BATCH_SIZE = 50
MAX_WORKERS = 10

# ============================================
# PINECONE
# ============================================

pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)

# ============================================
# HELPERS
# ============================================

def clean(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def parse_level(value):

    value = clean(value)

    if value == "":
        return 0

    m = re.search(r"\d+", value)

    if m:
        return int(m.group())

    return 0


def parse_classes(value):

    value = clean(value)

    if value == "":
        return []

    classes = []

    for part in value.split(","):

        part = part.strip()

        part = re.sub(r"\s*\(.*?\)", "", part)

        if part:
            classes.append(part)

    return classes


def build_spell_text(row):

    fields = [
        "Name",
        "Level",
        "School",
        "Casting Time",
        "Range",
        "Duration",
        "Components",
        "Classes",
        "Subclasses",
        "Text",
        "At Higher Levels"
    ]

    lines = []

    for field in fields:

        if field in row:

            value = clean(row[field])

            if value:

                lines.append(f"{field}: {value}")

    return "\n\n".join(lines)

# ============================================
# LOAD CSV FILES
# ============================================

def load_documents(folder):

    documents = []

    for filename in os.listdir(folder):

        if not filename.lower().endswith(".csv"):
            continue

        path = os.path.join(folder, filename)

        print(f"\nReading {filename}")

        df = pd.read_csv(path)

        for i, row in df.iterrows():

            row = row.fillna("")

            text = build_spell_text(row)

            file_lower = filename.lower()

            if "spell" in file_lower:
                data_type = "spell"
            elif "monster" in file_lower:
                data_type = "monster"
            elif "item" in file_lower:
                data_type = "item"
            elif "feat" in file_lower:
                data_type = "feat"
            else:
                data_type = "document"

            metadata = {

                "type": data_type,

                "name": clean(row.get("Name")),

                "level": parse_level(row.get("Level")),

                "school": clean(row.get("School")),

                "classes": parse_classes(row.get("Classes")),

                "casting_time": clean(row.get("Casting Time")),

                "duration": clean(row.get("Duration")),

                "range": clean(row.get("Range")),

                "source": clean(row.get("Source")),

                "source_file": filename,

                "chunk_text": text
            }

            documents.append({

                "id": f"{filename}-{i}",

                "text": text,

                "metadata": metadata

            })

    return documents

# ============================================
# EMBEDDINGS
# ============================================

def create_batches(documents):

    batches = []

    for i in range(0, len(documents), BATCH_SIZE):

        batch = documents[i:i+BATCH_SIZE]

        texts = [d["text"] for d in batch]

        embeddings = pc.inference.embed(

            model=MODEL_ENGINE,

            inputs=texts,

            parameters={
                "input_type": "passage"
            }

        )

        vectors = []

        for j, doc in enumerate(batch):

            vectors.append({

                "id": doc["id"],

                "values": embeddings[j].values,

                "metadata": doc["metadata"]

            })

        batches.append(vectors)

    return batches

# ============================================
# UPLOAD
# ============================================

def upload_batch(batch, number):

    try:

        index.upsert(

            namespace=NAMESPACE,

            vectors=batch

        )

        print(f"Batch {number} uploaded.")

        return True

    except Exception as e:

        print(e)

        return False


def upload(folder):

    documents = load_documents(folder)

    print(f"\nLoaded {len(documents)} documents.")

    batches = create_batches(documents)

    print(f"Uploading {len(batches)} batches...\n")

    success = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {

            executor.submit(upload_batch, batch, i+1): i

            for i, batch in enumerate(batches)

        }

        for future in as_completed(futures):

            if future.result():

                success += 1

    print()

    print(f"Finished.")

    print(f"{success}/{len(batches)} batches uploaded.")

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    folder = r"C:\Users\sayan\Desktop\dnd_bot"

    upload(folder)