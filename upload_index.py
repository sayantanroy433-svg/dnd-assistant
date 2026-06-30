import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pinecone import Pinecone

# ============================================
# CONFIGURATION
# ============================================

API_KEY = "pcsk_2J1n2y_2GERKijKxDfndAYVGApkPduPWFKBy83abPgN3VA99qxzfresLNrZopMT1bGAreq"

INDEX_NAME = "dnd-index"

NAMESPACE = "dnddocs"

MODEL_ENGINE = "multilingual-e5-large"

BATCH_SIZE = 5
MAX_WORKERS = 10
MAX_TEXT_CHARS = 4000

# ============================================
# INITIALIZE PINECONE
# ============================================

pc = Pinecone(api_key=API_KEY)

index = pc.Index(INDEX_NAME)

# ============================================
# EXTRACT TITLE
# ============================================

def extract_title(chunk):

    match = re.search(r"^#\s+(.*)", chunk, re.MULTILINE)

    if match:
        return match.group(1).strip()

    return "Untitled"

# ============================================
# CHUNK MARKDOWN
# ============================================

def chunk_markdown_by_headers(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    header_pattern = re.compile(
        r'(^#{1,6}\s+.*$)',
        re.MULTILINE
    )

    tokens = header_pattern.split(content)

    chunks = []

    current_chunk = ""

    for token in tokens:

        if not token.strip():
            continue

        if header_pattern.match(token):

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            current_chunk = token + "\n"

        else:

            current_chunk += token

    if current_chunk.strip():

        chunks.append(current_chunk.strip())

    final_chunks = []

    for chunk in chunks:

        if len(chunk) <= MAX_TEXT_CHARS:

            final_chunks.append(chunk)

        else:

            print(
                f"Large section detected ({len(chunk)} chars)"
            )

            for i in range(
                0,
                len(chunk),
                MAX_TEXT_CHARS
            ):

                final_chunks.append(
                    chunk[i:i + MAX_TEXT_CHARS]
                )

    return final_chunks

# ============================================
# LOAD DOCUMENTS
# ============================================

def load_documents(folder):

    documents = []

    for filename in os.listdir(folder):

        if not filename.endswith(".md"):
            continue

        print(f"Reading {filename}")

        path = os.path.join(folder, filename)

        chunks = chunk_markdown_by_headers(path)

        for idx, chunk in enumerate(chunks):

            title = extract_title(chunk)

            documents.append({

                "id": f"{filename}-{idx}",

                "text": chunk,

                "metadata": {

                    "type": "rule",

                    "title": title,

                    "source_file": filename,

                    "chunk_index": idx,

                    "chunk_text": chunk

                }

            })

    return documents

# ============================================
# CREATE EMBEDDINGS
# ============================================

def create_batches(documents):

    batches = []

    for i in range(
        0,
        len(documents),
        BATCH_SIZE
    ):

        batch = documents[i:i+BATCH_SIZE]

        texts = [
            doc["text"]
            for doc in batch
        ]

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
# UPLOAD FUNCTIONS
# ============================================

def upload_batch(batch, batch_number):

    try:

        index.upsert(
            namespace=NAMESPACE,
            vectors=batch
        )

        print(f"✅ Batch {batch_number} uploaded successfully.")

        return True

    except Exception as e:

        print(f"❌ Batch {batch_number} failed.")
        print(e)

        return False


# ============================================
# MAIN UPLOAD PIPELINE
# ============================================

def upload_documents(folder):

    documents = load_documents(folder)

    if not documents:

        print("No markdown files found.")

        return

    print(f"\nLoaded {len(documents)} chunks.")

    batches = create_batches(documents)

    print(f"Created {len(batches)} batches.")

    print(f"\nUploading using {MAX_WORKERS} workers...\n")

    success = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {

            executor.submit(
                upload_batch,
                batch,
                i + 1
            ): i

            for i, batch in enumerate(batches)

        }

        for future in as_completed(futures):

            if future.result():

                success += 1

    print("\n======================================")

    print("UPLOAD COMPLETE")

    print("======================================")

    print(f"Successful batches : {success}")

    print(f"Total batches      : {len(batches)}")

    print(f"Namespace          : {NAMESPACE}")

    print("======================================\n")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    MARKDOWN_FOLDER = r"C:\Users\sayan\Desktop\dnd_bot"

    upload_documents(MARKDOWN_FOLDER)