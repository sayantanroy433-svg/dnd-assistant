import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key="pcsk_2J1n2y_2GERKijKxDfndAYVGApkPduPWFKBy83abPgN3VA99qxzfresLNrZopMT1bGAreq")

index = pc.Index("dnd-index")

index.delete(
    namespace="csv-docs",
    delete_all=True
)

print("✅ Namespace cleared successfully!")