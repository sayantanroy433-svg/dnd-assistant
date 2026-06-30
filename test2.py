from pinecone import Pinecone

API_KEY = "pcsk_2J1n2y_2GERKijKxDfndAYVGApkPduPWFKBy83abPgN3VA99qxzfresLNrZopMT1bGAreq"

pc = Pinecone(api_key=API_KEY)

index = pc.Index("dnd-index")

stats = index.describe_index_stats()

print(stats.namespaces)