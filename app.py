from rag import ask

history = []

print("========================================")
print("     D&D Documentation Assistant")
print("========================================")
print("Type 'exit' to quit.\n")

while True:

    q = input("You: ")

    if q.lower() in ["exit", "quit"]:
        break

    print("\nAssistant:", end=" ", flush=True)

    answer = ask(q, history)

    history.append({
        "question": q,
        "answer": answer
    })

    # Keep only the last 5 exchanges
    history = history[-5:]

    print("\n")