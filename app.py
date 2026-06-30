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

    print("\nAssistant:")

    answer, sources = ask(q, history)


    history.append({
        "question": q,
        "answer": answer
    })

    history = history[-5:]

    print()
