from utils.rag_pipeline import RAGPipeline

rag = RAGPipeline()

question = "How can I reduce anxiety before public speaking?"

result = rag.ask(question)

print("\n==============================")
print("Question")
print("==============================")
print(question)

print("\n==============================")
print("Answer")
print("==============================")
print(result["answer"])

print("\n==============================")
print("Retrieved Sources")
print("==============================")

for source in result["sources"]:
    print(f"{source['source']} | Page {source['page']} | Similarity {source['similarity']}%")