from utils.rag_pipeline import RAGPipeline

rag = RAGPipeline()

stats = rag.build_index()

print("\nIndex built successfully!")
print(stats)