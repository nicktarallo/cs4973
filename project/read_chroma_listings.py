import chromadb
from datasets import Dataset

resumes = Dataset.from_json(
    'project/resumes.jsonl'
)
# Create a ChromaDB client
chroma_client = chromadb.PersistentClient(path="project/client")

# Access the existing collection by name
collection = chroma_client.get_collection(name="job_listing_collection")

results = collection.query(
    query_texts=resumes[12]['resume'],
    n_results=5  # How many results to return
)

print(results)

for i, doc_id in enumerate(results['ids'][0]):
    doc_text = results['documents'][0][i]
    distance = results['distances'][0][i]
    metadata = results['metadatas'][0][i]
    print(f"ID: {doc_id}, Distance: {distance}, Metadata: {metadata}")
