import chromadb
from datasets import Dataset

listings = Dataset.from_json(
    'job_listings.jsonl'
)
resumes = Dataset.from_json(
    'resumes.jsonl'
)

documents = []
metadatas = []
ids = []
for listing in listings:
    ids.append(str(listing['id']))
    documents.append(listing['description'])
    metadatas.append({
        'title': listing['title'],
        'company': listing['company'],
    })


# Create a ChromaDB client
chroma_client = chromadb.Client()

# Create a collection
collection = chroma_client.create_collection(name="job_listing_collection")

# Add documents to the collection
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
)

results = collection.query(
    query_texts=resumes[0]['resume'],
    n_results=5  # How many results to return
)

for i, doc_id in enumerate(results['ids'][0]):
    doc_text = results['documents'][0][i]
    distance = results['distances'][0][i]
    print(f"ID: {doc_id}, Distance: {distance}")