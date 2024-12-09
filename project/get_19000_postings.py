import pandas as pd
import kagglehub

# Download latest version
path = kagglehub.dataset_download("madhab/jobposts")

print("Path to dataset files:", path)

# Load the CSV file into a DataFrame
# Adjust the filename as needed based on your specific dataset
csv_file = f"{path}/data job posts.csv"  # Replace 'file_name.csv' with the actual filename
df = pd.read_csv(csv_file)


import chromadb
from datasets import Dataset

resumes = Dataset.from_json(
    'project/resumes.jsonl'
)

documents = []
metadatas = []
ids = []
for index, listing in df.iterrows():
    if index == 5000:
        break
    ids.append(str(index))
    documents.append(listing['jobpost'])
    metadatas.append({
        'title': listing['Title'],
        'company': listing['Company'],
    })


# Create a ChromaDB client
chroma_client = chromadb.PersistentClient(path="project/client")

# Create a collection
collection = chroma_client.create_collection(name="job_listing_collection_5000_real_jobs", metadata={"hnsw:batch_size":10000})

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
    metadata = results['metadatas'][0][i]
    print(f"ID: {doc_id}, Distance: {distance}, Metadata: {metadata}")