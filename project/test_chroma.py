import chromadb

# Create a ChromaDB client
chroma_client = chromadb.Client()

# Create a collection
collection = chroma_client.create_collection(name="my_collection")

# Add documents to the collection
collection.add(
    documents=["This is a document about pineapple", "This is a document about oranges"],
    ids=["id1", "id2"]
)

# Query the collection
results = collection.query(
    query_texts=["This is a query document about hawaii"],
    n_results=2  # How many results to return
)

print(results)

# Process and print the results
for i, doc_id in enumerate(results['ids'][0]):
    doc_text = results['documents'][0][i]
    distance = results['distances'][0][i]
    print(f"ID: {doc_id}, Text: {doc_text}, Distance: {distance}")

