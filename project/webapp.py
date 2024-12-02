import gradio as gr
from openai import OpenAI
from agent import Agent
import chromadb

# Initialize the OpenAI client and load flights dataset from your script
BASE_URL = "http://199.94.61.113:8000/v1/"
API_KEY = "tarallo.n@northeastern.edu:OQEgetkQ6LBofgWW4jsC"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


# Create a ChromaDB client
chroma_client = chromadb.PersistentClient(path="project/client")

# Access the existing collection by name
collection = chroma_client.get_collection(name="job_listing_collection")

# Initialize the agent from your script
agent = Agent(client, collection)

with open('project/temp/resume.txt', 'r') as f:
    results = collection.query(
        query_texts='hi',
        n_results=5  # How many results to return
    )

# Function to maintain and get response from the agent
def respond_to_chat(message, chat_history):
    response = agent.say(message)
    print(response)
    return response.text

# Set up Gradio chat interface
iface = gr.ChatInterface(
    respond_to_chat,
    title="LLM Job Helper",
    description="Chat with the agent to find your next great opportunity!",
    examples=[
        # ["Find a flight from Dallas to San Francisco on October 15, 2023"],
        # ["Book the second flight from the previous search"]
    ]
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()