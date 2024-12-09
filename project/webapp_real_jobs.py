import gradio as gr
from openai import OpenAI
from agent import Agent
import chromadb

# Initialize the OpenAI client and load flights dataset from your script
BASE_URL = "http://199.94.61.113:8000/v1/"
API_KEY = "tarallo.n@northeastern.edu:OQEgetkQ6LBofgWW4jsC"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Create a ChromaDB client
chroma_client = chromadb.PersistentClient(path="project/real_jobs_client")

# Access the existing collection by name
collection = chroma_client.get_collection(name="job_listing_collection_5000_real_jobs")

print(collection.count)

# Initialize the agent from your script
agent = Agent(client, collection)

# Function to handle file upload and load resume
def load_resume(file):
    with open(file.name, 'r') as f:
        resume_content = f.read()
    agent.set_resume(resume_content)
    return "Resume uploaded successfully!"

def clear_resume():
    agent.set_resume(None)
    return "Resume deleted."

# Function to maintain and get response from the agent
def respond_to_chat(message, chat_history):
    response = agent.say(message)
    print(response)
    return response.text

print(gr.__version__)
# Set up Gradio chat interface with an additional File input component
with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("# LLM Job Helper")
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload your resume", file_types=['text'])
        with gr.Column():
            file_output = gr.Textbox(label="File upload status"), # height=100)
            
    chat_box = gr.ChatInterface(
        respond_to_chat,
        # title="Chat with the agent",
        description="Find your next great opportunity!",
    )

    def handle_file_upload(file): 
        return load_resume(file) 
    def handle_file_clear(file): 
        return clear_resume() 
    file_input.upload(handle_file_upload, file_input, file_output) 
    file_input.clear(handle_file_clear, file_input, file_output)

    # File upload interface
    
    
    # Launch the combined interface
    # chat_box.render()
    # file_input.render()
    # file_output.render()
    # demo.render()

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
