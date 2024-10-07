import gradio as gr
from openai import OpenAI
from typing import List
from datasets import load_dataset
from datetime import date
import dataclasses
from travel_agent import Agent, FindFlightsResponse, BookFlightResponse, load_flights_dataset

# Initialize the OpenAI client and load flights dataset from your script
BASE_URL = "http://199.94.61.113:8000/v1/"
API_KEY = "tarallo.n@northeastern.edu:OQEgetkQ6LBofgWW4jsC"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
flights = load_flights_dataset()

# Initialize the agent from your script
agent = Agent(client, flights)

# Function to maintain and get response from the agent
def respond_to_chat(message, chat_history):
    response = agent.say(message)
    return response.text

# Set up Gradio chat interface
iface = gr.ChatInterface(
    respond_to_chat,
    title="LLM Travel Agent",
    description="Chat with Thomas, your travel assistant. Ask any travel-related questions or make bookings.",
    examples=[
        ["Find a flight from New York to Los Angeles on Dec 20, 2023"],
        ["Book the second flight from the previous search"]
    ]
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()