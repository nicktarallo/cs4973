from openai import OpenAI
from datasets import Dataset
import re

BASE_URL = "http://199.94.61.113:8000/v1/"
API_KEY = "tarallo.n@northeastern.edu:OQEgetkQ6LBofgWW4jsC"
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def generate_system_prompt(title, company, description):
    PROMPT = f"""
Job Title: {title}
Job Company: {company}
Job Description: {description}

INSTRUCTIONS: Generate a resume for a person that would be a good fit with this job description based on skills and experience level. Use markdown format. Please include only the resume, no other additional words in your response. Use real companies/schools in the experience section.
"""
    return PROMPT
    
ids = []
associated_listing_ids = []
resumes = []

listings = Dataset.from_json(
    'job_listings.jsonl'
)

resume_id = 0
for listing in listings:
    conversation = [{
        'role': 'user',
        'content': generate_system_prompt(listing['title'], listing['company'], listing['description'])
    }]
    resp = client.chat.completions.create(
        messages=conversation,
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        temperature=0.5,
    ).choices[0].message.content
    
    ids.append(resume_id)
    associated_listing_ids.append(listing['id'])
    resumes.append(resp)
    
    print(resp)
    print(resume_id)

    resume_id += 1
            


dataset_dict = {'id': ids, 'associated_job_id': associated_listing_ids, 'resume': resumes}
dataset = Dataset.from_dict(dataset_dict)
dataset.to_json('resumes.jsonl')