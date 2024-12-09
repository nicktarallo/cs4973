from openai import OpenAI
from datasets import Dataset
import re

industries = [
    'software',
    'food service',
    'entertainment',
    'music',
    'sports',
    'education',
    'retail',
    'accounting',
    'consulting',
    'medical',
    'insurance',
    'manufacturing',
    'auto',
    'transportation',
    'real estate',
    'mining',
    'banking',
    'data science',
    'hospitality',
    'farming',
]

BASE_URL = "http://199.94.61.113:8000/v1/"
API_KEY = "tarallo.n@northeastern.edu:OQEgetkQ6LBofgWW4jsC"
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def generate_system_prompt(industry):
    PROMPT = f"""Generate a job listing that corresponds with the following industry: {industry}
You can use real company names.
    
The listing should be in the following format DO NOT USE MARKDOWN AROUND "Title:", "Company:", or "Description:"

Title: [Job Title]
Company: [Company]
Description: [Job Description (Should be multiple lines, and should be formatted in markdown)]"""
    return PROMPT
    
ids = []
titles = []
companies = []
descriptions = []

job_id = 0
for i in range(100):  # Adjust this range as needed
    for industry in industries:
        conversation = [{
            'role': 'user',
            'content': generate_system_prompt(industry)
        }]
        resp = client.chat.completions.create(
            messages=conversation,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            temperature=0.7,
        ).choices[0].message.content
        
        try:
            title = resp.split("Title:")[1].split("Company:")[0].strip()
            company = resp.split("Company:")[1].split("Description:")[0].strip()
            description = resp.split("Description:")[1].strip()

            print(resp)

            print(title)
            print(company)
            print(description)
            
            titles.append(title)
            companies.append(company)
            descriptions.append(description)
            ids.append(job_id)
        except:
            print(f"Skipped {job_id} {industry}")
        print(job_id)
        job_id += 1
            


dataset_dict = {'id': ids, 'title': titles, 'company': companies, 'description': descriptions}
dataset = Dataset.from_dict(dataset_dict)
dataset.to_json('job_listings_2000.jsonl')