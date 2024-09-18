from openai import OpenAI

URL = "http://199.94.61.113:8000/v1/"
KEY = "tarallo.n@northeastern.edu:OQEgetkQ6LBofgWW4jsC"

client = OpenAI(base_url=URL, api_key=KEY)

from typing import List, Optional

# TASK 1
def prompt_zero_shot(problem: str) -> str:
    # Your code here
    pass

def extract_zero_shot(completion: str) -> Optional[int]:
    # Your code here
    pass

def solve_zero_shot(problem: str) -> Optional[int]:
    resp = client.completions.create(
        model="meta-llama/Llama-3.1-8B",
        temperature=0.2,
        prompt=prompt_zero_shot(problem)
    )
    return extract_zero_shot(resp.choices[0].text)


# TASK 2
def accuracy_zero_shot(problems: List[dict]) -> float:
    # Your code here
    pass


# TASK 4
def prompt_few_shot(problem: str) -> str:
    # Your code here
    pass

def extract_few_shot(completion: str) -> Optional[int]:
    # Your code here
    pass

def solve_few_shot(problem: str) -> Optional[int]:
    # Your code here
    pass

def accuracy_few_shot(problems: List[dict]) -> float:    
    # Your code here
    pass


# TASK 5
def prompt_cot(problem: str) -> str:
    # Your code here
    pass

def extract_cot(completion: str) -> Optional[int]:
    # Your code here
    pass

def solve_cot(problem: str) -> Optional[int]:
    # Your code here
    pass

def accuracy_cot(problems: List[dict]) -> float:    
    # Your code here
    pass


# TASK 6:
def prompt_pal(problem: str) -> str:
    # Your code here
    pass

def extract_pal(completion: str) -> Optional[int]:
    # Your code here. Use exec and eval.
    pass

def solve_pal(problem: str) -> Optional[int]:
    # Your code here
    pass

def accuracy_pal(problems: List[dict]) -> float:    
    # Your code here
    pass


