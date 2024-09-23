from openai import OpenAI
from tqdm.auto import tqdm

URL = "http://199.94.61.113:8000/v1/"
KEY = "tarallo.n@northeastern.edu:OQEgetkQ6LBofgWW4jsC"

client = OpenAI(base_url=URL, api_key=KEY)

from typing import List, Optional

# TASK 1

def prompt_zero_shot(problem: str) -> str:
    return problem + "\n\nAnswer without units ="

def extract_zero_shot(completion: str) -> Optional[int]:
    try:
        return int(completion.strip())
    except:
        print(completion)
        return None

def solve_zero_shot(problem: str) -> Optional[int]:
    resp = client.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B",
        temperature=0.2,
        prompt=prompt_zero_shot(problem),
        max_tokens=2
    )
    return extract_zero_shot(resp.choices[0].text)

def accuracy_zero_shot(problems: List[dict]) -> float:
    num_correct = 0
    for problem in tqdm(problems):
        for i in range(5):
            result = solve_zero_shot(problem['question'])
            if result == problem['answer']:
                num_correct += 1
            else:
                print(f'Expected: {problem["answer"]}')
                print(f'Got: {result}')

    return num_correct / (5 * len(problems))


# TASK 2
def prompt_few_shot(problem: str) -> str:
    EXAMPLES = """
        John had 9 chocolate bars and ate 3 of them. How many does he have left?

        Answer without units = 6



        Alan has 4 cats and 3 dogs. How many total pets does he have?
        
        Answer without units = 7
    
        

        Katie has five pencils. Rebecca has three times as many. How many pencils does Rebecca have?
        
        Answer without units = 15



    """
    PROMPT = EXAMPLES + problem + '\n\nAnswer without units ='
    return PROMPT
    

# TASK 4
def extract_few_shot(completion: str) -> Optional[int]:
    try:
        return int(completion.strip())
    except:
        print(completion)
        return None

def solve_few_shot(problem: str) -> Optional[int]:
    resp = client.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B",
        temperature=0.2,
        prompt=prompt_few_shot(problem),
        max_tokens=2
    )
    return extract_few_shot(resp.choices[0].text)

def accuracy_few_shot(problems: List[dict]) -> float:    
    num_correct = 0
    for problem in tqdm(problems):
        for i in range(5):
            result = solve_few_shot(problem['question'])
            if result == problem['answer']:
                num_correct += 1
            else:
                print(f'Expected: {problem["answer"]}')
                print(f'Got: {result}')

    return num_correct / (5 * len(problems))


# TASK 5
def prompt_cot(problem: str) -> str:
    EXAMPLES = """Input: Sarah has 15 candies. She wants to share them equally between herself and 4 friends. How many candies will each person get?
 
    Reasoning: Sarah has 15 candies, and she needs to divide them equally among herself and 4 friends, which makes 5 people in total. To find out how many candies each person gets, we divide the total number of candies by the number of people. 15 ÷ 5 = 3 candies per person. So, each person gets 3 candies.

    Answer without units: 3

    Done
    

    Input: A bus can seat 40 people. If 5 buses are filled with passengers and there are 10 more people waiting to board, how many people in total need transportation?

    Reasoning: Each bus can hold 40 people. There are 5 buses, so the total number of people already seated in the buses is 5 * 40 = 200 people. Now, there are 10 more people waiting, so the total number of people that need transportation is the sum of those already seated and those waiting. Therefore, there are 200 + 10 = 210 people in total.

    Answer without units: 210

    Done


    Input: A library has 6 shelves. Each shelf holds 24 books. If the librarian adds 12 more books to one shelf, how many books are there in total now?
  
    Reasoning: The library initially has 6 shelves with 24 books on each shelf. The total number of books in the library before any additions is 6 * 24 = 144 books. Now, the librarian adds 12 more books to one shelf. So, the new total number of books becomes 144 + 12 = 156 books.

    Answer without units: 156

    Done

    

    """
    PROMPT = EXAMPLES + "Input: " + problem + "\n\nReasoning:"
    return PROMPT


def extract_cot(completion: str) -> Optional[int]:
    items = completion.split("Answer without units: ")
    if len(items) < 2:
        return None
    try:
        return int(items[1].strip())
    except:
        return None

def solve_cot(problem: str) -> Optional[int]:
    resp = client.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B",
        temperature=0.2,
        prompt=prompt_cot(problem),
        max_tokens=200,
        stop=['Done'],
    )
    return extract_cot(resp.choices[0].text)

def accuracy_cot(problems: List[dict]) -> float:    
    num_correct = 0
    for problem in tqdm(problems):
        for i in range(5):
            result = solve_cot(problem['question'])
            if result == problem['answer']:
                num_correct += 1
            else:
                print(f'Expected: {problem["answer"]}')
                print(f'Got: {result}')

    return num_correct / (5 * len(problems))


# TASK 6:
def prompt_pal(problem: str) -> str:
    return f"""# Question: Sarah has 15 candies. She wants to share them equally between herself and 4 friends. How many candies will each person get?

def question():
    # Sarah has 15 candies
    total_candies = 15
    # The total number of people is herself plus four of her friends
    num_people = 1 + 4
    # The amount of candies each person gets, which is the answer, is
    candies_per_person = total_candies // num_people
    return candies_per_person



# Question: A bus can seat 40 people. If 5 buses are filled with passengers and there are 10 more people waiting to board, how many people in total need transportation?

def question():
    # Each bus can hold 40 people
    bus_capacity = 40
    # There are 5 buses
    num_buses = 5
    # The amount of passengers already seated is
    passengers_on_buses = num_buses * bus_capacity
    # There are 10 people waiting to board
    waiting_people = 10
    # The total amount of people, which is the answer, is
    total_people = passengers_on_buses + waiting_people
    return total_people



# Question: A library has 6 shelves. Each shelf holds 24 books. If the librarian adds 12 more books to one shelf, how many books are there in total now?

def question():
    # The library initially has 6 shelves
    shelves = 6
    # The library initially has 24 books on each shelf
    books_per_shelf = 24
    # The total number of books on shelves is initially
    total_books = shelves * books_per_shelf
    # 12 books were added to a shelf
    added_books = 12
    # The total number of books on shelves at the end, which is the answer, is
    total_books_now = total_books + added_books
    return total_books_now 


# Question: {problem}

def question():"""

def extract_pal(completion: str) -> Optional[int]:
    CODE = "def question():\n    " + completion.strip()
    try:
        # print(CODE)
        exec(CODE)
        result = question()
        return int(result)
    except:
        print(CODE)
        return None

def solve_pal(problem: str) -> Optional[int]:
    resp = client.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B",
        temperature=0.2,
        prompt=prompt_pal(problem),
        max_tokens=300,
        stop=['# Question:'],
    )
    return extract_pal(resp.choices[0].text)
    

def accuracy_pal(problems: List[dict]) -> float:    
    num_correct = 0
    for problem in tqdm(problems):
        for i in range(5):
            result = solve_pal(problem['question'])
            if result == problem['answer']:
                num_correct += 1
            else:
                print(f'Expected: {problem["answer"]}')
                print(f'Got: {result}')

    return num_correct / (5 * len(problems))


