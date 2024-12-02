from openai import OpenAI
from typing import List
from datasets import load_dataset
from datetime import date, time, datetime
import dataclasses
from typing import List, Optional
import yaml
import statistics
from pathlib import Path
import chromadb

BASE_URL = "http://199.94.61.113:8000/v1/"
API_KEY = "tarallo.n@northeastern.edu:OQEgetkQ6LBofgWW4jsC"

temp_path = 'project/temp'

@dataclasses.dataclass
class Application:
    pass

@dataclasses.dataclass
class Listing:
    id: int
    title: str
    description: str
    company: str
    applications: List[Application] = None

    # def matches_search(self, origins: str, destinations: str, start_date: date, end_date: date) -> bool:
    #     return self.origin in origins and self.destination in destinations and self.date >= start_date and self.date <= end_date
    
    # def matches_id(self, id: int) -> bool:
    #     return self.id == id
    
    # def book_flight(self) -> Optional[int]:
    #     if self.available_seats > 0:
    #         self.available_seats -= 1
    #         return self.id
    #     return None
    
    def format_main_details(self):
        return f"""Company: {self.company}
Job Title: {self.title}
Job ID: {self.id}"""

# def parse_flight(flight):
#     return Flight(
#         id=flight["id"],
#         date=datetime.strptime(flight["date"], "%Y-%m-%d").date(),
#         airline=flight["airline"],
#         flight_number=flight["flight_number"],
#         origin=flight["origin"],
#         destination=flight["destination"],
#         departure_time=datetime.strptime(flight["departure_time"], "%H:%M").time(),
#         arrival_time=datetime.strptime(flight["arrival_time"], "%H:%M").time(),
#         available_seats=flight["available_seats"],
#     )


# def load_flights_dataset() -> List[Flight]:
#     return [
#         parse_flight(flight)
#         for flight in load_dataset("nuprl/llm-systems-flights", split="train")
#     ]


@dataclasses.dataclass
class AgentResponse:
    """
    The superclass for all agent responses.
    """
    text: str

@dataclasses.dataclass
class FindJobsResponse(AgentResponse):
    jobs_found: List[Listing]

@dataclasses.dataclass
class SkillsGapResponse(AgentResponse):
    pass

@dataclasses.dataclass
class CoverLetterResponse(AgentResponse):
    pass

@dataclasses.dataclass
class TextResponse(AgentResponse):
    pass

SYSTEM_PROMPT = """You help people with their job search. If the user asks to find jobs based on their resume, respond with code that uses this function:

```python
def find_jobs(self):
    ...
```

If the user asks to do a skills gap analysis for a specific job, respond with code that uses this function:

```python
def perform_skills_analysis(self, job_id: int):
    ...
```

If the user asks you to write a cover letter for a specific job, respond with code that uses this function:

```python
def write_cover_letter(self, job_id: int):
    ...
```

Otherwise, just respond with text

When responding with code, make sure to set the variable `result` equal to the return value of the function call.

If you respond with code, do NOT include any other text outside of the code.

In any code block you write, assume result is originally set to None.

Make sure if you respond with code to use markdown backticks with python.

YOU DO NOT NEED TO DEFINE THOSE SPECIFIED FUNCTIONS, THEY ALREADY EXIST. ALSO, IF THEY ARE DOING ONE OF THOSE TASKS, MAKE SURE YOU RESPOND WITH CODE, DON'T TRY TO MAKE UP JOBS."""

USER_1 = "Find me some jobs that would fit me based on my resume."
ASSISTANT_1 = """```python
result = self.find_jobs()
```"""
USER_2 = """Company: Company A
Job Title: Software Engineer
Job ID: 87

Company: Company X
Job Title: Junior Software Engineer
Job ID: 92

Perform a skills gap analysis for the first one please."""
ASSISTANT_2 = """```python
result = self.perform_skills_analysis(87)
```"""
USER_3 = """Now write me a cover letter for that job"""
ASSISTANT_3 = """```python
result = self.write_cover_letter(87)
```"""

UPLOAD_RESUME_TEXT = 'Please upload your resume.'

class Agent:

    # The complete conversation with the LLM, including the system prompt.
    conversation: List[dict]
    # The formatted response from the last tool call.
    text_prefix: Optional[str]
    collection: chromadb.Collection
    # jobs: chromadb.Collection
    client: OpenAI
    # Global variables used in tool calls.
    program_state: dict
    # The tool that was used last in this prompt (if a tool was used)
    most_recent_tool: Optional[str]

    def __init__(self, client, collection):
        self.conversation = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': USER_1},
            {'role': 'assistant', 'content': ASSISTANT_1},
            {'role': 'user', 'content': USER_2},
            {'role': 'assistant', 'content': ASSISTANT_2},
            {'role': 'user', 'content': USER_3},
            {'role': 'assistant', 'content': ASSISTANT_3},
        ]
        self.text_prefix = None
        self.client = client
        self.collection = collection
        self.program_state = {
            'result': None, 
            'find_jobs': self.find_jobs, 
            'perform_skills_analysis': self.perform_skills_analysis,
            'write_cover_letter': self.write_cover_letter,
            'self': self,
            'FindJobsResponse': FindJobsResponse,
            'SkillsGapResponse': SkillsGapResponse,
            'CoverLetterResponse': CoverLetterResponse,
        } # 'book_flight': self.book_flight, 'Flight': Flight}
        self.most_recent_tool = None
    
    def get_resume(self, path_extension = '') -> Optional[str]:
        try:
            with open(temp_path + f'/resume{"_" + path_extension if path_extension != "" else ""}.txt', 'r') as file:
                resume = file.read()
        except:
            return None
        return resume

    def find_jobs(self, amount=5, resume=None) -> List[Listing]:
        print('0here')
        if resume is None:
            resume = self.get_resume()
        print('1here')
        self.most_recent_tool = 'find-jobs'
        listings = []
        if resume is not None:
            print('2here')
            print(resume)
            results = self.collection.query(
                query_texts=[resume],
                n_results=amount  # How many results to return
            )
            print('3here')
            
            for i, id in enumerate(results['ids'][0]):
                description = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                company = metadata['company']
                title = metadata['title']
                listings.append(Listing(
                    id=id,
                    title=title,
                    description=description,
                    company=company,
                ))
                print('4here')
            response_text = f'Here are some matching jobs:'
            self.text_prefix = ''
            print('5here')
            for listing in listings:
                response_text += f'\n\n{listing.format_main_details()}'
                self.text_prefix += f'\n\n{listing.format_main_details()}'
            print('6here')
        else:
            response_text = UPLOAD_RESUME_TEXT
        return FindJobsResponse(
            response_text,
            listings
        )

    def perform_skills_analysis(self, job_id: int, resume=None) -> str:
        if resume is None:
            resume = self.get_resume()
        self.most_recent_tool = 'perform-skills-analysis'
        if resume is not None:
            print('here100')
            results = self.collection.get(
                ids=[str(job_id)],
            )
            print('here101')
            print(results)
            PROMPT = f"""
    Job Title: {results['metadatas'][0]['title']}
    Job Company: {results['metadatas'][0]['company']}
    Job Description: {results['documents'][0]}

    ----------------------------
    MY RESUME:
    {resume}


    INSTRUCTIONS: You have been given a job description and my resume. Please provide a skills-gap analysis, highlighting the skills that I might want to improve on to have a better chance of getting the job, or more importantly, skills (or experience-level) that I don't have at all that are either required or preferred qualifications for the job. Use markdown format and list out the skills (text bolded) in bullet points, providing an explanation for each skill afterwards (this should not be bolded). Write it in second-person.
    """
            conversation = [{
                'role': 'user',
                'content': PROMPT
            }]
            response_text = self.client.chat.completions.create(
                messages=conversation,
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                temperature=0.5,
            ).choices[0].message.content

        else:
            response_text = UPLOAD_RESUME_TEXT
        return SkillsGapResponse(response_text)
    
    def write_cover_letter(self, job_id: int, resume=None) -> str:
        if resume is None:
            resume = self.get_resume()
        if resume is not None:

            results = self.collection.get(
                ids=[str(job_id)],
            )
            PROMPT = f"""
    Job Title: {results['metadatas'][0]['title']}
    Job Company: {results['metadatas'][0]['company']}
    Job Description: {results['documents'][0]}

    ----------------------------
    MY RESUME:
    {resume}


    INSTRUCTIONS: You have been given a job description and my resume. Please write a cover letter for me. Provide NOTHING in your response except for the cover letter (no additional text/response)
    """
            conversation = [{
                'role': 'user',
                'content': PROMPT
            }]
            cover_letter = self.client.chat.completions.create(
                messages=conversation,
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                temperature=0.5,
            ).choices[0].message.content
            response_text = f'Here is a cover letter I wrote for you:\n\n{cover_letter}'
        else:
            response_text = UPLOAD_RESUME_TEXT
        return CoverLetterResponse(response_text)
        
    
        
    # def book_flight(self, flight_id: int) -> Optional[int]:
    #     self.most_recent_tool = 'book-flight'
    #     for flight in self.flights:
    #         if flight.matches_id(flight_id):
    #             return flight.book_flight()
    #     return None

    def say(self, user_message: str) -> AgentResponse:
        self.conversation.append(
            {
                'role': 'user',
                'content': (f'{self.text_prefix}\n\n' if self.text_prefix is not None else '') + user_message,
            }
        )
        self.text_prefix = None
        self.most_recent_tool = None
        resp = self.client.chat.completions.create(
            messages = self.conversation,
            model = "meta-llama/Meta-Llama-3.1-8B-Instruct",
            temperature=0.3,
        ).choices[0].message.content
        self.conversation.append({
            'role': 'assistant',
            'content': resp,
        })
        try:
            if resp[0:9] == '```python':
                print('here')
                self.program_state['result'] = None
                print('here2')
                exec(resp[9:-3].strip(), self.program_state)
                print('here3')
                # if self.program_state['result'] is not None:
                if self.program_state['result'] is not None and isinstance(self.program_state['result'], AgentResponse):
                    print('here4')
                    return self.program_state['result']
                else:
                    print('here5')
                    return TextResponse(resp)
            else:
                print('here6')
                return TextResponse(resp)
        except Exception as e:
            print(repr(e))
            print('here7')
            return TextResponse(resp)
        
        

@dataclasses.dataclass
class EvaluationResult:
    """
    The result from evaluating a test case
    """
    score: float
    conversation: List[dict]


# def eval_agent(client: OpenAI, benchmark_file: str, flights: List[Flight]) -> float:
#     """
#     Evaluate the agent on the given benchmark YAML file.
#     """
#     agent = Agent(client, flights)
#     with open(benchmark_file, "r") as file:
#         steps = yaml.safe_load(file)
#     for n, step in enumerate(steps):
#         response = agent.say(step["prompt"])
#         match step["expected_type"]:
#             case "text":
#                 if not isinstance(response, TextResponse):
#                     return EvaluationResult(n / len(steps), agent.conversation)
#             case "find-flights":
#                 if not isinstance(response, FindFlightsResponse):
#                     return EvaluationResult(n / len(steps), agent.conversation)
#                 if set(response.available_flights) != set(step["expected_result"]):
#                     return EvaluationResult(n / len(steps), agent.conversation)
#             case "book-flight":
#                 if not isinstance(response, BookFlightResponse):
#                     return EvaluationResult(n / len(steps), agent.conversation)
#                 if response.booked_flight != step["expected_result"]:
#                     return EvaluationResult(n / len(steps), agent.conversation)
#     return EvaluationResult(1.0, agent.conversation)

