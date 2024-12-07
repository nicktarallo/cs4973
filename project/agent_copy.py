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
    resume: str
    cover_letter: Optional[str]

@dataclasses.dataclass
class Listing:
    id: int
    title: str
    description: str
    company: str
    applications: List[Application] = None
    
    def format_main_details(self):
        return f"""Company: {self.company}
Job Title: {self.title}
Job ID: {self.id}"""
    
    def format_all_details(self):
        return f"""Company: {self.company}
Job Title: {self.title}
Job ID: {self.id}

DESCRIPTION:
{self.description}
"""

@dataclasses.dataclass
class AgentResponse:
    """
    The superclass for all agent responses.
    """
    text: str

@dataclasses.dataclass
class FindJobsResponse(AgentResponse):
    jobs_found: Optional[List[int]]

@dataclasses.dataclass
class SkillsGapResponse(AgentResponse):
    job_id: Optional[int]

@dataclasses.dataclass
class CoverLetterResponse(AgentResponse):
    job_id: Optional[int]

@dataclasses.dataclass
class ShowJobDescriptionResponse(AgentResponse):
    job_id: Optional[int]

@dataclasses.dataclass
class ApplyResponse(AgentResponse):
    job_id: Optional[int]
    used_custom_cover_letter: bool

# This is used when the model writes code that throws an error:
@dataclasses.dataclass
class ErrorResponse(AgentResponse):
    e: Exception

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

If you are asked to show the full job description or details of a specific job, respond with code that uses this function:

```python
def show_job_description(self, job_id: int):
    ...
```

If you are asked to apply to a job, respond with code using this function:
```python
def apply(self, job_id: int):
    ...
```

Otherwise, just respond with text

When responding with code, make sure to set the variable `result` equal to the return value of the function call.

If you respond with code, do NOT include any other text outside of the code.

In any code block you write, assume result is originally set to None.

Make sure if you respond with code to use markdown backticks with python.

YOU DO NOT NEED TO DEFINE THOSE SPECIFIED FUNCTIONS, THEY ALREADY EXIST. ALSO, IF THEY ARE DOING ONE OF THOSE TASKS, MAKE SURE YOU RESPOND WITH CODE, DON'T TRY TO MAKE UP JOBS."""

USER_0 = """Hi"""
ASSISTANT_0 = """Hi! I can help with your job search."""
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
USER_4 = """Show me the job description"""
ASSISTANT_4 = """```python
result = self.show_job_description(87)
```"""
USER_5 = """Apply to the job"""
ASSISTANT_5 = """```python
result = self.apply(87)
```"""

UPLOAD_RESUME_TEXT = 'Please upload your resume.'
JOB_NOT_FOUND_TEXT = 'Job listing not found.'
ERROR_TEXT = 'Error processsing your request, please try another request.'

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
    user_resume: Optional[str]
    cover_letters = dict[int, str]
    jobs_applied_to = set[int]

    def __init__(self, client, col1, resume=None):
        self.conversation = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': USER_0},
            {'role': 'assistant', 'content': ASSISTANT_0},
            {'role': 'user', 'content': USER_1},
            {'role': 'assistant', 'content': ASSISTANT_1},
            {'role': 'user', 'content': USER_2},
            {'role': 'assistant', 'content': ASSISTANT_2},
            {'role': 'user', 'content': USER_3},
            {'role': 'assistant', 'content': ASSISTANT_3},
            {'role': 'user', 'content': USER_4},
            {'role': 'assistant', 'content': ASSISTANT_4},
            {'role': 'user', 'content': USER_5},
            {'role': 'assistant', 'content': ASSISTANT_5},
        ]
        self.text_prefix = None
        self.client = client
        self.collection = col1
        self.program_state = {
            'result': None, 
            'find_jobs': self.find_jobs, 
            'perform_skills_analysis': self.perform_skills_analysis,
            'write_cover_letter': self.write_cover_letter,
            'self': self,
            'FindJobsResponse': FindJobsResponse,
            'SkillsGapResponse': SkillsGapResponse,
            'CoverLetterResponse': CoverLetterResponse,
            'ShowJobDescriptionResponse': ShowJobDescriptionResponse,
            'ApplyResponse': ApplyResponse,
        }
        self.most_recent_tool = None
        self.user_resume = resume
        self.job_specific_resumes = {}
        self.cover_letters = {}
        self.jobs_applied_to = set()
    
    def get_resume(self, job_id: Optional[int] = None) -> Optional[str]:
        if job_id is not None and int(job_id) in self.job_specific_resumes:
            return self.job_specific_resumes[int(job_id)]
        return self.user_resume
    
    def get_listing_from_id(self, job_id: int) -> Optional[Listing]:
        results = self.collection.get(
            ids=[str(job_id)],
        )
        try:
            return Listing(job_id, results['metadatas'][0]['job_title'], results['documents'][0], results['metadatas'][0]['company'])
        except:
            return None

    def set_resume(self, r: str) -> None:
        print('setting resume')
        self.user_resume = r

    def find_jobs(self, amount=5, resume=None) -> List[Listing]:
        if resume is None:
            resume = self.get_resume()
        self.most_recent_tool = 'find-jobs'
        listings = []
        if resume is not None:
            results = self.collection.query(
                query_texts=[resume],
                n_results=amount  # How many results to return
            )
            
            for i, id in enumerate(results['ids'][0]):
                description = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                company = metadata['company']
                title = metadata['job_title']
                listings.append(Listing(
                    id=int(id),
                    title=title,
                    description=description,
                    company=company,
                ))
            response_text = f'Here are some matching jobs:'
            self.text_prefix = ''
            for listing in listings:
                response_text += f'\n\n{listing.format_main_details()}'
                self.text_prefix += f'\n\n{listing.format_main_details()}'
        else:
            response_text = UPLOAD_RESUME_TEXT
        return FindJobsResponse(
            response_text,
            [listing.id for listing in listings]
        )

    def perform_skills_analysis(self, job_id: int, resume=None) -> str:
        if resume is None:
            resume = self.get_resume()
        self.most_recent_tool = 'perform-skills-analysis'
        if resume is not None:
            results = self.collection.get(
                ids=[str(job_id)],
            )
            try:
                PROMPT = f"""
        Job Title: {results['metadatas'][0]['job_title']}
        Job Company: {results['metadatas'][0]['company']}
        Job Description: {results['documents'][0]}
        Job Level: {results['metadatas'][0]['job_level']}
        Typical Job Skills: {set(results['metadatas'][0]['job_skills'])}

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
                result = job_id
            except:
                response_text = JOB_NOT_FOUND_TEXT
                result = None

        else:
            response_text = UPLOAD_RESUME_TEXT
            result = None
        return SkillsGapResponse(response_text, result)
    
    def write_cover_letter(self, job_id: int, resume=None) -> str:
        if resume is None:
            resume = self.get_resume()
        if resume is not None:

            results = self.collection.get(
                ids=[str(job_id)],
            )
            try:
                PROMPT = f"""
        Job Title: {results['metadatas'][0]['title']}
        Job Company: {results['metadatas'][0]['company']}
        Job Description: {results['documents'][0]}
        Job Level: {results['metadatas'][0]['job_level']}
        Typical Job Skills: {set(results['metadatas'][0]['job_skills'])}

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
                self.cover_letters[job_id] = cover_letter
                result = job_id
            except:
                response_text = JOB_NOT_FOUND_TEXT
                result = None
        else:
            response_text = UPLOAD_RESUME_TEXT
            result = None
        return CoverLetterResponse(response_text, result)
        
    def show_job_description(self, job_id: int):
        listing = self.get_listing_from_id(job_id)
        if listing:
            text_prefix = f'''Job {listing.id} Description:
    {listing.description}'''
            response_text = f'''Here's the full description:
    {listing.format_all_details()}'''
            return ShowJobDescriptionResponse(response_text, listing.id)
        return ShowJobDescriptionResponse(JOB_NOT_FOUND_TEXT, None)
    
    def apply(self, job_id: int):
        used_custom_cover_letter = False
        if job_id in self.jobs_applied_to:
            result = None
            result_text = 'You already applied to this job.'
        else:
            resume = self.get_resume()
            if resume is not None:
                extraction = self.extract_name_and_email(resume)
                if extraction is not None:
                    name = extraction[0]
                    email = extraction[1]
                else:
                    name = None
                    email = None
                if job_id in self.cover_letters:
                    cover_letter = self.cover_letters[job_id]
                    used_custom_cover_letter = True
                else:
                    cover_letter = None
                results = self.collection.get(
                    ids=[str(job_id)],
                )
                old_metadata = results['metadatas'][0]
                old_metadata['application_resume'] = resume
                result_text = 'Successfully applied to job.'
                if cover_letter is not None:
                    result_text = 'Successfully applied to job with cover letter.'
                    old_metadata['application_cover_letter'] = cover_letter
                if name is not None:
                    old_metadata['application_name'] = name
                if email is not None:
                    old_metadata['application_email'] = email
                self.collection.update(ids=str(job_id), metadatas=old_metadata)
                result = job_id
                self.jobs_applied_to.add(job_id)
                
            else:
                result_text = UPLOAD_RESUME_TEXT
                result = None
        return ApplyResponse(result_text, result, used_custom_cover_letter)
    
    def extract_name_and_email(self, resume: str):
        PROMPT = f"""RESUME:
Jordan Banks 456 Finance Avenue Boston, MA 02118 Phone: (617) 555-6789 Email: jordan.banks@email.com

Professional Experience
Chase Bank Senior Financial Analyst March 2017 – Present

Analyzed and interpreted complex financial data to provide strategic recommendations.

Developed financial models to support investment decisions and risk management.

Led a team of junior analysts, providing training and mentorship.

Wells Fargo Financial Analyst July 2013 – February 2017

Conducted detailed financial analysis and forecasting for corporate clients.

Collaborated with cross-functional teams to support business development initiatives.

Bank of America Junior Financial Analyst June 2011 – June 2013

Supported senior analysts in preparing financial reports and presentations.

Contributed to the development of financial models and valuation analyses.

Education
Harvard University Master of Business Administration (MBA) September 2009 – June 2011

Specialized in Finance and Investment Management.

Relevant coursework: Corporate Finance, Investment Analysis, Financial Markets.

Relevant coursework: Financial Accounting, Economics, Portfolio Management.

Skills
Financial Analysis: Advanced Excel, Financial Modeling, Valuation Techniques

Investment Management: Asset Allocation, Portfolio Management, Risk Assessment

Certifications
Chartered Financial Analyst (CFA)

Certified Financial Planner (CFP)

Professional Affiliations
Member, Chartered Financial Analyst Institute

Projects
Strategic Financial Planning for Corporate Clients

Developed and implemented financial strategies to optimize client profitability.

Conducted market analysis to identify investment opportunities.

Mergers and Acquisitions Analysis

Assisted in due diligence and financial analysis for successful M&A transactions.

Contributed to the valuation and integration planning of acquired companies.

EXTRACTION:
NAME: Jordan Banks
EMAIL: jordan.banks@email.com

RESUME:
{resume}

EXTRACTION:"""
        
        resp = self.client.completions.create(
            prompt=PROMPT,
            model = "meta-llama/Meta-Llama-3.1-8B-Instruct",
            temperature=0.3,
        ).choices[0].text.strip()
        splitted = resp.split('\n')
        try:
            name = splitted[0][6:]
            email = splitted[1][7:]
        except:
            return None
        return name, email
        
            
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
                self.program_state['result'] = None
                exec(resp[9:-3].strip(), self.program_state)
                # if self.program_state['result'] is not None:
                if self.program_state['result'] is not None and isinstance(self.program_state['result'], AgentResponse):
                    return self.program_state['result']
                else:
                    return TextResponse(resp)
            else:
                return TextResponse(resp)
        except Exception as e:
            print(repr(e))
            return ErrorResponse(ERROR_TEXT, e)
        
        

@dataclasses.dataclass
class EvaluationResult:
    """
    The result from evaluating a test case
    """
    score: float
    conversation: List[dict]


def eval_agent(client: OpenAI, benchmark_file: str, collection: chromadb.Collection, resume: str) -> float:
    """
    Evaluate the agent on the given benchmark YAML file.
    """
    agent = Agent(client, collection, resume)
    with open(benchmark_file, "r") as file:
        steps = yaml.safe_load(file)
    for n, step in enumerate(steps):
        response = agent.say(step["prompt"])
        match step["expected_type"]:
            case "text":
                if not isinstance(response, TextResponse):
                    print(type(response), "expected: TextResponse")
                    return EvaluationResult(n / len(steps), agent.conversation)
            case "find-jobs":
                if not isinstance(response, FindJobsResponse):
                    print(type(response), "expected: FindJobsResponse")
                    return EvaluationResult(n / len(steps), agent.conversation)
                if response.jobs_found != step["expected_result"]:
                    print(response.jobs_found, f"expected: {step['expected_result']}")
                    return EvaluationResult(n / len(steps), agent.conversation)
            case "skills-gap":
                if not isinstance(response, SkillsGapResponse):
                    print(type(response), "expected: SkillsGapResponse")
                    return EvaluationResult(n / len(steps), agent.conversation)
                if response.job_id != step["expected_result"]:
                    print(response.job_id, f"expected: {step['expected_result']}")
                    return EvaluationResult(n / len(steps), agent.conversation)
            case "cover-letter":
                if not isinstance(response, CoverLetterResponse):
                    print(type(response), "expected: CoverLetterResponse")
                    return EvaluationResult(n / len(steps), agent.conversation)
                if response.job_id != step["expected_result"]:
                    print(response.job_id, f"expected: {step['expected_result']}")
                    return EvaluationResult(n / len(steps), agent.conversation)
            case "show-job-description":
                if not isinstance(response, ShowJobDescriptionResponse):
                    print(type(response), "expected: ShowJobDescriptionResponse")
                    return EvaluationResult(n / len(steps), agent.conversation)
                if response.job_id != step["expected_result"]:
                    print(response.job_id, f"expected: {step['expected_result']}")
                    return EvaluationResult(n / len(steps), agent.conversation)
            case "apply":
                if not isinstance(response, ApplyResponse):
                    print(type(response), "expected: ApplyResponse")
                    return EvaluationResult(n / len(steps), agent.conversation)
                if response.job_id != step["expected_result"] or response.used_custom_cover_letter != step["cover_letter"]:
                    print(response.job_id, f"expected: {step['expected_result']}")
                    print(response.used_custom_cover_letter, f"expected: {step['cover_letter']}")
                    return EvaluationResult(n / len(steps), agent.conversation)
    return EvaluationResult(1.0, agent.conversation)
