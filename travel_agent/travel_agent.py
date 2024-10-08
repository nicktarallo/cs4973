from openai import OpenAI
from typing import List
from datasets import load_dataset
from datetime import date, time, datetime
import dataclasses
from typing import List, Optional
import yaml
import statistics
from pathlib import Path

BASE_URL = "http://199.94.61.113:8000/v1/"
API_KEY = "tarallo.n@northeastern.edu:OQEgetkQ6LBofgWW4jsC"



@dataclasses.dataclass
class Flight:
    id: int
    date: date
    airline: str
    flight_number: str
    origin: str
    destination: str
    departure_time: time
    arrival_time: time
    available_seats: int

    def matches_search(self, origins: str, destinations: str, start_date: date, end_date: date) -> bool:
        return self.origin in origins and self.destination in destinations and self.date >= start_date and self.date <= end_date
    
    def matches_id(self, id: int) -> bool:
        return self.id == id
    
    def book_flight(self) -> Optional[int]:
        if self.available_seats > 0:
            self.available_seats -= 1
            return self.id
        return None
    
    def format_flight(self):
        return f"""Date: {self.date.isoformat()}
Airline: {self.airline}
Flight Number: {self.flight_number}
Origin: {self.origin}
Destination: {self.destination}
Departure Time: {self.departure_time.isoformat()}
Arrival Time: {self.arrival_time.isoformat()}
Available Seats: {self.available_seats}"""

def parse_flight(flight):
    return Flight(
        id=flight["id"],
        date=datetime.strptime(flight["date"], "%Y-%m-%d").date(),
        airline=flight["airline"],
        flight_number=flight["flight_number"],
        origin=flight["origin"],
        destination=flight["destination"],
        departure_time=datetime.strptime(flight["departure_time"], "%H:%M").time(),
        arrival_time=datetime.strptime(flight["arrival_time"], "%H:%M").time(),
        available_seats=flight["available_seats"],
    )


def load_flights_dataset() -> List[Flight]:
    return [
        parse_flight(flight)
        for flight in load_dataset("nuprl/llm-systems-flights", split="train")
    ]


@dataclasses.dataclass
class AgentResponse:
    """
    The superclass for all agent responses.
    """
    text: str

@dataclasses.dataclass
class FindFlightsResponse(AgentResponse):
    """
    The agent used the `find_flights` tool and found the following flights.
    """
    available_flights: List[int]

@dataclasses.dataclass
class BookFlightResponse(AgentResponse):
    """
    The agent used the `book_flight` tool and booked the following flight.
    """
    booked_flight: Optional[int]

@dataclasses.dataclass
class TextResponse(AgentResponse):
    pass

SYSTEM_PROMPT = """
You are Thomas, a helpful travel agent. If the user asks to search for flights, respond with code that uses this function:

def find_flights(origins: List[str], destinations: List[str], start_date: datetime.date, end_date: datetime.date) -> List[Flight]:
    ...

If the user asks to book a flight, respond with code that uses this function:

def book_flight(flight_id: int) -> Optional[int]:
    ...

Otherwise, just respond with text

Return the result in a variable called result.

Today's date is September 1, 2023. However, if the user asks you to, you can book flights in the past.

If your response uses code to find or book flights, do NOT include any other text outside of the code.

In any code block you write, assume result is originally set to None.

If the person asks to book a flight from an airline that isn't available from the flights that are found, you should not attempt to book and should refuse with text. 

You can book flights from previous searches if the user asks.

Make sure if you respond with code to use markdown backticks with python.

When calling find_flights, make sure to only include the range of dates that the user actually wants to fly on. Don't get confused with other dates.
"""

USER_1 = "as a test, find me a flight from New York to Denver between December 29 and December 31. Don't assume I will always want to leave from New York, though"
ASSISTANT_1 = """```python
result = find_flights(['JFK'], ['DEN'], date(2023, 12, 29), date(2023, 12, 31))
```"""
USER_2 = """[Flight(id=27799, date=datetime.date(2023, 12, 30), airline='American', flight_number='AA5890', origin='JFK', destination='DEN', departure_time=datetime.time(12, 22), arrival_time=datetime.time(14, 22), available_seats=10), Flight(id=27800, date=datetime.date(2023, 12, 30), airline='United', flight_number='UA6409', origin='JFK', destination='DEN', departure_time=datetime.time(2, 56), arrival_time=datetime.time(4, 56), available_seats=101)]

book the United flight
"""
ASSISTANT_2 = """```python
result = book_flight(27800)
```"""


class Agent:

    # The complete conversation with the LLM, including the system prompt.
    conversation: List[dict]
    # The formatted response from the last tool call.
    text_prefix: Optional[str]
    # The current database of flights. The tools update this database.
    flights: List[Flight]
    client: OpenAI
    # Global variables used in tool calls.
    program_state: dict
    # The tool that was used last in this prompt (if a tool was used)
    most_recent_tool: Optional[str]

    def __init__(self, client, flights):
        self.conversation = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': USER_1},
            {'role': 'assistant', 'content': ASSISTANT_1},
            {'role': 'user', 'content': USER_2},
            {'role': 'assistant', 'content': ASSISTANT_2},
        ]
        self.text_prefix = None
        self.flights = flights
        self.client = client
        self.program_state = {'result': None, 'date': date, 'find_flights': self.find_flights, 'book_flight': self.book_flight, 'Flight': Flight}
        self.most_recent_tool = None,


    def find_flights(self, origins: List[str], destinations: List[str], start_date: date, end_date: date) -> List[Flight]:
        self.most_recent_tool = 'find-flights'
        flights = [flight for flight in self.flights if flight.matches_search(origins, destinations, start_date, end_date)]
        return flights
        
    def book_flight(self, flight_id: int) -> Optional[int]:
        self.most_recent_tool = 'book-flight'
        for flight in self.flights:
            if flight.matches_id(flight_id):
                return flight.book_flight()
        return None

    def say(self, user_message: str) -> AgentResponse:
        self.conversation.append(
            {
                'role': 'user',
                'content': (f'{self.text_prefix}\n\n' if self.text_prefix is not None else '') + user_message,
            }
        )
        self.most_recent_tool = None
        resp = self.client.chat.completions.create(
            messages = self.conversation,
            model = "meta-llama/Meta-Llama-3.1-8B-Instruct",
            temperature=0.2,
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
                self.text_prefix = str(self.program_state['result'])
                # print(self.program_state['result'])
                if self.most_recent_tool == 'find-flights':
                    flights = self.program_state['result']
                    if len(flights) == 0:
                        text = 'No flights found.'
                    else:
                        text = 'Here are the flights I found:'
                        for flight in flights:
                            text += f'\n\n{flight.format_flight()}'
                    return FindFlightsResponse(
                        text,
                        [flight.id for flight in flights]
                    )
                elif self.most_recent_tool == 'book-flight':
                    flight_id = self.program_state['result']
                    if flight_id is None:
                        text = 'Flight could not be booked'
                    else:
                        for flight in self.flights:
                            if flight.matches_id(flight_id):
                                break
                        text = f'Booked the following flight:\n\n{flight.format_flight()}'
                    return BookFlightResponse(
                        text,
                        flight_id,
                    )
                else:
                    return TextResponse(resp)
            else:
                return TextResponse(resp)
        except Exception as e:
            # print(e)
            return TextResponse(resp)
        self.text_prefix = None
        

@dataclasses.dataclass
class EvaluationResult:
    """
    The result from evaluating a test case
    """
    score: float
    conversation: List[dict]


def eval_agent(client: OpenAI, benchmark_file: str, flights: List[Flight]) -> float:
    """
    Evaluate the agent on the given benchmark YAML file.
    """
    agent = Agent(client, flights)
    with open(benchmark_file, "r") as file:
        steps = yaml.safe_load(file)
    for n, step in enumerate(steps):
        response = agent.say(step["prompt"])
        match step["expected_type"]:
            case "text":
                if not isinstance(response, TextResponse):
                    return EvaluationResult(n / len(steps), agent.conversation)
            case "find-flights":
                if not isinstance(response, FindFlightsResponse):
                    return EvaluationResult(n / len(steps), agent.conversation)
                if set(response.available_flights) != set(step["expected_result"]):
                    return EvaluationResult(n / len(steps), agent.conversation)
            case "book-flight":
                if not isinstance(response, BookFlightResponse):
                    return EvaluationResult(n / len(steps), agent.conversation)
                if response.booked_flight != step["expected_result"]:
                    return EvaluationResult(n / len(steps), agent.conversation)
    return EvaluationResult(1.0, agent.conversation)

