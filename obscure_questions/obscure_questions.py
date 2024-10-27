from functools import cache
from typing import List
from openai import OpenAI
import math
import datasets
import numpy as np

def answer_query(question: str, choices: List[str], documents: List[str]) -> str:
    """
    Answers a multiple choice question using retrieval augmented generation.

    `question` is the text of the question. `choices` is the list of choices
     with leading letters. For example:

     ```
     ["A. Choice 1", "B. Choice 2", "C. Choice 3", "D. Choice 4"]
     ```

     `documents` is the list of documents to use for retrieval augmented
     generation.

     The result should be the just the letter of the correct choice, e.g.,
     `"A"` but not `"A."` and not `"A. Choice 1"`.
     """
    
    BASE_URL = "http://199.94.61.113:8000/v1/"
    API_KEY = "tarallo.n@northeastern.edu:OQEgetkQ6LBofgWW4jsC"
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    doc_string = "\n\n".join(documents)
    question_text = f'Based on these texts, answer this question with the corresponding letter only: Question: {question}'
    answers = '\n'.join(choices)
    prompt = f'{doc_string}\n\n{question_text}\n{answers}\nAnswer:'

    resp = client.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        temperature=0.2,
        prompt=prompt,
        max_tokens=2
    )

    return resp.choices[0].text.strip()


def term_frequency2(term:str, document: str):
    return  document.count(term)


@cache
def inverse_document_frequency(term: str, nuwiki: datasets.arrow_dataset.Dataset) -> float:
    num_docs_with_term = 0
    for item in nuwiki:
        if term in item["text"].split():
            num_docs_with_term += 1
    if num_docs_with_term == 0:
        return 0
    return math.log(len(nuwiki) / num_docs_with_term)


def tf_idf_vector(terms, doc, nuwiki: datasets.arrow_dataset.Dataset):
    vec =  np.array([term_frequency2(term, doc['text']) * inverse_document_frequency(term, nuwiki) for term in terms])
    normalized_vec = vec / np.linalg.norm(vec)
    return normalized_vec


def rank_by_tf_idf(query: str, n: int, nuwiki: datasets.arrow_dataset.Dataset) -> list:
    terms = query.split()
    query_vec = tf_idf_vector(terms, { "text": query }, nuwiki)
    ranked_docs = sorted(
        nuwiki,
        key=lambda doc: tf_idf_vector(terms, doc, nuwiki).dot(query_vec),
        reverse=True
    )
    return ranked_docs[:n]

print(answer_query("What is 2+2", ["A. 1", "B. 2", "C. 3", "D. 4"], []))



