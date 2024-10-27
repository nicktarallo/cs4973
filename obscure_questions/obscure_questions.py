from functools import cache
from typing import List
from openai import OpenAI
import math
import datasets
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from parser import parser

# # Load pre-trained BERT model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# model = AutoModel.from_pretrained('bert-base-uncased')

# # Ensure the model is in evaluation mode
# model.eval()

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
        max_tokens=1
    )

    return resp.choices[0].text.strip()


class RAG:
    def __init__(self, nuwiki):
        self.nuwiki = nuwiki
        # Load pre-trained BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')

        # Ensure the model is in evaluation mode
        self.model.eval()

    def term_frequency2(self, term:str, document: str):
        return  document.count(term)


    @cache
    def inverse_document_frequency(self, term: str) -> float:
        num_docs_with_term = 0
        for item in self.nuwiki:
            if term in item["text"].split():
                num_docs_with_term += 1
        if num_docs_with_term == 0:
            return 0
        return math.log(len(self.nuwiki) / num_docs_with_term)


    def tf_idf_vector(self, terms, doc):
        doc_text = ' '.join(parser(doc['text']))
        # doc_text = doc['text']
        vec =  np.array([self.term_frequency2(term, doc_text) * self.inverse_document_frequency(term) for term in terms])
        normalized_vec = vec / np.linalg.norm(vec)
        return normalized_vec


    # TODO: We may want to remove punctuation from query (maybe lowercase as well but unlikely)
    def rank_by_tf_idf(self, query: str, n: int) -> list:
        # terms = query.split()
        terms = parser(query)
        query_vec = self.tf_idf_vector(terms, { "text": query })
        ranked_docs = sorted(
            self.nuwiki,
            key=lambda doc: self.tf_idf_vector(terms, doc).dot(query_vec),
            reverse=True
        )
        return ranked_docs[:n]
    
    def get_normalized_bert_embedding(self, text: str) -> torch.Tensor:
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # print(inputs)
        # Run the model
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the last hidden state
        last_hidden_state = outputs.last_hidden_state
        
        # Get the embedding of the last token (excluding padding tokens)
        last_token_embedding = last_hidden_state[0, -1]
        normalized_embedding = last_token_embedding / last_token_embedding.norm()
        return normalized_embedding

    def chunk_document(self, document_text: str, max_tokens: int = 512):
        tokens = self.tokenizer.tokenize(document_text, max_length=512, truncation=True, return_tensors="pt")
        chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
        chunk_texts = [self.tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]
        return chunk_texts
    
    def rank_by_bert(self, query: str, documents, n: int) -> list:
        chunks = []
        for doc in documents:
            chunks.extend(self.chunk_document(doc['text'], 250))
        
        query_vec = self.get_normalized_bert_embedding(query)
        ranked_docs = sorted(
            chunks,
            key=lambda chunk: self.get_normalized_bert_embedding(chunk).dot(query_vec),
            reverse=True
        )
        return ranked_docs[:n]
    
    def combined_rank(self, query):
        tf_idf_ranked_docs = self.rank_by_tf_idf(query, 10)
        bert_ranked_chunks = self.rank_by_bert(query, tf_idf_ranked_docs, 10)
        return bert_ranked_chunks


if __name__ == '__main__':
    print(answer_query("What is 2+2", ["A. 1", "B. 2", "C. 3", "D. 4"], []))

    wiki_dataset = datasets.load_dataset("nuprl/engineering-llm-systems", name="wikipedia-northeastern-university", split="test")
    rag = RAG(wiki_dataset)
    # print([article['title'] for article in rag.rank_by_tf_idf("Who is the Dean of Northeastern computer science?", 10)])

    chunks = rag.combined_rank("Who is the Dean of Northeastern computer science?")
    for chunk in chunks:
        print(chunk)
        print()
        print()