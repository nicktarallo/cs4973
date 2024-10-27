from functools import cache
from nltk.corpus import stopwords
import string
import nltk

@cache
def parser(text: str):
    """
    split by space, parse all the stop words, and remove punctuation
    """
    terms = text.split()
    # terms = [t.lower() for t in terms]
    stopword_set = set(stopwords.words('english'))

    parsed = [t for t in terms if t.lower() not in stopword_set]
    parsed = terms

    # out = parsed.translate(None, string.punctuations)
    out = ' '.join(parsed).translate(str.maketrans('','',string.punctuation))

    return out.split()
