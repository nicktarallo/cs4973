from nltk.corpus import stopwords
import string

def parser(text: str):
    """
    split by space, parse all the stop words, and remove punctuation
    """
    terms = text.split()
    terms = [t.lower() for t in terms]

    nltk.download('stopwords')
    parsed = [t for t in terms if t not in stopwords.words('english')]

    # out = parsed.translate(None, string.punctuations)
    out = parsed.translate(str.maketrans('','',string.punctuations))

    return out
