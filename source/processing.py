from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer, LancasterStemmer

sw = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')

def process_text(review, stem='p'):
    """
    Given a text, the function converts the text into lowercase,
    removes the stopwords, removes punctuation, tokenize the text,
    performs stemming and returns the processed text
    :param review: row text
    :param stem: Stemmer - 'p' for PorterStemmer and 'l' for LancasterStemmer
    :return: processed text
    """
    # Convert text to lower
    review = review.lower()
    # Word tokenize the review
    tokens = word_tokenize(review)
    # Remove stopwords
    tokens = [t for t in tokens if t not in sw]
    # Remove punctuations
    tokens = [tokenizer.tokenize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 0]
    tokens = ["".join(t) for t in tokens]

    # Create stemmer
    if stem == 'p':
        stemmer = PorterStemmer()
    elif stem == 'l':
        stemmer = LancasterStemmer()
    else:
        raise Exception("stem has to be either 'p' for PorterStemmer or 'l' for LancasterStemmer")

    # Stemming
    tokens = [stemmer.stem(t) for t in tokens]

    # Return clean string
    return " ".join(tokens)