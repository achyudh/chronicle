from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

english_stopwords = stopwords.words("english")


def tokenize_top_n(text, top_n=2, stemming=True):
    """
    Tokenizes only the top_n lines for the given text
    """
    split_string = text.split('.')
    top_n_string = ""
    for string in range(min(len(split_string), top_n)):
        top_n_string += (split_string[0].strip() + ". ")
    return tokenize(top_n_string, stemming)


def tokenize(text, stemming=True):
    words = [word.lower() for word in word_tokenize(text) if word.lower() not in english_stopwords]
    if stemming:
        words = [PorterStemmer().stem(word) for word in words]
    return words
