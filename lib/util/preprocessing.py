from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

english_stopwords = stopwords.words("english")


def tokenize(text, stemming=True):
    words = [word.lower() for word in word_tokenize(text) if word.lower() not in english_stopwords]
    if stemming:
        words = [PorterStemmer().stem(word) for word in words]
    return words
