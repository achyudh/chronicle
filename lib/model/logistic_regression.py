from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
import numpy as np

import lib.data.fetch
import lib.util.preprocessing

english_stopwords = stopwords.words("english")


def train(train_x, train_y, single_label=True, random_state=37):
    np.random.seed(random_state)
    vectorizer = TfidfVectorizer(stop_words=english_stopwords,
                                 tokenizer=lib.util.preprocessing.tokenize)
    train_x = vectorizer.fit_transform(train_x)
    if single_label:
        classifier = LogisticRegression(random_state=random_state)
    else:
        classifier = OneVsRestClassifier(LogisticRegression(random_state=random_state))
    classifier.fit(train_x, train_y)
    return classifier, vectorizer


def predict(classifier, vectorizer, predict_x):
    predict_x = vectorizer.transform(predict_x)
    return classifier.predict(predict_x)


def evaluate(classifier, vectorizer, evaluate_x, evaluate_y):
    evaluate_x = vectorizer.transform(evaluate_x)
    predict_y = classifier.predict(evaluate_x)
    accuracy = accuracy_score(evaluate_y, predict_y)
    precision = precision_score(evaluate_y, predict_y, average='micro')
    recall = recall_score(evaluate_y, predict_y, average='micro')
    f1 = f1_score(evaluate_y, predict_y, average='micro')
    return accuracy, precision, recall, f1

