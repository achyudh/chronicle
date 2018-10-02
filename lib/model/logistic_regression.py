from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
import numpy as np

import lib.data.fetch
import lib.util.preprocessing

english_stopwords = stopwords.words("english")


def train(train_x, train_y):
    vectorizer = TfidfVectorizer(stop_words=english_stopwords,
                                 tokenizer=lib.util.preprocessing.tokenize)
    train_x = vectorizer.fit_transform(train_x)
    classifier = OneVsRestClassifier(LogisticRegression(random_state=37))
    classifier.fit(train_x, train_y)
    return classifier, vectorizer


def predict(classifier, vectorizer, predict_x):
    predict_x = vectorizer.transform(predict_x)
    return classifier.predict(predict_x)


def evaluate(classifier, vectorizer, evaluate_x, evaluate_y):
    predict_y = classifier.predict(classifier, vectorizer, evaluate_x)
    accuracy = accuracy_score(evaluate_y, predict_y)
    precision = precision_score(evaluate_y, predict_y, average='micro')
    recall = recall_score(evaluate_y, predict_y, average='micro')
    f1 = f1_score(evaluate_y, predict_y, average='micro')
    print(accuracy, precision, recall, f1)


if __name__ == '__main__':
    train_split, validation_split, test_split = lib.data.fetch.reuters()
    train_x = [x[1] for x in train_split]
    train_y = np.array([x[0] for x in train_split])
    classifier, vectorizer = train(train_x, train_y)

    validation_x = [x[1] for x in validation_split]
    validation_y = np.array([x[0] for x in validation_split])
    evaluate(classifier, vectorizer, validation_x, validation_y)

    test_x = [x[1] for x in train_split]
    test_y = np.array([x[0] for x in test_split])
    evaluate(classifier, vectorizer, test_x, test_y)
