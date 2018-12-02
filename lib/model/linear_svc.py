from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
import numpy as np

import lib.data.fetch
import lib.util.preprocessing

english_stopwords = stopwords.words("english")


def train(train_x, train_y, single_label=True, random_state=37):
    vectorizer = TfidfVectorizer(stop_words=english_stopwords,
                                 tokenizer=lib.util.preprocessing.tokenize)
    train_x = vectorizer.fit_transform(train_x)
    if single_label:
        classifier = LinearSVC(random_state=random_state)
    else:
        classifier = OneVsRestClassifier(LinearSVC(random_state=random_state))
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
    print(accuracy, precision, recall, f1)


if __name__ == '__main__':
    train_split, validation_split, test_split = lib.data.fetch.reuters()
    train_x = [x[1] for x in train_split]
    train_y = np.array([x[0] for x in train_split])
    classifier, vectorizer = train(train_x, train_y)

    validation_x = [x[1] for x in validation_split]
    validation_y = np.array([x[0] for x in validation_split])
    evaluate(classifier, vectorizer, validation_x, validation_y)

    test_x = [x[1] for x in test_split]
    test_y = np.array([x[0] for x in test_split])
    evaluate(classifier, vectorizer, test_x, test_y)
