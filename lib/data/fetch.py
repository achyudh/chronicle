import os
import json

from sklearn.model_selection import train_test_split


def reuters():
    train_split = list()
    validation_split = list()
    test_split = list()
    with open(os.path.join('data', 'reuters', 'reuters_train.tsv')) as tsv_file:
        for line in tsv_file:
            label, text = line.split('\t')
            label = [int(x) for x in label]
            train_split.append((label, text))
    with open(os.path.join('data', 'reuters', 'reuters_validation.tsv')) as tsv_file:
        for line in tsv_file:
            label, text = line.split('\t')
            label = [int(x) for x in label]
            validation_split.append((label, text))
    with open(os.path.join('data', 'reuters', 'reuters_test.tsv')) as tsv_file:
        for line in tsv_file:
            label, text = line.split('\t')
            label = [int(x) for x in label]
            test_split.append((label, text))
    return train_split, validation_split, test_split


def aapd():
    train_split = list()
    validation_split = list()
    test_split = list()
    with open(os.path.join('data', 'aapd', 'aapd_train.tsv')) as tsv_file:
        for line in tsv_file:
            label, text = line.split('\t')
            label = [int(x) for x in label]
            train_split.append((label, text))
    with open(os.path.join('data', 'aapd', 'aapd_validation.tsv')) as tsv_file:
        for line in tsv_file:
            label, text = line.split('\t')
            label = [int(x) for x in label]
            validation_split.append((label, text))
    with open(os.path.join('data', 'aapd', 'aapd_test.tsv')) as tsv_file:
        for line in tsv_file:
            label, text = line.split('\t')
            label = [int(x) for x in label]
            test_split.append((label, text))
    return train_split, validation_split, test_split


def yelp14():
    train_split = list()
    validation_split = list()
    test_split = list()
    with open(os.path.join('data', 'aapd', 'aapd_train.tsv')) as tsv_file:
        for line in tsv_file:
            label, text = line.split('\t')
            label = [int(x) for x in label]
            train_split.append((label, text))
    with open(os.path.join('data', 'aapd', 'aapd_validation.tsv')) as tsv_file:
        for line in tsv_file:
            label, text = line.split('\t')
            label = [int(x) for x in label]
            validation_split.append((label, text))
    with open(os.path.join('data', 'aapd', 'aapd_test.tsv')) as tsv_file:
        for line in tsv_file:
            label, text = line.split('\t')
            label = [int(x) for x in label]
            test_split.append((label, text))
    return train_split, validation_split, test_split


def imdb():
    train_split = list()
    validation_split = list()
    test_split = list()
    with open(os.path.join('data', 'imdb', 'imdb_train.tsv')) as tsv_file:
        for line in tsv_file:
            label, text = line.split('\t')
            label = [int(x) for x in label]
            train_split.append((label, text))
    with open(os.path.join('data', 'imdb', 'imdb_validation.tsv')) as tsv_file:
        for line in tsv_file:
            label, text = line.split('\t')
            label = [int(x) for x in label]
            validation_split.append((label, text))
    with open(os.path.join('data', 'imdb', 'imdb_test.tsv')) as tsv_file:
        for line in tsv_file:
            label, text = line.split('\t')
            label = [int(x) for x in label]
            test_split.append((label, text))
    return train_split, validation_split, test_split