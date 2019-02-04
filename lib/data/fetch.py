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
    with open(os.path.join('data', 'yelp14', 'yelp2014_train.tsv')) as tsv_file:
        for line in tsv_file:
            label, text = line.split('\t')
            label = [int(x) for x in label]
            train_split.append((label, text))
    with open(os.path.join('data', 'yelp14', 'yelp2014_validation.tsv')) as tsv_file:
        for line in tsv_file:
            label, text = line.split('\t')
            label = [int(x) for x in label]
            validation_split.append((label, text))
    with open(os.path.join('data', 'yelp14', 'yelp2014_test.tsv')) as tsv_file:
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


def robust04():
    train_split, validation_split = dict(), dict()
    topics = ['307', '310', '321', '325', '330', '336', '341', '344', '345', '347', '350', '353', '354', '355', '356',
              '362', '363', '367', '372', '375', '378', '379', '389', '393', '394', '397', '399', '400', '404', '408',
              '414', '416', '419', '422', '423', '426', '427', '433', '435', '436', '439', '442', '443', '445', '614',
              '620', '626', '646', '677', '690']

    for topic in topics:
        train_split[topic] = list()
        validation_split[topic] = list()
        with open(os.path.join('data', 'trec', 'robust04_train_%s.tsv' % topic)) as tsv_file:
            for line in tsv_file:
                label, docid, text = line.split('\t')
                label = 0 if label == '01' else 1
                train_split[topic].append((label, text))
        with open(os.path.join('data', 'trec', 'robust04_dev_%s.tsv' % topic)) as tsv_file:
            for line in tsv_file:
                label, docid, text = line.split('\t')
                label = 0 if label == '01' else 1
                validation_split[topic].append((label, text))
    return train_split, validation_split, topics


def robust45():
    train_split, validation_split = dict(), dict()
    topics = ['307', '310', '321', '325', '330', '336', '341', '344', '345', '347', '350', '353', '354', '355', '356',
              '362', '363', '367', '372', '375', '378', '379', '389', '393', '394', '397', '399', '400', '404', '408',
              '414', '416', '419', '422', '423', '426', '427', '433', '435', '436', '439', '442', '443', '445', '614',
              '620', '626', '646', '677', '690']

    for topic in topics:
        train_split[topic] = list()
        validation_split[topic] = list()
        with open(os.path.join('data', 'trec', 'robust45_train_%s.tsv' % topic)) as tsv_file:
            for line in tsv_file:
                label, docid, text = line.split('\t')
                label = 0 if label == '01' else 1
                train_split[topic].append((label, text))
        with open(os.path.join('data', 'trec', 'robust45_dev_%s.tsv' % topic)) as tsv_file:
            for line in tsv_file:
                label, docid, text = line.split('\t')
                label = 0 if label == '01' else 1
                validation_split[topic].append((label, text))
    return train_split, validation_split, topics


def robust05():
    train_split, validation_split = dict(), dict()
    topics = ['307', '310', '325', '330', '336', '341', '344', '345', '347', '353', '354', '362', '363', '367', '372',
              '375', '378', '389', '393', '394', '397', '399', '404', '408', '416', '419', '426', '427', '433', '435',
              '436', '439', '443']

    for topic in topics:
        train_split[topic] = list()
        validation_split[topic] = list()
        with open(os.path.join('data', 'trec', 'robust05_train_%s.tsv' % topic)) as tsv_file:
            for line in tsv_file:
                label, docid, text = line.split('\t')
                label = 0 if label == '01' else 1
                train_split[topic].append((label, text))
        with open(os.path.join('data', 'trec', 'robust05_dev_%s.tsv' % topic)) as tsv_file:
            for line in tsv_file:
                label, docid, text = line.split('\t')
                label = 0 if label == '01' else 1
                validation_split[topic].append((label, text))
    return train_split, validation_split, topics