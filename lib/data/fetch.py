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


def imdb():
    dataset = list()
    for filename in ['part1.json', 'part2.json', 'part3.json']:
        with open(os.path.join('data', 'imdb', filename)) as json_file:
            documents = json.load(json_file)
            for document in documents:
                if isinstance(document["review"], dict):
                    try:
                        text = document["review"]["title"] + " " + document["review"]["review"]
                        dataset.append((document["review"]["rating"], text))
                    except:
                        print("Irregular formatting:", document)
                        text = document["title"] + " " + document["review"]
                        dataset.append((document["rating"], text))
                else:
                    text = document["title"] + " " + document["review"]
                    dataset.append((document["rating"], text))
    print(len(dataset))
    train_split, test_split = train_test_split(dataset, test_size=0.3, random_state=37)
    train_split, validation_split = train_test_split(dataset, test_size=0.25, random_state=59)
    return train_split, validation_split, test_split


if __name__ == "__main__":
    imdb()
