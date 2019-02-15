from nltk import tokenize
from collections import defaultdict


def num_samples(data_x):
    return len(data_x)


def avg_num_words(data_x):
    word_counts = [len(tokenize.word_tokenize(x)) for x in data_x]
    return sum(word_counts)/len(word_counts)


def avg_num_sentences(data_x):
    sent_counts = [len(tokenize.sent_tokenize(x)) for x in data_x]
    return sum(sent_counts)/len(sent_counts)


def label_freq(data_y):
    counts = defaultdict(int)
    total = 0
    for label in data_y:
        for digit in range(len(label)):
            if label[digit] == '1':
                counts[digit] += 1
        total += 1
    return [count/total for count in counts.values()]
