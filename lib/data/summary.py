from nltk import tokenize


def num_samples(data_x):
    return len(data_x)


def avg_num_words(data_x):
    word_counts = [len(tokenize.word_tokenize(x)) for x in data_x]
    return sum(word_counts)/len(word_counts)


def avg_num_sentences(data_x):
    sent_counts = [len(tokenize.sent_tokenize(x)) for x in data_x]
    return sum(sent_counts)/len(sent_counts)


def label_skew(data_y):
    return sum(data_y)/len(data_y)