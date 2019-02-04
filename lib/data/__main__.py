from lib.data import summary
from lib.data import fetch
from lib.data.args import get_args


if __name__ == '__main__':
    args = get_args()

    if args.dataset.lower() == 'reuters':
        train_split, validation_split, test_split = fetch.reuters()
        multiple_topics = False
    elif args.dataset.lower() == 'aapd':
        train_split, validation_split, test_split = fetch.aapd()
        multiple_topics = False
    elif args.dataset.lower() == 'imdb':
        train_split, validation_split, test_split = fetch.imdb()
        multiple_topics = False
    elif args.dataset.lower() == 'yelp2014':
        train_split, validation_split, test_split = fetch.yelp14()
        multiple_topics = False
    elif args.dataset.lower() == 'robust04':
        train_split, validation_split, topics = fetch.robust04()
        multiple_topics = True
    elif args.dataset.lower() == 'robust05':
        train_split, validation_split, topics = fetch.robust05()
        multiple_topics = True
    elif args.dataset.lower() == 'robust45':
        train_split, validation_split, topics = fetch.robust45()
        multiple_topics = True
    else:
        raise Exception("Unsupported dataset")

    if multiple_topics:
        a = dict()
        for topic in topics:
            data_x = [x[1] for x in train_split[topic]]
            data_x.extend([x[1] for x in validation_split[topic]])

            data_y = [x[0] for x in train_split[topic]]
            data_y.extend([x[0] for x in validation_split[topic]])

            print("Topic:", topic)
            print("Number of samples:", summary.num_samples(data_x))
            print("Label skew:", summary.label_skew(data_y))
            # print("Average number of words in a sample:", summary.avg_num_words(data_x))
            # print("Average number of sentences in a sample:", summary.avg_num_sentences(data_x))
            print("-----")
    else:
        data_x = [x[1] for x in train_split]
        data_x.extend([x[1] for x in test_split])
        data_x.extend([x[1] for x in validation_split])

        print("Number of samples:", summary.num_samples(data_x))
        print("Average number of words in a sample:", summary.avg_num_words(data_x))
        print("Average number of sentences in a sample:", summary.avg_num_sentences(data_x))




