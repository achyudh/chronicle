from lib.data import summary
from lib.data import fetch
from lib.data.args import get_args


if __name__ == '__main__':
    args = get_args()

    if args.dataset.lower() == 'reuters':
        train_split, validation_split, test_split = fetch.reuters()
        single_label = False
    elif args.dataset.lower() == 'aapd':
        train_split, validation_split, test_split = fetch.aapd()
        single_label = False
    elif args.dataset.lower() == 'imdb':
        train_split, validation_split, test_split = fetch.imdb()
        single_label = True
    elif args.dataset.lower() == 'yelp2014':
        train_split, validation_split, test_split = fetch.yelp14()
        single_label = True
    else:
        raise Exception("Unsupported dataset")

    data_x = [x[1] for x in train_split]
    data_x.extend([x[1] for x in test_split])
    data_x.extend([x[1] for x in validation_split])

    print("Number of samples:", summary.num_samples(data_x))
    print("Average number of words in a sample:", summary.avg_num_words(data_x))
    print("Average number of sentences in a sample:", summary.avg_num_sentences(data_x))




