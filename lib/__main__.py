from lib.model import linear_svc, logistic_regression
from lib.data import fetch
from lib.args import get_args
import numpy as np

if __name__ == '__main__':
    args = get_args()
    accuracy_values = list()
    precision_values = list()
    recall_values = list()
    f1_values = list()

    if args.dataset.lower == 'reuters':
        train_split, validation_split, test_split = fetch.reuters()
        single_label = False
    elif args.dataset.lower == 'aapd':
        train_split, validation_split, test_split = fetch.aapd()
        single_label = False
    elif args.dataset.lower == 'imdb':
        train_split, validation_split, test_split = fetch.imdb()
        single_label = True
    elif args.dataset.lower == 'yelp14':
        train_split, validation_split, test_split = fetch.yelp14()
        single_label = True
    else:
        raise Exception("Unsupported dataset")

    train_x = [x[1] for x in train_split]
    train_y = np.array([x[0] for x in train_split])
    test_x = [x[1] for x in test_split]
    test_y = np.array([x[0] for x in test_split])
    validation_x = [x[1] for x in validation_split]
    validation_y = np.array([x[0] for x in validation_split])

    if single_label:
        train_y = [np.where(r == 1)[0][0] for r in train_y]
        test_y = [np.where(r == 1)[0][0] for r in test_y]
        validation_y = [np.where(r == 1)[0][0] for r in validation_y]

    if args.model == 'LogisticRegression':
        model = logistic_regression
    elif args.model == 'LinearSVC':
        model = linear_svc
    else:
        raise Exception("Unsupported model")

    for random_state in (1, 3, 34, 345, 3454):
        classifier, vectorizer = model.train(train_x, train_y, single_label=single_label, random_state=random_state)
        print("Dev:", model.evaluate(classifier, vectorizer, validation_x, validation_y))
        accuracy, precision, recall, f1 = model.evaluate(classifier, vectorizer, test_x, test_y)
        print("Test:", accuracy, precision, recall, f1)
        accuracy_values.append(accuracy)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    for metric_name, metric_values in (("Accuracy:", accuracy_values), ("Precision:", precision_values),
                                       ("Recall:", recall_values), ("F-measure:", f1_values)):
        metric_median = np.median(np.array(metric_values))
        metric_range = (min(metric_values) - metric_median, max(metric_values) - metric_median)
        print("\n" + metric_name)
        print(metric_median, metric_range)
