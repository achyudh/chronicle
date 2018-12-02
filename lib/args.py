from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Document classification baselines")
    parser.add_argument('--model', type=str, default='logistic_regression', choices=['LogisticRegression', 'LinearSVM'])
    parser.add_argument('--dataset', type=str, default='reuters', choices=['Reuters', 'AAPD', 'IMDB', 'Yelp2014'])

    args = parser.parse_args()
    return args