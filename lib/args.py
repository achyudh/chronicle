from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Document classification baselines")
    parser.add_argument('--model', type=str, default='LogisticRegression', choices=['LogisticRegression', 'LinearSVC'])
    parser.add_argument('--dataset', type=str, default='Reuters', choices=['Reuters', 'AAPD', 'IMDB', 'Yelp2014'])

    args = parser.parse_args()
    return args