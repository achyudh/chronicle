from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Chronicle - lib/data")
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--dataset', type=str, default='Reuters', choices=['Reuters', 'AAPD', 'IMDB', 'Yelp2014', 'Robust04', 'Robust45', 'Robust05'])

    args = parser.parse_args()
    return args