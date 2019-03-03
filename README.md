# Chronicle
This project is aimed at establishing the baselines scores for document classification datasets such as Reuters (RCV1), AAPD, IMDB, and Yelp 2014 using basic feature selection techniques and classifiers from Scikit-Learn. The datasets are stored separately in this repository: https://git.uwaterloo.ca/arkeshav/Castor-Data. 

Available datasets are `Reuters`, `AAPD`, `IMDB`, `Yelp2014` and `AGNews`, and available models include `LogisticRegression` and `LinearSVC`. For instance, to perform Logistic Regression on the IMDB dataset, the following command can be used: `python -m lib --model LogisticRegression --dataset IMDB`.

Dataset statistics can also be printed for the aforementioned datasets, and for the TREC Robust 2004, Robust 2005 and Robust 2004+05 datasets using `python -m lib.data --summary --dataset <dataset_name>`.

## References:
* Lewis, David D., et al. "Rcv1: A new benchmark collection for text categorization research." Journal of machine learning research 5.Apr (2004): 361-397.
* Tang, Duyu, Bing Qin, and Ting Liu. "Document modeling with gated recurrent neural network for sentiment classification." Proceedings of the 2015 conference on empirical methods in natural language processing. 2015.
