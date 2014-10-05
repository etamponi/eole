import numpy
from sklearn import preprocessing

from analysis.dataset_utils import ArffLoader
import evaluation


__author__ = 'Emanuele Tamponi'


def main():
    for dataset_name in evaluation.dataset_names():
        dataset_path = "evaluation/datasets/{}.arff".format(dataset_name)
        X, y = ArffLoader(dataset_path).load_dataset()
        X = preprocessing.MinMaxScaler().fit_transform(X)

        n, n_feats = X.shape
        n_classes = len(numpy.unique(y))
        precisions = pow(X.var(axis=0), -1)
        mean_precision = numpy.mean(precisions)
        precision_var = numpy.var(precisions)

        numpy.set_printoptions(precision=3, suppress=True)
        print dataset_name
        print "\t          size:{: 5d}    ;           features: {: 6d}    ; classes: {}".format(n, n_feats, n_classes)
        print "\tmean precision: {: 8.3f}; precision variance: {:10.3f}".format(mean_precision, precision_var)
        print "\tprecisions:\n{}".format(precisions)
        print "\n\n"


if __name__ == "__main__":
    main()
