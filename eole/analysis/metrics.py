import numpy
import sklearn

__author__ = 'Emanuele Tamponi'


def precision_recall_f1_score(y_true, y_pred):
    precisions, recalls, f1s, counts = sklearn.metrics.precision_recall_fscore_support(
        y_true, y_pred, average=None, warn_for=()
    )
    return (
        numpy.average(precisions, weights=counts),
        numpy.average(recalls, weights=counts),
        numpy.average(f1s, weights=counts)
    )
