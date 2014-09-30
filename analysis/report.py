from copy import deepcopy
import os
import cPickle

import numpy
from sklearn.metrics.metrics import accuracy_score

from analysis.metrics import precision_recall_f1_score


__author__ = 'Emanuele Tamponi'


class Report(object):

    def __init__(self, experiment):
        self.experiment = deepcopy(experiment)
        self.sample_size = experiment.n_folds * experiment.n_repetitions
        shape = (self.sample_size, experiment.ensemble.n_experts)
        self.accuracy_sample = numpy.zeros(shape)
        self.precision_sample = numpy.zeros(shape)
        self.recall_sample = numpy.zeros(shape)
        self.f1_sample = numpy.zeros(shape)
        self.current_run = 0

    def analyze_run(self, prediction_matrix, labels):
        for j in range(self.accuracy_sample.shape[1]):
            predictions = prediction_matrix[:, j]
            self.accuracy_sample[self.current_run][j] = accuracy_score(labels, predictions)

            precision, recall, f1 = precision_recall_f1_score(labels, predictions)
            self.precision_sample[self.current_run][j] = precision
            self.recall_sample[self.current_run][j] = recall
            self.f1_sample[self.current_run][j] = f1
        self.current_run += 1

    def synthesis(self):
        if not self.is_ready():
            raise ReportNotReady()
        ret = {}
        for score in ["accuracy", "precision", "recall", "f1"]:
            sample = self.__dict__[score+"_sample"]
            ret[score] = {
                "mean": sample.mean(axis=0),
                "variance": sample.var(axis=0)
            }
        return ret

    def is_ready(self):
        return self.current_run == self.sample_size

    def dump(self, directory):
        if not os.path.isdir(directory):
            raise ReportDirectoryError()
        file_name = "{}/{}.rep".format(directory, self.experiment.name)
        if os.path.isfile(file_name):
            raise ReportFileAlreadyExists()
        with open(file_name, "w") as f:
            cPickle.dump(self, f)

    @staticmethod
    def load(file_name):
        with open(file_name) as f:
            return cPickle.load(f)


class ReportNotReady(Exception):
    pass


class ReportDirectoryError(Exception):
    pass


class ReportFileAlreadyExists(Exception):
    pass
