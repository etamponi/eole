from copy import deepcopy
import os

import numpy
from pymc.six.moves import cPickle
from sklearn.metrics.metrics import accuracy_score


__author__ = 'Emanuele Tamponi'


class Report(object):

    def __init__(self, experiment):
        self.experiment = deepcopy(experiment)
        self.sample_size = experiment.folds * experiment.repetitions
        self.accuracy_sample = numpy.zeros((self.sample_size, experiment.ensemble.n_experts))
        self.current_run = 0

    def analyze_run(self, prediction_matrix, labels):
        self.accuracy_sample[self.current_run] = numpy.asarray(
            [accuracy_score(labels, prediction_matrix[:, i]) for i in range(self.accuracy_sample.shape[1])]
        )
        self.current_run += 1

    def synthesis(self):
        if not self.is_ready():
            raise ReportNotReady()
        mean = self.accuracy_sample.mean(axis=0)
        variance = self.accuracy_sample.var(axis=0)
        return mean, variance

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
