import numpy

__author__ = 'Emanuele Tamponi'


class Report(object):

    def __init__(self, experiment):
        pass

    def analyze_run(self, prediction_matrix, labels):
        self.accuracy_sample = [numpy.asarray([0.5, 0.75, 0.75])]