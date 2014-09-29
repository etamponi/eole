import numpy
from scipy.spatial import distance

from eole.interfaces import Weigher


__author__ = 'Emanuele'


class InverseDistance(Weigher):

    def __init__(self, power, offset):
        self.power = power
        self.offset = offset

    def train(self, instances):
        pass

    def _get_weight(self, x, centroid):
        return 1.0 / (distance.euclidean(x, centroid) + self.offset)**self.power

    def get_weights(self, instances, centroid):
        return [self._get_weight(x, centroid) for x in instances]


class FakeEstimator(object):

    def __init__(self, n_labels, label_cycle, prob):
        self.n_labels = n_labels
        self.label_cycle = label_cycle
        self.prob = prob

    def predict(self, instances):
        n = len(self.label_cycle)
        ret = [self.label_cycle[i % n] for i in range(len(instances))]
        return numpy.asarray(ret)

    def predict_proba(self, instances):
        n = len(self.label_cycle)
        ret = []
        for i in range(len(instances)):
            probs = (1.0 - self.prob) / (self.n_labels - 1.0) * numpy.ones(self.n_labels)
            probs[self.label_cycle[i % n]] = self.prob
            ret.append(probs)
        return numpy.asarray(ret)
