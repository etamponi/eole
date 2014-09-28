import numpy
from scipy.spatial import distance

from eole.trainable import Trainable


__author__ = 'Emanuele'


class InverseDistance(Trainable):

    def train(self, instances):
        pass

    def __init__(self, power, offset):
        self.power = power
        self.offset = offset

    def apply_to(self, x, c):
        return 1.0 / (distance.euclidean(x, c) + self.offset)**self.power

    def apply_to_all(self, instances, c):
        return [self.apply_to(x, c) for x in instances]


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
