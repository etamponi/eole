import numpy

from eole.core.interfaces import BaseWeigher


__author__ = 'Emanuele Tamponi'


class RandomSampler(BaseWeigher):

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def get_weights(self, instances, centroid):
        weights = numpy.random.randint(self.low, self.high, len(instances)).astype(float)
        return weights

    def train(self, instances):
        pass
