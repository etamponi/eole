import numpy
from scipy.spatial import distance

from core.interfaces import BaseWeigher


__author__ = 'Emanuele'


class ExponentialWeigher(BaseWeigher):

    def __init__(self, precision, power, dist_measure=distance.euclidean, sample_percent=None):
        self.precision = precision
        self.power = power
        self.dist_measure = dist_measure
        self.sample_percent = None if sample_percent is None else float(sample_percent)/100

    def train(self, instances):
        pass

    def _get_weight(self, x, centroid):
        return numpy.exp(-0.5 * self.precision * self.dist_measure(x, centroid)**self.power)

    def get_weights(self, instances, centroid):
        return numpy.asarray([self._get_weight(x, centroid) for x in instances])

    def get_sample_weights(self, instances, centroid):
        sample = self.get_weights(instances, centroid)
        if self.sample_percent is not None:
            choices = numpy.random.choice(len(instances), size=int(len(instances)*self.sample_percent), replace=False)
            mask = numpy.zeros(len(instances))
            mask[choices] = 1
            sample = mask * sample
        return sample
