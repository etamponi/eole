import numpy

from eole.interfaces import Weigher


__author__ = 'Emanuele Tamponi'


class AutoExponentialWeigher(Weigher):

    def __init__(self, scale):
        self.variance_matrix = None
        self.scale = scale

    def train(self, instances):
        self.variance_matrix = numpy.diag(instances.var(axis=0))

    def _get_weight(self, x, centroid):
        x = x - centroid
        return numpy.exp(-0.5 * self.scale * numpy.dot(x, numpy.dot(self.variance_matrix, x)))

    def get_weights(self, instances, centroid):
        return numpy.asarray([self._get_weight(x, centroid) for x in instances])
