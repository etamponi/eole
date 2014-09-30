import numpy

from core.interfaces import BaseWeigher


__author__ = 'Emanuele Tamponi'


class AutoExponentialWeigher(BaseWeigher):

    def __init__(self, scale):
        self.precision_matrix = None
        self.scale = scale

    def train(self, instances):
        self.precision_matrix = numpy.diag(instances.var(axis=0)**-1)

    def _get_weight(self, x, centroid):
        x = x - centroid
        return numpy.exp(-0.5 * self.scale * numpy.dot(x, numpy.dot(self.precision_matrix, x)))

    def get_weights(self, instances, centroid):
        return numpy.asarray([self._get_weight(x, centroid) for x in instances])
