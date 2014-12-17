import numpy
from scipy.spatial import distance

from eole.core.interfaces import BaseWeigher


__author__ = 'Emanuele Tamponi'


class AutoExponentialWeigher(BaseWeigher):

    def __init__(self, scale=1., power=2., take_average=False):
        self.precisions = None
        self.scale = scale
        self.power = power
        self.take_average = take_average

    def train(self, instances):
        self.precisions = pow(instances.var(axis=0), -1)
        if self.take_average:
            self.precisions[:] = numpy.mean(self.precisions)

    def _get_weight(self, x, centroid):
        return numpy.exp(-0.5 * self.scale * distance.wminkowski(x, centroid, self.power, self.precisions))

    def get_weights(self, instances, centroid):
        return numpy.asarray([self._get_weight(x, centroid) for x in instances])
