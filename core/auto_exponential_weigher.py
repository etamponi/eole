import numpy
from scipy.spatial import distance

from core.interfaces import BaseWeigher


__author__ = 'Emanuele Tamponi'


class AutoExponentialWeigher(BaseWeigher):

    def __init__(self, scale=1., power=2., average=False):
        self.precisions = None
        self.scale = scale
        self.power = power
        self.average = average

    def train(self, instances):
        self.precisions = self.scale * instances.var(axis=0)**-1
        if self.average:
            self.precisions[:] = self.precisions.mean()

    def _get_weight(self, x, centroid):
        return numpy.exp(-0.5 * distance.wminkowski(x, centroid, self.power, self.precisions))

    def get_weights(self, instances, centroid):
        return numpy.asarray([self._get_weight(x, centroid) for x in instances])
