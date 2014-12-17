import numpy
from scipy.spatial import distance

from eole.core.interfaces import SamplerWeigher


__author__ = 'Emanuele Tamponi'


class CrazySampler(SamplerWeigher):

    def __init__(self, precision=1, power=2):
        self.precision = precision
        self.power = power

    def train(self, instances):
        pass

    def get_sample_weights(self, instances, centroid):
        if isinstance(centroid, int):
            c_index = centroid
        else:
            distances = numpy.asarray([distance.euclidean(x, centroid) for x in instances])
            c_index = distances.argmin()
        return numpy.asarray([self._get_weight(i, c_index, len(instances)) for i in range(len(instances))])

    def _get_weight(self, i, c_index, size):
        factor = size / 2
        relative_pos = float(i - c_index)
        if relative_pos > factor:
            relative_pos -= size
        if relative_pos < -factor:
            relative_pos += size
        relative_pos /= factor
        return numpy.exp(-0.5 * (self.precision * relative_pos)**self.power)

    def get_weights(self, instances, centroid):
        return 1.0


class CrazyPicker(object):

    def pick(self, instances, labels, n_centroids):
        indices = numpy.linspace(0, len(instances)-1, n_centroids).astype(int)
        return indices