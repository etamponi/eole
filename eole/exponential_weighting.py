import numpy

__author__ = 'Emanuele'


class ExponentialWeighting(object):

    def __init__(self, precision, power):
        self.precision = precision
        self.power = power

    def apply_to(self, x, centroid):
        return numpy.exp(-0.5 * self.precision * numpy.linalg.norm(x - centroid)**self.power)

    def apply_to_all(self, instances, centroid, normalize=False):
        weights = numpy.array([self.apply_to(x, centroid) for x in instances])
        if normalize:
            weights /= weights.sum()
        return weights

    def generate_sample(self, instances, centroid):
        return self.apply_to_all(instances, centroid)
