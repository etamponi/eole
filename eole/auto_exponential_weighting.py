import numpy

from eole.trainable import Trainable


__author__ = 'Emanuele Tamponi'


class AutoExponentialWeighting(Trainable):

    def __init__(self, scale):
        self.variance_matrix = None
        self.scale = scale

    def train(self, instances):
        self.variance_matrix = numpy.diag(instances.var(axis=0))

    def apply_to_all(self, instances, centroid):
        instances = instances - centroid
        return numpy.exp(
            -0.5 * self.scale * numpy.asarray([numpy.dot(x, numpy.dot(self.variance_matrix, x)) for x in instances])
        )

    def generate_sample(self, instances, centroid):
        return self.apply_to_all(instances, centroid)
