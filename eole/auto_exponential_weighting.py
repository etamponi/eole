import numpy

from eole.trainable import Trainable


__author__ = 'Emanuele Tamponi'


class AutoExponentialWeighting(Trainable):

    def __init__(self):
        self.variance_matrix = None

    def train(self, instances):
        self.variance_matrix = numpy.diag(instances.var(axis=0))

    def apply_to_all(self, instances, centroid):
        instances = instances - centroid
        return numpy.asarray([numpy.dot(x, numpy.dot(self.variance_matrix, x.transpose())) for x in instances])
