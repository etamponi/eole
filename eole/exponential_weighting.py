import numpy
from scipy.spatial import distance

from eole.trainable import Trainable


__author__ = 'Emanuele'


class ExponentialWeighting(Trainable):

    def train(self, instances):
        pass

    def __init__(self, precision, power, sample_percent=None):
        self.precision = precision
        self.power = power
        self.sample_percent = None if sample_percent is None else float(sample_percent)/100

    def apply_to(self, x, centroid):
        return numpy.exp(-0.5 * self.precision * distance.euclidean(x, centroid)**self.power)

    def apply_to_all(self, instances, centroid, normalize=False):
        weights = numpy.array([self.apply_to(x, centroid) for x in instances])
        if normalize:
            weights /= weights.sum()
        return weights

    def generate_sample(self, instances, centroid):
        sample = self.apply_to_all(instances, centroid)
        if self.sample_percent is not None:
            choices = numpy.random.choice(len(instances), size=int(len(instances)*self.sample_percent), replace=False)
            mask = numpy.zeros(len(instances))
            mask[choices] = 1
            sample = mask * sample
        return sample
