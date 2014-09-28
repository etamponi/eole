import numpy

from eole.trainable import Trainable


__author__ = 'Emanuele Tamponi'


class DeterministicSampling(Trainable):

    def train(self, instances):
        self.weighting.train(instances)

    def __init__(self, sample_percent, weighting):
        self.sample_percent = float(sample_percent) / 100
        self.weighting = weighting

    def generate_sample(self, instances, centroid):
        target_size = int(len(instances) * self.sample_percent)
        weights = self.weighting.apply_to_all(instances, centroid)
        mul_a = 0
        mul_b = 1.
        sample = numpy.asarray(weights, dtype=int).sum()
        while sample.sum() < target_size:
            mul_b *= 2
            sample = numpy.asarray(weights * mul_b, dtype=int)

        while abs(sample.sum() - target_size) > 1:
            mul_c = (mul_a + mul_b) / 2
            sample = numpy.asarray(weights * mul_c, dtype=int)
            if sample.sum() < target_size:
                mul_a = mul_c
            else:
                mul_b = mul_c

        return sample
