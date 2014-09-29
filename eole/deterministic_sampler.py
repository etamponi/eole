import numpy

from eole.interfaces import Sampler


__author__ = 'Emanuele Tamponi'


class DeterministicSampler(Sampler):

    def __init__(self, sample_percent, weigher):
        super(DeterministicSampler, self).__init__(weigher)
        self.sample_percent = float(sample_percent) / 100

    def generate_sample(self, instances, centroid):
        target_size = int(len(instances) * self.sample_percent)
        weights = self.get_weights(instances, centroid)
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