import numpy

from core.interfaces import BaseSampler


__author__ = 'Emanuele Tamponi'


class DeterministicSampler(BaseSampler):

    def __init__(self, sample_percent, weigher):
        super(DeterministicSampler, self).__init__(weigher)
        self.sample_percent = float(sample_percent) / 100

    def get_sample_weights(self, instances, centroid):
        target_size = int(len(instances) * self.sample_percent)
        weights = self.get_weights(instances, centroid)
        mul_a = 0
        mul_b = 1.
        sample = numpy.asarray(weights, dtype=int).sum()
        while sample.sum() < target_size:
            mul_b *= 2
            sample = numpy.asarray(weights * mul_b, dtype=int)

        prev_sum = 0
        while True:
            curr_sum = sample.sum()
            if curr_sum == target_size:
                break
            if curr_sum == prev_sum:
                break
            mul_c = (mul_a + mul_b) / 2
            sample = numpy.asarray(weights * mul_c, dtype=int)
            if curr_sum < target_size:
                mul_a = mul_c
            else:
                mul_b = mul_c
            prev_sum = curr_sum

        return sample
