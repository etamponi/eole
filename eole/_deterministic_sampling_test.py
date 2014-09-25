import unittest

import numpy

from eole.deterministic_sampling import DeterministicSampling
from eole.exponential_weighting import ExponentialWeighting


__author__ = 'Emanuele Tamponi'


class DeterministicSamplingTest(unittest.TestCase):

    def test_generate_sample(self):
        instances = numpy.random.randn(10, 3)
        centroid = numpy.asarray([0, 0, 0])
        sampler = DeterministicSampling(sample_percent=1000, weighting=ExponentialWeighting(1, 2))
        sample = sampler.generate_sample(instances, centroid)
        self.assertTrue(abs(sample.sum() - 100) <= 1)