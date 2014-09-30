import unittest

import numpy

from core.deterministic_sampler import DeterministicSampler
from core.exponential_weigher import ExponentialWeigher


__author__ = 'Emanuele Tamponi'


class DeterministicSamplerTest(unittest.TestCase):

    def test_get_sample_weights(self):
        instances = numpy.random.randn(10, 3)
        centroid = instances[0]
        sampler = DeterministicSampler(sample_percent=1000, weigher=ExponentialWeigher(1, 2))
        sample = sampler.get_sample_weights(instances, centroid)
        self.assertTrue(abs(sample.sum() - 100) <= 1)