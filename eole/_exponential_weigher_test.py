import unittest

import numpy

from eole.exponential_weigher import ExponentialWeigher


__author__ = 'Emanuele'


class ExponentialWeigherTest(unittest.TestCase):

    def test_get_weights(self):
        centroid = numpy.array([0, 0])
        weighting = ExponentialWeigher(precision=1, power=2)
        weights = weighting.get_weights([[3, 4], [6, 8]], centroid)
        self.assertTrue(numpy.all([numpy.exp(-0.5 * 25), numpy.exp(-0.5 * 100)] == weights))

    def test_sample_percent(self):
        instances = numpy.random.randn(100, 2)
        centroid = instances[5]
        weighting = ExponentialWeigher(precision=1, power=2, sample_percent=70)
        sample = weighting.generate_sample(instances, centroid)
        self.assertEqual(sum(sample != 0), 70)
