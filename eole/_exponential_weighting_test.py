import unittest

import numpy

from eole.exponential_weighting import ExponentialWeighting


__author__ = 'Emanuele'


class ExponentialWeightingTest(unittest.TestCase):

    def test_apply_to(self):
        centroid = numpy.array([1])
        weighting = ExponentialWeighting(precision=1, power=2)
        weight = weighting.apply_to([2], centroid=centroid)
        self.assertEqual(numpy.exp(-0.5 * (2 - 1)**2), weight)

        centroid = numpy.array([0, 0])
        weighting = ExponentialWeighting(precision=2, power=2)
        weight = weighting.apply_to([3, 4], centroid)
        self.assertEqual(numpy.exp(-0.5 * 2 * (9 + 16)), weight)

    def test_apply_to_all(self):
        centroid = numpy.array([0, 0])
        weighting = ExponentialWeighting(precision=1, power=2)
        weights = weighting.apply_to_all([[3, 4], [6, 8]], centroid)
        self.assertTrue(numpy.all([numpy.exp(-0.5 * 25), numpy.exp(-0.5 * 100)] == weights))

    def test_sample_percent(self):
        instances = numpy.random.randn(100, 2)
        centroid = instances[5]
        weighting = ExponentialWeighting(precision=1, power=2, sample_percent=70)
        sample = weighting.generate_sample(instances, centroid)
        self.assertEqual(sum(sample != 0), 70)
