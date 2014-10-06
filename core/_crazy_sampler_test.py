import unittest

import numpy

from core.crazy_sampler import CrazySampler


__author__ = 'Emanuele Tamponi'


class CrazySamplerTest(unittest.TestCase):

    def test_sample_generation(self):
        instances = numpy.random.randn(3, 8)
        centroid = instances[1]
        expected_weights = numpy.exp([
            -0.5 * (0 - 1)**2, -0.5 * (1 - 1)**2, -0.5 * (2 - 1)**2
        ])
        sampler = CrazySampler()
        numpy.testing.assert_array_almost_equal(expected_weights, sampler.get_sample_weights(instances, centroid))

    def test_precision_and_power(self):
        instances = numpy.random.randn(3, 8)
        centroid = instances[1]
        expected_weights = numpy.exp([
            -0.5 * (3 * (0 - 1))**4, -0.5 * (3 * (1 - 1))**4, -0.5 * (3 * (2 - 1))**4
        ])
        sampler = CrazySampler(precision=3, power=4)
        numpy.testing.assert_array_almost_equal(expected_weights, sampler.get_sample_weights(instances, centroid))

    def test_factor(self):
        instances = numpy.random.randn(5, 8)
        centroid = instances[2]
        factor = float(2)
        expected_weights = numpy.exp([
            -0.5 * ((0 - 2)/factor)**2,
            -0.5 * ((1 - 2)/factor)**2,
            -0.5 * ((2 - 2)/factor)**2,
            -0.5 * ((3 - 2)/factor)**2,
            -0.5 * ((4 - 2)/factor)**2
        ])
        sampler = CrazySampler()
        numpy.testing.assert_array_almost_equal(expected_weights, sampler.get_sample_weights(instances, centroid))