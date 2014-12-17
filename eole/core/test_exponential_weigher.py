import unittest

import numpy

from eole.core.exponential_weigher import ExponentialWeigher


__author__ = 'Emanuele'


class ExponentialWeigherTest(unittest.TestCase):

    def test_get_weights(self):
        centroid = numpy.array([0, 0])
        weigher = ExponentialWeigher(precision=3, power=2)
        instances = numpy.asarray([
            [3., 4.],
            [6., 8.]
        ])
        expected_weights = numpy.exp([-0.5 * 3 * 25, -0.5 * 3 * 100])
        numpy.testing.assert_array_almost_equal(expected_weights, weigher.get_weights(instances, centroid))

    def test_sample_percent(self):
        instances = numpy.random.randn(100, 2)
        centroid = instances[5]
        weigher = ExponentialWeigher(precision=1, power=2, sample_percent=70)
        self.assertEqual((weigher.get_sample_weights(instances, centroid) != 0).sum(), 70)
