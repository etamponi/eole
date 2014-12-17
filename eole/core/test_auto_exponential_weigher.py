import unittest

import numpy

from eole.core.auto_exponential_weigher import AutoExponentialWeigher


__author__ = 'Emanuele Tamponi'


class AutoExponentialWeigherTest(unittest.TestCase):

    def test_train(self):
        instances = numpy.asarray([
            [+0.7, +2.0],
            [-0.5, -1.0],
            [+1.3, +0.5]
        ])
        expected_precision_matrix = numpy.asarray([0.56, 1.50])**-1
        weigher = AutoExponentialWeigher()
        weigher.train(instances)
        numpy.testing.assert_array_almost_equal(expected_precision_matrix, weigher.precisions)

    def test_weights(self):
        train_instances = numpy.asarray([
            [+2.0, +2.0],
            [-1.0, -1.0],
            [+0.5, +0.5]
        ])
        weigher = AutoExponentialWeigher(scale=4, power=1)
        weigher.train(train_instances)
        test_instances = numpy.asarray([
            [0.0, 1.0],
            [1.0, -1.],
            [1.0, 2.0]
        ])
        centroid = numpy.zeros(2)
        factor = -0.5 * 4.0 / 1.5  # -0.5 * scale / variance
        expected_weights = numpy.exp([factor * 1.0, factor * 2.0, factor * 3.0])
        numpy.testing.assert_array_almost_equal(expected_weights, weigher.get_weights(test_instances, centroid))
