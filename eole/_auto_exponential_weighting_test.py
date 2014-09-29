import unittest

import numpy

from eole.auto_exponential_weighting import AutoExponentialWeighting


__author__ = 'Emanuele Tamponi'


class AutoExponentialWeightingTest(unittest.TestCase):

    def test_train(self):
        instances = numpy.asarray([
            [+0.7, +2.0],
            [-0.5, -1.0],
            [+1.3, +0.5]
        ])
        expected_variances = numpy.diag(numpy.asarray([0.56, 1.50]))
        weighting = AutoExponentialWeighting(scale=1)
        weighting.train(instances)
        numpy.testing.assert_array_almost_equal(expected_variances, weighting.variance_matrix)
