import unittest

import numpy

from core.crazy_sampler import CrazyPicker


__author__ = 'Emanuele Tamponi'


class CrazyPickerTest(unittest.TestCase):

    def test_picker(self):
        instances = numpy.random.randn(101, 8)
        expected_centroids = numpy.asarray([
            instances[0],
            instances[10],
            instances[20],
            instances[30],
            instances[40],
            instances[50],
            instances[60],
            instances[70],
            instances[80],
            instances[90],
            instances[100]
        ])
        picker = CrazyPicker()
        numpy.testing.assert_array_equal(expected_centroids, instances[picker.pick(instances, None, 11)])