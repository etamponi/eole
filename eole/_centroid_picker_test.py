import unittest

import numpy

from eole.centroid_picker import RandomCentroidPicker, AlmostRandomCentroidPicker


__author__ = 'Emanuele'


class CentroidPickerTest(unittest.TestCase):

    def test_pick(self):
        for cp in [AlmostRandomCentroidPicker(), RandomCentroidPicker()]:
            instances = numpy.random.rand(10, 2)
            centroids = cp.pick(instances, 3)
            self.assertEqual((3, 2), centroids.shape)
            self.assertTrue(all(c in instances for c in centroids))

    def test_repeatability_and_randomness(self):
        for cp in [AlmostRandomCentroidPicker(), RandomCentroidPicker()]:
            instances = numpy.random.rand(10, 2)
            numpy.random.seed(1)
            c1 = cp.pick(instances, 3)
            c2 = cp.pick(instances, 3)
            self.assertTrue(numpy.any(c1 != c2))
            numpy.random.seed(1)
            c2 = cp.pick(instances, 3)
            self.assertTrue(numpy.all(c1 == c2))
