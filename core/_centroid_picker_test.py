import unittest

import numpy

from core.centroid_picker import RandomCentroidPicker, AlmostRandomCentroidPicker


__author__ = 'Emanuele'


class CentroidPickerTest(unittest.TestCase):

    def setUp(self):
        self.implementations = [
            AlmostRandomCentroidPicker(), RandomCentroidPicker()
        ]

    def test_pick(self):
        for cp in self.implementations:
            instances = numpy.random.rand(100, 2)
            labels = numpy.random.choice(["a", "b"], size=100)
            centroids = cp.pick(instances, labels, 10)
            self.assertEqual((10, 2), centroids.shape)
            for c in centroids:
                self.assertIn(c, instances, "{} not present".format(c))

    def test_repeatability_and_randomness(self):
        exclude = {}
        for cp in self.implementations:
            if cp.__class__ in exclude:
                continue
            instances = numpy.random.rand(100, 2)
            labels = numpy.random.choice(["a", "b"], size=100)
            numpy.random.seed(1)
            c1 = cp.pick(instances, labels, 3)
            c2 = cp.pick(instances, labels, 3)
            self.assertTrue(numpy.any(c1 != c2))
            numpy.random.seed(1)
            c2 = cp.pick(instances, labels, 3)
            self.assertTrue(numpy.all(c1 == c2))
