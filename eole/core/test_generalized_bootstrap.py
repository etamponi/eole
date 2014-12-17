import unittest

import numpy

from eole.core.exponential_weigher import ExponentialWeigher
from eole.core.generalized_bootstrap import GeneralizedBootstrap


__author__ = 'Emanuele'


class GeneralizedBootstrapTest(unittest.TestCase):

    def setUp(self):
        self.centroid = numpy.asarray([0, 0])
        self.gb = GeneralizedBootstrap(sample_percent=500, weigher=ExponentialWeigher(precision=1, power=2))
        self.instances = numpy.random.rand(100, 2)

    def test_get_sample_weights(self):
        sample = self.gb.get_sample_weights(self.instances, self.centroid)
        self.assertEqual(100, len(sample))
        self.assertEqual(500, sample.sum())

    def test_repeatability(self):
        numpy.random.seed(1)
        sample1 = self.gb.get_sample_weights(self.instances, self.centroid)
        numpy.random.seed(1)
        sample2 = self.gb.get_sample_weights(self.instances, self.centroid)
        self.assertTrue(numpy.all(sample1 == sample2))

    def test_randomness(self):
        numpy.random.seed(1)
        sample1 = self.gb.get_sample_weights(self.instances, self.centroid)
        sample2 = self.gb.get_sample_weights(self.instances, self.centroid)
        self.assertTrue(numpy.any(sample1 != sample2))
