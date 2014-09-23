import unittest

import numpy

from eole.exponential_weighting import ExponentialWeighting
from eole.generalized_bootstrap import GeneralizedBootstrap


__author__ = 'Emanuele'


class GeneralizedBootstrapTest(unittest.TestCase):

    def setUp(self):
        self.centroid = [0, 0]
        weighting = ExponentialWeighting(precision=1, power=2)
        self.gb = GeneralizedBootstrap(sample_percent=500, weighting=weighting)
        self.instances = numpy.random.rand(10, 2)

    def test_generate_sample(self):
        sample = self.gb.generate_sample(self.instances, self.centroid)
        self.assertEqual(10, len(sample))
        self.assertEqual(50, sample.sum())

    def test_repeatability(self):
        numpy.random.seed(1)
        sample1 = self.gb.generate_sample(self.instances, self.centroid)
        numpy.random.seed(1)
        sample2 = self.gb.generate_sample(self.instances, self.centroid)
        self.assertTrue(numpy.all(sample1 == sample2))

    def test_randomness(self):
        numpy.random.seed(1)
        sample1 = self.gb.generate_sample(self.instances, self.centroid)
        sample2 = self.gb.generate_sample(self.instances, self.centroid)
        self.assertTrue(numpy.any(sample1 != sample2))
