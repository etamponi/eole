import unittest

from sympy.external.tests.test_autowrap import numpy

from core.interfaces import BaseWeigher
from core.local_expert import LocalExpert


__author__ = 'Emanuele'


class LocalExpertTest(unittest.TestCase):

    def test_train_returns_self(self):
        instances = numpy.random.randn(100, 3)
        labels = numpy.random.choice(["a", "b"], size=100)
        expert = LocalExpert(base_estimator=FakeEstimator(), sampler_weigher=FakeWeigher())
        self.assertEqual(expert, expert.train(instances, labels, instances[0]))


class FakeEstimator(object):

    def fit(self, instances, labels, **kwargs):
        return self


class FakeWeigher(BaseWeigher):

    def train(self, instances):
        pass

    def get_weights(self, instances, centroid):
        return numpy.asarray([centroid[0] * x[0] for x in instances])