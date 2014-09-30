import unittest

import numpy
from sklearn.tree.tree import DecisionTreeClassifier

from core.generalized_bootstrap import GeneralizedBootstrap
from core.local_expert import LocalExpert
from core.exponential_weigher import ExponentialWeigher


__author__ = 'Emanuele'


class LocalExpertTest(unittest.TestCase):

    def test_fit(self):
        lt = LocalExpert(
            base_estimator=DecisionTreeClassifier(),
            sampler=ExponentialWeigher(precision=1, power=2)
        )
        instances = numpy.concatenate((numpy.random.randn(10, 2) - 1, numpy.random.randn(10, 2) + 1))
        labels = numpy.concatenate((numpy.zeros(10), numpy.ones(10)))
        lt.train([0, 0], instances, labels)
        x = numpy.random.randn(3, 2)
        y = lt.predict(x)
        self.assertEqual(3, len(y))
        self.assertTrue(all(l in (0, 1) for l in y))

    def test_repeatability_and_randomness(self):
        lt1 = LocalExpert(
            base_estimator=DecisionTreeClassifier(max_features="log2"),
            sampler=GeneralizedBootstrap(500, ExponentialWeigher(1, 2))
        )
        lt2 = LocalExpert(
            base_estimator=DecisionTreeClassifier(max_features="log2"),
            sampler=GeneralizedBootstrap(500, ExponentialWeigher(1, 2))
        )
        centroid = numpy.zeros(8)
        instances = numpy.random.randn(50, 8)
        labels = numpy.random.choice(["a", "b"], size=50)
        tests = numpy.random.randn(50, 8)
        numpy.random.seed(1)
        lt1.train(centroid, instances, labels)
        lt2.train(centroid, instances, labels)
        self.assertFalse(numpy.array_equal(lt1.predict(tests), lt2.predict(tests)))
        numpy.random.seed(1)
        lt2.train(centroid, instances, labels)
        self.assertTrue(numpy.array_equal(lt1.predict(tests), lt2.predict(tests)))
