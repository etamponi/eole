import unittest

import numpy
from sklearn.tree.tree import DecisionTreeClassifier

from eole.generalized_bootstrap import GeneralizedBootstrap
from eole.local_expert import LocalExpert
from eole.exponential_weighting import ExponentialWeighting


__author__ = 'Emanuele'


class LocalExpertTest(unittest.TestCase):

    def test_fit(self):
        lt = LocalExpert(
            base_estimator=DecisionTreeClassifier(),
            centroid=[0, 0],
            sampling=ExponentialWeighting(precision=1, power=2)
        )
        instances = numpy.concatenate((numpy.random.randn(10, 2) - 1, numpy.random.randn(10, 2) + 1))
        labels = numpy.concatenate((numpy.zeros(10), numpy.ones(10)))
        lt.train(instances, labels)
        x = numpy.random.randn(3, 2)
        y = lt.predict(x)
        self.assertEqual(3, len(y))
        self.assertTrue(all(l in (0, 1) for l in y))

    def test_repeatability_and_randomness(self):
        lt1 = LocalExpert(
            base_estimator=DecisionTreeClassifier(max_features="log2"),
            centroid=[0, 0],
            sampling=GeneralizedBootstrap(500, ExponentialWeighting(1, 2))
        )
        lt2 = LocalExpert(
            base_estimator=DecisionTreeClassifier(max_features="log2"),
            centroid=[0, 0],
            sampling=GeneralizedBootstrap(500, ExponentialWeighting(1, 2))
        )
        instances = numpy.random.randn(10, 2)
        labels = list("aaaaa") + list("bbbbb")
        tests = numpy.random.randn(10, 2)
        numpy.random.seed(1)
        lt1.train(instances, labels)
        lt2.train(instances, labels)
        self.assertTrue(numpy.any(lt1.predict(tests) != lt2.predict(tests)))
        numpy.random.seed(1)
        lt2.train(instances, labels)
        self.assertTrue(numpy.all(lt1.predict(tests) == lt2.predict(tests)))
