import unittest

import numpy
from sklearn.tree.tree import DecisionTreeClassifier

from eole.core.centroid_picker import RandomCentroidPicker
from eole.core.ensemble_trainer import EnsembleTrainer
from eole.core.exponential_weigher import ExponentialWeigher


__author__ = 'Emanuele'


class EnsembleTrainerTest(unittest.TestCase):

    def test_train_generates_random_experts(self):
        instances, labels = numpy.random.randn(50, 10), numpy.random.choice(["a", "b", "c"], size=50)
        tests = numpy.random.randn(50, 10)
        ft = EnsembleTrainer(
            base_estimator=DecisionTreeClassifier(),
            centroid_picker=RandomCentroidPicker(),
            weigher_sampler=ExponentialWeigher(1, 2)
        )
        experts = ft.train(2, instances, labels)
        self.assertEqual(2, len(experts))
        self.assertFalse(numpy.array_equal(experts[0].centroid, experts[1].centroid))
        self.assertFalse(numpy.array_equal(experts[0].predict(tests), experts[1].predict(tests)))
