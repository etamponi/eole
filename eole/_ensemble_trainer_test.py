import unittest

import numpy
from sklearn.tree.tree import DecisionTreeClassifier

from eole.centroid_picker import RandomCentroidPicker
from eole.ensemble_trainer import EnsembleTrainer
from eole.exponential_weighting import ExponentialWeighting


__author__ = 'Emanuele'


class EnsembleTrainerTest(unittest.TestCase):

    def test_train_generates_random_trees(self):
        instances, labels = numpy.random.randn(50, 10), numpy.random.choice(["a", "b", "c"], size=50)
        tests = numpy.random.randn(10, 10)
        ft = EnsembleTrainer(
            base_estimator=DecisionTreeClassifier(),
            centroid_picker=RandomCentroidPicker(), sampling=ExponentialWeighting(1, 2)
        )
        experts = ft.train(2, instances, labels)
        self.assertEqual(2, len(experts))
        self.assertNotEqual(experts[0].centroid.tolist(), experts[1].centroid.tolist())
        self.assertNotEqual(experts[0].predict(tests).tolist(), experts[1].predict(tests).tolist())
