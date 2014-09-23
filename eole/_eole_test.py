import unittest
from eole.centroid_picker import CentroidPicker
from eole.exponential_weighting import ExponentialWeighting
from eole.generalized_bootstrap import GeneralizedBootstrap

__author__ = 'Emanuele'


class EOLETest(unittest.TestCase):

    def test_predict(self):
        eole = EOLE(
            n_experts=3,
            centroid_picker=CentroidPicker(),
            sampling=GeneralizedBootstrap(500, weighting=ExponentialWeighting(1, 2)),
            expert_weighting=ExponentialWeighting(1, 2)
        )
