import unittest

import numpy

from analysis.dataset_utils import CoupledShuffle


__author__ = 'Emanuele Tamponi'


class CoupledShuffleTest(unittest.TestCase):

    def test_shuffle_does_not_modify_input(self):
        X = numpy.random.randn(100, 3)
        y = numpy.random.choice(["a", "b"], size=100)
        shuffler = CoupledShuffle(X, y)
        Xs, ys = shuffler.shuffle()
        self.assertFalse(numpy.array_equal(X, Xs))
        self.assertFalse(numpy.array_equal(y, ys))

    def test_shuffles_are_seedable(self):
        X = numpy.random.randn(10, 3)
        y = numpy.random.choice(["a", "b"], size=10)
        shuffler = CoupledShuffle(X, y)
        Xs1a, ys1a = shuffler.shuffle(1)
        Xs1b, ys1b = shuffler.shuffle(1)
        numpy.testing.assert_array_equal(Xs1a, Xs1b)
        numpy.testing.assert_array_equal(ys1a, ys1b)

    def test_shuffles_are_independent_of_previous_shuffles(self):
        X = numpy.random.randn(10, 3)
        y = numpy.random.choice(["a", "b"], size=10)
        shuffler1 = CoupledShuffle(X, y)
        shuffler1.shuffle(1)
        Xs1, ys1 = shuffler1.shuffle(2)
        shuffler2 = CoupledShuffle(X, y)
        Xs2, ys2 = shuffler2.shuffle(2)
        numpy.testing.assert_array_equal(Xs1, Xs2)
        numpy.testing.assert_array_equal(ys1, ys2)