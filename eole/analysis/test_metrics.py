import unittest

import numpy

from eole.analysis.metrics import precision_recall_f1_score


__author__ = 'Emanuele Tamponi'


class MetricsTest(unittest.TestCase):

    def test_precision_recall_f1_standard_cases(self):
        y_true = numpy.asarray(["A", "A", "B", "B", "A"])
        y_pred = numpy.asarray(["B", "B", "B", "A", "A"])
        a_prec = 1. / 2.
        b_prec = 1. / 3.
        avg_prec = (3*a_prec + 2*b_prec)/5
        prec, rec, f1 = precision_recall_f1_score(y_true, y_pred)
        self.assertAlmostEqual(avg_prec, prec)

    def test_precision_something_is_zero(self):
        y_true = numpy.asarray(["A", "A", "B"])
        y_pred = numpy.asarray(["B", "B", "B"])
        a_prec = 0.
        b_prec = 1. / 3.
        avg_prec = (2*a_prec + 1*b_prec)/3
        prec, rec, f1 = precision_recall_f1_score(y_true, y_pred)
        self.assertAlmostEqual(avg_prec, prec)