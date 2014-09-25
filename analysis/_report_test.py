import unittest

import numpy

from analysis import Experiment
from analysis.report import Report
from eole import EOLE


__author__ = 'Emanuele Tamponi'


class ReportTest(unittest.TestCase):

    def test_analyze_run(self):
        report = Report(
            Experiment("test_experiment", EOLE(3, None, None, None, False, False), None, folds=2, repetitions=3)
        )
        prediction_matrix = numpy.asarray([
            ["A", "A", "B"],
            ["A", "B", "B"],
            ["B", "A", "A"],
            ["B", "A", "B"]
        ])
        labels = numpy.asarray([
            "A", "B", "A", "B"
        ])
        expected_accuracy = numpy.asarray([0.5, 0.75, 0.75])
        report.analyze_run(prediction_matrix, labels)
        numpy.testing.assert_array_almost_equal(expected_accuracy, report.accuracy_sample[0])