import os
import unittest

import numpy

from eole.analysis.experiment import Experiment
from eole.analysis.report import Report, ReportNotReady, ReportDirectoryError, ReportFileAlreadyExists
from eole.core.flt import EOLE


__author__ = 'Emanuele Tamponi'


class ReportTest(unittest.TestCase):

    def setUp(self):
        self.experiment = Experiment(
            "test_experiment", EOLE(3, None, None, False, False), None, n_folds=2, n_repetitions=5
        )
        self.report = Report(self.experiment)
        self.prediction_matrix = numpy.asarray([
            ["A", "A", "B"],
            ["A", "B", "B"],
            ["B", "A", "A"],
            ["B", "A", "B"]
        ])
        self.labels = numpy.asarray([
            "A", "B", "A", "B"
        ])

    def test_report_samples_follows_experiment_specification(self):
        self.assertEqual(10, self.report.sample_size)
        self.assertEqual((2*5, 3), self.report.accuracy_sample.shape, "report samples should follow experiment specs")
        self.assertEqual(self.experiment.name, self.report.experiment.name, "report should contain the same experiment")
        self.assertIsNot(self.experiment, self.report.experiment, "report should contain a copy of the experiment")

    def test_analyze_run(self):
        expected_accuracy = numpy.asarray([0.5, 0.75, 0.75])
        self.report.analyze_run(self.prediction_matrix, self.labels)
        numpy.testing.assert_array_almost_equal(expected_accuracy, self.report.accuracy_sample[0])

        # second run
        prediction_matrix = numpy.asarray([
            ["A", "A", "B"],
            ["A", "B", "B"],
            ["B", "A", "A"],
            ["B", "A", "B"]
        ])
        labels = numpy.asarray([
            "A", "B", "B", "A"
        ])
        expected_accuracy = numpy.asarray([0.5, 0.75, 0.25])
        self.report.analyze_run(prediction_matrix, labels)
        numpy.testing.assert_array_almost_equal(
            expected_accuracy, self.report.accuracy_sample[1], err_msg="second run didn't work as expected"
        )

    def test_synthesis_raises_if_not_ready(self):
        self.assertRaises(ReportNotReady, self.report.synthesis)

    def test_synthesis_works(self):
        for i in range(self.report.sample_size):
            self.report.analyze_run(self.prediction_matrix, self.labels)
        expected_mean = numpy.asarray([0.5, 0.75, 0.75])
        expected_variance = numpy.zeros(3)
        synthesis = self.report.synthesis()
        numpy.testing.assert_array_almost_equal(expected_mean, synthesis["accuracy"]["mean"])
        numpy.testing.assert_array_almost_equal(expected_variance, synthesis["accuracy"]["variance"])

    def test_ready(self):
        for i in range(self.report.sample_size - 1):
            self.report.analyze_run(self.prediction_matrix, self.labels)
        self.assertFalse(self.report.is_ready())

        self.report.analyze_run(self.prediction_matrix, self.labels)
        self.assertTrue(self.report.is_ready())

    def test_dump_filename(self):
        for i in range(self.report.sample_size):
            self.report.analyze_run(self.prediction_matrix, self.labels)
        try:
            self.report.dump("test_files/")
            self.assertTrue(os.path.isfile("test_files/test_experiment.rep"))
        finally:
            os.remove("test_files/test_experiment.rep")

    def test_dump_directory_error(self):
        for i in range(self.report.sample_size):
            self.report.analyze_run(self.prediction_matrix, self.labels)
        self.assertRaises(ReportDirectoryError, self.report.dump, "not_exists/")
        self.assertRaises(ReportDirectoryError, self.report.dump, "test_files/file_test")

    def test_load_works_on_dump(self):
        for i in range(self.report.sample_size):
            self.report.analyze_run(self.prediction_matrix, self.labels)
        try:
            self.report.dump("test_files/")
            loaded = Report.load("test_files/test_experiment.rep")
            numpy.testing.assert_array_equal(self.report.sample_size, loaded.sample_size)
            numpy.testing.assert_array_equal(self.report.accuracy_sample, loaded.accuracy_sample)
        finally:
            os.remove("test_files/test_experiment.rep")

    def test_dump_does_not_override_files(self):
        for i in range(self.report.sample_size):
            self.report.analyze_run(self.prediction_matrix, self.labels)
        try:
            self.report.dump("test_files/")
            self.assertRaises(ReportFileAlreadyExists, self.report.dump, "test_files/")
        finally:
            os.remove("test_files/test_experiment.rep")
