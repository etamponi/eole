import glob
import re

from scipy.stats.stats import ttest_ind

from analysis.report import Report


__author__ = 'Emanuele Tamponi'


def main():
    dataset_name = "mushroom"

    report_rf = Report.load("evaluation/reports/{}_random_forest.rep".format(dataset_name))
    accuracy_full_rf = report_rf.accuracy_sample[:, -1]
    print "Random Forest: {:.3f} +- {:.3f}".format(
        accuracy_full_rf.mean(), accuracy_full_rf.std()
    )
    for dataset_path in glob.glob("evaluation/reports/{}_*eole*.rep".format(dataset_name)):
        report_eole = Report.load(dataset_path)
        eole_name = re.search(r"{}_([\w_]+)\.rep".format(dataset_name), dataset_path).group(1)
        # n_max = report_eole.synthesis()["accuracy"]["mean"].argmax()
        n_max = -1
        accuracy_best_eole = report_eole.accuracy_sample[:, n_max]
        print "- {}: {:.3f} +- {:.3f} ({})".format(
            eole_name, accuracy_best_eole.mean(), accuracy_best_eole.std(), n_max
        )
        p = one_side_test(accuracy_best_eole, accuracy_full_rf)
        print "  P(EOLE > RF) = {:.3f} => {}".format(p, p > 0.95)


def one_side_test(first, second):
    value, p = ttest_ind(first, second, equal_var=False)
    if value < 0:
        return 0.0
    else:
        return 1 - p / 2


if __name__ == "__main__":
    main()
