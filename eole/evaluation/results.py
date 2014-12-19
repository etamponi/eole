from collections import Counter, defaultdict
import glob
import re

from scipy.stats import ttest_ind

from eole.analysis.report import Report
from eole.evaluation import dataset_names


__author__ = 'Emanuele Tamponi'


def main():
    improvs = Counter()
    improvs_stat = Counter()
    improvs_percent = defaultdict(list)
    improv_by_dataset = {}
    for dataset_name in dataset_names():
        improv_by_dataset[dataset_name] = ("", 0, False)
        report_rf = Report.load("reports/small/{}_small_random_forest.rep".format(dataset_name))
        accuracy_full_rf = report_rf.accuracy_sample[:, -1]
        print "COMPARISON FOR: {}".format(dataset_name.upper())
        print "Random Forest: {:.3f} +- {:.3f}".format(
            accuracy_full_rf.mean(), accuracy_full_rf.std()
        )
        for dataset_path in glob.glob("reports/small/{}_small_eole_*.rep".format(dataset_name)):
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
            if p > 0:
                imp = accuracy_best_eole.mean() - accuracy_full_rf.mean()
                improvs[eole_name] += 1
                improvs_percent[eole_name].append(
                    str(round(100*imp, 1))
                )
                if imp > improv_by_dataset[dataset_name][1]:
                    improv_by_dataset[dataset_name] = [eole_name, imp, False]
            if p > 0.95:
                improvs_stat[eole_name] += 1
                improvs_percent[eole_name][-1] += "*"
                if improv_by_dataset[dataset_name][0] == eole_name:
                    improv_by_dataset[dataset_name][2] = True
    total = len(dataset_names())
    for eole in improvs:
        print "{}: {} {} ({}) = {}".format(eole, total, improvs[eole], improvs_stat[eole], improvs_percent[eole])
    i = 1
    j = 0
    for dataset_name in sorted(improv_by_dataset.keys()):
        data = improv_by_dataset[dataset_name]
        if data[1] == 0:
            continue
        print "{:2d} {:15s}: {} = {:.1f} {}".format(i, dataset_name, data[0], data[1]*100, data[2])
        if data[2]:
            j += 1
        i += 1
    print "Significant total improvements:", j


def one_side_test(first, second):
    value, p = ttest_ind(first, second, equal_var=False)
    if value < 0:
        return 0.0
    else:
        return 1 - p / 2


if __name__ == "__main__":
    main()
