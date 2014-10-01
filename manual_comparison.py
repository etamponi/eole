from scipy.stats.stats import ttest_ind

from analysis.report import Report


__author__ = 'Emanuele Tamponi'


def main():
    dataset_names = [
        "anneal",
        "audiology",
        "autos",
        "balance-scale",
        "breast-cancer",
        "heart-c",
        "credit-a",
        "credit-g",
        "glass",
        "heart-statlog",
        "hepatitis",
        "colic",
        "heart-h",
        "hypothyroid",
        "ionosphere",
        "iris",
        "labor",
        "letter",
        "lymph",
        "diabetes",
        "primary-tumor",
        "segment",
        "sonar",
        "soybean",
        "splice",
        "vehicle",
        "vote",
        "vowel",
        "waveform-5000",
        "breast-w",
        "zoo"
    ]

    ensemble_names = [
        "random_forest",
        "exponential_eole_1",
        "exponential_eole_2",
        "exponential_eole_3",
        "exponential_eole_4",
        "exponential_eole_5",
        "exponential_eole_6",
        "exponential_eole_7",
        "exponential_eole_8",
        "exponential_eole_9",
    ]

    dataset_name = "splice"
    eole_name = "exponential_eole"

    report_rf = Report.load("evaluation/reports/{}_random_forest.rep".format(dataset_name))
    accuracy_full_rf = report_rf.accuracy_sample[:, -1]
    for i in range(1, 10):
        report_eole = Report.load("evaluation/reports/{}_{}_{}.rep".format(dataset_name, eole_name, i))
        n_max = report_eole.synthesis()["accuracy"]["mean"].argmax()
        accuracy_best_eole = report_eole.accuracy_sample[:, n_max]
        print "RF: {:.3f} - {}_{}: {:.3f} ({})".format(
            accuracy_full_rf.mean(), eole_name, i, accuracy_best_eole.mean(), n_max
        )
        _, p = ttest_ind(accuracy_full_rf, accuracy_best_eole, equal_var=False)
        print "RF == EOLE: {:.3f} => {}".format(p, p >= 0.05)


if __name__ == "__main__":
    main()