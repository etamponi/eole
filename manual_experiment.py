from scipy.spatial import distance
from scipy.stats.stats import ttest_ind
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from analysis.dataset_utils import ArffLoader
from analysis.experiment import Experiment
from core.centroid_picker import RandomCentroidPicker, AlmostRandomCentroidPicker
from core.ensemble_trainer import EnsembleTrainer
from core.eole import EOLE
from core.exponential_weigher import ExponentialWeigher
from core.generalized_bootstrap import GeneralizedBootstrap


__author__ = 'Emanuele Tamponi'


def main():
    n_experts = 100
    n_inner_experts = 1
    # base_estimator = RandomForestClassifier(n_estimators=n_inner_experts, max_features="auto")
    base_estimator = DecisionTreeClassifier(max_features="log2")
    centroid_picker = AlmostRandomCentroidPicker(dist_measure=distance.chebyshev)
    weigher_sampler = GeneralizedBootstrap(
        sample_percent=100,
        weigher=ExponentialWeigher(precision=5, power=1, dist_measure=distance.chebyshev)
    )

    eole = make_eole(n_experts, base_estimator, centroid_picker, weigher_sampler)
    rf = make_random_forest(n_experts, n_inner_experts)

    loader = ArffLoader("evaluation/datasets/vowel.arff")
    n_folds = 5
    n_repetitions = 1

    experiment_eole = Experiment("eole", eole, loader, n_folds, n_repetitions)
    experiment_rf = Experiment("rf", rf, loader, n_folds, n_repetitions)

    report_eole = experiment_eole.run()
    accuracy_eole = report_eole.synthesis()["accuracy"]["mean"]
    print "EOLE: {:.3f} {:.3f} ({})".format(accuracy_eole[-1], accuracy_eole.max(), accuracy_eole.argmax())

    report_rf = experiment_rf.run()
    accuracy_rf = report_rf.synthesis()["accuracy"]["mean"]
    print "RF: {:.3f} {:.3f} ({})".format(accuracy_rf[-1], accuracy_rf.max(), accuracy_rf.argmax())

    p = one_side_test(report_eole.accuracy_sample[:, -1], report_rf.accuracy_sample[:, -1])
    print "P(EOLE > RF) = {:.3f} => {}".format(p, p > 0.95)

    # pyplot.plot(
    #     numpy.linspace(1, n_experts, n_experts), accuracy_eole, "g",
    #     numpy.linspace(1, n_experts, n_experts), accuracy_rf, "r"
    # )
    # pyplot.grid()
    # pyplot.show()


def make_eole(n_experts, base_estimator, centroid_picker, weigher_sampler):
    return EOLE(
        n_experts=n_experts,
        ensemble_trainer=EnsembleTrainer(
            base_estimator=base_estimator,
            centroid_picker=centroid_picker,
            weigher_sampler=weigher_sampler
        ),
        preprocessor=MinMaxScaler(),
        use_probs=True,
        use_competences=False
    )


def make_random_forest(n_experts, n_inner_experts):
    return EOLE(
        n_experts=1,
        ensemble_trainer=EnsembleTrainer(
            base_estimator=RandomForestClassifier(n_estimators=n_experts*n_inner_experts, max_features="auto"),
            centroid_picker=RandomCentroidPicker(),
            weigher_sampler=ExponentialWeigher(precision=0, power=1)
        ),
        preprocessor=None,
        use_probs=True,
        use_competences=False
    )


def one_side_test(first, second):
    value, p = ttest_ind(first, second, equal_var=False)
    if value < 0:
        return 0.0
    else:
        return 1 - p / 2


if __name__ == "__main__":
    main()
