from scipy.spatial import distance
from scipy.stats.stats import ttest_ind
from sklearn import preprocessing

from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from analysis.dataset_utils import ArffLoader
from analysis.experiment import Experiment
from core.centroid_picker import RandomCentroidPicker, ClusterCenterPicker
from core.ensemble_trainer import EnsembleTrainer
from core.eole import EOLE
from core.exponential_weigher import ExponentialWeigher


__author__ = 'Emanuele Tamponi'


def main():
    dataset = "balance-scale"
    dataset_path = "evaluation/datasets/{}.arff".format(dataset)

    n_experts = 10
    n_inner_experts = 1
    if n_inner_experts == 1:
        base_estimator = DecisionTreeClassifier(max_features=0.1, max_leaf_nodes=25)
    else:
        base_estimator = RandomForestClassifier(max_features=0.3, n_estimators=n_inner_experts)
    centroid_picker = ClusterCenterPicker()
    weigher_sampler = ExponentialWeigher(precision=1, power=1, dist_measure=distance.euclidean, sample_percent=None)

    eole = make_eole(n_experts, base_estimator, centroid_picker, weigher_sampler)
    rf = make_random_forest(n_experts, n_inner_experts)

    loader = ArffLoader(dataset_path)
    n_folds = 10
    n_repetitions = 10

    experiment_eole = Experiment("{}_eole".format(dataset), eole, loader, n_folds, n_repetitions)
    experiment_rf = Experiment("{}_rf".format(dataset), rf, loader, n_folds, n_repetitions)

    report_eole = experiment_eole.run()
    accuracy_eole = report_eole.synthesis()["accuracy"]["mean"]
    print "EOLE: {:.3f} {:.3f} ({})".format(accuracy_eole[-1], accuracy_eole.max(), accuracy_eole.argmax())

    report_rf = experiment_rf.run()
    accuracy_rf = report_rf.synthesis()["accuracy"]["mean"]
    print "EOLE: {:.3f} {:.3f} ({})".format(accuracy_eole[-1], accuracy_eole.max(), accuracy_eole.argmax())
    print "RF: {:.3f} {:.3f} ({})".format(accuracy_rf[-1], accuracy_rf.max(), accuracy_rf.argmax())

    p = one_side_test(report_eole.accuracy_sample[:, -1], report_rf.accuracy_sample[:, -1])
    print "P(EOLE > RF) = {:.3f} => {}".format(p, p > 0.95)


def make_eole(n_experts, base_estimator, centroid_picker, weigher_sampler):
    return EOLE(
        n_experts=n_experts,
        ensemble_trainer=EnsembleTrainer(
            base_estimator=base_estimator,
            centroid_picker=centroid_picker,
            weigher_sampler=weigher_sampler
        ),
        preprocessor=preprocessing.MinMaxScaler(),
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
