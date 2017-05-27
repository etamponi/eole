import math
from scipy.spatial import distance
from scipy.stats import ttest_ind
from sklearn import preprocessing
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC

from eole.analysis.dataset_utils import ArffLoader
from eole.analysis.experiment import Experiment
from eole.core.centroid_picker import RandomCentroidPicker, AlmostRandomCentroidPicker
from eole.core.ensemble_trainer import EnsembleTrainer
from eole.core.flt import EOLE
from eole.core.generalized_bootstrap import GeneralizedBootstrap
from eole.core.exponential_weigher import ExponentialWeigher


__author__ = 'Emanuele Tamponi'


def main():
    dataset = "autos"
    dataset_path = "datasets/{}.arff".format(dataset)

    n_experts = 10
    n_inner_experts = 1
    if n_inner_experts == 1:
        base_estimator = DecisionTreeClassifier(max_features=0.1, max_leaf_nodes=None)
    else:
        base_estimator = RandomForestClassifier(max_features=0.3, n_estimators=n_inner_experts)
    centroid_picker = AlmostRandomCentroidPicker(dist_measure=distance.euclidean)
    weigher_sampler = ExponentialWeigher(precision=1, power=1, dist_measure=distance.euclidean, sample_percent=None)

    eole = make_eole(n_experts, base_estimator, centroid_picker, weigher_sampler)
    # rf = make_random_forest(n_experts, n_inner_experts)
    # rf = make_bagging(DecisionTreeClassifier(max_features=None))
    rf = make_boosting(SVC())

    loader = ArffLoader(dataset_path)
    n_folds = 10
    n_repetitions = 10

    experiment_eole = Experiment("{}_eole".format(dataset), eole, loader, n_folds, n_repetitions)
    experiment_rf = Experiment("{}_rf".format(dataset), rf, loader, n_folds, n_repetitions)

    report_eole = experiment_eole.run()
    accuracy_eole = report_eole.synthesis()["accuracy"]["mean"]
    stddev_eole = math.sqrt(report_eole.synthesis()["accuracy"]["variance"][-1])
    print "EOLE: {:.3f} {:.3f}".format(accuracy_eole[-1], stddev_eole)

    report_rf = experiment_rf.run()
    accuracy_rf = report_rf.synthesis()["accuracy"]["mean"]
    stddev_rf = math.sqrt(report_rf.synthesis()["accuracy"]["variance"][-1])
    print "EOLE: {:.3f} {:.3f}".format(accuracy_eole[-1], stddev_eole)
    print "  RF: {:.3f} {:.3f}".format(accuracy_rf[-1], stddev_rf)

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


def make_boosting(base_estimator):
    return EOLE(
        n_experts=1,
        ensemble_trainer=EnsembleTrainer(
            base_estimator=AdaBoostClassifier(base_estimator=base_estimator, n_estimators=10),
            centroid_picker=RandomCentroidPicker(),
            weigher_sampler=GeneralizedBootstrap(sample_percent=100, weigher=ExponentialWeigher(precision=0, power=1))
        ),
        preprocessor=None,
        use_probs=False,
        use_competences=False
    )


def make_bagging(base_estimator):
    return EOLE(
        n_experts=10,
        ensemble_trainer=EnsembleTrainer(
            base_estimator=base_estimator,
            centroid_picker=RandomCentroidPicker(),
            weigher_sampler=GeneralizedBootstrap(sample_percent=100, weigher=ExponentialWeigher(precision=0, power=1))
        ),
        preprocessor=None,
        use_probs=False,
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


class TransformerLoader(object):

    def __init__(self, loader, transform):
        self.loader = loader
        self.transform = transform

    def load_dataset(self):
        X, y = self.loader.get_dataset()
        return self.transform.fit_transform(X), y


if __name__ == "__main__":
    main()
