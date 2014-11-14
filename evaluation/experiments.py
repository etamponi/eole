import os

from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.tree.tree import DecisionTreeClassifier

from analysis.dataset_utils import ArffLoader
from analysis.experiment import Experiment
from core.centroid_picker import RandomCentroidPicker, AlmostRandomCentroidPicker
from core.ensemble_trainer import EnsembleTrainer
from core.eole import EOLE
from core.exponential_weigher import ExponentialWeigher
from core.generalized_bootstrap import GeneralizedBootstrap
import evaluation


__author__ = 'Emanuele Tamponi'


def main():
    n_folds = 10
    repetitions = 10
    n_groups = 2
    group = 1
    ensembles = [
        ("small_random_forest", make_random_forest()),
        ("small_eole_10_050_50", make_eole(0.1, 50, 50)),
        ("small_eole_10_100_50", make_eole(0.1, 100, 50)),
        ("small_eole_10_Nil_50", make_eole(0.1, None, 50)),
        ("small_eole_30_050_50", make_eole(0.3, 50, 50)),
        ("small_eole_30_100_50", make_eole(0.3, 100, 50)),
        ("small_eole_30_Nil_50", make_eole(0.3, None, 50)),
        ("small_eole_50_050_50", make_eole(0.5, 50, 50)),
        ("small_eole_50_100_50", make_eole(0.5, 100, 50)),
        ("small_eole_50_Nil_50", make_eole(0.5, None, 50)),
    ]

    for dataset_name in evaluation.dataset_names(n_groups, group)[::-1]:
        print "Start experiments on: {}".format(dataset_name)
        for ens_name, ensemble in ensembles:
            exp_name = "{}_{}".format(dataset_name, ens_name)
            if os.path.isfile("reports/{}.rep".format(exp_name)):
                print "Experiment {} already done, going to next one.".format(exp_name)
                continue
            print "Start experiment: {}".format(exp_name)
            experiment = Experiment(
                name=exp_name,
                ensemble=ensemble,
                dataset_loader=ArffLoader("datasets/{}.arff".format(dataset_name)),
                n_folds=n_folds,
                n_repetitions=repetitions
            )
            report = experiment.run()
            report.dump("reports/")


def make_random_forest():
    return EOLE(
        n_experts=10,
        ensemble_trainer=EnsembleTrainer(
            base_estimator=DecisionTreeClassifier(max_features="auto"),
            centroid_picker=RandomCentroidPicker(),
            weigher_sampler=GeneralizedBootstrap(sample_percent=100, weigher=ExponentialWeigher(precision=0, power=1))
        ),
        preprocessor=None,
        use_probs=False,
        use_competences=False
    )


def make_eole(max_feature_percent, max_leaf_nodes, sample_percent=None):
    return EOLE(
        n_experts=10,
        ensemble_trainer=EnsembleTrainer(
            base_estimator=DecisionTreeClassifier(max_features=max_feature_percent, max_leaf_nodes=max_leaf_nodes),
            centroid_picker=AlmostRandomCentroidPicker(dist_measure=distance.euclidean),
            weigher_sampler=ExponentialWeigher(precision=1, power=1, dist_measure=distance.euclidean,
                                               sample_percent=sample_percent)
        ),
        preprocessor=preprocessing.MinMaxScaler(),
        use_probs=True,
        use_competences=False
    )


if __name__ == "__main__":
    main()
