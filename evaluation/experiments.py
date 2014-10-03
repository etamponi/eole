from scipy.spatial import distance
from sklearn.preprocessing.data import MinMaxScaler
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
    n_folds = 5
    repetitions = 20
    n_groups = 3
    group = 1
    ensembles = [
        ("random_forest", make_random_forest()),
        ("bootstrap_eole_0100_01", make_eole(100, 1)),
        ("bootstrap_eole_1000_01", make_eole(1000, 1)),
        ("bootstrap_eole_0100_05", make_eole(100, 5)),
        ("bootstrap_eole_1000_05", make_eole(1000, 5)),
        ("bootstrap_eole_0100_10", make_eole(100, 10)),
        ("bootstrap_eole_1000_10", make_eole(1000, 10)),
        ("bootstrap_eole_0100_20", make_eole(100, 20)),
        ("bootstrap_eole_1000_20", make_eole(1000, 20))
    ]

    for dataset_name in evaluation.dataset_names():
        print "Start experiments on: {}".format(dataset_name)
        for ens_name, ensemble in ensembles:
            exp_name = "{}_{}".format(dataset_name, ens_name)
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
        n_experts=100,
        ensemble_trainer=EnsembleTrainer(
            base_estimator=DecisionTreeClassifier(max_features="auto"),
            centroid_picker=RandomCentroidPicker(),
            weigher_sampler=GeneralizedBootstrap(sample_percent=100, weigher=ExponentialWeigher(precision=0, power=1))
        ),
        preprocessor=None,
        use_probs=False,
        use_competences=False
    )


def make_eole(sample_percent, precision):
    return EOLE(
        n_experts=100,
        ensemble_trainer=EnsembleTrainer(
            base_estimator=DecisionTreeClassifier(max_features="auto"),
            centroid_picker=AlmostRandomCentroidPicker(dist_measure=distance.chebyshev),
            weigher_sampler=GeneralizedBootstrap(
                sample_percent=sample_percent,
                weigher=ExponentialWeigher(precision=precision, power=1, dist_measure=distance.chebyshev)
            )
        ),
        preprocessor=MinMaxScaler(),
        use_probs=True,
        use_competences=False
    )


if __name__ == "__main__":
    main()
