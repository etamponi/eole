import os

from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB

from eole.analysis.dataset_utils import ArffLoader
from eole.analysis.experiment import Experiment
from eole.core.centroid_picker import RandomCentroidPicker, AlmostRandomCentroidPicker
from eole.core.ensemble_trainer import EnsembleTrainer
from eole.core.flt import EOLE
from eole.core.exponential_weigher import ExponentialWeigher
from eole.core.generalized_bootstrap import GeneralizedBootstrap
from eole.evaluation import dataset_names, run_parallel


__author__ = 'Emanuele Tamponi'


def main():
    arg_list = [(dataset_name, ) for dataset_name in dataset_names()]
    run_parallel(experiments_on_dataset, arg_list)


# noinspection PyTypeChecker
def experiments_on_dataset(dataset_name):
    n_folds = 10
    repetitions = 10
    ensembles = [
        ("small_boosting_sgd", make_boosting(base_estimator=SGDClassifier(loss="modified_huber"))),
        ("small_bagging_sgd", make_bagging(base_estimator=SGDClassifier(loss="modified_huber"))),
        ("small_boosting_logreg", make_boosting(base_estimator=LogisticRegression())),
        ("small_bagging_logreg", make_bagging(base_estimator=LogisticRegression())),
        ("small_boosting_naive", make_boosting(base_estimator=BernoulliNB())),
        ("small_bagging_naive", make_bagging(base_estimator=BernoulliNB())),
        ("small_random_forest", make_random_forest()),
        ("small_boosting", make_boosting()),
        ("small_bagging", make_bagging()),
        # ("small_eole_10_050", make_eole(0.1, 50)),
        # ("small_eole_10_100", make_eole(0.1, 100)),
        # ("small_eole_10_Nil", make_eole(0.1, None)),
        ("small_eole_30_050", make_eole(0.3, 50)),
        # ("small_eole_30_100", make_eole(0.3, 100)),
        # ("small_eole_30_Nil", make_eole(0.3, None)),
        # ("small_eole_50_050", make_eole(0.5, 50)),
        # ("small_eole_50_100", make_eole(0.5, 100)),
        # ("small_eole_50_Nil", make_eole(0.5, None)),
        # ("small_eole_10_050_50", make_eole(0.1, 50, 50)),
        # ("small_eole_10_100_50", make_eole(0.1, 100, 50)),
        # ("small_eole_10_Nil_50", make_eole(0.1, None, 50)),
        # ("small_eole_30_050_50", make_eole(0.3, 50, 50)),
        # ("small_eole_30_100_50", make_eole(0.3, 100, 50)),
        # ("small_eole_30_Nil_50", make_eole(0.3, None, 50)),
        # ("small_eole_50_050_50", make_eole(0.5, 50, 50)),
        # ("small_eole_50_100_50", make_eole(0.5, 100, 50)),
        # ("small_eole_50_Nil_50", make_eole(0.5, None, 50)),
    ]
    print "Start experiments on: {}".format(dataset_name)
    for ens_name, ensemble in ensembles:
        exp_name = "{}_{}".format(dataset_name, ens_name)
        if os.path.isfile("reports/small/{}.rep".format(exp_name)):
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
        report.dump("reports/small/")


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


def make_boosting(base_estimator=DecisionTreeClassifier()):
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


def make_bagging(base_estimator=DecisionTreeClassifier(max_features=None)):
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
