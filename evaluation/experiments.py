from sklearn.preprocessing.data import MinMaxScaler
from sklearn.tree.tree import DecisionTreeClassifier

from analysis.dataset_utils import ArffLoader
from analysis.experiment import Experiment
from core.centroid_picker import RandomCentroidPicker
from core.ensemble_trainer import EnsembleTrainer
from core.eole import EOLE
from core.exponential_weigher import ExponentialWeigher
from core.generalized_bootstrap import GeneralizedBootstrap


__author__ = 'Emanuele Tamponi'


def main():
    ds_names = [
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

    ensembles = [
        ("random_forest", make_bootstrap_ensemble(0, 100)),
        ("random_eole_1", make_exponential_ensemble(1)),
        ("random_eole_2", make_exponential_ensemble(2)),
        ("random_eole_3", make_exponential_ensemble(3)),
        ("random_eole_4", make_exponential_ensemble(4)),
        ("random_eole_5", make_exponential_ensemble(5))
    ]

    n_folds = 5
    repetitions = 20

    for ds_name in ds_names:
        print "Start experiments on: {}".format(ds_name)
        for ens_name, ensemble in ensembles:
            exp_name = "{}_{}".format(ds_name, ens_name)
            print "Start experiment: {}".format(exp_name)
            experiment = Experiment(
                name=exp_name,
                ensemble=ensemble,
                dataset_loader=ArffLoader("datasets/{}.arff".format(ds_name)),
                folds=n_folds,
                repetitions=repetitions
            )
            report = experiment.run()
            report.dump("reports/")


def make_bootstrap_ensemble(precision, sample_percent):
    return EOLE(
        n_experts=100,
        ensemble_trainer=EnsembleTrainer(
            base_estimator=DecisionTreeClassifier(max_features="auto"),
            centroid_picker=RandomCentroidPicker(),
            sampling=GeneralizedBootstrap(
                sample_percent=sample_percent,
                weigher=ExponentialWeigher(precision=precision, power=1)
            )
        ),
        expert_weighting=ExponentialWeigher(precision=precision, power=1),
        preprocessor=MinMaxScaler(),
        use_probs=False,
        use_competences=False
    )


def make_exponential_ensemble(precision):
    return EOLE(
        n_experts=100,
        ensemble_trainer=EnsembleTrainer(
            base_estimator=DecisionTreeClassifier(max_features="auto"),
            centroid_picker=RandomCentroidPicker(),
            sampling=ExponentialWeigher(precision=precision, power=1)
        ),
        expert_weighting=ExponentialWeigher(precision=precision, power=1),
        preprocessor=MinMaxScaler(),
        use_probs=False,
        use_competences=False
    )


if __name__ == "__main__":
    main()
