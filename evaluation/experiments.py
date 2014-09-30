from sklearn.preprocessing.data import MinMaxScaler
from sklearn.tree.tree import DecisionTreeClassifier

from analysis.dataset_utils import ArffLoader
from analysis.experiment import Experiment
from core.centroid_picker import RandomCentroidPicker, AlmostRandomCentroidPicker
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

    centroid_picker = AlmostRandomCentroidPicker()

    ensembles = [
        ("random_forest", make_random_forest()),
        ("exponential_eole_1", make_eole(centroid_picker, ExponentialWeigher(1, 2))),
        ("exponential_eole_2", make_eole(centroid_picker, ExponentialWeigher(2, 2))),
        ("exponential_eole_3", make_eole(centroid_picker, ExponentialWeigher(3, 2))),
        ("exponential_eole_4", make_eole(centroid_picker, ExponentialWeigher(4, 2))),
        ("exponential_eole_5", make_eole(centroid_picker, ExponentialWeigher(5, 2)))
    ]

    n_folds = 10
    repetitions = 10

    for ds_name in ds_names:
        print "Start experiments on: {}".format(ds_name)
        for ens_name, ensemble in ensembles:
            exp_name = "{}_{}".format(ds_name, ens_name)
            print "Start experiment: {}".format(exp_name)
            experiment = Experiment(
                name=exp_name,
                ensemble=ensemble,
                dataset_loader=ArffLoader("datasets/{}.arff".format(ds_name)),
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
            weigher_sampler=GeneralizedBootstrap(
                sample_percent=100,
                weigher=ExponentialWeigher(precision=0, power=1)
            )
        ),
        preprocessor=None,
        use_probs=False,
        use_competences=False
    )


def make_eole(centroid_picker, weigher_sampler):
    return EOLE(
        n_experts=100,
        ensemble_trainer=EnsembleTrainer(
            base_estimator=DecisionTreeClassifier(max_features="auto"),
            centroid_picker=centroid_picker,
            weigher_sampler=weigher_sampler
        ),
        preprocessor=MinMaxScaler(),
        use_probs=True,
        use_competences=False
    )


if __name__ == "__main__":
    main()
