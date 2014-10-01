from matplotlib import pyplot
import numpy
from scipy.stats.stats import ttest_ind
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
    base_estimator = DecisionTreeClassifier(max_features="auto")
    centroid_picker = AlmostRandomCentroidPicker()
    weigher_sampler = ExponentialWeigher(precision=4, power=2)

    eole = make_eole(n_experts, base_estimator, centroid_picker, weigher_sampler)
    rf = make_random_forest(n_experts, base_estimator)

    loader = ArffLoader("evaluation/datasets/heart-c.arff")
    n_folds = 10
    n_repetitions = 10

    experiment_eole = Experiment("eole", eole, loader, n_folds, n_repetitions)
    experiment_rf = Experiment("rf", rf, loader, n_folds, n_repetitions)

    report_rf = experiment_rf.run()
    report_eole = experiment_eole.run()

    accuracy_eole = report_eole.synthesis()["accuracy"]["mean"]
    accuracy_rf = report_rf.synthesis()["accuracy"]["mean"]
    print "EOLE: {:.3f} {:.3f} ({})".format(accuracy_eole[-1], accuracy_eole.max(), accuracy_eole.argmax())
    print "RF: {:.3f} {:.3f} ({})".format(accuracy_rf[-1], accuracy_rf.max(), accuracy_rf.argmax())
    _, p = ttest_ind(report_eole.accuracy_sample[:, -1], report_rf.accuracy_sample[:, -1], equal_var=False)
    print "EOLE == RF: {:.5f} => {}".format(p, p >= 0.05)

    pyplot.plot(
        numpy.linspace(1, n_experts, n_experts), accuracy_eole, "r",
        numpy.linspace(1, n_experts, n_experts), accuracy_rf, "g"
    )
    pyplot.grid()
    pyplot.show()


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


def make_random_forest(n_experts, base_estimator):
    return EOLE(
        n_experts=n_experts,
        ensemble_trainer=EnsembleTrainer(
            base_estimator=base_estimator,
            centroid_picker=RandomCentroidPicker(),
            weigher_sampler=GeneralizedBootstrap(sample_percent=100, weigher=ExponentialWeigher(precision=0, power=1))
        ),
        preprocessor=None,
        use_probs=False,
        use_competences=False
    )


if __name__ == "__main__":
    main()
