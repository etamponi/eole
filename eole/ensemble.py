import numpy
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets.samples_generator import make_classification
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics.metrics import accuracy_score
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.tree.tree import DecisionTreeClassifier

from eole import matrix_utils
from eole.ensemble_gate import EnsembleGate
from eole.centroid_picker import AlmostRandomCentroidPicker
from eole.ensemble_trainer import EnsembleTrainer
from eole.exponential_weighting import ExponentialWeighting


__author__ = 'Emanuele Tamponi'


class EOLE(object):

    def __init__(self, ensemble_trainer, expert_weighting, preprocessor, use_probs, use_competences):
        self.ensemble_trainer = ensemble_trainer
        self.expert_weighting = expert_weighting
        self.preprocessor = preprocessor
        self.use_probs = use_probs
        self.use_competences = use_competences
        self.labels = None
        self.gate = None

    def train(self, instances, labels):
        self.labels, labels = numpy.unique(labels, return_inverse=True)
        instances = self.preprocessor.fit_transform(instances)
        self.gate = EnsembleGate(self.ensemble_trainer.train(instances, labels), self.expert_weighting)

    def predict(self, instances):
        return matrix_utils.prediction_matrix(self.predict_probs(instances), self.labels)

    def predict_probs(self, instances):
        instances = self.preprocessor.transform(instances)

        competence_matrix = self.gate.competence_matrix(instances)
        probability_matrix = self.gate.probability_matrix(instances)

        # Sort the probability matrix in decreasing competence order
        indices = matrix_utils.order(competence_matrix)
        competence_matrix = matrix_utils.sort_matrix(competence_matrix, indices)
        probability_matrix = matrix_utils.sort_matrix(probability_matrix, indices)

        if not self.use_probs:
            probability_matrix = matrix_utils.sharpen_probability_matrix(probability_matrix)
        if self.use_competences:
            ensemble_probs = matrix_utils.partial_row_average_matrix(probability_matrix, competence_matrix)
        else:
            ensemble_probs = matrix_utils.partial_row_average_matrix(probability_matrix)
        return ensemble_probs


if __name__ == "__main__":
    numpy.random.seed(1)

    # iris = load_iris()
    # instances = iris.data
    # labels = iris.target_names[iris.target]

    instances, labels = make_classification(1000, n_classes=2, n_informative=10, n_clusters_per_class=5)

    skf = StratifiedKFold(labels, n_folds=10)
    train_indices, test_indices = next(iter(skf))
    train_instances = instances[train_indices]
    train_labels = labels[train_indices]
    test_instances = instances[test_indices]
    test_labels = labels[test_indices]

    ensemble = EOLE(
        ensemble_trainer=EnsembleTrainer(
            base_estimator=DecisionTreeClassifier(
                max_features="auto",
                max_depth=20
            ),
            n_experts=500,
            centroid_picker=AlmostRandomCentroidPicker(),
            sampling=ExponentialWeighting(precision=4, power=1, sample_percent=None)
        ),
        expert_weighting=ExponentialWeighting(precision=5, power=1),
        preprocessor=MinMaxScaler(),
        use_probs=True,
        use_competences=False
    )

    ensemble.train(train_instances, train_labels)
    prediction_matrix = ensemble.predict(test_instances)
    print numpy.asarray(
        [accuracy_score(test_labels, prediction_matrix[:, j]) for j in range(0, prediction_matrix.shape[1], 10)]
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="auto"
    ).fit(train_instances, train_labels)
    print "RF:", accuracy_score(test_labels, rf.predict(test_instances))
