import numpy

from eole import matrix_utils
from eole.ensemble_gate import EnsembleGate


__author__ = 'Emanuele'


class EOLE(object):

    def __init__(self, n_experts, ensemble_trainer, expert_weighting, preprocessor, use_probs, use_competences):
        self.n_experts = n_experts
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
        self.gate = EnsembleGate(self.ensemble_trainer.train(self.n_experts, instances, labels), self.expert_weighting)

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
