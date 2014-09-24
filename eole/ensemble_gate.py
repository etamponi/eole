import numpy

__author__ = 'Emanuele'


class EnsembleGate(object):

    def __init__(self, experts, expert_weighting):
        self.experts = experts
        self.expert_weighting = expert_weighting

    def competence_matrix(self, instances):
        ret = numpy.zeros((len(instances), len(self.experts)))
        for j, e in enumerate(self.experts):
            ret[:, j] = self.expert_weighting.apply_to_all(instances, e.centroid)
        return ret

    def probability_matrix(self, instances):
        n_classes = len(self.experts[0].predict_probs([instances[0]])[0])
        ret = numpy.zeros((len(instances), len(self.experts), n_classes))
        for j, e in enumerate(self.experts):
            ret[:, j, :] = e.predict_probs(instances)
        return ret
