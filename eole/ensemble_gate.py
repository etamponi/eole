import numpy

__author__ = 'Emanuele'


class EnsembleGate(object):

    def __init__(self, experts, expert_weigher):
        self.experts = experts
        self.expert_weigher = expert_weigher

    def competence_matrix(self, instances):
        ret = numpy.zeros((len(instances), len(self.experts)))
        for j, e in enumerate(self.experts):
            ret[:, j] = self.expert_weigher.get_weights(instances, e.centroid)
        return ret

    def probability_matrix(self, instances):
        n_classes = len(self.experts[0].predict_probs([instances[0]])[0])
        ret = numpy.zeros((len(instances), len(self.experts), n_classes))
        for j, e in enumerate(self.experts):
            ret[:, j, :] = e.predict_probs(instances)
        return ret
