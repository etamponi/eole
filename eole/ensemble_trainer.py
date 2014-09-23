import sklearn

from eole.local_expert import LocalExpert


__author__ = 'Emanuele'


class EnsembleTrainer(object):

    def __init__(self, base_estimator, n_experts, centroid_picker, sampling):
        self.base_estimator = base_estimator
        self.n_experts = n_experts
        self.centroid_picker = centroid_picker
        self.sampling = sampling

    def train(self, instances, labels):
        experts = []
        for c in self.centroid_picker.pick(instances, self.n_experts):
            expert = LocalExpert(sklearn.clone(self.base_estimator), c, self.sampling)
            expert.train(instances, labels)
            experts.append(expert)
        return experts
