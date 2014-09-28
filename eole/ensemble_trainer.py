import sklearn

from eole.local_expert import LocalExpert


__author__ = 'Emanuele'


class EnsembleTrainer(object):

    def __init__(self, base_estimator, centroid_picker, sampling):
        self.base_estimator = base_estimator
        self.centroid_picker = centroid_picker
        self.sampling = sampling

    def train(self, n_experts, instances, labels):
        experts = []
        self.sampling.train(instances)
        for centroid in self.centroid_picker.pick(instances, n_experts):
            expert = LocalExpert(sklearn.clone(self.base_estimator), self.sampling)
            expert.train(centroid, instances, labels)
            experts.append(expert)
        return experts
