import sklearn

from core.local_expert import LocalExpert


__author__ = 'Emanuele'


class EnsembleTrainer(object):

    def __init__(self, base_estimator, centroid_picker, weigher_sampler):
        self.base_estimator = base_estimator
        self.centroid_picker = centroid_picker
        self.weigher_sampler = weigher_sampler

    def train(self, n_experts, instances, labels):
        experts = []
        self.weigher_sampler.train(instances)
        for centroid in self.centroid_picker.pick(instances, labels, n_experts):
            expert = LocalExpert(sklearn.clone(self.base_estimator), self.weigher_sampler)
            expert.train(instances, labels, centroid)
            experts.append(expert)
        return experts
