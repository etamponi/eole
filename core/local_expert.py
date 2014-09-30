import numpy

__author__ = 'Emanuele Tamponi'


class LocalExpert(object):

    def __init__(self, base_estimator, sampler_weigher):
        self.base_estimator = base_estimator
        self.sampler_weigher = sampler_weigher
        self.centroid = None
        self.sample_centroid = None

    def train(self, instances, labels, centroid):
        self.centroid = centroid
        sample_weights = self.sampler_weigher.get_sample_weights(instances, centroid)
        self.sample_centroid = numpy.average(instances, axis=0, weights=sample_weights)
        self.base_estimator.fit(instances, labels, sample_weight=sample_weights)
        return self

    def predict(self, instances):
        return self.base_estimator.predict(instances)

    def predict_probs(self, instances):
        return self.base_estimator.predict_proba(instances)

    def competence(self, instances):
        return self.sampler_weigher.get_weights(instances, self.sample_centroid)
