import numpy

__author__ = 'Emanuele Tamponi'


class LocalExpert(object):

    def __init__(self, base_estimator, sampler):
        self.base_estimator = base_estimator
        self.sampler = sampler
        self.centroid = None

    def train(self, centroid, instances, labels):
        self.centroid = numpy.asarray(centroid)
        sample = self.sampler.generate_sample(instances, self.centroid)
        self.base_estimator.fit(instances, labels, sample_weight=sample)

    def predict(self, instances):
        return self.base_estimator.predict(instances)

    def predict_probs(self, instances):
        return self.base_estimator.predict_proba(instances)
