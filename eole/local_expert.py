import numpy

__author__ = 'Emanuele Tamponi'


class LocalExpert(object):

    def __init__(self, base_estimator, sampling):
        self.base_estimator = base_estimator
        self.sampling = sampling
        self.centroid = None

    def train(self, centroid, instances, labels):
        self.centroid = numpy.asarray(centroid)
        sample = self.sampling.generate_sample(instances, self.centroid)
        self.base_estimator.fit(instances, labels, sample_weight=sample)

    def predict(self, instances):
        return self.base_estimator.predict(instances)

    def predict_probs(self, instances):
        return self.base_estimator.predict_proba(instances)
