import numpy

from core.interfaces import BaseSampler


__author__ = 'Emanuele'


class GeneralizedBootstrap(BaseSampler):

    def __init__(self, sample_percent, weigher):
        super(GeneralizedBootstrap, self).__init__(weigher)
        self.sample_percent = float(sample_percent) / 100

    def generate_sample(self, instances, centroid):
        size = int(len(instances) * self.sample_percent)
        probs = self.get_weights(instances, centroid)
        probs = probs / probs.sum()
        choices = numpy.random.choice(len(instances), size=size, replace=True, p=probs)
        sample = numpy.array([sum(choices == i) for i in range(len(instances))])
        return sample
