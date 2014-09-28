import numpy

from eole.trainable import Trainable


__author__ = 'Emanuele'


class GeneralizedBootstrap(Trainable):

    def train(self, instances):
        self.weighting.train(instances)

    def __init__(self, sample_percent, weighting):
        self.sample_percent = float(sample_percent) / 100
        self.weighting = weighting

    def generate_sample(self, instances, centroid):
        size = int(len(instances) * self.sample_percent)
        choices = numpy.random.choice(
            len(instances),
            size=size,
            replace=True,
            p=self.weighting.apply_to_all(instances, centroid, normalize=True)
        )
        sample = numpy.array([sum(choices == i) for i in range(len(instances))])
        return sample
