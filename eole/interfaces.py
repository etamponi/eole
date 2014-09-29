import abc

__author__ = 'Emanuele Tamponi'


class Trainable(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, instances):
        pass


class Weigher(Trainable):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_weights(self, instances, centroid):
        pass

    def generate_sample(self, instances, centroid):
        return self.get_weights(instances, centroid)


class Sampler(Trainable):
    __metaclass__ = abc.ABCMeta

    def __init__(self, weigher):
        self.weigher = weigher

    def train(self, instances):
        self.weigher.train(instances)

    def get_weights(self, instances, centroid):
        return self.weigher.get_weights(instances, centroid)

    @abc.abstractmethod
    def generate_sample(self, instances, centroid):
        pass
