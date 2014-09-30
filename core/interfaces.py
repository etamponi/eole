import abc

__author__ = 'Emanuele Tamponi'


class SamplerWeigher(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, instances):
        pass

    @abc.abstractmethod
    def get_weights(self, instances, centroid):
        pass

    @abc.abstractmethod
    def get_sample_weights(self, instances, centroid):
        pass


class BaseWeigher(SamplerWeigher):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, instances):
        pass

    @abc.abstractmethod
    def get_weights(self, instances, centroid):
        pass

    def get_sample_weights(self, instances, centroid):
        return self.get_weights(instances, centroid)


class BaseSampler(SamplerWeigher):
    __metaclass__ = abc.ABCMeta

    def __init__(self, weigher):
        self.weigher = weigher

    def train(self, instances):
        self.weigher.train(instances)

    def get_weights(self, instances, centroid):
        return self.weigher.get_weights(instances, centroid)

    @abc.abstractmethod
    def get_sample_weights(self, instances, centroid):
        pass
