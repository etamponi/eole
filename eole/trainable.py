from abc import ABCMeta
import abc

__author__ = 'Emanuele Tamponi'


class Trainable(object):
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def train(self, instances):
        pass
