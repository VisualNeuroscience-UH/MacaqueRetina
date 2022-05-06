from abc import ABCMeta, abstractmethod

class AnalysisBase(metaclass = ABCMeta):

    @property
    @abstractmethod
    def context():
        pass

    @property
    @abstractmethod
    def data_io():
        pass

