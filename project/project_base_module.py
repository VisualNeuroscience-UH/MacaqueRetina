from abc import ABCMeta, abstractmethod

class ProjectBase(metaclass = ABCMeta):


    @property
    @abstractmethod
    def context(self):
        pass

    @context.setter
    @abstractmethod
    def context(self, value):
        pass

    @property
    @abstractmethod
    def data_io(self):
        pass

    @data_io.setter
    @abstractmethod
    def data_io(self, value):
        pass

    @property
    @abstractmethod
    def construct(self):
        pass

    @data_io.setter
    @abstractmethod
    def construct(self, value):
        pass



