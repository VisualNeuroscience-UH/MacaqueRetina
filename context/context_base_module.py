from abc import ABCMeta, abstractmethod


class ContextBase(metaclass=ABCMeta):

    @property
    @abstractmethod
    def set_context():
        pass

    @property
    @abstractmethod
    def generate_hash():
        pass
