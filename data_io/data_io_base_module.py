from abc import ABCMeta, abstractmethod

class DataIOBase(metaclass = ABCMeta):

    @property
    @abstractmethod
    def context():
        pass

    @abstractmethod
    def get_data():
        pass

    @abstractmethod
    def listdir_loop():
        pass

    @abstractmethod
    def most_recent():
        pass

    @abstractmethod
    def parse_path():
        pass


    
