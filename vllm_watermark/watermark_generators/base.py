from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    @abstractmethod
    def sample_next(self, *args, **kwargs):
        pass
