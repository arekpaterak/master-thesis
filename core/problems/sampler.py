from abc import ABC, abstractmethod


class BatchSampler[T](ABC):
    @abstractmethod
    def __call__(self, batch_size: int) -> T: ...
