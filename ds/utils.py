from torch.utils.data import Dataset, DataLoader, \
    Sampler, WeightedRandomSampler, RandomSampler, BatchSampler, DistributedSampler
from torch import Tensor, Generator

import torch


class _InfiniteSampler(Sampler):
    def __init__(self, sampler: Sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoaderCore:
    def __init__(self, dataset: Dataset, weights: Tensor, batch_size: int, num_workers: int, pin_memory: bool, generator: Generator):
        if weights is not None:
            self.sampler = WeightedRandomSampler(weights, replacement=True, num_samples=batch_size, generator=generator)
        else:
            self.sampler = RandomSampler(dataset, replacement=True, generator=generator)

        if weights == None:
            weights = torch.ones(len(dataset))

        self.batch_sampler = BatchSampler(self.sampler, batch_size=batch_size, drop_last=True)

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        raise ValueError


class InfiniteDataLoader(InfiniteDataLoaderCore):
    def __init__(self, dataset: Dataset, weights: Tensor, batch_size: int, num_workers: int, pin_memory: bool, generator: Generator):
        super().__init__(dataset, weights, batch_size, num_workers, pin_memory, generator)

        self._infinite_iterator = iter(
            DataLoader(dataset, num_workers=num_workers, pin_memory=pin_memory, 
                       batch_sampler=_InfiniteSampler(self.batch_sampler), generator=generator))
    
    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return super().__len__()


class _RepeatSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class DistributedInfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)