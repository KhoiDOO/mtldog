from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler, RandomSampler, BatchSampler
from torch import Tensor, Generator

import torch


class _InfiniteSampler(Sampler):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset: Dataset, weights: Tensor, batch_size: int, num_workers: int, pin_memory: bool, generator: Generator):
        super().__init__()

        if weights is not None:
            sampler = WeightedRandomSampler(weights, replacement=True, num_samples=batch_size, generator=generator)
        else:
            sampler = RandomSampler(dataset, replacement=True, generator=generator)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)

        self._infinite_iterator = iter(
            DataLoader(dataset, num_workers=num_workers, pin_memory=pin_memory, 
                       batch_sampler=_InfiniteSampler(batch_sampler), generator=generator)
        )

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError