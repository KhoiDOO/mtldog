from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler, RandomSampler, BatchSampler, DistributedSampler

import torch
import math


class _InfiniteSampler(Sampler):
    def __init__(self, sampler: Sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoaderCore:
    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int, pin_memory: bool):
        self.dataset = dataset
        self.sampler = RandomSampler(dataset, replacement=True)
        self.batch_sampler = BatchSampler(self.sampler, batch_size=batch_size, drop_last=True)

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        raise ValueError


class InfiniteDataLoader(InfiniteDataLoaderCore):
    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int, pin_memory: bool):
        super().__init__(dataset, batch_size, num_workers, pin_memory)

        self._infinite_iterator = iter(
            DataLoader(dataset, num_workers=num_workers, pin_memory=pin_memory, 
                       batch_sampler=_InfiniteSampler(self.batch_sampler)))
    
    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return super().__len__()


class RepeatSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class DistributedInfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class SmartDistributedSampler(DistributedSampler):
    def __iter__(self):
        self.round = 0
        g = torch.Generator()
        g.manual_seed(self.seed + self.round)

        n = int((len(self.dataset) - self.rank - 1) / self.num_replicas) + 1
        idx = torch.randperm(n, generator=g)
        if not self.shuffle:
            idx = idx.sort()[0]

        idx = idx.tolist()
        if self.drop_last:
            idx = idx[: self.num_samples]
        else:
            padding_size = self.num_samples - len(idx)
            if padding_size <= len(idx):
                idx += idx[:padding_size]
            else:
                idx += (idx * math.ceil(padding_size / len(idx)))[:padding_size]

        return iter(idx)

    def set_round(self, round):
        self.round = round