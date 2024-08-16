import random
import numpy as np

import torch
from torch.utils.data import DataLoader

class MetaLoader(object):
    """ wraps multiple data loaders """
    def __init__(self, loaders, accum_steps=1, distributed=False):
        assert isinstance(loaders, dict)
        self.name2iter = {}
        self.name2loader = {}
        self.sampling_pools = []
        self.name2index = {}
        self.probs = []
        self.names = []
        for i, (n, l) in enumerate(loaders.items()):
            l, p = l
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.name2index[n] = i
            self.probs.append(p)
            self.names.append(n)


        self.accum_steps = accum_steps
        self.distributed = distributed
        self.step = 0
        self.probs = np.array(self.probs, dtype=np.float32)
        self.probs = self.probs / self.probs.sum()
        
        assert len(self.names) == len(self.probs)
        for n, p in zip(self.names, self.probs):
            print(f'Name: {n}\tProb: {p} ')
    
    def update_prob(self, tasks, probs):
        for task, prob in zip(tasks, probs):
            index = self.name2index[task]
            self.probs[index] = prob
        self.probs = self.probs / self.probs.sum()


    def __iter__(self):
        """ this iterator will run indefinitely """
        task = np.random.choice(self.names, p=self.probs)
        while True:
            task = np.random.choice(self.names, p=self.probs)
            self.step += 1
            iter_ = self.name2iter[task]
            try:
                batch = next(iter_)
            except StopIteration:
                iter_ = iter(self.name2loader[task])
                batch = next(iter_)
                self.name2iter[task] = iter_

            yield task, batch

    """ wraps multiple data loaders """
class IterLoader(object):
    def __init__(self, loaders, accum_steps=1, distributed=False):
        self.loaders=loaders
    def __iter__(self):
        """ this iterator will run indefinitely """
        iter_=iter(self.loaders)
        while True:
            try:
                batch = next(iter_)
            except StopIteration:
                iter_ = iter(self.loaders)
                batch = next(iter_)
            yield  batch