import pickle
import dareblopy as db
from threading import Thread, Lock, Event
import random
import threading

import numpy as np
import torch
import torch.tensor
import torch.utils
import torch.utils.data
import time

import sys
sys.path.append('../')
from dlutils.batch_provider import batch_provider
from dlutils.shuffle import shuffle_ndarray

cpu = torch.device('cpu')

class TFRecordsDataset:
    def __init__(self, cfg, logger, rank=0, buffer_size_mb=200):
        self.cfg = cfg
        self.logger = logger
        self.part_size = cfg.DATASET.SIZE // cfg.DATASET.PART_COUNT
        self.iterator = None
        self.filenames = {}
        self.batch_size = 512
        self.features = {}
        self.rank = rank #一次执行几个文件块
        self.part_count_local = cfg.DATASET.PART_COUNT
        assert self.part_count_local % 1 == 0


        for r in range(2, cfg.DATASET.MAX_RESOLUTION_LEVEL + 1):
            files = []
            for i in range(self.part_count_local * rank, self.part_count_local * (rank + 1)):
                file = cfg.DATASET.PATH % (r, i)
                files.append(file)
            self.filenames[r] = files

        self.buffer_size_b = 1024 ** 2 * buffer_size_mb

        self.current_filenames = []

    def reset(self, lod, batch_size):
        assert lod in self.filenames.keys()
        self.current_filenames = self.filenames[lod]
        self.batch_size = batch_size

        img_size = 2 ** lod

        self.features = {
            # 'shape': db.FixedLenFeature([3], db.int64),
            'data': db.FixedLenFeature([3, img_size, img_size], db.uint8)
        }
        buffer_size = self.buffer_size_b // (3 * img_size * img_size)

        self.iterator = db.ParsedTFRecordsDatasetIterator(self.current_filenames, self.features, self.batch_size, buffer_size, seed=np.uint64(time.time() * 1000))

    def __iter__(self):
        return self.iterator

    def __len__(self):
        return self.part_count_local * self.part_size


def make_dataloader(cfg, logger, dataset, GPU_batch_size, local_rank):
    class BatchCollator(object):
        def __init__(self, device=torch.device("cpu")):
            self.device = device

        def __call__(self, batch):
            with torch.no_grad():
                x, = batch
                x = torch.tensor(x, requires_grad=True, device=torch.device(self.device), dtype=torch.float32)
                return x
    batches = db.data_loader(iter(dataset), BatchCollator(local_rank), len(dataset) // GPU_batch_size)
    return batches
