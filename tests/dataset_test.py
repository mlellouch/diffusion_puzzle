import unittest

import torch
from puzzle_generation import grid_puzzle
from diffusion_training.dataset import TranslatingPuzzleDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from multiprocessing import cpu_count


class DatasetCase(unittest.TestCase):

    def test_sanity(self):
        dataset = TranslatingPuzzleDataset(puzzle_size=(496, 496), pad_size=(16, 16), grid_size=8, total_steps=100)
        for i in tqdm(range(300)):
            img, vector_field, step = dataset[i]
            assert type(img) == torch.Tensor
            assert type(vector_field) == torch.Tensor
            assert type(step) == int

    def test_mutiple_loaders(self):
        dataset = TranslatingPuzzleDataset(puzzle_size=(496, 496), pad_size=(16, 16), grid_size=16, total_steps=64, workers=cpu_count())
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=cpu_count())
        for img, field, i in dataloader:
            pass


if __name__ == '__main__':
    dataset = TranslatingPuzzleDataset(puzzle_size=(496, 496), pad_size=(16, 16), grid_size=8, total_steps=300)
    for i in tqdm(range(300)):
        img, vector_field, step = dataset[i]
    # unittest.main()
