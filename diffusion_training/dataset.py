from torch.utils.data import Dataset
from pathlib import Path
import os
import random
from puzzle_generation import grid_puzzle, moving_puzzle
from puzzle import TranslatingPuzzle
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize


faces_path = Path(__file__).parent.parent.joinpath(Path('datasets/faces/train'))

import time
class TranslatingPuzzleDataset(Dataset):
    """A dataset that uses translating puzzles"""

    def __init__(self, puzzle_size, pad_size, grid_size, total_steps, images_dir=str(faces_path), workers=1, start_at_random_step=True):
        self.puzzle_size = puzzle_size
        self.pad_size = pad_size
        self.grid_size = grid_size
        self.total_steps = total_steps
        self.images_dir = images_dir
        self.all_images = os.listdir(images_dir)
        self.start_at_random_step = start_at_random_step

        self.image_transforms = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.field_transforms = Compose([
            ToTensor(),
            # Normalize((0,0,0), (0.005, 0.005, 0.005))
        ])

        self.worker_states = []
        for w in range(workers):
            new_puzzle = self._load_new_puzzle()
            current_step = 0
            if self.start_at_random_step:
                current_step = random.randint(0, self.total_steps - 2)
                new_puzzle.set_current_step(current_step)
                new_puzzle.draw_with_vector_field()

            self.worker_states.append({
                'current_step': current_step,
                'current_puzzle': new_puzzle
            })



    def __len__(self):
        return len(self.all_images) * self.total_steps

    def _load_new_puzzle(self):
        image = random.choice(self.all_images)
        puzzle = grid_puzzle.image_to_grid_puzzle(image_path=os.path.join(self.images_dir, image), grid_size=self.grid_size,
                                                  puzzle_size=self.puzzle_size, puzzle_pad=self.pad_size)

        new_puzzle = moving_puzzle.create_random_translating_puzzle(puzzle, total_steps=self.total_steps)
        return new_puzzle


    def __getitem__(self, idx):
        """
        index is a dummy variable. What we'll do instead is each time load a new puzzle, run it, and return for each step
        :param idx:
        :return:
        """

        current_worker = self.worker_states[idx % len(self.worker_states)]

        if current_worker['current_step'] == self.total_steps: # we reached the end of the run
            current_worker['current_puzzle'] = self._load_new_puzzle()
            current_worker['current_step'] = 0

        current_puzzle = current_worker['current_puzzle']
        current_worker['current_step'] += 1
        current_puzzle.set_current_step(current_worker['current_step'])
        img, mask, vector_field = current_puzzle.draw_with_vector_field()
        return self.image_transforms((img / 255.0).astype(np.float32)), self.field_transforms((vector_field / 255.0).astype(np.float32)), current_worker['current_step']
