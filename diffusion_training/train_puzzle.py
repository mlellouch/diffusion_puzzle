from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from diffusion_training.dataset import TranslatingPuzzleDataset
from diffusion_training.denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from puzzle import TranslatingPuzzle
import torch
import neptune.new as neptune
from torch.optim import Adam
from multiprocessing import cpu_count
import torch.nn as nn
from tqdm import tqdm
import configargparse
import os
from typing import List
import cv2
import numpy as np

from puzzle_generation import grid_puzzle, moving_puzzle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = configargparse.ArgParser()
    parser.add_argument('-c', '--config', required=True, is_config_file=True, help='config file path')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dataset_path', type=str, default=Path(__file__).parent.parent.joinpath(Path('datasets/faces')))

    parser.add_argument('--test_per_epoch', type=int, default=1)
    parser.add_argument('--test_initial_steps', nargs='+', default=[1, 5, 10, 16, 32, 64])
    parser.add_argument('--save_model_per_epoch', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--experiment_name', type=str, required=True)

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--parrallel_loading', action='store_true', default=True)

    parser.add_argument('--puzzle_size', type=int, default=228)
    parser.add_argument('--pad_size', type=int, default=28)
    parser.add_argument('--grid_size', type=int, default=8)
    parser.add_argument('--total_steps', type=int, default=64)


    return parser.parse_args()


def save_model(epoch:int , model: nn.Module, experiment_dir: str):
    path = os.path.join(experiment_dir, f'{epoch}.pt')
    torch.save(model.state_dict(), path)


def load_model(experiment_dir:str , model:nn.Module):
    all_files = os.listdir(experiment_dir)
    all_epochs = [int(i[:i.index('.pt')]) for i in all_files if i.endswith('.pt')]
    if len(all_epochs) == 0:
        return model, 0

    max_epoch = max(all_epochs)
    path = os.path.join(experiment_dir, f'{max_epoch}.pt')
    model.load_state_dict(torch.load(path))
    model.eval()
    return model, max_epoch


def build_test_puzzles(args):
    images = os.path.join(args.dataset_path, 'test')
    puzzles = []
    puzzle_size = (args.puzzle_size, args.puzzle_size)
    pad_size = (args.pad_size, args.pad_size)
    for image in os.listdir(images):
        puzzle = grid_puzzle.image_to_grid_puzzle(image_path=os.path.join(images, image), grid_size=args.grid_size,
                                                  puzzle_size=puzzle_size, puzzle_pad=pad_size)

        new_puzzle = moving_puzzle.create_random_translating_puzzle(puzzle, total_steps=args.total_steps)
        puzzles.append(new_puzzle)
    return puzzles

def test_model(model, puzzles_to_fix: List[TranslatingPuzzle], initial_steps:List[int], experiment_dir, epoch):
    for idx, puzzle in enumerate(puzzles_to_fix):
        for initial_step in initial_steps:
            puzzle.set_current_step(initial_step)
            img, canvas = puzzle.draw()
            cv2.imwrite(os.path.join(experiment_dir, f'puzzle_{idx}_at_{initial_step}.jpg'), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            final_image = puzzle.run_model_on_puzzle(model, initial_step=initial_step)
            cv2.imwrite(os.path.join(experiment_dir, f'puzzle_{idx}_epoch_{epoch}_initial_step_{initial_step}.jpg'), cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))



def run(args):
    experiment_dir = os.path.join(args.log_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    )

    model, first_epoch = load_model(experiment_dir, model)
    test_puzzles = build_test_puzzles(args)

    if torch.cuda.is_available():
        model = model.cuda()

    run = neptune.init(
        project="michael.lellouch/DiffusionPuzzle",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmVjNDRjMy0wNjQwLTRhOWItODZlZi1mNzAyOTBmNmZjMjUifQ==",
    )  # your credentials

    train_lr = args.lr
    params = {"learning_rate": train_lr, "optimizer": "Adam"}
    run["parameters"] = params

    puzzle_size = (args.puzzle_size, args.puzzle_size)
    pad_size = (args.pad_size, args.pad_size)
    workers = cpu_count() if args.parrallel_loading else 0
    train_path = os.path.join(args.dataset_path, 'train')
    dataset = TranslatingPuzzleDataset(puzzle_size=puzzle_size, pad_size=pad_size, grid_size=args.grid_size, images_dir=train_path,
                                       total_steps=args.total_steps, workers=cpu_count())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=train_lr, betas=(0.9, 0.99))
    criterion = nn.L1Loss()
    model.train()

    for epoch in tqdm(range(first_epoch, args.epochs)):
        running_loss = 0.0
        model.train()
        for i, data in tqdm(enumerate(dataloader), total=len(dataset) // 4, leave=True):
            # get the inputs; data is a list of [inputs, labels]
            img, vector_field, time = data
            img, vector_field, time = img.to(device=device), vector_field.to(device=device), time.type(torch.int32).to(device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(img, time)
            loss = criterion(outputs, vector_field)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        if (epoch+1) % args.test_per_epoch == 0:
            model.eval()
            test_model(model=model, puzzles_to_fix=test_puzzles, initial_steps=args.test_initial_steps, experiment_dir=experiment_dir, epoch=epoch)

        if (epoch + 1) % args.save_model_per_epoch == 0:
            save_model(epoch, model, experiment_dir)

        run["train/loss"].log(running_loss)

    run.stop()


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    args = parse_args()
    run(args)
