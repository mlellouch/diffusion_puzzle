from pathlib import Path
from torch.utils.data import DataLoader
from diffusion_training.dataset import TranslatingPuzzleDataset
from diffusion_training.denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch
import neptune.new as neptune
from torch.optim import Adam
from multiprocessing import cpu_count
import torch.nn as nn
from tqdm import tqdm
import configargparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = configargparse.ArgParser()
    parser.add_argument('-c', '--config', required=True, is_config_file=True, help='config file path')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dataset_path', type=str, default=Path(__file__).parent.parent.joinpath(Path('datasets/faces')))
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--puzzle_size', type=int, default=228)
    parser.add_argument('--pad_size', type=int, default=28)
    parser.add_argument('--grid_size', type=int, default=8)
    parser.add_argument('--total_steps', type=int, default=64)
    parser.add_argument('--parrallel_loading', action='store_true', default=True)
    return parser.parse_args()

def run(args):
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    )

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
    dataset = TranslatingPuzzleDataset(puzzle_size=puzzle_size, pad_size=pad_size, grid_size=args.grid_size,
                                       total_steps=args.total_steps, workers=cpu_count())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=workers)

    optimizer = Adam(model.parameters(), lr=train_lr, betas=(0.9, 0.99))
    criterion = nn.L1Loss()
    model.train()

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in tqdm(enumerate(dataloader), total=len(dataset)):
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

        run["train/loss"].log(running_loss)

    run.stop()


if __name__ == '__main__':
    args = parse_args()
    run(args)