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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
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

    train_lr = 1e-4
    params = {"learning_rate": train_lr, "optimizer": "Adam"}
    run["parameters"] = params

    dataset = TranslatingPuzzleDataset(puzzle_size=(496, 496), pad_size=(16, 16), grid_size=16, total_steps=64,
                                       workers=cpu_count())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    optimizer = Adam(model.parameters(), lr=train_lr, betas=(0.9, 0.99))
    criterion = nn.L1Loss()
    model.train()

    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in tqdm(enumerate(dataloader)):
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

    run.stop()