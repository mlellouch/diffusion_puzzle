from pathlib import Path
from diffusion_training.denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch

if __name__ == '__main__':
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l1'            # L1 or L2
    )

    if torch.cuda.is_available():
        model = model.cuda()
        diffusion = diffusion.cuda()



    images_path = Path(__file__).parent.parent.joinpath(Path('datasets')).joinpath('faces')

    trainer = Trainer(
        diffusion,
        str(images_path),
        train_batch_size = 2,
        train_lr = 8e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True                        # turn on mixed precision
    )

    trainer.train()