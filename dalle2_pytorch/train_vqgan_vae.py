from math import sqrt
import copy
from random import choice
from pathlib import Path
from shutil import rmtree
from PIL import Image

import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, random_split

from PIL import Image
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image

from einops import rearrange

from dalle2_pytorch.train import EMA
from dalle2_pytorch.vqgan_vae import VQGanVAE
from dalle2_pytorch.optimizer import get_optimizer

# helpers

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data
    
def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# classes

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=('jpg', 'png', 'png'),
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        
        print(f'Found {len(self.paths)} training images in {folder}')
        
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = self.transform(img)
        return img
    
# main trainer class

class VQGanVAETrainer(nn.Module):
    def __init__(
        self,
        vae:VQGanVAE,
        *,
        num_train_steps,
        lr,
        batch_size,
        folder,
        grad_accum_every,
        wd = 0.,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        valid_frac = 0.05,
        random_split_seed = 42,
        ema_beta = 0.995,
        ema_update_after_step = 2000,
        ema_update_every = 10,
        apply_grad_penalty_every = 4,
        amp = False
    ):
        super().__init__()
        assert isinstance(vae, VQGanVAE), f'{vae} is not a VQGanVAE'
        image_size = vae.image_size
        
        self.vae = vae
        self.ema_vae = EMA(vae, beta=ema_beta, update_after_step=ema_update_after_step, update_every=ema_update_every)
        
        self.register_buffer('steps', torch.tensor([0]))
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every
        
        all_parameters = set(self.vae.parameters())
        discr_parameters = set(self.vae.discr.parameters())
        vae_parameters = all_parameters - discr_parameters
        
        self.optim = get_optimizer(vae_parameters, lr=lr, wd=wd)
        self.discr_optim = get_optimizer(discr_parameters, lr=lr, wd=wd)
        
        self.amp = amp
        self.scaler = GradScaler(enabled=self.amp)
        self.discr_scaler = GradScaler(enabled=self.amp)
        
        # create dataset
        
        self.ds = ImageDataset(folder, image_size)
        
        # split into train and valid
        
        if valid_frac > 0:
            train_size = int(len(self.ds) * (1 - valid_frac))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, (train_size, valid_size), generator=torch.Generator().manual_seed(random_split_seed))
            print(f'Split dataset into {train_size} train and {valid_size} valid samples')
        else:
            self.valid_ds = self.ds
            print(f'training with shared training and validation set of {len(self.ds)} samples')
        
        # dataloader
        
        self.dl = cycle(DataLoader(
            self.ds,
            batch_size=batch_size,
            shuffle=True,
        ))
        
        self.valid_dl = cycle(DataLoader(
            self.valid_ds,
            batch_size=batch_size,
            shuffle=True,
        ))
        
        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        
        self.apply_grad_penalty_every = apply_grad_penalty_every
        
        self.results_folder = Path(results_folder)
        
        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))
                
        self.results_folder.mkdir(parents=True, exist_ok=True)
        
    def train_step(self):
        device = next(self.vae.parameters()).device
        steps = int(self.steps.item())
        apply_grad_penalty = steps % self.apply_grad_penalty_every == 0
        
        self.vae.train()
        
        # logs
        logs = {}
        
        # update vae (generator)
        
        for _ in range(self.grad_accum_every):
            img = next(self.dl).to(device)
            
            with autocast(enabled=self.amp):
                loss = self.vae(
                    img,
                    return_loss=True,
                    apply_grad_penalty=apply_grad_penalty,
                )
                self.scaler.scale(loss / self.grad_accum_every).backward()
            
            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})
            
        
        self.scaler.step(self.optim)
        self.scaler.update()
        self.optim.zero_grad()
        
        # update vae (discriminator)
        
        if exists(self.vae.discr):
            for _ in range(self.grad_accum_every):
                img = next(self.dl).to(device)
                
                with autocast(enabled=self.amp):
                    loss = self.vae(
                        img,
                        return_discr_loss=True,
                    )
                    self.discr_scaler.scale(loss / self.grad_accum_every).backward()
                    
                accum_log(logs, {'discr_loss': loss.item() / self.grad_accum_every})
                (loss / self.grad_accum_every).backward()
            
            self.discr_scaler.step(self.discr_optim)
            self.discr_scaler.update()
            self.discr_optim.zero_grad()
            
            # log
            
            print(f'Step {steps}: vae loss: {logs["loss"]:.4f} - discr loss {logs["discr_loss"]:.4f}')
            
        # update ema vae
        
        self.ema_vae.update()
        
        # sample results every so often
        
        if steps % self.save_results_every == 0:
            for model, filename in ((self.ema_vae.ema_model, f'{steps}.ema'), (self.vae, str(steps))):
                model.eval()
                
                imgs = next(self.valid_dl).to(device)
                recons = model(imgs)
                nrows = int(sqrt(self.batch_size))
                
                imgs_and_recons = torch.stack([imgs, recons], dim=0)
                imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')
                
                imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0, 1)
                grid = make_grid(imgs_and_recons, nrow=nrows, normalize=True, value_range=(0, 1))
                
                logs['reconstructions'] = grid
                
                save_image(grid, self.results_folder / f'{filename}.png')
            
            print(f'Saved results at step {steps} to {self.results_folder}')
            
        # save model every so often
        
        if steps % self.save_model_every == 0:
            state_dict = self.vae.state_dict()
            model_path = str(self.results_folder / f'vae.{steps}.pt')
            torch.save(state_dict, model_path)

            ema_state_dict = self.ema_vae.state_dict()
            model_path = str(self.results_folder / f'vae.{steps}.ema.pt')
            torch.save(ema_state_dict, model_path)

            print(f'Saved model at step {steps} to {self.results_folder}')
            
        self.steps += 1
        return logs

    def train(self, log_fn=noop):
        device = next(self.vae.parameters()).device
        
        while self.steps < self.max_steps:
            logs = self.train_step()
            log_fn(logs)
        
        print('Training finished')