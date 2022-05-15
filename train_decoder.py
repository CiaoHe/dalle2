import os
import math
import argparse
from pathlib import Path
import time
from omegaconf import OmegaConf
from tqdm import tqdm
import copy

import wandb
os.environ["WANDB_SILENT"] = "true"

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from dalle2_pytorch import OpenAIClipAdapter, Unet, Decoder
from dalle2_pytorch.train import print_ribbon, DecoderTrainer
from dalle2_pytorch.tokenizer import tokenizer

from mm_data import get_dataset
from resize_right import resize

# helper
def exists(val):
    return val is not None

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def resize_image_to(image, target_image_size):
    orig_h, orig_w = image.shape[-2:]
    
    if orig_h == target_image_size and orig_w == target_image_size:
        return image
    
    scale_factors = (target_image_size / orig_h, target_image_size / orig_w)
    
    return resize(image, scale_factors = scale_factors)

@torch.no_grad()
def tensor2grid(tensor, unnormalize=True):
    """
    Convert a tensor into a grid of images.
        unormalize: if True, will unormalize the image from (-1,1)->(0,1)
    """
    if unnormalize:
        tensor = tensor / 2 + 0.5
    tensor = (tensor * 255).to(torch.uint8)
    sample = make_grid(tensor, nrow=int(math.sqrt(tensor.shape[0]))).cpu().numpy()
    sample = sample.transpose((1, 2, 0))
    return sample
    

# dataset helper

def get_dl(mode, dataset, batch_size, shuffle, image_resize=256):
    # sorry for limited support
    assert dataset == 'CC_3M', "Only CC_3M dataset now is supported"
    # define transforms
    if mode== "train":
        tf = transforms.Compose([
        transforms.Resize((image_resize, image_resize), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_resize),
        _convert_image_to_rgb,
        transforms.ToTensor(),
    ])
    elif mode == "val":
        tf = transforms.Compose([
            transforms.Resize((image_resize, image_resize), interpolation=transforms.InterpolationMode.BICUBIC),
            _convert_image_to_rgb,
            transforms.ToTensor(),
        ])
    # get dataset
    dataset, _ = get_dataset(dataset=dataset, split=mode, image_transforms=tf)
    # get dataloader
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    print("Dataset in mode-{} has {} samples".format(mode, len(dataset)))
    
    return dl

def load_decoder_model(path, device, load_clip=False, clip=None):
    decoder_path = Path(path)
    assert decoder_path.exists(), f"Decoder model path {decoder_path} does not exist"
    loaded_obj = torch.load(decoder_path, map_location = 'cpu')
    
    # Get hyperparameters for decoder
    decoder_config = loaded_obj['hparams']['decoder']
    # Get hyperparameters for unets
    unet_configs = [None] * len(loaded_obj['hparams']['unets'])
    for i, unet_name in enumerate(loaded_obj['hparams']['unets'].keys()):
        unet_configs[i] = loaded_obj['hparams']['unets'][unet_name]
        
    # Load CLIP (optional) and create it
    if load_clip:
        assert not exists(clip), "load trained clip, no need provide initialized clip"
        CLIP_MODELTYPE = loaded_obj['clip']['ModelType']
        clip:nn.Module = CLIP_MODELTYPE(loaded_obj['clip'].get('hparams', None))
        clip = clip.load_state_dict(loaded_obj['clip']['weights']).to(device)
    else:
        assert exists(clip), "no clip will provided in ckpt, we need parse it from outer"
    
    # Create Unets
    unets = [None] * len(unet_configs)
    for i in range(len(unet_configs)):
        unets[i] = Unet(**unet_configs[i]).to(device)
    unets = tuple(unets)
    
    # Create Decoder
    decoder = Decoder(
        unet = unets,
        clip = clip,
        **decoder_config
    ).to(device)
    
    # Loader from ckpt
    decoder.load_state_dict(loaded_obj['decoder'], strict=False)
    return decoder

def save_decoder_model(save_path, decoder:Decoder, clip, hparams, step):
    # avoid save decoder's clip
    decoder = copy.deepcopy(decoder)
    if exists(decoder.clip):
        del decoder.clip
    # Saving State Dict
    print_ribbon('Saving checkpoint')
    state_dict = dict(
        decoder = decoder.state_dict(),
        hparams = hparams,
    )
    if exists(clip):
        if clip.__class__.__name__ != "OpenAIClipAdapter":
            state_dict['clip'] = {
                'ModelType': clip.__class__.__name__,
                'weights': clip.state_dict(),
            }
    torch.save(state_dict, save_path+'/'+str(step)+'_saved_model.pth')
    print_ribbon('Saved checkpoint to '+save_path+'/'+str(step)+'_saved_model.pth')
    

def eval(trainer:DecoderTrainer, val_dl, device, cond_scale=1.0):
    trainer.eval()
    clip = trainer.decoder.clip
    for image, text in val_dl:
        image = image = image.to(device)
        text = tokenizer.tokenize(text).to(device)

        image_embed = clip.embed_image(image).image_embed
        samples = trainer.sample(image_embed, text=text, cond_scale=cond_scale)
        sample = tensor2grid(samples)
        break
    return sample
    
def train(trainer:DecoderTrainer, train_dl, val_dl, cfg, device):
    # set Logger
    # NEED automatically create log/ckpt dir
    if not os.path.exists(cfg.wandb_dir):
        os.makedirs(cfg.wandb_dir, exist_ok=True)
    wandb_config = {
            'lr': cfg.learning_rate,
            'wd': cfg.weight_decay,
            'max_gradient_clipping_norm': cfg.max_gradient_clipping_norm,
            'batch_size': cfg.batch_size,
            'max_batch_size': cfg.max_batch_size,
            'num_epochs': cfg.epochs,
        }
    wandb_config.update(**cfg.mconfig)
    wandb.init(
        project=cfg.wandb_project, 
        config = wandb_config,
        dir = cfg.wandb_dir,
    )
    
    # set save path
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path, exist_ok=True)
    
    # Train Loop
    epochs = cfg.epochs
    step = 0
    t = time.time()
    
    for _ in range(epochs):
        
        for image, text in tqdm(train_dl):
            trainer.train()
            # resize image and tokenize text
            image = image.to(device)
            text = tokenizer.tokenize(text).to(device)

            # forward
            step += 1
            loss_log = {}
            
            for unet_index in range(len(trainer.decoder.unets)):
                unet_number = unet_index + 1
                loss = trainer(image, text=text, unet_number=unet_number, max_batch_size=cfg.max_batch_size)
                
                trainer.update(unet_number)
                loss_log[f"Training loss - Unet{unet_number}"] = loss
                
            # Log to wandb
            wandb.log(loss_log) 
            
            # eval time - sample
            if step > 0 and step % cfg.eval_interval == 0:
                print_ribbon('Evaluating')
                sample = eval(trainer, val_dl, device, cond_scale=cfg.cond_scale)
                # wandb visualize
                wandb.log({"sample": wandb.Image(sample)}, step=step)
            
            #save checkpoint every save_interval minutes
            if (int(time.time() - t) >= cfg.save_interval * 60):
                print_ribbon(f"Saving checkpoint at {int(time.time() - t)} seconds")
                t = time.time()
                save_decoder_model(
                    cfg.save_path, 
                    decoder = trainer.decoder, 
                    clip = trainer.decoder.clip, 
                    hparams = dict(
                        decoder=cfg.mconfig.Decoder,
                        unets=cfg.mconfig.Unets,
                    ),
                    step=step
                )
    
def main():
    # argument parser
    
    parser = argparse.ArgumentParser()
    # Logging
    parser.add_argument("--wandb-project", type=str, default="dalle2-decoder")
    # Confg Path
    parser.add_argument("--config-path", type=str, default="./configs/decoder.yaml")
    # Dataset
    parser.add_argument("--dataset", type=str, default="CC_3M")
    # Train Hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=10**4)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--amp", type=bool, default=False)
    parser.add_argument("--cond-scale", type=float, default=1.0)
    parser.add_argument('--ema_update_after_step', type=int, default=1000)
    parser.add_argument('--ema_update_every', type=int, default=10)
    # set which clip to use
    parser.add_argument("--clip", type=str, default=None)    
    # Evaluation sampling during training
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=1000)
    # Model checkpointing interval(in minutes)
    parser.add_argument("--save-interval", type=int, default=30)
    parser.add_argument("--save-path", type=str, default="./decoder_checkpoints")
    # Saved model path 
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrained-model-path", type=str, default=None)
    
    args = parser.parse_args()
    mconfig = OmegaConf.load(args.config_path)
    
    train_config = OmegaConf.create({
        "learning_rate": args.learning_rate,
        "weight_decay":args.weight_decay,
        "max_gradient_clipping_norm":args.max_grad_norm,
        "batch_size":args.batch_size,
        "max_batch_size":args.max_batch_size,
        "eval_batch_size":args.eval_batch_size,
        "eval_interval":args.eval_interval,
        "epochs": args.num_epochs,
        "amp": args.amp,
        "cond_scale": args.cond_scale,
        "mconfig": mconfig,
        "wandb_project": args.wandb_project,
        "save_path": args.save_path,
        "save_interval": args.save_interval,
        "wandb_dir": './wandb_decoder',
    })
    
    # set dataloader
    train_dl = get_dl(mode="train", dataset=args.dataset, batch_size=args.batch_size, shuffle=True)
    val_dl = get_dl(mode="val", dataset=args.dataset, batch_size=args.eval_batch_size, shuffle=False)
    
    # Obtain the utillized device
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        device = torch.device("cuda:0")
        
    # init clip
    clip = None
    if exists(args.clip):
        if args.clip == "openai_clip":
            clip = OpenAIClipAdapter().to(device)
        
    # resume if needed
    # Check if DECODER_PATH exists(saved model path)
    if args.resume:
        DECODER_PATH = Path(args.pretrained_model_path)
        if DECODER_PATH.exists():
            print_ribbon("Loading pretrained model from {}".format(DECODER_PATH))
            decoder = load_decoder_model(DECODER_PATH, device=device, 
                                        load_clip=not args.clip == "openai_clip", clip=clip)
    else:
        # init unets
        unets = []
        for unet_name in mconfig.Unets:
            unets.append(Unet(**mconfig.Unets[unet_name]).to(device))
        
        # init decoder
        decoder = Decoder(
            unet = tuple(unets),
            clip = clip,
            **mconfig.Decoder
        ).to(device)
        
    decoder_trainer = DecoderTrainer(
        decoder = decoder,
        use_ema=True,
        lr = args.learning_rate,
        wd = args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        ema_beta = 0.9999,
        ema_update_after_step = args.ema_update_after_step,
        ema_update_every = args.ema_update_every,
        amp = args.amp,
    )
    
    # train loop
    train(decoder_trainer, train_dl, val_dl, train_config, device)

if __name__ == "__main__":
    main()