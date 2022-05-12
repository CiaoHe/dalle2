import os
import math
import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter
from dalle2_pytorch.train import load_diffusion_model, save_diffusion_model, print_ribbon
from dalle2_pytorch.tokenizer import tokenizer
from dalle2_pytorch.optimizer import get_optimizer
from torch.cuda.amp import autocast, GradScaler

from mm_data import get_dataset
from resize_right import resize

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def resize_image_to(image, target_image_size):
    orig_h, orig_w = image.shape[-2:]
    
    if orig_h == target_image_size and orig_w == target_image_size:
        return image
    
    scale_factors = (target_image_size / orig_h, target_image_size / orig_w)
    
    return resize(image, scale_factors = scale_factors)


import time
from tqdm import tqdm

import wandb
os.environ["WANDB_SILENT"] = "true"
NUM_TEST_EMBEDDINGS = 100 # for cosine similarity reporting during training
REPORT_METRICS_EVERY = 100 # for cosine similarity report metrics every this many steps

def eval_model(model, clip, device, val_dl, loss_type, phase="Validation",):
    model.eval()
    with torch.no_grad():
        total_loss = 0.
        total_samples = 0       
        
        for image, text in tqdm(val_dl, desc=f"{phase}", leave=False):
            image = resize_image_to(image, clip.image_size).to(device)
            text = tokenizer.tokenize(text).to(device)
            
            batches = image.shape[0]
            
            loss = model(text = text, image = image)
            
            total_loss += loss.item() * batches
            total_samples += batches
            
        avg_loss = total_loss / total_samples
        wandb.log({f"{phase} {loss_type} Loss": avg_loss})
        
    
def report_cosine_sims(diffusion_prior:DiffusionPrior, clip, device, val_dataset):
    diffusion_prior.eval()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    # fetch images and texts, convert to embeddings
    val_dl = DataLoader(val_dataset, batch_size=NUM_TEST_EMBEDDINGS, shuffle=False, num_workers=1)
    image, text = next(iter(val_dl))
    image = resize_image_to(image, clip.image_size).to(device)
    text = tokenizer.tokenize(text).to(device)
    
    image_embed, _ = clip.embed_image(image)
    text_embed, _, _ = clip.embed_text(text)
    
    # make a copy of the text embeddings for shuffling
    text_embed_shuffled = text_embed.clone()
    
    # roll the text embeddings to simulate "unrelated" captions
    rolled_idx = torch.roll(torch.arange(NUM_TEST_EMBEDDINGS), 1)
    text_embed_shuffled = text_embed_shuffled[rolled_idx]
    text_embed_shuffled = text_embed_shuffled / torch.norm(text_embed_shuffled, dim=1, keepdim=True) # normalize
    test_text_shuffled_cond = dict(text_embed=text_embed_shuffled)
    
    # prepare the text embedding
    text_embed = text_embed / torch.norm(text_embed, dim=1, keepdim=True) # normalize
    test_text_cond = dict(text_embed=text_embed)
    
    # prepare the image embeddings
    test_image_embeddings = image_embed
    test_image_embeddings = test_image_embeddings / torch.norm(test_image_embeddings, dim=1, keepdim=True) # normalize
    
    # predict[^] on the unshuffled text embedding
    predicted_image_embeddings = diffusion_prior.p_sample_loop((NUM_TEST_EMBEDDINGS, image_embed.shape[-1]), test_text_cond)
    predicted_image_embeddings = predicted_image_embeddings / torch.norm(predicted_image_embeddings, dim=1, keepdim=True) # normalize 
    
    # predicted[^] on the shuffled text embedding
    predicted_unrelated_embeddings = diffusion_prior.p_sample_loop((NUM_TEST_EMBEDDINGS, image_embed.shape[-1]), test_text_shuffled_cond)
    predicted_unrelated_embeddings = predicted_unrelated_embeddings / torch.norm(predicted_unrelated_embeddings, dim=1, keepdim=True) # normalize
    
    # calculate similarities
    original_similarity = cos(text_embed, test_image_embeddings).cpu().numpy()
    predicted_similarity = cos(text_embed, predicted_image_embeddings).cpu().numpy()
    unrelated_similarity = cos(text_embed, predicted_unrelated_embeddings).cpu().numpy()
    predicted_img_similarity = cos(test_image_embeddings, predicted_image_embeddings).cpu().numpy()
    
    wandb.log(
        {"CosineSimilarity(text_embed,image_embed)": np.mean(original_similarity)})
    wandb.log({"CosineSimilarity(text_embed,predicted_image_embed)": np.mean(
        predicted_similarity)})
    wandb.log({"CosineSimilarity(text_embed,predicted_unrelated_embed)": np.mean(
        unrelated_similarity)})
    wandb.log({"CosineSimilarity(image_embed,predicted_image_embed)": np.mean(
        predicted_img_similarity)})
        
def train(
    image_embed_dim,
    batch_size,
    num_epochs,
    dp_loss_type,  # diffusion prior loss type
    clip,          # CLIP used
    dp_condition_on_text_encodings,
    dp_timesteps,  # T
    dp_normformer,
    dp_cond_drop_prob,
    dpn_depth,
    dpn_dim_head,
    dpn_heads,
    save_interval,
    save_path,
    device,
    RESUME,
    DPRIOR_PATH,
    config,
    wandb_project,
    learning_rate=0.001,
    max_grad_norm=0.5,
    weight_decay=0.01,
    dropout=0.05,
    amp=False
):  
    # clip
    if clip is not None:
        clip = OpenAIClipAdapter()
        print(f"Using open ai clip adapter with image size {clip.image_size}")
    
    # diffusion prior network
    prior_network = DiffusionPriorNetwork(
        dim = image_embed_dim,
        depth = dpn_depth,
        dim_head = dpn_dim_head,
        heads = dpn_heads,
        normformer = dp_normformer,
        attn_dropout = dropout,
        ff_dropout = dropout,
    ).to(device)
    
    # DiffusionPrior with text embeddings and image embeddings pre-computed
    diffusion_prior = DiffusionPrior(
        net = prior_network,
        clip = clip,
        image_embed_dim = image_embed_dim,
        timesteps=dp_timesteps,
        cond_drop_prob=dp_cond_drop_prob,
        loss_type=dp_loss_type,
        condition_on_text_encodings=dp_condition_on_text_encodings,
    ).to(device)
    
    # Load pre-trained model from DPRIOR_PATH
    if RESUME:
        diffusion_prior=load_diffusion_model(DPRIOR_PATH, device)   
        wandb.init(project=wandb_project, config=config) 
    
    # Create save_path if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Get dataset and dataloader
    tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        _convert_image_to_rgb,
        transforms.ToTensor(),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        _convert_image_to_rgb,
        transforms.ToTensor(),
    ])
    train_dataset, _ = get_dataset(dataset='CC_3M', split='train', image_transforms=tf)
    val_dataset, _ = get_dataset(dataset='CC_3M', split='val', image_transforms=tf_val)
    
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print("Number of training samples:", len(train_dl.dataset))

        
    # Train code
    scaler = GradScaler(enabled=amp)
    optimizer = get_optimizer(diffusion_prior.net.parameters(), learning_rate, weight_decay)
    epochs = num_epochs
    
    step = 0
    t = time.time()
    
    for _ in range(epochs):
        
        for image, text in tqdm(train_dl):
            diffusion_prior.train()
            image = resize_image_to(image, clip.image_size).to(device)
            text = tokenizer.tokenize(text).to(device)
            
            with autocast(enabled=amp):
                loss = diffusion_prior(text=text, image=image)
                scaler.scale(loss).backward()
                
            # Samples per second
            step += 1
            samples_per_sec = batch_size * step / (time.time() - t)
            
            # save checkpoint every save_interval minutes
            if (int(time.time() - t) >= save_interval * 60):
                print_ribbon(f"Saving checkpoint at {int(time.time() - t)} seconds")
                t = time.time()
                save_diffusion_model(
                    save_path,
                    diffusion_prior,
                    optimizer,
                    scaler,
                    config,
                    image_embed_dim
                )
            
            # Log to wandb
            wandb.log({"Training loss": loss.item(), "samples_per_sec": samples_per_sec, "steps": step})    
            
            # Log cosineSim(text_embed,predicted_image_embed) - cosineSim(text_embed,image_embed)
            # Use NUM_TEST_EMBEDDINGS samples from the test set each time
            # Get embeddings from the most recently saved model
            if (step % REPORT_METRICS_EVERY == 0):
                diff_cosine_sim = report_cosine_sims(diffusion_prior=diffusion_prior, clip=clip, device=device, val_dataset=val_dataset)
                wandb.log({"CosineSimilarity difference": diff_cosine_sim})
            
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(diffusion_prior.parameters(), max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        # Evaluate model (validation run)
        eval_model(model=diffusion_prior, clip=clip, device=device, val_dl=val_dl, loss_type=dp_loss_type, phase='Validation')
            
        
def main():
    parser = argparse.ArgumentParser()
    # Logging
    parser.add_argument("--wandb-entity", type=str, default="laion")
    parser.add_argument("--wandb-project", type=str, default="diffusion-prior")
    parser.add_argument("--wandb-name", type=str, default="laion-dprior")
    parser.add_argument("--wandb-dataset", type=str, default="LAION-5B")
    parser.add_argument("--wandb-arch", type=str, default="DiffusionPrior")
    # Hyperparameters
    parser.add_argument("--learning-rate", type=float, default=1.1e-4)
    parser.add_argument("--weight-decay", type=float, default=6.02e-2)
    parser.add_argument("--dropout", type=float, default=5e-2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=10**4)
    parser.add_argument("--num-epochs", type=int, default=5)
    # Image embed dimesion
    parser.add_argument("--image-embed-dim", type=int, default=512)
    # DiffusionPriorNetwork(dpn) parameters
    parser.add_argument("--dpn-depth", type=int, default=6)
    parser.add_argument("--dpn-dim-head", type=int, default=64)
    parser.add_argument("--dpn-heads", type=int, default=8)
    # DiffusionPrior(dp) parameters
    parser.add_argument("--dp-condition-on-text-encodings", type=bool, default=False)
    parser.add_argument("--dp-timesteps", type=int, default=100)
    parser.add_argument("--dp-normformer", type=bool, default=False)
    parser.add_argument("--dp-cond-drop-prob", type=float, default=0.1)
    parser.add_argument("--dp-loss-type", type=str, default="l2")
    parser.add_argument("--clip", type=str, default=None)
    parser.add_argument("--amp", type=bool, default=False)
    # Model checkpointing interval(in minutes)
    parser.add_argument("--save-interval", type=int, default=30)
    parser.add_argument("--save-path", type=str, default="./diffusion_prior_checkpoints")
    # Saved model path 
    parser.add_argument("--pretrained-model-path", type=str, default=None)
    
    args = parser.parse_args()
    
    config = ({"learning_rate": args.learning_rate,
        "architecture": args.wandb_arch,
        "dataset": args.wandb_dataset,
        "weight_decay":args.weight_decay,
        "max_gradient_clipping_norm":args.max_grad_norm,
        "batch_size":args.batch_size,
        "epochs": args.num_epochs,
        "diffusion_prior_network":{
            "depth":args.dpn_depth,
            "dim_head":args.dpn_dim_head,
            "heads":args.dpn_heads,
            "normformer":args.dp_normformer,
            "attn_dropout":args.dropout,
            "ff_dropout":args.dropout,
        },
        "diffusion_prior":{
            "condition_on_text_encodings": args.dp_condition_on_text_encodings,
            "timesteps": args.dp_timesteps,
            "cond_drop_prob":args.dp_cond_drop_prob,
            "loss_type":args.dp_loss_type,
            "clip":args.clip
        }
    })
    
    RESUME = False
    # Check if DPRIOR_PATH exists(saved model path)
    DPRIOR_PATH = args.pretrained_model_path
    if(DPRIOR_PATH is not None):
        RESUME = True
    else:
        wandb.init(
            project=args.wandb_project,
            config=config
        )
    
    # Obtain the utillized device
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    
    # Train loop
    train(
        image_embed_dim=args.image_embed_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        dp_loss_type=args.dp_loss_type,
        clip=args.clip,
        dp_condition_on_text_encodings=args.dp_condition_on_text_encodings,
        dp_timesteps=args.dp_timesteps,
        dp_normformer=args.dp_normformer,
        dp_cond_drop_prob=args.dp_cond_drop_prob,
        dpn_depth=args.dpn_depth,
        dpn_dim_head=args.dpn_dim_head,
        dpn_heads=args.dpn_heads,
        save_interval=args.save_interval,
        save_path=args.save_path,
        device=device,
        RESUME=RESUME,
        DPRIOR_PATH=DPRIOR_PATH,
        config=config,
        wandb_project=args.wandb_project,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        amp=args.amp,
    )

if __name__ == "__main__":
    main()