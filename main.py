import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST, CIFAR10

import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

import os
import argparse
import pickle
import time
import json
import pprint
from functools import partial

from optim import Adam, RMSprop

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(dest='model', help='Select model architecture.', required=True)
# pixelcnn args
parser_a = subparsers.add_parser('pixelcnn')
parser_a.add_argument('--n_channels', default=128, type=int, help='Number of channels for gated residual convolutional layers.')
parser_a.add_argument('--n_out_conv_channels', default=1024, type=int, help='Number of channels for outer 1x1 convolutional layers.')
parser_a.add_argument('--n_res_layers', default=12, type=int, help='Number of Gated Residual Blocks.')
parser_a.add_argument('--kernel_size', default=5, type=int, help='Kernel size for the gated residual convolutional blocks.')
parser_a.add_argument('--norm_layer', default=True, type=eval, help='Add a normalization layer in every Gated Residual Blocks.')
# pixelcnn++ args
parser_b = subparsers.add_parser('pixelcnnpp')
parser_b.add_argument('--n_channels', default=128, type=int, help='Number of channels for residual blocks.')
parser_b.add_argument('--n_res_layers', default=5, type=int, help='Number of residual blocks at each stage.')
parser_b.add_argument('--n_logistic_mix', default=10, type=int, help='Number of of mixture components for logistics output.')
# pixelsnail args
parser_c = subparsers.add_parser('pixelsnail')
parser_c.add_argument('--n_channels', default=256, type=int, help='Number of channels for residual blocks.')
parser_c.add_argument('--n_res_layers', default=5, type=int, help='Number of residual blocks in each attention layer.')
parser_c.add_argument('--attn_n_layers', default=12, type=int, help='Number of attention layers.')
parser_c.add_argument('--attn_nh', default=1, type=int, help='Number of attention heads.')
parser_c.add_argument('--attn_dq', default=16, type=int, help='Size of attention queries and keys.')
parser_c.add_argument('--attn_dv', default=128, type=int, help='Size of attention values.')
parser_c.add_argument('--attn_drop_rate', default=0, type=float, help='Dropout rate on attention logits.')
parser_c.add_argument('--n_logistic_mix', default=10, type=int, help='Number of of mixture components for logistics output.')

# action
parser.add_argument('--train', action='store_true', help='Train model.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate model.')
parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
parser.add_argument('--cuda', type=int, help='Which cuda device to use.')
parser.add_argument('--mini_data', action='store_true', help='Truncate dataset to mini_data number of examples.')
# data params
parser.add_argument('--dataset', choices=['mnist', 'colored-mnist', 'cifar10'])
parser.add_argument('--n_cond_classes', type=int, help='Number of classes for class conditional model.')
parser.add_argument('--n_bits', type=int, default=4, help='Number of bits of input data.')
parser.add_argument('--image_dims', type=int, nargs='+', default=(3,28,28), help='Dimensions of the input data.')
parser.add_argument('--data_path', default='./data/mnist-hw1.pkl', help='Location of datasets.')
# training param
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
parser.add_argument('--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization.')
parser.add_argument('--polyak', type=float, default=0.9995, help='Polyak decay for parameter exponential moving average.')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--step', type=int, default=0, help='Current step of training (number of minibatches processed).')
parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
parser.add_argument('--log_interval', type=int, default=50, help='How often to show loss statistics and save samples.')
parser.add_argument('--eval_interval', type=int, default=10, help='How often to evaluate and save samples.')
# generation param
parser.add_argument('--n_samples', type=int, default=8, help='Number of samples to generate.')


# --------------------
# Data
# --------------------

def fetch_dataloaders(args):
    # preprocessing transforms
    transform = T.Compose([T.ToTensor(),                                            # tensor in [0,1]
                           lambda x: x.mul(255).div(2**(8-args.n_bits)).floor(),    # lower bits
                           partial(preprocess, n_bits=args.n_bits)])                # to model space [-1,1]
    target_transform = (lambda y: torch.eye(args.n_cond_classes)[y]) if args.n_cond_classes else None

    if args.dataset=='mnist':
        args.image_dims = (1,28,28)
        train_dataset = MNIST(args.data_path, train=True, transform=transform, target_transform=target_transform)
        valid_dataset = MNIST(args.data_path, train=False, transform=transform, target_transform=target_transform)
    elif args.dataset=='cifar10':
        args.image_dims = (3,32,32)
        train_dataset = CIFAR10(args.data_path, train=True, transform=transform, target_transform=target_transform)
        valid_dataset = CIFAR10(args.data_path, train=False, transform=transform, target_transform=target_transform)
    elif args.dataset=='colored-mnist':
        args.image_dims = (3,28,28)
        # NOTE -- data is quantized to 2 bits and in (N,H,W,C) format
        with open(args.data_path, 'rb') as f:  # return dict {'train': np array; 'test': np array}
            data = pickle.load(f)
        # quantize to n_bits to match the transforms for other datasets and construct tensors in shape N,C,H,W
        train_data = torch.from_numpy(np.floor(data['train'].astype(np.float32) / (2**(2 - args.n_bits)))).permute(0,3,1,2)
        valid_data = torch.from_numpy(np.floor(data['test'].astype(np.float32) / (2**(2 - args.n_bits)))).permute(0,3,1,2)
        # preprocess to [-1,1] and setup datasets -- NOTE using 0s for labels to have a symmetric dataloader
        train_dataset = TensorDataset(preprocess(train_data, args.n_bits), torch.zeros(train_data.shape[0]))
        valid_dataset = TensorDataset(preprocess(valid_data, args.n_bits), torch.zeros(valid_data.shape[0]))
    else:
        raise RuntimeError('Dataset not recognized')

    if args.mini_data:  # dataset to a single batch
        if args.dataset=='colored-mnist':
            train_dataset = train_dataset.tensors[0][:args.batch_size]
        else:
            train_dataset.data = train_dataset.data[:args.batch_size]
            train_dataset.targets = train_dataset.targets[:args.batch_size]
        valid_dataset = train_dataset

    print('Dataset {}\n\ttrain len: {}\n\tvalid len: {}\n\tshape: {}\n\troot: {}'.format(
        args.dataset, len(train_dataset), len(valid_dataset), train_dataset[0][0].shape, args.data_path))

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=(args.device.type=='cuda'), num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, pin_memory=(args.device.type=='cuda'), num_workers=4)

    # save a sample
    data_sample = next(iter(train_dataloader))[0]
    writer.add_image('data_sample', make_grid(data_sample, normalize=True, scale_each=True), args.step)
    save_image(data_sample, os.path.join(args.output_dir, 'data_sample.png'), normalize=True, scale_each=True)

    return train_dataloader, valid_dataloader

def preprocess(x, n_bits):
    # 1. convert data to float
    # 2. normalize to [0,1] given quantization
    # 3. shift to [-1,1]
    return x.float().div(2**n_bits - 1).mul(2).add(-1)

def deprocess(x, n_bits):
    # 1. shift to [0,1]
    # 2. quantize to n_bits
    # 3. convert data to long
    return x.add(1).div(2).mul(2**n_bits - 1).long()

def save_json(data, filename, args):
    with open(os.path.join(args.output_dir, filename + '.json'), 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# --------------------
# Train, evaluate, generate
# --------------------

def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, epoch, writer, args):
    model.train()

    with tqdm(total=len(dataloader), desc='epoch {}/{}'.format(epoch, args.start_epoch + args.n_epochs)) as pbar:
        for x,y in dataloader:
            args.step += 1

            x = x.to(args.device)
            logits = model(x, y.to(args.device) if args.n_cond_classes else None)
            loss = loss_fn(logits, x, args.n_bits).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()

            pbar.set_postfix(bits_per_dim='{:.4f}'.format(loss.item() / (np.log(2) * np.prod(args.image_dims))))
            pbar.update()

            # record
            if args.step % args.log_interval == 0:
                writer.add_scalar('train_bits_per_dim', loss.item() / (np.log(2) * np.prod(args.image_dims)), args.step)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], args.step)

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, args):
    model.eval()

    losses = 0
    for x,y in tqdm(dataloader, desc='Evaluate'):
        x = x.to(args.device)
        logits = model(x, y.to(args.device) if args.n_cond_classes else None)
        losses += loss_fn(logits, x, args.n_bits).mean(0).item()
    return losses / len(dataloader)

@torch.no_grad()
def generate(model, generate_fn, args):
    model.eval()
    if args.n_cond_classes:
        samples = []
        for h in range(args.n_cond_classes):
            h = torch.eye(args.n_cond_classes)[h,None].to(args.device)
            samples += [generate_fn(model, args.n_samples, args.image_dims, args.device, h=h)]
        samples = torch.cat(samples)
    else:
        samples = generate_fn(model, args.n_samples, args.image_dims, args.device)
    return make_grid(samples.cpu(), normalize=True, scale_each=True, nrow=args.n_samples)

def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, loss_fn, generate_fn, writer, args):
    for epoch in range(args.start_epoch, args.start_epoch + args.n_epochs):
        # train
        train_epoch(model, train_dataloader, optimizer, scheduler, loss_fn, epoch, writer, args)

        if (epoch+1) % args.eval_interval == 0:
            # save model
            torch.save({'epoch': epoch,
                        'global_step': args.step,
                        'state_dict': model.state_dict()},
                        os.path.join(args.output_dir, 'checkpoint.pt'))
            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, 'optim_checkpoint.pt'))
            if scheduler: torch.save(scheduler.state_dict(), os.path.join(args.output_dir, 'sched_checkpoint.pt'))

            # swap params to ema values
            optimizer.swap_ema()

            # evaluate
            eval_loss = evaluate(model, test_dataloader, loss_fn, args)
            print('Evaluate bits per dim: {:.3f}'.format(eval_loss.item() / (np.log(2) * np.prod(args.image_dims))))
            writer.add_scalar('eval_bits_per_dim', eval_loss.item() / (np.log(2) * np.prod(args.image_dims)), args.step)

            # generate
            samples = generate(model, generate_fn, args)
            writer.add_image('samples', samples, args.step)
            save_image(samples, os.path.join(args.output_dir, 'generation_sample_step_{}.png'.format(args.step)))

            # restore params to gradient optimized
            optimizer.swap_ema()

# --------------------
# Main
# --------------------

if __name__ == '__main__':
    args = parser.parse_args()
    args.output_dir = os.path.dirname(args.restore_file) if args.restore_file else \
                        os.path.join('results', args.model, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
    writer = SummaryWriter(log_dir = args.output_dir)

    # save config
    if not os.path.exists(os.path.join(args.output_dir, 'config.json')): save_json(args.__dict__, 'config', args)
    writer.add_text('config', str(args.__dict__))
    pprint.pprint(args.__dict__)

    args.device = torch.device('cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_dataloader, test_dataloader = fetch_dataloaders(args)

    if args.model=='pixelcnn':
        import pixelcnn
        model = pixelcnn.PixelCNN(args.image_dims, args.n_bits, args.n_channels, args.n_out_conv_channels, args.kernel_size,
                                  args.n_res_layers, args.n_cond_classes, args.norm_layer).to(args.device)
        # images need to be deprocessed to [0, 2**n_bits) for loss fn
        loss_fn = lambda scores, targets, n_bits: pixelcnn.loss_fn(scores, deprocess(targets, n_bits))
        # multinomial sampling needs to be processed to [-1,1] at generation
        generate_fn = partial(pixelcnn.generate_fn, preprocess_fn=preprocess, n_bits=args.n_bits)
        optimizer = RMSprop(model.parameters(), lr=args.lr, polyak=args.polyak)
        scheduler = None
    elif args.model=='pixelcnnpp':
        import pixelcnnpp
        model = pixelcnnpp.PixelCNNpp(args.image_dims, args.n_channels, args.n_res_layers, args.n_logistic_mix,
                                      args.n_cond_classes).to(args.device)
        loss_fn = pixelcnnpp.loss_fn
        generate_fn = pixelcnnpp.generate_fn
        optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.95, 0.9995), polyak=args.polyak)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)
    elif args.model=='pixelsnail':
        import pixelsnail, pixelcnnpp
        model = pixelsnail.PixelSNAIL(args.image_dims, args.n_channels, args.n_res_layers, args.attn_n_layers, args.attn_nh, 
                args.attn_dq, args.attn_dv, args.attn_drop_rate, args.n_logistic_mix, args.n_cond_classes).to(args.device)
        loss_fn = pixelcnnpp.loss_fn
        generate_fn = pixelcnnpp.generate_fn
        optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.95, 0.9995), polyak=args.polyak, eps=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

#    print(model)
    print('Model parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))

    if args.restore_file:
        model_checkpoint = torch.load(args.restore_file, map_location=args.device)
        model.load_state_dict(model_checkpoint['state_dict'])
        optimizer.load_state_dict(torch.load(os.path.dirname(args.restore_file) + '/optim_checkpoint.pt', map_location=args.device))
        if scheduler:
            scheduler.load_state_dict(torch.load(os.path.dirname(args.restore_file) + '/sched_checkpoint.pt', map_location=args.device))
        args.start_epoch = model_checkpoint['epoch'] + 1
        args.step = model_checkpoint['global_step']

    if args.train:
        train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, loss_fn, generate_fn, writer, args)

    if args.evaluate:
        if args.step > 0: optimizer.swap_ema()
        eval_loss = evaluate(model, test_dataloader, loss_fn, args)
        print('Evaluate bits per dim: {:.3f}'.format(eval_loss.item() / (np.log(2) * np.prod(args.image_dims))))
        if args.step > 0: optimizer.swap_ema()

    if args.generate:
        if args.step > 0: optimizer.swap_ema()
        samples = generate(model, generate_fn, args)
        writer.add_image('samples', samples, args.step)
        save_image(samples, os.path.join(args.output_dir, 'generation_sample_step_{}.png'.format(args.step)))
        if args.step > 0: optimizer.swap_ema()

