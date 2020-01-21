import argparse
import configparser
import numpy as np
import os
import time
import shutil
import json
import glob

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
import warnings

from lib.dataset import MegaDepthDataset
from lib.exceptions import NoGradientError
from lib.loss import D2Loss_hpatches
from lib.model import D2Net
from lib.system import gct
from lib.dataset_hpatches import (
    HPatchesDataset,
    Normalize,
    Rescale,
    LargerRescale,
    RandomCrop,
    ToTensor
)
from lib.tensor import parse_batch, parse_output
from lib.utils import count_parameters_in_MB, AvgrageMeter, create_exp_dir


# Define epoch function
def process_epoch(
        epoch_idx,
        model, loss_function, optimizer, dataloader, device,
        log_file, args, train=True
):
    epoch_loss = AvgrageMeter()

    torch.set_grad_enabled(train)
    dataset_len = len(dataloader)
    step_base = step_base = epoch_idx * dataset_len

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, batch in progress_bar:
        if train:
            optimizer.zero_grad()

        # if batch_idx == 73:
        #     print(' ')
        batch = parse_batch(batch, device)
        output = parse_output(model(batch))
        try:
            loss = loss_function(batch, output, plot=args.plot)
        except NoGradientError:
            continue

        epoch_loss.update(loss.item())

        progress_bar.set_postfix(loss=('%.4f' % epoch_loss.avg))

        if train:
            loss.backward()
            optimizer.step()

            if args.log:
                vis_writer.add_scalar('Loss/train_batch', loss.item(), step_base + batch_idx)

        # if batch_idx % args.log_interval == 0:
        #     print(gct(), f'[{batch_idx:03d} / {dataset_len:03d}]', f'Loss: {loss.item():.03f}',
        #           f'Avg loss: {epoch_loss.avg:.03f}')

    return epoch_loss.avg


# CUDA
if not torch.cuda.is_available():
    print(gct(), 'no gpu device available')
    exit(1)

# Argument parsing
parser = argparse.ArgumentParser(description='Training script')

parser.add_argument('--cpu', action='store_true', help='Only use cpu.')
parser.add_argument('--lr', type=float, default=1e-3,help='initial learning rate')

parser.add_argument('--seed', type=int, default=-1, help='random seed')
parser.add_argument('--log_interval', type=int, default=10, help='loss logging interval')

parser.add_argument('--log_file', type=str, default='log.txt', help='loss logging file')

parser.add_argument('--plot', dest='plot', action='store_true', help='plot training pairs')
parser.set_defaults(plot=False)

parser.add_argument(
    '--checkpoint_directory', type=str, default='checkpoints',
    help='directory for training checkpoints'
)
parser.add_argument(
    '--checkpoint_prefix', type=str, default='d2',
    help='prefix for training checkpoints'
)
parser.add_argument('--dataset', default='view', type=str, help='Optional dataset: [view, view_e, view_m, view_h]')
parser.add_argument('--log', action='store_true', help='Logs save path(defualt: none)')
parser.add_argument('--resume', default=None, type=str, help='Latest checkpoint (default: none)')

args = parser.parse_args()

print(gct(), 'Args = %s', args)

cfg = configparser.ConfigParser()
t = cfg.read('settings.conf')

if args.cpu:
    device = torch.device('cpu')
    use_cuda = False
else:
    device = torch.device('cuda')
    use_cuda = True

if args.log:
    log_folder = 'logs/{}'.format(time.strftime('model_hpatches_%Y%m%d-%H%M%S'))
    log_path = os.path.join(log_folder, 'log')
    model_path = os.path.join(log_folder, 'model')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    create_exp_dir(log_folder, scripts_to_save=glob.glob('*.py'))
    create_exp_dir(log_folder, scripts_to_save=glob.glob('lib/*.py'))

    vis_writer = SummaryWriter(log_path)

else:
    vis_writer = None

criterion = D2Loss_hpatches(scaling_steps=3, device=device).to(device)

model = D2Net(model_file=args.resume, use_cuda=use_cuda)

# Optimizer
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
)

print(gct(), 'Param size = %fMB', count_parameters_in_MB(model))

# Resume training
if args.resume:
    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_loss = checkpoint['loss']
        seed = checkpoint['seed']
        print(f'{gct()} Loaded checkpoint {args.resume} (epoch {start_epoch-1})')
    else:
        print(f'{gct()} No checkpoint found at {args.resume}')
        raise IOError
else:
    start_epoch = 0
    best_loss = 10000000

    if args.seed < 0:
        seed = int(time.time())
        print('\tRandom seed:', seed)
    else:
        seed = args.seed
# Seed

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Create the folders for plotting if need be
# if args.plot:
#     plot_path = 'train_vis'
#     if os.path.isdir(plot_path):
#         print('[Warning] Plotting directory already exists.')
#     else:
#         os.mkdir(plot_path)


if args.dataset[:4] == 'view':
    csv_file = cfg['hpatches']['view_csv']
    root_dir = cfg['hpatches'][f'{args.dataset}_root']

dataset = HPatchesDataset(
    csv_file=csv_file,
    root_dir=root_dir,
    transform=transforms.Compose([
        LargerRescale((960, 1280)),
        RandomCrop((720, 960)),
        Rescale((720, 960)),  # 360x480
        ToTensor(),
        Normalize(
            mean=json.loads(cfg['hpatches']['view_mean']),
            std=json.loads(cfg['hpatches']['view_std']))
    ]),
)

print(gct(), 'Load training dataset.')
print(gct(), f'Root dir: {root_dir}. #Image pair: {len(dataset)}')

dataset_len = len(dataset)
split_idx = list(range(dataset_len))
part_pos = [
    int(dataset_len * float(cfg['hpatches']['train_part_pos'])),
    int(dataset_len * float(cfg['hpatches']['eval_part_pos']))]
training_dataloader = DataLoader(
    dataset=dataset,
    batch_size=int(cfg['params']['train_bs']),
    sampler=sampler.SubsetRandomSampler(split_idx[:part_pos[0]]),
    pin_memory=True,
    num_workers=4  # Unknown broken pipe error
)

validation_dataloader = DataLoader(
    dataset=dataset,
    batch_size=int(cfg['params']['train_bs']),
    sampler=sampler.SubsetRandomSampler(split_idx[part_pos[0]:part_pos[1]]),
    pin_memory=True,
    num_workers=4  # Unknown broken pipe error
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.5, start_epoch - 1)

# Dataset


# validation_dataset = MegaDepthDataset(
#     scene_list_path='megadepth_utils/valid_scenes.txt',
#     scene_info_path=args.scene_info_path,
#     base_path=args.dataset_path,
#     train=False,
#     preprocessing=args.preprocessing,
#     pairs_per_scene=25
# )
# validation_dataloader = DataLoader(
#     validation_dataset,
#     batch_size=args.batch_size,
#     num_workers=args.num_workers
# )

# training_dataset = MegaDepthDataset(
#     scene_list_path='megadepth_utils/train_scenes.txt',
#     scene_info_path=args.scene_info_path,
#     base_path=args.dataset_path,
#     preprocessing=args.preprocessing
# )
# training_dataloader = DataLoader(
#     training_dataset,
#     batch_size=args.batch_size,
#     num_workers=args.num_workers
# )


# Create the checkpoint directory
# if os.path.isdir(args.checkpoint_directory):
#     print('[Warning] Checkpoint directory already exists.')
# else:
#     os.mkdir(args.checkpoint_directory)


# Open the log file for writing
# if os.path.exists(args.log_file):
#     print('[Warning] Log file already exists.')
# log_file = open(args.log_file, 'a+')

# Initialize the history

train_loss_history = AvgrageMeter()
validation_loss_history = AvgrageMeter()

best_loss = process_epoch(
    0,
    model, criterion, optimizer, validation_dataloader, device,
    vis_writer, args,
    train=False
)

# Start the training
for epoch_idx in range(start_epoch, int(cfg['params']['total_epochs'])):
    print('Epoch:', epoch_idx)
    # Process epoch

    train_loss = process_epoch(
        epoch_idx,
        model, criterion, optimizer, training_dataloader, device,
        vis_writer, args
    )
    if args.log:
        vis_writer.add_scalar('Loss/train_epoch', train_loss, epoch_idx)

    valid_loss = process_epoch(
        epoch_idx,
        model, criterion, optimizer, validation_dataloader, device,
        vis_writer, args,
        train=False
    )
    if args.log:
        vis_writer.add_scalar('Loss/valid_epoch', valid_loss, epoch_idx)

    if args.log and (epoch_idx == 0 or best_loss > valid_loss):
        best_loss = valid_loss
        checkpoint = {
            'epoch': epoch_idx,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': best_loss,
            'seed': seed
        }

        checkpoint_path = f'{model_path}/NAS_model_epoch_{epoch_idx:03d}_loss_{best_loss:.03f}.pth.tar'
        torch.save(checkpoint, checkpoint_path)

        print('Model saved.')
    else:
        print('')

    scheduler.step()
