import argparse
import glob
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from pathlib import Path
from core.raft_stereo import RAFTStereo
from dataset import stereo_datasets
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import time


def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


DEVICE = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help="number of epoch.")
parser.add_argument('--train_iters', type=int, default=7,
                    help="number of updates to the disparity field in each forward pass.")
parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720],
                    help="size of the random image crops used during training.")
parser.add_argument('--train_datasets', nargs='+', default=['tartan_air'], help="training datasets.")
parser.add_argument('--valid_datasets', nargs='+', default=['sceneflow', ], help="validation datasets.")
parser.add_argument('--val_rate', type=int, default=1, help="validation epoch")
parser.add_argument('--batch_size', type=int, default=21, help="batch size used during training.")
# Architecture choices
parser.add_argument('--valid_iters', type=int, default=7, help='number of flow-field updates during forward pass')
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                    help="hidden state and context dimensions")
parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
                    help="correlation volume implementation")
parser.add_argument('--shared_backbone', action='store_true',
                    help="use a single backbone for the context and feature encoders")
parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
parser.add_argument('--n_downsample', type=int, default=3, help="resolution of the disparity field (1/2^K)")
parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
parser.add_argument('--n_gru_layers', type=int, default=2, help="number of hidden GRU levels")

# Data augmentation
parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
parser.add_argument('--do_flip', default=False, choices=['h', 'v'],
                    help='flip the images horizontally or vertically')
parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')

args = parser.parse_args()
data_root = '/media/anas/ssd4to/denzel_data'

args.shared_backbone = True
args.slow_fast_gru = True
args.mixed_precision = True

model = torch.nn.DataParallel(RAFTStereo(args))
model.module.load_state_dict(torch.load('last.pt'))
model.cuda()
model.train()
model.module.freeze_bn()

scaler = GradScaler(enabled=args.mixed_precision)

train_loader = stereo_datasets.fetch_dataloader(args=args, root=data_root)
val_loaders = stereo_datasets.fetch_validation_dataloader(args=args, root=data_root)
optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=.00001, eps=1e-8)
# scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, 200000 + 100,
#                                               pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

best_metric = {'avg': np.inf}

for dataset in args.valid_datasets:
    best_metric[dataset] = np.inf

for epoch in range(args.epochs):
    print(f'Epoch {epoch}')
    print(('\n' + '%10s') % 'loss')
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    mean_loss = []
    mean_metric = []
    model.train()
    model.module.freeze_bn()

    for i_batch, (_, *data_blob) in pbar:
        optimizer.zero_grad()
        image1, image2, flow, valid = [x.cuda() for x in data_blob]
        assert model.training
        flow_predictions = model(image1, image2, iters=args.train_iters)
        assert model.training
        loss, metrics = sequence_loss(flow_predictions, flow, valid)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        mean_loss.append(loss.detach().cpu().numpy())
        mean_metric.append(metrics['epe'])
        pbar.set_description('%10.4g' % loss)
    print(f'train loss: {np.mean(mean_loss)}')
    print(f'train metric: {np.mean(mean_metric)}')
    torch.save(model.module.state_dict(), 'last.pt')

    if epoch % args.val_rate == 0:
        model.eval()

        with torch.no_grad():
            for i, val_loader in enumerate(val_loaders):
                print('Validation: ', args.valid_datasets[i])
                temp_mean_val_metric = []

                for (_, *data_blob) in tqdm(val_loader, total=len(val_loader)):
                    image1, image2, flow, valid = [x.cuda() for x in data_blob]
                    flow_predictions = model(image1, image2, iters=args.train_iters)
                    loss, metrics = sequence_loss(flow_predictions, flow, valid)
                    temp_mean_val_metric.append(metrics['epe'])

                temp_val_metric = np.mean(temp_mean_val_metric)
                if temp_val_metric < best_metric[args.valid_datasets[i]]:
                    best_metric[args.valid_datasets[i]] = temp_val_metric
                    torch.save(model.module.state_dict(), f'best_{args.valid_datasets[i]}.pt')
                print(f'Epoch {epoch} {args.valid_datasets[i]} val_metric: {temp_val_metric}')
