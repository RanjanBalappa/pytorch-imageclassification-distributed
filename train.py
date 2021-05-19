import argparse
import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import models



from dp.loader import ImageDataset
from nn.classifier import Classifier
from utils import AverageMeter, Accuracy
from ddp_utils import all_gather




parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()




def train_epoch(epoch, model, loss_fn, optimizer, train_data_loader, name):
    model.train()
    losses = AverageMeter()
    if args.local_rank == 0:
        iterator = tqdm(train_data_loader)
    else:
        iterator = train_data_loader

    for index, data in enumerate(iterator):
        images = data['image'].cuda(non_blocking=True)
        labels = data['label'].cuda(non_blocking=True)

        if 'inception' in name:
            outputs, aux_outputs  = model(images)
            loss1 = loss_fn(outputs, labels)
            loss2 = loss_fn(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2

        else:
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        #calculate accuracy
        with torch.no_grad():
            #combine the loss from all process and take average
            reduced_loss = loss.clone()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            reduced_loss /= dist.get_world_size()
            losses.update(reduced_loss, images.size(0))


        if args.local_rank == 0:
            iterator.set_description(f'Epoch: {epoch}; Loss {losses.val:.4f}|({losses.avg:.4f})')


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        



def val_epoch(epoch, model, val_data_loader):
    model.eval()
    avgaccuracy = []

    with torch.no_grad():
        for index, data in enumerate(val_data_loader):
            images = data['image'].cuda(non_blocking=True)
            labels = data['label'].cuda(non_blocking=True)

            outputs = model(images)
            for batch in range(images.size(0)):
                accuracy = Accuracy(outputs[batch].unsqueeze(0), labels[batch]).cpu().numpy()
                avgaccuracy.append(accuracy * 100)

        comb_accuracy = np.concatenate(all_gather(np.asarray(avgaccuracy))).mean()

        if args.local_rank == 0:
            print(f'Validation Accuracy {comb_accuracy}')

    return comb_accuracy

args.world_size = 1

#initialize distributed process
pg = dist.init_process_group('nccl', rank=args.local_rank)
args.world_size = dist.get_world_size()
args.distributed = True if args.world_size else False

torch.cuda.set_device(args.local_rank)

#prepare data
DATA_DIR = ''
RESIZE_SIZE = 299
BATCH_SIZE = 4
train_dataset = ImageDataset(DATA_DIR, 'train', RESIZE_SIZE)
train_sampler = DistributedSampler(train_dataset)
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True, sampler=train_sampler)

val_dataset = ImageDataset(DATA_DIR, 'valid', RESIZE_SIZE)
val_sampler = DistributedSampler(val_dataset)
val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=6, pin_memory=True, sampler=val_sampler)


#initialize model
name = 'inceptionv3'
model = Classifier(name=name, num_classes=train_dataset.num_classes)
model = nn.SyncBatchNorm.convert_sync_batchnorm(model, pg)
model = model.cuda()
params = model.parameters()
optimizer = Adam(params, lr=0.5e-5)
model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)


#load model checkpoint
best_model_snapshot = 'best_model'
latest_model_snapshot = 'latest_model'
best_score = 0
start_epoch = 0
checkpoint_path = os.path.join('dtmodel', 'cp', name, best_model_snapshot)
if os.path.exists(checkpoint_path):
    if args.local_rank == 0:
        print(f'Loading Checkpoint from {best_model_snapshot}')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    loaded_dict = checkpoint['state_dict']
    state_dict = model.state_dict() 
    for key in state_dict:
        if key in loaded_dict:
            state_dict[key] = loaded_dict[key]

    model.load_state_dict(state_dict)
    start_epoch = checkpoint['epoch']
    best_score = checkpoint['best_score']

    if args.local_rank == 0:
        print(f'Loaded Checkpoint: {best_model_snapshot}, with epoch {start_epoch} and best score {best_score}')


scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.5)
weight = torch.tensor([3, 3, 10, 1, 4, 4, 5], requires_grad=False, dtype=torch.float32).cuda()
loss_fn = nn.CrossEntropyLoss(weight=weight)


for epoch in range(100):

    #train loop
    train_sampler.set_epoch(epoch)
    train_epoch(epoch, model, loss_fn, optimizer, train_data_loader, name)
    scheduler.step()

    torch.cuda.empty_cache()


    #validation loop
    val_accuracy = val_epoch(epoch, model, val_data_loader)
    if val_accuracy > best_score and args.local_rank == 0:
        print(f'Model improved to {val_accuracy} so storing checkpoint')
        best_score = val_accuracy
        torch.save({
            'epoch': epoch,
            'best_score': val_accuracy,
            'state_dict': model.state_dict()
        }, os.path.join('dtmodel', 'cp', name, best_model_snapshot))


    if args.local_rank == 0 and epoch % 5 == 0:
        torch.save({
            'epoch': epoch,
            'best_score': best_score,
            'state_dict': model.state_dict()
        }, os.path.join('dtmodel', 'cp', name, latest_model_snapshot))