from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses

import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch

np.random.seed(42)

from torchvision import transforms
from torch import optim
import re
import time
import torch.nn as nn
from natsort import natsorted
import argparse
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    output_folder = "outputs/transmorph/"
    os.makedirs(output_folder, exist_ok=True)

    parser = argparse.ArgumentParser(description='Train a model with optional continued training.')
    parser.add_argument('--cont_training', action='store_true', help='Enable continued training from a checkpoint.')
    args = parser.parse_args()

    batch_size = 1
    train_dir = 'Dataset/MGRegBench/train'
    models_dir = output_folder + 'models/'
    logs_dir = output_folder + 'logs/'
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    sys.stdout = Logger(logs_dir)
    lr = 0.0001 # learning rate
    epoch_start = 0
    max_epoch = 1501 #max traning epoch
    cont_training = False #if continue training

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if args.cont_training:
        model_files = natsorted(os.listdir(models_dir))

        second_last_model = model_files[-2]
        print(f'The chosen model: {second_last_model}')

        epoch_pattern = re.compile(r'epoch_(\d+)\.pth\.tar')
        match = epoch_pattern.search(second_last_model)

        epoch_start = int(match.group(1))
        print(f'The epoch number: {epoch_start}')

        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)

        best_model = torch.load(os.path.join(models_dir, second_last_model), weights_only=False)['state_dict']
        model.load_state_dict(best_model)
        print(f'The model {second_last_model} successfully loaded!')
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
    train_set = datasets.MammoDataset(train_dir, transforms=train_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    loss_sim = torch.nn.MSELoss(reduction='mean')
    loss_smooth = losses.Grad2d(penalty='l2')

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = nn.MSELoss()
    criterions = [criterion]
    criterions += [losses.Grad2d(penalty='l2')]
    writer = SummaryWriter(log_dir=logs_dir)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t = time.time()

    for epoch in range(epoch_start, max_epoch):
        steps = 0
        print('Training Starts')
        '''
        Training
        '''
        loss_all = 0
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = data.cuda()  # [B, 2, H, W]
            x = data[:, 0:1, :, :]  # fixed: [B, 1, H, W]
            y = data[:, 1:2, :, :]  # moving: [B, 1, H, W]
            input = torch.cat([y, x], 1) 
            moved, field = model(input)

            loss = 0
            loss1 = loss_sim(x, moved)
            loss2 = loss_smooth(field)

            loss = loss1 + loss2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all += loss
            steps += 1
            print('Iter {} of {} loss {:.14f}'.format(idx, len(train_loader), loss))

        writer.add_scalar('Loss/train', loss_all/steps, epoch)
        print('Epoch {} loss {:.14f}'.format(epoch, loss_all/steps))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, save_dir=models_dir, filename=f'epoch_{epoch}.pth.tar')

    if torch.cuda.is_available():
        peak_memory_allocated = torch.cuda.max_memory_allocated("cuda") 
        peak_memory_allocated = peak_memory_allocated / 1024 / 1024 / 1024 #Gb
    else:
        peak_memory_allocated = 0

    print(f"time: {time.time() - t}\n memory:{peak_memory_allocated}")

    writer.close()

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 4
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()