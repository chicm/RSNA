from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

#from loss import FocalLoss
from retinanet import RetinaNet

from torch.autograd import Variable
import time
import logging as log
from imgdataset import get_train_loader, get_small_train_loader
from dcom import get_train_ids, get_val_ids
import settings

batch_size = 4

log.basicConfig(
        filename = 'trainlog.txt', 
        format   = '%(asctime)s : %(message)s',
        datefmt  = '%Y-%m-%d %H:%M:%S', 
        level = log.DEBUG)

def run_train(args):
    assert torch.cuda.is_available(), 'Error: CUDA not found!'
    start_epoch = 0  # start from epoch 0 or last epoch

    trainloader = get_train_loader(get_train_ids(), batch_size=batch_size)
    print(trainloader.num)
    

    # Model
    net = RetinaNet()
    net.train_class_only = True
    net.load_state_dict(torch.load('./models/net.pth'))
    #net.load_state_dict(torch.load('./ckps/best_0.pth'))
    #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    #optimizer = optim.Adam(net.parameters(), lr=args.lr)

    iter_save = 200
    iter_val = 500
    bgtime = time.time()
    # Training
    for epoch in range(start_epoch, start_epoch+100):
        print('\nEpoch: %d' % epoch)
        net.train()
        net.freeze_bn()
        for batch_idx, (inputs, img_label, _, _) in enumerate(trainloader):
            inputs, img_label = inputs.cuda(), img_label.cuda()

            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, img_label)
            loss.backward()
            optimizer.step()

            preds = output.max(1, keepdim=True)[1]
            corrects = preds.eq(img_label.view_as(preds)).sum().item()

            sample_num = (batch_idx+1)*batch_size
            print('Epoch: {}, num: {}/{} train_loss: {:.3f} | corrects: {:d} min: {:.1f}'.format(
                epoch, sample_num, trainloader.num, loss.item(), corrects, (time.time() - bgtime)/60), end='\r')

            if batch_idx % iter_save == 0:
                torch.save(net.state_dict(), './ckps/best_{}.pth'.format(batch_idx//iter_save % 5))
                #log.info('batch: {}, loss: {:.4f}'.format(batch_idx, avg_loss))

            if batch_idx % iter_val == 0:
                val_corrects, val_num = validate(net, criterion)
                print(f'val corrects: {val_corrects} / {val_num}, {val_corrects*100./val_num:.2f}%')
                net.train()
                net.freeze_bn()

# Test
def validate(net, criterion):
    print('\nTest')
    valloader = get_train_loader(get_val_ids()[:500], batch_size=batch_size)
    net.eval()
    corrects = 0
    for batch_idx, (inputs, img_label, _, _) in enumerate(valloader):
        inputs, img_label = inputs.cuda(), img_label.cuda()
        output = net(inputs)
        #loss = criterion(output, img_label)
        preds = output.max(1, keepdim=True)[1]
        corrects += preds.eq(img_label.view_as(preds)).sum().item()

    return corrects, valloader.num

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    
    run_train(args)
    #test(epoch)
