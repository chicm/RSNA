# ====================================================================================================== #
# The MIT License (MIT)
# Copyright (c) Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ====================================================================================================== #

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from dcom import get_balanced_train_ids, get_val_ids
from imgdataset import get_train_loader

# This example only tested on GPU, it could be extremly slow if you try to run it on cpu.
# Please adjust the batch size per your GPU Memory capacity.

H, W = (512, 512)
batch_size = 16

def create_resnet():
    # Pretrained resnet model https://arxiv.org/abs/1512.03385
    model = models.resnet50(pretrained = True)
    # Customize full connection layer for CIFAR10 dataset after model weights loaded
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    print(model.fc)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def save_model(args, model, filename):
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    torch.save(model.state_dict(), os.path.join(args.model_dir, filename))

def validate(model, device, val_loader, criterion):
    print('\nvalidating..')
    model.eval()
    val_loss = 0
    corrects = 0
    with torch.no_grad():
        for i, (inputs, label, _, _) in enumerate(val_loader):
            inputs, label = inputs.to(device), label.to(device)
            #print(f'label: {label}')
            output = model(inputs)
            #print(f'output: {output}')
            val_loss += criterion(output, label).item()
            preds = output.max(1, keepdim=True)[1]
            #print(f'preds: {preds}')
            corrects += preds.eq(label.view_as(preds)).sum().item()

            print(f'{i*batch_size}/{val_loader.num}', end='\r')
    print('')
    return val_loss, corrects, len(val_loader.dataset)

def train(args):
    #train_loader = get_data_loader(args, True)
    train_ids = get_balanced_train_ids()#[10:14]
    #print(train_ids)
    trainloader = get_train_loader(train_ids, batch_size=batch_size, shuffle=True)
    valloader = get_train_loader(get_val_ids(), batch_size=batch_size, shuffle=False)
    #val_loader = get_data_loader(args, False, shuffle = False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_resnet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_corrects = 0

    val_iter = 200

    for epoch in range(args.epochs):
        model.train()
        count = 0
        for batch_idx, (inputs, label, _, _) in enumerate(trainloader):
            #print(label)
            inputs, label = inputs.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            count += len(inputs)
            print('Epoch {}: ({}/{})\tLoss: {:.4f}'.format(epoch, count, trainloader.num, loss.item()), end='\r')

            if batch_idx % val_iter == 0:
                val_loss, corrects, num = validate(model, device, valloader, criterion)
                val_loss /= (num / args.batch_size)
                print('\nValidation loss: {:.4f}, Validation accuracy: {}/{} ({:.02f}%)\n'.format(
                    val_loss, corrects, num, 100. * corrects / num))

                if corrects > best_corrects:
                    save_model(args, model, 'best_model.pth')
                    best_corrects = corrects
                
                model.train()

def test():
    x = torch.randn(2,3,512,512).cuda()
    net = create_resnet().cuda()
    net(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate", required=False)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size", required=False)
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs', required=False)
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory to put training data', required=False)
    parser.add_argument('--model_dir', type=str, default='model', help='Directory to save models', required=False)

    args, unknown = parser.parse_known_args()

    train(args)
    #test()
