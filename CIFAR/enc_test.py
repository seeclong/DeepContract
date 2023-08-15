#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models import *
from utils import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10')


parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')

parser.add_argument('--model', default='resnet', type=str,
                    help='model structure (resnet or vgg)')
parser.add_argument('--model-path', default='./checkpoint/resnet18_93.36.pth', help='original model path')


parser.add_argument('--enc-layers', default=-1, type=int, metavar='N',
                    help='number of encrypted layers (default: number of all conv layers)')

parser.add_argument('--seed', default=666, type=int, help='random seed')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

assert args.model == 'resnet' or args.model == 'vgg', 'model structure can only be resnet or vgg'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # Load model
    print('==> Loading the model {}'.format(args.model_path))
    if args.model == 'resnet':
        original_model = ResNet18()
        print("ResNet is adopted")
    else:
        original_model = VGG('VGG16')
        print("VGG is adopted")

    assert os.path.isfile(
        args.model_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.model_path)
    original_model = original_model.to(device)
    original_model = torch.nn.DataParallel(original_model)
    original_model.load_state_dict(checkpoint)
    original_model.eval()


    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel()
                                       for p in original_model.parameters()) / 1000000.0))
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print('==> Loading the dataset')

    dataloader = datasets.CIFAR10

    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    testset = dataloader(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


    original_accuracy = easy_test(testloader, original_model)
    print('The inference accuracy of the original model is  %.3f' % original_accuracy)
   
    encrypted_model, authorization_key = EncryptModel(original_model, enc_layers_num=-1)

    encrypted_accuracy = easy_test(testloader, encrypted_model)
    print('The inference accuracy of the encrypted model is  %.3f' % encrypted_accuracy)

    decrypted_model = DecryptModel(encrypted_model, authorization_key)
    decrypted_accuracy = easy_test(testloader, decrypted_model)
    print('The inference accuracy of the decrypted model is  %.3f' % decrypted_accuracy)

if __name__ == '__main__':
    main()


