#!/bin/bash


# ResNet encryption and decryption testing
python3 enc_test.py --workers 2 --gpu-id 0 --test-batch 128 --model 'resnet'  --model-path './checkpoint/resnet18_93.36.pth'  --enc-layers -1

# VGG encryption and decryption testing
python3 enc_test.py --workers 2 --gpu-id 0 --test-batch 128 --model 'vgg'  --model-path './checkpoint/vgg16_92.24.pth'  --enc-layers -1
