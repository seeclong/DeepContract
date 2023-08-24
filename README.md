# DeepContract

This is the official implementation of our paper **DeepContract: Controllable Authorization of Deep Learning Models**, accepted by the Annual Computer Security Applications Conference 2023 (ACSAC'23). This research project is developed based on Python 3 and Pytorch.


## Requirements

To install requirements:

```python
pip install -r requirements.txt
```

## Running Examples

### For CV tasks

Please follow the instructions in [main.sh](CIFAR/main.sh). For convenience, we  provide the checkpoints of well-trained models so that you can run the encryption and decryption of these models directly.

For example, to excute the process of encryption and decryption for a ResNet/VGG model, you can run the following command:

```python
# ResNet encryption and decryption testing
python3 enc_test.py --workers 2 --gpu-id 0 --test-batch 128 --model 'resnet'  --model-path './checkpoint/resnet18_93.36.pth'  --enc-layers -1

# VGG encryption and decryption testing
python3 enc_test.py --workers 2 --gpu-id 0 --test-batch 128 --model 'vgg'  --model-path './checkpoint/vgg16_92.24.pth'  --enc-layers -1
```

## License 

This project is licensed under the terms of the Apache License 2.0. See the LICENSE file for the full text.



