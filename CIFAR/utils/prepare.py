import numpy as np
import math
import random
import torch

__all__ = ['search_conv']

def search_conv(net):
    conv_names = []
    for key,value in net.state_dict().items():
        if len(value.shape)==4 and value.shape[2]!=1:
            conv_names.append(key)
    return conv_names