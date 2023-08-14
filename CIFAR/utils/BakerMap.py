import numpy as np
import math
import random
import torch
from . import prepare
__all__ = ['EncryptModel','DecryptModel']

def factor(nb,na):
    factors = []
    for_times = int(math.sqrt(nb))
    for i in range(for_times + 1)[1:]:
        if nb % i == 0:
            if na * i % nb == 0:
                factors.append(i)
            t = int(nb / i)
            if t != i and na * t % nb == 0:
                factors.append(t)
    factors.sort()
    return factors
 

def baker_makenlist(before,num=10):
    na = before.shape[0]
    nb = before.shape[1]
    if nb == 3:
        na, nb = nb, na
        factors = factor(nb,nb)
    else:
        # if na>nb :
        #     na,nb = nb,na
        factors = factor(nb,na)
    # if len(factors)>1 :
    #     factors = factors[1:]
    # nb>na 限制最小的ni
    if nb > na :
        nmin = int(nb/na)
        for i in range(len(factors)):
            if factors[i] >= nmin:
                factors = factors[i:]
                break
   
    if len(factors)>1 :
        factors=factors[:-1]
    factorstemp = factors
    nlist = []
    nbtemp = nb
    while nbtemp != 0 :
        newn = random.choice(factorstemp)
        while newn > nbtemp :
            factorstemp.remove(newn)
            newn = random.choice(factorstemp)
        nlist.append(newn)
        nbtemp = nbtemp - newn

    return nlist


def baker_encryption(before,nlist):

    if before.shape[1]==3:
        nb = before.shape[0]
        na = before.shape[1]

        new = torch.zeros_like(before)
        new = new.cpu()
        before = before.cpu()

        Ni = 0
        r0 = 0; s0 = 0
        r1 = 0; s1 = 0
        for ni in nlist:
            qi = int(nb/ni)
            for k in range(int(na/qi)):
                stemp = s0
                for i in range(ni):    
                    rtemp = r0            

                    for j in range(qi):
                        new[s1][r1] = before[s0][r0]
                        r0 = r0 + 1
                        s1 = s1 + 1
                        if s1==nb:
                            s1 = 0
                            r1 = r1 + 1
                    s0 = s0 + 1
                    r0 = rtemp

                s0 = stemp
                r0 = r0 + qi

            stemp = s0
            for i in range(ni):    
                rtemp = r0            
                for j in range(na-rtemp):
                    new[s1][r1] = before[s0][r0]
                    r0 = r0 + 1
                    s1 = s1 + 1
                    if s1==nb:
                        s1 = 0
                        r1 = r1 + 1   
                s0 = s0 + 1
                r0 = rtemp
            s0 = stemp
            
            Ni = Ni + ni
            s0 = s0 + ni
            r0 = 0


    else:
        na = before.shape[0]
        nb = before.shape[1]

        new = torch.zeros_like(before)
        new = new.cpu()
        before = before.cpu()
        before2 = torch.transpose(before,0,1)

        Ni = 0
        r0 = 0; s0 = 0
        for ni in nlist:
            qi = int(nb/ni)
            for k in range(int(na/qi)):
                rtemp = r0
                for i in range(ni):    
                    r1 = int( qi * ( r0 - Ni ) + s0 % qi)
                    s1 = int( ( s0 - s0 % qi )/qi + na / nb * Ni ) 
                
                    #print(r0,',',s0,'->',r1,',',s1)
                    new[s1][r1:r1+qi] = before2[r0][s0:s0+qi]
                    r0 = r0 + 1
                r0 = rtemp
                s0 = s0 + qi
            Ni = Ni + ni
            r0 = r0 + ni
            s0 = 0


    return new

def backer_decryption(before,nlist):

    if before.shape[1]==3:
        new = torch.zeros_like(before)

        nb = before.shape[0]
        na = before.shape[1]

        new = new.cpu()
        before = before.cpu()

        Ni = 0
        r0 = 0; s0 = 0
        r1 = 0; s1 = 0
        for ni in nlist:
            qi = int(nb/ni)
            for k in range(int(na/qi)):
                stemp = s0
                for i in range(ni):    
                    rtemp = r0            

                    for j in range(qi):
                        new[s0][r0] = before[s1][r1]
                        r0 = r0 + 1
                        s1 = s1 + 1
                        if s1==nb:
                            s1 = 0
                            r1 = r1 + 1
                    s0 = s0 + 1
                    r0 = rtemp

                s0 = stemp
                r0 = r0 + qi

            stemp = s0
            for i in range(ni):    
                rtemp = r0            
                for j in range(na-rtemp):
                    new[s0][r0] = before[s1][r1]
                    r0 = r0 + 1
                    s1 = s1 + 1
                    if s1==nb:
                        s1 = 0
                        r1 = r1 + 1   
                s0 = s0 + 1
                r0 = rtemp
            s0 = stemp
            
            Ni = Ni + ni
            s0 = s0 + ni
            r0 = 0


    else:
        na = before.shape[0]
        nb = before.shape[1]

        new = torch.zeros_like(before)
        new = torch.transpose(new,0,1)

        new = new.cpu()
        before = before.cpu()
        
        Ni = 0
        r0 = 0; s0 = 0
        for ni in nlist:
            qi = int(nb/ni)
            for k in range(int(na/qi)):
                rtemp = r0

                for i in range(ni):    
                    r1 = int( qi * ( r0 - Ni ) + s0 % qi)
                    s1 = int( ( s0 - s0 % qi )/qi + na / nb * Ni ) 
                
                    new[r0][s0:s0+qi] = before[s1][r1:r1+qi]
                    r0 = r0 + 1

                r0 = rtemp
                s0 = s0 + qi
            Ni = Ni + ni
            r0 = r0 + ni
            s0 = 0
        
        new = torch.transpose(new,0,1)

    return new

def EncryptModel(net, enc_layers_num = -1):
    conv_names = prepare.search_conv(net)
    if enc_layers_num == -1:
        enc_layers_names = conv_names
    else:
        assert enc_layers_num <= len(conv_names), 'The number of encrypted layers is larger than the number of conv layers! '
        enc_layers_names = random.sample(conv_names, k = enc_layers_num)

    dict = net.state_dict()
    key = []
    for name in enc_layers_names:
        nlist = baker_makenlist(dict[name])
        key.append((name, nlist))
        dict[name] = baker_encryption(dict[name], nlist)
    net.load_state_dict(dict)
    return net, key
        
def DecryptModel(net, key):
    dict = net.state_dict()
    for name, nlist in key:
        # print(name,nlist)
        dict[name] = backer_decryption(dict[name], nlist)
    net.load_state_dict(dict)
    return net 

