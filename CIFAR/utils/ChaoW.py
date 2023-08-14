import numpy as np

__all__ = ['ACM']

fab=[1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040, 1346269, 2178309, 3524578, 5702887, 9227465, 14930352, 24157817, 39088169, 63245986, 102334155, 165580141, 267914296]

def getA(r):
    A = np.array([[fab[2*r-1],fab[2*r]],[fab[2*r],fab[2*r+1]]])
    return A

def getB(r):
    B = np.array([[fab[2*r+1],-1*fab[2*r]],[-1*fab[2*r],fab[2*r-1]]])
    return B

def map(x,y,A):
    ori = np.array([[x],[y]])
    fin = np.dot(A,ori)
    return fin

def acm(bef,r,s):
    #new = torch.zeros_like(bef.data)
    new = bef.clone()
    if r>=0:
        A = getA(r)
    else:
        A = getB(-1*r)
    if s <= min(bef.shape[0],bef.shape[1]):
         for i in range(s):
             for j in range(s):
                 fin = map(i,j,A)
                 new[fin[0][0]%s][fin[1][0]%s] = bef[i][j]
    else:
        print("s is out of limit.")
    return new         

def map1(x,y,A):
    ori = np.array([x,y])
    fin = np.dot(A,ori)
    return fin

def acm1(bef,r,s):
    #new = torch.zeros_like(bef.data)
    new = bef.clone()
    if r>=0:
        A = getA(r)
    else:
        A = getB(-1*r)
    if s <= min(bef.shape[0],bef.shape[1]):
        fin = []
        for i in range(s):
            x = [i]*s
            y = []
            for j in range(s):
                y.append(j)
            fin.append(map1(x,y,A))  

        for i in range(s):
            for j in range(s):
                new[fin[i][0][j]%s][fin[i][1][j]%s] = bef[i][j]
    else:
        print("s is out of limit.")
    return new          


def ACM(bef,r,percent=1):
    new = bef.clone()
    bef = bef.cpu()
    new = new.cpu()
    if r>=0:
        A = getA(r)
    else:
        A = getB(-1*r)
    
    s = min(bef.shape[0],bef.shape[1])
    s = int(s*percent)
    fin_x = []
    fin_y = []
    for i in range(s):
        x = [i]*s
        y = range(s)
        ret = map1(x,y,A)
        fin_x.extend(ret[0])
        fin_y.extend(ret[1])
    
    cnt=0
    for i in range(s):
        for j in range(s):
            new[fin_x[cnt]%s][fin_y[cnt]%s] = bef[i][j]
            cnt = cnt+1
            #print('(',fin[0][0]%s,',',fin[1][0]%s,')','(',i,',',j,')')
    
    return new   

 
