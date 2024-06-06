import math

import numpy as np
import matplotlib.pyplot as plt

def generate_data(line,row,coeff):
    features=np.random.normal(size=(line,1))
    np.random.shuffle(features)
    poly_features=np.power(features,np.arange(row).reshape(1,-1))
    for i in range(row):
        poly_features[:,i]/=math.gamma(i+1)
    ncoeff=np.zeros(row)
    ncoeff[0:len(coeff)]=coeff
    labels=np.dot(poly_features,ncoeff)
    labels+=np.random.normal(scale=0.1,size=labels.shape)
    return poly_features,labels


if __name__== '__main__':
    features,labels=generate_data(200,20,[5, 1.2])
    print(f"features.shape:{features.shape}\n")
    print(f"labels.shape:{labels.shape}\n")

    print(f"features[0:2,:]:{features[0:2, :]}\n")
    print(f"labels[:2]:{labels[0:2]}\n")