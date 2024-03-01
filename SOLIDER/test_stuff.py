import pickle
import numpy as np
import os
import scipy.io

with open('./data/PETA/dataset_all.pkl', 'rb') as f:
    data = pickle.load(f)



# data = scipy.io.loadmat('./data/PETA/PETA.mat')


print(data['weight_trainval'][1].shape)