from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import scipy.io as sio


import numpy as np
import random,cPickle
import sys   
import h5py

from keras.layers import Input, Dense, TimeDistributedDense, Flatten, Lambda, merge, Dropout
from keras.models import Model
from keras import backend as K
from keras.regularizers import l1, l2, l1l2, activity_l1
from keras.optimizers import SGD
from keras.models import model_from_json
import h5py
startN = 0;
randS = 23;


def vladEncoder(feats):
	fDim = 512
	fK = 64
	fT = 64
	d_flat = feats[:, :fT*fK]
	x_flat = feats[:, fT*fK:fT*fK+fT*fDim]
	gd_flat = feats[:, fT*fK+fT*fDim:]
	D = K.reshape(d_flat, (-1, fT, fK))
	X = K.reshape(x_flat, (-1, fT, fDim))
	GD = K.reshape(gd_flat, (-1, fK, fDim))
	x = K.reshape(X, (-1, fDim))  # (samples * timesteps, input_dim)
	d = K.reshape(D, (-1, fK))  # (samples * timesteps, cluster_num)
	x_exp = K.tile(x, fK)
	d_exp = K.repeat_elements(d, fDim, axis=1)
	cc = K.repeat_elements(gd_flat, fT, axis=0)
	Y = d_exp*(x_exp - cc)  # (samples * timesteps, output_dim)
	Y1 = K.reshape(Y, (-1, fT, fDim*fK))
	Y2 = K.mean(Y1, axis=1)
	
	Y2 = Y2*Y2+1e-6
	vlPower = K.tanh(Y2)*K.sqrt(K.sqrt(Y2));
	vlL2 = K.sqrt(K.sum(vlPower*vlPower, axis=0))
	vlL2norm = vlPower/vlL2
	return vlL2norm
	
def vladEcnoder_output_shape(input_shape):
	fDim = 512
	fK = 64
	shape = list(input_shape)
	assert len(shape) == 2
	shape[-1] = fDim*fK
	return tuple(shape)
	
	
def expan(feats):
	feats = K.reshape(feats, (-1, 64))
	feats = K.softmax(feats)
	feats = K.reshape(feats, (-1, 64, 1))
	feats = 64*K.repeat_elements(feats, 512, axis=-1)

	return feats

	
def expan_output_shape(input_shape):
	#fDim = 512
	shape = list(input_shape)
	assert len(shape) == 3
	shape[-1] = 512
	return tuple(shape)

dim = 512
Kc = 64
timestep = 64 

main_input = Input(shape=(timestep,dim), name='main_input')
aux_input = Input(shape=(Kc*dim,), name='aux_input')

x1 = TimeDistributedDense(1, activation='relu')(main_input)

x1 = Lambda(expan, output_shape=expan_output_shape)(x1)
x = merge([x1, main_input], mode='mul')
dist = TimeDistributedDense(Kc, activation='softmax')(x)
dist = Dropout(0.01)(dist)
x = Flatten()(x)
dist = Flatten()(dist)

x_vlad_1 = merge([dist, x], mode='concat')
x_vlad_2 = merge([x_vlad_1, aux_input], mode='concat')
x_vlad = Lambda(vladEncoder, output_shape=vladEcnoder_output_shape, name='aux_output')(x_vlad_2)
predictions = Dense(55, W_regularizer=l2(0.0001), name='main_output')(x_vlad)

model = Model(input=[main_input, aux_input], output=[predictions, x_vlad])
sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.5, nesterov=True)
weightName = './weights/netVLAD_512_weights_'+str(startN).zfill(3)+'.h5'
model.load_weights(weightName)

model.compile(optimizer=sgd,
              loss={'main_output': 'hinge', 'aux_output': 'cosine_proximity'},
              loss_weights={'main_output': 1, 'aux_output': 5})


mainI = h5py.File(r'./data/data4train512.mat','r')
data = np.array(mainI['data'][:])
data = np.transpose(data)
del mainI

mainO = h5py.File('./data/label_512.mat')       #label
label = np.array(mainO['label'][:])
label = np.transpose(label)
label[label==0] = -1
del mainO

auxO = h5py.File(r'./data/gd_512.mat','r')
gd = np.array(auxO['gd'][:])
gd = np.transpose(gd)
del auxO

auxI = h5py.File(r'./data/centers_512.mat', 'r')      #centers
centers = np.array(auxI['centers'][:])
centers = np.transpose(centers)
del auxI



mainI = h5py.File(r'./data/data_hwtr4de.mat','r')
data1 = np.array(mainI['data'][:])
data1 = np.transpose(data1)
del mainI

mainO = h5py.File('./data/label_hwtr4de.mat')       #label
label1 = np.array(mainO['label'][:])
label1 = np.transpose(label1)
label1[label1==0] = -1
del mainO

auxO = h5py.File(r'./data/gd_hwtr4de.mat','r')
gd1 = np.array(auxO['gd'][:])
gd1 = np.transpose(gd1)
del auxO

auxI = h5py.File(r'./data/centers_hwtr4de.mat', 'r')      #centers
centers1 = np.array(auxI['centers'][:])
centers1 = np.transpose(centers1)
del auxI

data = np.concatenate((data,data1))
del data1
label = np.concatenate((label,label1))
del label1
centers = np.concatenate((centers,centers1))
del centers1
gd = np.concatenate((gd,gd1))
del gd1


np.random.seed(randS)
np.random.shuffle(data)
np.random.shuffle(centers)
np.random.shuffle(gd)
np.random.shuffle(label)


model.fit({'main_input': data, 'aux_input': centers},
		  {'main_output': label, 'aux_output': gd},
		  nb_epoch=1, batch_size=256, 
		  validation_split=0.005)
		 
weightName = './weights/netVLAD_512_weights_'+str(startN+1).zfill(3)+'.h5'
model.save_weights(weightName)
for i in range(startN+1, 100):
	weightName = './weights/netVLAD_512_weights_'+str(i).zfill(3)+'.h5'
	model.load_weights(weightName)
	model.compile(optimizer=sgd,
              loss={'main_output': 'hinge', 'aux_output': 'cosine_proximity'},
              loss_weights={'main_output': 1, 'aux_output': 10000*(1-200*i/50000)**1.6})
	sgd = SGD(lr=0.03*(1-200*i/50000)**0.6, decay=1e-6, momentum=0.8, nesterov=True)
	model.fit({'main_input': data, 'aux_input': centers},
			{'main_output': label, 'aux_output': gd},
			nb_epoch=200, batch_size=200, 
			validation_split=0.005)
	weightName = './weights/netVLAD_512_weights_'+str(i+1).zfill(3)+'.h5'
	model.save_weights(weightName)

