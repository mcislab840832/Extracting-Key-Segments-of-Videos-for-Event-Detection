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
from keras.regularizers import l1, l2, l1l2
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.models import model_from_json
from keras.utils import np_utils


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

def norm_l2(feats):
	feats_norm = K.l2_normalize(feats, axis=-1)
	return feats_norm

	
def norm_l2_output_shape(input_shape):
	shape = list(input_shape)
	assert len(shape) == 2
	return tuple(shape)

dim = 512
Kc = 64
timestep = 64
main_input = Input(shape=(timestep,dim), name='main_input')
aux_input = Input(shape=(Kc*dim,), name='aux_input')

x1 = TimeDistributedDense(1, activation='relu', trainable=False)(main_input)
x1 = Lambda(expan, output_shape=expan_output_shape)(x1)
x = merge([x1, main_input], mode='mul')
dist = TimeDistributedDense(Kc, activation='softmax', trainable=False)(x)
dist = Dropout(0.2)(dist)
x = Flatten()(x)
dist = Flatten()(dist)


x_vlad_1 = merge([dist, x], mode='concat')
x_vlad_2 = merge([x_vlad_1, aux_input], mode='concat')
x_vlad = Lambda(vladEncoder, output_shape=vladEcnoder_output_shape)(x_vlad_2)
predictions = Dense(55, W_regularizer=l2(0.001), name='main_output')(x_vlad)
model = Model(input=[main_input, aux_input], output=predictions)

weightName = './weights/netVLAD_512_weights_100.h5'
model.load_weights(weightName)
model = Model(input=[main_input, aux_input], output=vlad)

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(optimizer=sgd,
#              loss= 'multilabel_crossentropy',
#              metrics=['accuracy'])

model.compile(optimizer=sgd,
              loss= 'hinge',
              metrics=['accuracy'])

mainI = h5py.File(r'./data/data_medtr512.mat','r')
data = np.array(mainI['data'][:])
del mainI
data = np.transpose(data)
print (data.shape)



auxI = h5py.File(r'./data/centers_medtr512.mat','r')
centers = np.array(auxI['centers'][:])
del auxI
centers = np.transpose(centers)
print (centers.shape)


features = model.predict([data, centers])
np.savetxt("./scores/med_canet_vlad_train.txt",features)

del data
del centers
del features

mainI = h5py.File(r'./data/data_medts512.mat','r')
data = np.array(mainI['data'][:])
del mainI
data = np.transpose(data)
print (data.shape)



auxI = h5py.File(r'./data/centers_medts512.mat','r')
centers = np.array(auxI['centers'][:])
del auxI
centers = np.transpose(centers)
print (centers.shape)


features = model.predict([data, centers])

np.savetxt("./scores/med_canet_vlad_test.txt",features)

