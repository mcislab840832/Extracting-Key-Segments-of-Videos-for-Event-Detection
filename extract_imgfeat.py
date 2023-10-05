import os,sys,skimage,math
import numpy as np
import scipy.io as sio
sys.path.append('~/library/caffe-master/python')
import caffe
#import ipdb

workspace = '~/resnet/'
database = 'hw2ts_'

imgdir = workspace + database + 'pic/'
feadir = workspace + database + 'fea/'

caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net('ResNet-152-deploy.prototxt', 'ResNet-152-model.caffemodel', caffe.TEST)


transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load('imagenet_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

net.blobs['data'].reshape(1,3,224,224)

def mg(store):
    if not os.path.exists(feadir):
        os.makedirs(feadir)

    nv = 0
    for vid in open(store):
	simgd = vid
	if simgd[0] == '\t':
		simgd = simgd[1:]
	if simgd[-1] == '\n':
		simgd = simgd[:-1]
        nv = nv +1
	print simgd + ' ' + str(nv)

	temp1=simgd.split('.')[0]
	temp2=simgd.split('.')[1]
	vn = temp1+'.'+temp2
	matdir = feadir + simgd.split('/')[0] 

        
	if not os.path.exists(matdir):
		os.makedirs(matdir)

	if os.path.exists(feadir + vn+'.mat'): 
		continue

	net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(imgdir+simgd))
	net.forward()
	pool5 = net.blobs['pool5'].data
	pool5 = np.squeeze(pool5)
	fc1000 = net.blobs['fc1000'].data
	fc1000 = np.squeeze(fc1000)

	sio.savemat(feadir+vn, {'pool5':pool5,'fc1000':fc1000})

mg('input_list_hw2ts.txt')

