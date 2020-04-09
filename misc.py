import torch
import os
import sys
import numpy as np
import torchvision.transforms as transforms
from load_data import load_data
from vgg16 import Vgg16
import torchfile
import torch.nn as nn

def getLoader(dataroot,originalSize,imageSize,batchSize=64,workers=4,mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),split='train',shuffle=True,seed=None):
	if(split=='train'):
		dataset=load_data(root=dataroot,
						  transform=transforms.Compose([
						  	transforms.Resize(originalSize),
							transforms.RandomCrop(imageSize),
							transforms.RandomHorizontalFlip(),
							transforms.ToTensor(),
							transforms.Normalize(mean,std)]),
						  seed=seed)
	else:
		dataset=load_data(root=dataroot,
						  transform=transforms.Compose([
						    transforms.Resize(originalSize),
							transforms.RandomCrop(imageSize),
							transforms.RandomHorizontalFlip(),
							transforms.ToTensor(),
							transforms.Normalize(mean,std)]),
						  seed=seed)
	dataloader=torch.utils.data.DataLoader(dataset,batch_size=batchSize,shuffle=shuffle,num_workers=int(workers))
	return dataloader

def create_dir(dir):
	try:
		os.makedirs(dir)
		print('create dir: %s'%dir)

	except OSError:
		pass
	return True

def weights_init(m):
	classname=m.__class__.__name__
	if isinstance(m,nn.Conv2d):
		m.weight.data.normal_(0.0,0.02)
	elif isinstance(m,nn.BatchNorm2d):
		m.weight.data.normal_(1.0,0.02)
		m.bias.data.fill_(0)

class ImagePool:
	def __init__(self,pool_size=50):
		self.pool_size=pool_size
		if(pool_size>0):
			self.num_imgs=0
			self.images=[]

	def query(self,image):
		if(self.pool_size==0):
			return image
		if(self.num_imgs<self.pool_size):
			self.images.append(image.clone())
			self.num_imgs=self.num_imgs+1
			return image
		else:
			if(np.random.uniform(0,1)>0.5):
				random_id=np.random.randint(self.pool_size,size=1)[0]
				tmp=self.images[random_id].clone()
				self.images[random_id]=image.clone()
				return tmp
			else:
				return image
def adjust_learning_rate(optimizer,init_lr,epoch,every):
	lrd=epoch*init_lr/every
	old_lr=optimizer.param_groups[0]['lr']
	lr=old_lr-lrd
	if lr<0:lr=0
	for param_group in optimizer.param_groups:
		param_group['lr']=lr


def init_vgg16(model_folder):
#    if not os.path.exists(os.path.join(model_folder,'vgg16.weight')):
#        if not os.path.exists(os.path.join(model_folder,'vgg16.t7')):
#            os.system(
#            'wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7 -O'+ os.path.join(model_folder,'vgg16.t7'))
    vgglua=torchfile.load(os.path.join(model_folder,'vgg16.t7'))
    vgg=Vgg16()
    for (src,dst) in zip(vgglua.parameters()[0],vgg.parameters()):
        dst.data[:]=src
    torch.save(vgg.state_dict(),os.path.join(model_folder,'vgg16.weight'))
