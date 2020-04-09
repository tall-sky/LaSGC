import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import h5py
import glob
import scipy.ndimage

def default_loader(path):
	return Image.open(path).convert('RGB')

class load_data(data.Dataset):
	def __init__(self,root,transform=None,loader=default_loader,seed=None):
		self.root=root
		self.transform=transform
		self.loader=loader
		if seed is not None:
			np.random.seed(seed)

	def __getitem__(self,index):
		filename=self.root+'/'+str(index)+'.h5'
		img=h5py.File(filename,'r')

		haze_image=img['haze'][:].transpose((2,0,1))
		trans_image=img['trans'][:].transpose((2,0,1))
		ato_map=img['atom'][:].transpose((2,0,1))
		gt=img['gt'][:].transpose((2,0,1))

		return haze_image,trans_image,ato_map,gt

	def __len__(self):
		train_list=glob.glob(self.root+'/*h5')
		return len(train_list)

