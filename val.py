import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from vgg16 import Vgg16
from misc import *
from torch.autograd import Variable
from model.model1 import net
import numpy as np
import cv2
from PIL import Image

parser=argparse.ArgumentParser()
parser.add_argument('--dataroot',default='create_train/facades/val',help='path to val data')
parser.add_argument('--originalSize',type=int,default='286')
parser.add_argument('--valbatchSize',type=int,default=1)
parser.add_argument('--imageSize',type=int,default='128')
parser.add_argument('--workers',type=int,default=1)
parser.add_argument('--model_path',default='folder/epoch201.pth')
parser.add_argument('--gpu_id',default=5,type=int)

opt=parser.parse_args()

net=net()
device=torch.device('cuda:5')
net=net.to(device)

kwargs={'map_location':lambda storage,loc:storage.cuda(5)}

valdataloader=getLoader(opt.dataroot,opt.originalSize,opt.imageSize,opt.valbatchSize,opt.workers,
                        mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),split='val',shuffle=False)

def load_GPUS(model,model_path,kwargs):
    state_dict=torch.load(model_path,**kwargs)
    from collections import OrderedDict
    new_state_dict=OrderedDict()
    for k,v in state_dict.items():
        name=k[7:]
        new_state_dict[name]=v
    model.load_state_dict(new_state_dict)
    return model

model=load_GPUS(net,opt.model_path,kwargs)
model.eval()
with torch.no_grad():
    for i,data in enumerate(valdataloader):
        haze,trans,atom,target=data
        haze=haze.float()
        haze=haze.to(device)
        output=model(haze)
        output=torch.squeeze(output)
        image=output.cpu().numpy()
        image=np.transpose(image,(1,2,0))
        img_path='folder/result/'+str(i)+'.jpg'
        im=Image.fromarray(np.uint8(image*255))
        im.save(img_path)
