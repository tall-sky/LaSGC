import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark=True
cudnn.fastest=True
import torch.optim as optim
import torchvision.utils as utils
import torchvision.models as models
from model.model_dense_attention import net
from model.model_dense_attention  import *
from vgg16 import Vgg16
from misc import *
from torch.autograd import Variable
import faulthandler
faulthandler.enable()

parser=argparse.ArgumentParser()
parser.add_argument('--batchSize',type=int,default=16,help='input batch size')
parser.add_argument('--valbatchSize',type=int,default=16,help='input batch_size of val')
parser.add_argument('--originalSize', type=int,
  default=286, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
  default=128, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--exp',default='folder',help='folder to output image and checkpoint')
parser.add_argument('--dataroot',required=False,default='create_train/facades/train',help='path to try dataset')
parser.add_argument('--valdataroot',required=False,default='create_train/facades/val',help='path to val dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--netparam',default='temp',help='path to net')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate, default=0.0002')
parser.add_argument('--niter', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--resetstart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--device_ids',type=list,default=[4,5],help='the device chosen to train')



opt=parser.parse_args()


create_dir(opt.exp)

dataloader=getLoader(opt.dataroot,opt.originalSize,opt.imageSize,opt.batchSize,opt.workers,
                     mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),split='train',shuffle=True)
valdataloader=getLoader(opt.valdataroot,opt.originalSize,opt.imageSize,opt.valbatchSize,opt.workers,
                     mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5),split='val',shuffle=False)

net=net()
net.apply(weights_init)

#if(opt.netparam!=''):
#    net.load_state_dict(torch.load(opt.netparam))

net.train

BCE=nn.BCELoss()
CAE=nn.L1Loss(size_average=False)

vgg=Vgg16()
vgg_dict=vgg.state_dict()
vgg_ed=models.vgg16(pretrained=True)
vgg_ed_dict=vgg_ed.state_dict()
vgg_ed_dict={k:v for k,v in vgg_ed_dict.items() if k in vgg_dict}
vgg_dict.update(vgg_ed_dict)
vgg.load_state_dict(vgg_dict)
vgg.cuda()

net=nn.DataParallel(net,device_ids=opt.device_ids)
net.cuda(opt.device_ids[0])
vgg=nn.DataParallel(vgg,device_ids=opt.device_ids)
vgg.cuda(opt.device_ids[0])
CAE=CAE.cuda(opt.device_ids[0])



optimizer=optim.Adam(net.parameters(),lr=opt.lr,betas=(0.9,0.999),weight_decay=0.000005)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,verbose=True)


for epoch in range(opt.niter):
    sum_vgg_loss=0.0
    sum_L1_loss=0.0
    #if epoch >opt.resetstart:
       # adjust_learning_rate(optimizer,opt.lr,epoch,opt.niter)
    for i,data in enumerate(dataloader,0):
        inputt,trans, atom,target = data
        inputt,target=inputt.float().cuda(opt.device_ids[0]),target.float().cuda(opt.device_ids[0])
        optimizer.zero_grad()
        outputs=net(inputt)
        vgg_outputs=vgg(outputs)
        vgg_target=vgg(target)
        vgg_loss=CAE(vgg_outputs[-1],vgg_target[-1])*1000
        vgg_loss.backward(retain_graph=True)
        L1_loss=CAE(outputs,target)
        L1_loss.backward()
        optimizer.step()
        sum_vgg_loss=sum_vgg_loss+vgg_loss.item()
        sum_L1_loss=sum_L1_loss+L1_loss.item()
    print("Epoch%03d: vgg_loss = %.5f ,  L1_loss = %.5f " % (epoch + 1, sum_vgg_loss/8000,sum_L1_loss/8000))
    
    scheduler.step(sum_L1_loss)
    if(epoch%5==0):
        if(epoch%20==0):
            torch.save(net.state_dict(), opt.exp+'/'+"epoch" + str(epoch + 1) + ".pth")
        with torch.no_grad():
            v_loss=0.0
            l_loss=0.0
            for i,data in enumerate(valdataloader,0):
                haze,trans,atom,target=data
                haze,target=haze.float().cuda(opt.device_ids[0]),target.float().cuda(opt.device_ids[0])
                outputs=net(haze)
                v_loss=v_loss+1000*CAE(vgg(outputs)[-1],vgg(target)[-1]).item()
                l_loss=l_loss+CAE(outputs,target).item()
            print('#'*20)
            print("Val  Epoch%03d: vgg_loss = %.10f ,  L1_loss = %.10f " % (epoch + 1, v_loss/792,l_loss/792))
            print('#'*20)








