import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models

class Residual_block(nn.Module):
	def __init__(self,in_planes,grow_rate,kernel_size):
		super(Residual_block,self).__init__()
		self.bn1=nn.BatchNorm2d(in_planes)
		self.conv1=nn.Conv2d(in_planes,grow_rate*4,kernel_size=kernel_size,padding=int(kernel_size/2),bias=False)
		self.bn2=nn.BatchNorm2d(grow_rate*4)
		self.conv2=nn.Conv2d(4*grow_rate,grow_rate,kernel_size=kernel_size,padding=int(kernel_size/2),bias=False)

	def forward(self,x):
		out=self.conv1(F.relu(self.bn1(x)))
		out=self.conv2(F.relu(self.bn2(out)))
		out=out+x
		out=F.relu(out)

		return out


class Conv_relu_block(nn.Module):
	def __init__(self,in_planes,grow_rate,kernel_size,droprate=0.0):
		super(Conv_relu_block,self).__init__()
		self.bn1=nn.BatchNorm2d(in_planes)
		self.conv1=nn.Conv2d(in_planes,grow_rate*4,kernel_size=kernel_size,padding=int(kernel_size/2),bias=False)
		self.bn2=nn.BatchNorm2d(grow_rate*4)
		self.conv2=nn.Conv2d(4*grow_rate,grow_rate,kernel_size=kernel_size,padding=int(kernel_size/2),bias=False)
		self.droprate=droprate

	def forward(self,x):
		out=self.conv1(F.relu(self.bn1(x)))
		if(self.droprate>0):
			out=F.dropout(out,p=self.droprate)
		out=self.conv2(F.relu(self.bn2(out)))

		return out

class sub_net(nn.Module):
	def __init__(self,in_planes,out_planes,if_b,kernel_size):
		super(sub_net,self).__init__()
		self.conv_relu1=Conv_relu_block(in_planes,16,kernel_size,droprate=0.5)
		self.residual_block1=Residual_block(16,16,kernel_size)
		self.conv_relu2=Conv_relu_block(16,16,kernel_size,droprate=0.3)
		self.upsample=F.upsample_nearest
		self.if_b=if_b

	def forward(self,sub_x,x):
		out=self.conv_relu1(x)
		out=out+sub_x
		out=self.conv_relu2(out)
		out=self.residual_block1(out)
		out=self.conv_relu2(out)
		if(self.if_b):
			return out
		else:
			return self.upsample(out,scale_factor=2)

class net(nn.Module):
	def __init__(self):
		super(net,self).__init__()
		self.subnet1=sub_net(16,16,False,3)
		self.subnet2=sub_net(16,16,False,3)
		self.subnet3=sub_net(16,16,True,3)
		self.residual_block1=Residual_block(16,16,3)
		self.conv_relu1=Conv_relu_block(16,16,3,droprate=0.3)
		self.conv_relu2=Conv_relu_block(16,3,3,droprate=0.3)
		self.conv1=nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1)
		self.batchnorm1=nn.BatchNorm2d(8)
		self.conv2=nn.Conv2d(8,16,kernel_size=3,stride=1,padding=1)
		self.batchnorm2=nn.BatchNorm2d(16)

	def forward(self,x):
		out=self.conv1(x)
		out=self.batchnorm1(out)
		out=self.conv2(out)
		out=self.batchnorm2(out)
		input3=out
		input2=F.avg_pool2d(input3,2)
		input1=F.avg_pool2d(input2,2)
		out1=self.subnet1(input1,input1)
		out2=self.subnet2(out1,input2)
		out3=self.subnet3(out2,input3)
		out=self.residual_block1(out3)
		out=self.conv_relu1(out)
		out=self.conv_relu1(out)
		out=self.conv_relu2(out)

		return out








