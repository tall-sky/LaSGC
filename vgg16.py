import  torch
import  torch.nn as nn
import  torch.nn.functional as F


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16,self).__init__()
        self.conv1_1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,)
        self.conv1_2=nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)

        self.conv2_1=nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1,)
        self.conv2_2=nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,)

        self.conv3_1=nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1,)
        self.conv3_2=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,)
        self.conv3_3=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,)

        self.conv4_1=nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1,)
        self.conv4_2=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,)
        self.conv4_3=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,)

        self.conv5_1=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,)
        self.conv5_2=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,)
        self.conv5_3=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,)

    def forward(self,x):
        out=F.relu(self.conv1_1(x))
        out=F.relu(self.conv1_2(out))
        relu1=out
        out=F.max_pool2d(out,kernel_size=2,stride=2)

        out=F.relu(self.conv2_1(out))
        out=F.relu(self.conv2_2(out))
        relu2=out
        out=F.max_pool2d(out,kernel_size=2,stride=2)

        out=F.relu(self.conv3_1(out))
        out=F.relu(self.conv3_2(out))
        out=F.relu(self.conv3_3(out))
        relu3=out
        out=F.max_pool2d(out,kernel_size=2,stride=2)

        out=F.relu(self.conv4_1(out))
        out=F.relu(self.conv4_2(out))
        out=F.relu(self.conv4_3(out))
        relu4=out

        return [relu1,relu2,relu3,relu4]

