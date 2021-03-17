import torch.nn as nn
import torch.nn.functional as F

"""
LeNet with MNIST
"""
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) #input is a single channel image -- change according to dataset, now LeNet is specified for mnist dataset only
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
"""
VGG witj CIFAR-10
"""
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG13(nn.Module):
    def __init__(self):
          super(VGG13, self).__init__()
          self.features = self.make_layers()
          self.fc1 = nn.Linear(512,10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = F.softmax(out,dim=1)
        return out

    def make_layers(self,config = cfg['VGG13']):
        in_channels = 3
        layers = []
        for c in config:
            if c == 'M':
                layers += [nn.MaxPool2d(2,2)]
            else:
                conv_layer = nn.Conv2d(in_channels=in_channels,out_channels=c,kernel_size=3,stride=1,padding=1,dilation=1)
                batch_norm = nn.BatchNorm2d(num_features=c)
                activation = nn.ReLU(inplace=True) 
                #inplace=True means that it will modify the input directly, without allocating any additional output. True -- save memory
                layers += [conv_layer,batch_norm,activation]
                in_channels = c #update -- current layer output channels = next layer input channel

        return nn.Sequential(*layers)

def create_model(model_name):
    if model_name.lower() == 'lenet':
        return LeNet()

    if model_name.lower() == 'vgg13':
        return VGG13()