import numpy as np
from torchvision import transforms

class Config(object):
    def __init__(self):
        self.imgroot = 'D:/VOCdevkit/VOC2007/JPEGImages'
        self.xmlroot = 'D:/VOCdevkit/VOC2007/Annotations'
        self.batchsize = 1
        self.transform = transforms.Compose([transforms.CenterCrop((512, 512)), transforms.ToTensor()])
