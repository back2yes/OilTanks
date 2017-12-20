import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import pandas as pd
from PIL import Image

xml_data = open('D:/VOCdevkit/VOC2007/Annotations/000001.xml').read()

"""
Person: person
Animal: bird, cat, cow, dog, horse, sheep
Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor
"""
synsets = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4,
           'horse': 5, 'sheep': 6, 'aeroplane': 7, 'bicycle': 8, 'boat': 9,
           'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13, 'bottle': 14,
           'chair': 15, 'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19}


def parseXML(xml_path):
    with open(xml_path, 'r') as f:
        xml_data = f.read()
    root = ET.XML(xml_data)
    size = []
    bndboxes = []
    classes = []
    # print(root)  # root is 'annotation'
    for ii, child in enumerate(root):
        if child.tag == 'size':  # w, h, c
            size = [int(child[0].text), int(child[1].text), int(child[2].text)]

        if child.tag == 'object':
            # print(child) # nodes inside the root
            for jj, sub_child in enumerate(child):
                # print(sub_child)
                if sub_child.tag == 'bndbox':  # a 'bndbox' consists of 4 integers, xmin, ymin, xmax, ymax
                    xmin = int(sub_child[0].text)
                    ymin = int(sub_child[1].text)
                    xmax = int(sub_child[2].text)
                    ymax = int(sub_child[3].text)
                    # print(xmin, ymin, xmax, ymax)
                    bndboxes.append([xmin, ymin, xmax, ymax])
                if sub_child.tag == 'name':
                    classes.append(synsets[sub_child.text])
    return size, classes, bndboxes


# print(parseXML(xml_data))

class PascalDataset(Dataset):
    def __init__(self, dataroot):
        self.dataroot = dataroot

    def __getitem__(self, item):


if __name__ == '__main__':
    import glob, os.path

    for ii, fn in enumerate(sorted(glob.glob(os.path.join('D:/VOCdevkit/VOC2007/Annotations', '*.xml')))):
        # fn = fn.replace('\\', '/')
        # with open(fn) as f:

        print(fn)
        print(parseXML(fn))
