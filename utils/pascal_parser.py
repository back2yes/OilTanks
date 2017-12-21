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


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(imgdir, xmldir):
    datalist = []
    imgdir = os.path.expanduser(imgdir)
    for target in sorted(os.listdir(imgdir)):
        imgfn = reformatSlash(os.path.join(imgdir, target))
        if not is_image_file(imgfn):
            continue
        # xmlfn = imgfn.replace('Annotations', 'JPEGImages')[:-4] + '.xml'
        xmlfn = reformatSlash(os.path.join(xmldir, renameExt(target, 'xml')))
        xml_data = parseXML(xmlfn)
        datalist.append((imgfn, xml_data))

    return datalist

def reformatSlash(string):
    return string.replace('\\', '/')

def renameExt(fn, ext='xml'):
    return fn.rsplit('/', 1)[-1][:-4] + '.' + ext


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# print(parseXML(xml_data))

class PascalDataset(Dataset):
    def __init__(self, dataroot):
        self.dataroot = dataroot
        self.datalist = make_dataset()


    def __getitem__(self, index):
        imgfn, annot = self.datalist[index]



if __name__ == '__main__':
    import glob
    import os.path

    # for ii, fn in enumerate(sorted(glob.glob(os.path.join('D:/VOCdevkit/VOC2007/Annotations', '*.xml')))):
    # print(fn)
    # print(parseXML(fn))
    xmldir = 'D:/VOCdevkit/VOC2007/Annotations'
    imgdir = 'D:/VOCdevkit/VOC2007/JPEGImages'
    for item in make_dataset(imgdir, xmldir):
        print(item)
