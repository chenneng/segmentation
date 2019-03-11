import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def recursive_glob(rootdir='.', suffix=''):
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                    for filename in filenames if filename.endswith(suffix)]

class segmentationData(Dataset):
    def __init__(self, root, split, dataTransform = None, labelTransform = None):
        self.root = root
        self.split = split
        self.dataTransform = dataTransform
        self.labelTransform = labelTransform
        self.files = {}
        self.n_classes = 19

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine',self.split)

        self.files[split] = recursive_glob(rootdir = self.images_base, suffix = '.png')
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_name = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base, img_path.split(os.sep)[-2], 
                                    os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')
            
        img = Image.open(img_path).convert('RGB')
        tmp = np.array(Image.open(lbl_path), dtype = np.uint8)
        tmp = self.encode_segmap(tmp)
        target = Image.fromarray(tmp)

        if self.dataTransform:
            img = self.dataTransform(img)

        if self.labelTransform:
            target = self.labelTransform(target)
            target = target.to(torch.long)

        return img, target

    
    def encode_segmap(self, mask):
        for voidc in self.void_classes:
            mask[mask == voidc] = self.ignore_index

        for validc in self.valid_classes:
            mask[mask == validc] = self.class_map[validc]

        return mask

