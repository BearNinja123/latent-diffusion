from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch, glob, os

class ImgTextDataset(Dataset):
    def __init__(self, img_folder_path, caption_file, crop_size=128):
        super().__init__()
        # note: img_fnames must correspond to img_captions by index, i.e. the text of img_captions[0] must describe the image with filename=img_fnames[0]
        self.img_fnames = sorted(glob.glob(f'{img_folder_path}/*jpg') + glob.glob(f'{img_folder_path}/*png'))
        self.img_captions = open(caption_file).read().strip().split('\n')
        self.crop_size = crop_size
        assert len(self.img_fnames) == len(self.img_captions)

    def __len__(self):
        return len(self.img_fnames)

    def __getitem__(self, idx):
        img_fname = self.img_fnames[idx]
        img_caption = self.img_captions[idx]
        img = np.array(Image.open(img_fname))
        H, W, _C = img.shape
        img = img[(H-self.crop_size)//2:(H+self.crop_size)//2,
                (W-self.crop_size)//2:(W+self.crop_size)//2] # center crop
        return img, img_caption
