'''
Purify adversarial images within l_inf <= 16/255
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
import os, imageio
import numpy as np
import argparse
import cv2
from networks import *
from utils import *

parser = argparse.ArgumentParser(description='Purify Images')
parser.add_argument('--dir', default= 'adv_images/')
parser.add_argument('--purifier', type=str, default= 'NRP',  help ='NPR, NRP_resG')
parser.add_argument('--dynamic', action='store_true', help='Dynamic inferrence (in case of whitebox attack)')
args = parser.parse_args()
print(args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if args.purifier == 'NRP':
    netG = NRP(3,3,64,23)
    netG.load_state_dict(torch.load('pretrained_purifiers/NRP.pth'))
if args.purifier == 'NRP_resG':
    netG = NRP_resG(3, 3, 64, 23)
    netG.load_state_dict(torch.load('pretrained_purifiers/NRP_resG.pth'))
netG = netG.to(device)
netG.eval()

print('Parameters (Millions):',sum(p.numel() for p in netG.parameters() if p.requires_grad)/1000000)


dataset = custom_dataset(args.dir)
test_loader = torch.utils.data.DataLoader(dataset,batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


if not os.path.exists('purified_imgs'):
    os.mkdir('purified_imgs')
for i, (img, path) in enumerate(test_loader):
    img = img.to(device)

    if args.dynamic:
        eps = 16/255
        img_m = img + torch.randn_like(img) * 0.05
        #  Projection
        img_m = torch.min(torch.max(img_m, img - eps), img + eps)
        img_m = torch.clamp(img_m, 0.0, 1.0)
    else:
        img_m = img

    purified = netG(img_m).detach()

    save_img(tensor2img(purified), os.path.join('purified_imgs', path[0]))

    print('Number of processed images:', i+1)
