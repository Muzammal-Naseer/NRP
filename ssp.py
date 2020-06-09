'''
Self supervised attack is purely based on maximizing the perceptual feature difference.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
import os, imageio, argparse
import numpy as np

parser = argparse.ArgumentParser(description='SSP Attack')
parser.add_argument('--sourcedir', default='clean_imgs')
parser.add_argument('--targetdir', default='ssp_adv')
parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
parser.add_argument('--eps', type=int, default= 16,  help ='pertrbation budget')
parser.add_argument('--step_size', type=float, default=0.01, help='Step size')
parser.add_argument('--iters', type=int, default=100, help='Number of SSP Iterations')
parser.add_argument('--ssp_layer', type=int, default=16, help='VGG layer that is going to be used in SSP')

args = parser.parse_args()
print(args)

# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

all_classes = sorted(os.listdir(args.sourcedir))

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

batch_size = args.batch_size
data_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.sourcedir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])),
    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# We use keras model to create dataset and run different experiments
class perceptual_criteria(nn.Module):
    def __init__(self):
        super(perceptual_criteria, self).__init__()
        # you can try other models
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16 = nn.Sequential(*list(vgg16.features))[:args.ssp_layer].eval()
        # you can try other losses
        self.mse = nn.MSELoss()

    def forward(self, adv, org):
        vgg_out = self.mse(self.vgg16(adv), self.vgg16(org))
        return vgg_out
criterion = perceptual_criteria()
criterion.to(device)

def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t

eps = args.eps / 255
step = args.step_size

counter = 0
current_class = None
current_class_files = None
big_img = []

# labels are just to store the image properly
for i, (img, label) in enumerate(data_loader):
    img = img.to(device)

    adv = torch.randn(img.shape).to(device)
    adv.requires_grad = True

    for t in range(100):
        adv1 = adv + 0
        loss = criterion(normalize(adv1), normalize(img.clone().detach()))
        loss.backward()

        adv.data = adv.data + step * adv.grad.sign()
        adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
        adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()

    # Distance between image and adversary
    print((adv-img).max()*255)

    # Courtesy of: https://github.com/rgeirhos/Stylized-ImageNet/blob/master/code/preprocess_imagenet.py
    for img_index in range(adv.size()[0]):
        source_class = all_classes[label[img_index]]
        source_classdir = os.path.join(args.sourcedir, source_class)
        assert os.path.exists(source_classdir)

        target_classdir = os.path.join(args.targetdir, source_class)
        if not os.path.exists(target_classdir):
            os.makedirs(target_classdir)

        if source_class != current_class:
            # moving on to new class:
            # start counter (=index) by 0, update list of files
            # for this new class
            counter = 0
            current_class_files = sorted(os.listdir(source_classdir))

        current_class = source_class

        target_img_path = os.path.join(target_classdir,
                                       current_class_files[counter]).replace(".JPEG", ".png")


        adv_to_save = np.transpose(adv[img_index, :, :, :].detach().cpu().numpy(), (1, 2, 0))


        adv_to_save = (adv_to_save * 255).round().astype(np.uint8)
        imageio.imwrite(target_img_path, adv_to_save, format='png')


        # imageio.imwrite(target_img_path, adv_to_save.astype(np.uint8))
        # save_image(tensor=adv[img_index, :, :, :],
        #            filename=target_img_path)
        counter += 1
    #
    # del(img)
    # del(adv)
    # del(adv1)

    print('Number of Images Processed:', (i + 1) * batch_size)
