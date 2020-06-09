''''
An example of how to by pass NRP. Solution to this problem is dynamic infernce as discussed in the paper.
Dynamic inference is achieved by perturbing the incoming sample with random noise.
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
from networks import *

parser = argparse.ArgumentParser(description='By Pass NRP')
parser.add_argument('--test_dir', default= 'val/')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size for evaluation')
parser.add_argument('--model_type', type=str, default= 'res152',  help ='incv3, res152')
parser.add_argument('--eps', type=int, default= 16,  help ='pertrbation budget')
parser.add_argument('--purifier', type=str, default= 'NRP',  help ='NPR, NRP_resG')
args = parser.parse_args()
print(args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Setup-Data
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t

test_dir = args.test_dir
test_set = datasets.ImageFolder(test_dir, data_transform)
test_size = len(test_set)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


# Load Purifier
if args.purifier == 'NRP':
    netG = NRP(3,3,64,23)
    netG.load_state_dict(torch.load('pretrained_purifiers/NRP.pth'))
if args.purifier == 'NRP_resG':
    netG = NRP_resG(3, 3, 64, 23)
    netG.load_state_dict(torch.load('pretrained_purifiers/NRP_resG.pth'))
netG = netG.to(device)
netG.eval()
netG = torch.nn.DataParallel(netG)

# Load Backbone model
model = torchvision.models.resnet152(pretrained=True)
model = model.to(device)
model.eval()
model = torch.nn.DataParallel(model)

# Loss Criteria
criterion = nn.CrossEntropyLoss()
eps = args.eps / 255
iters = 10
step = 2/255

counter = 0
current_class = None
current_class_files = None
big_img = []

sourcedir = args.test_dir
targetdir = '{}_{}'.format(args.model_type, args.eps)

all_classes = sorted(os.listdir(sourcedir))

# Generate labels
# Courtesy of: https://github.com/carlini/breaking_efficient_defenses/blob/master/test.py
def get_labs(y):
    l = np.zeros((len(y),1000))
    for i in range(len(y)):
        r = np.random.random_integers(0,999)
        while r == np.argmax(y[i]):
            r = np.random.random_integers(0,999)
        l[i,r] = 1
    return l


out = 0
for i, (img, label) in enumerate(test_loader):
    img = img.to(device)
    label = label.to(device)

    # Random Target labels
    new_label = torch.from_numpy(get_labs(label.detach().cpu().numpy()).argmax(axis=-1)).to(device)

    adv = img.detach()
    adv.requires_grad = True
    for j in range(iters):
        adv1 = netG(adv)
        adv1 = torch.clamp(adv1, 0.0, 1.0)
        output = model(normalize(adv1))
        loss = criterion(output, new_label)
        loss.backward()

        adv.data = adv.data - step * adv.grad.sign()
        adv.data = torch.min(torch.max(adv.data, img - eps), img + eps)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()

    print((adv-img).max().item()*255)
    # Courtesy of: https://github.com/rgeirhos/Stylized-ImageNet/blob/master/code/preprocess_imagenet.py
    for img_index in range(adv.size()[0]):
        source_class = all_classes[label[img_index]]
        source_classdir = os.path.join(sourcedir, source_class)
        assert os.path.exists(source_classdir)

        target_classdir = os.path.join(targetdir, source_class)
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

        # if size_error == 1:
        #     big_img.append(target_img_path)

        adv_to_save = np.transpose(adv[img_index, :, :, :].detach().cpu().numpy(), (1, 2, 0))*255
        imageio.imwrite(target_img_path, adv_to_save.astype(np.uint8))
        # save_image(tensor=adv[img_index, :, :, :],
        #            filename=target_img_path)
        counter += 1
    #
    # del(img)
    # del(adv)
    # del(adv1)
    print('Number of Images Processed:', (i + 1) * args.batch_size)
