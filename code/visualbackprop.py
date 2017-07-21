# stdlib
import os

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms

# 3rd party packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# modules
from net.models.vgg import *
from net.models.alexnet import *
from net.utility.tools import *

maps = []
layers=[]
hooks=[]


def visual_feature(self, input, output):
    # The hook function that show you the feature maps while forward propagate

    vis_square(output.data[0,:])

def save_feature_maps(self,input,output):
    # The hook function that saves feature maps while forward propagate

    map = output.data
    maps.append(map)

def save_image_maps(save_dir, img, layers, maps):

    '''
    This function saves feature maps locally for examine

    :param save_dir: the destination of the saved feature maps
    :param img: the origin image
    :param layers: the saved layers by using pytorch hook
    :param maps: the saved feature maps by using pytorch hook
    :return: this function will not return anything, instead it will save all your feature maps locally
    '''

    os.makedirs(save_dir,exist_ok=True)
    cv2.imwrite(save_dir + '/input.png', img)

    num_layers = len(layers)
    for n in range(num_layers):
        layer = layers[n]
        os.makedirs(save_dir+'/'+layer[0],exist_ok=True)

        map = maps[n].numpy()
        map = map[0,:]

        if map.ndim==1:
            l = len(map)
            if l==1:
                with open(save_dir+'/'+ layer[0] + '/%f'%map[0], 'w') as f:
                    pass

            else:
                w = int(l**0.5)
                h = int(np.ceil(l/w))

                f = map
                fmax=np.max(f)
                fmin=np.min(f)
                f = ((f-fmin)/(fmax-fmin+1e-12)*255).astype(np.uint8)

                f1 = np.zeros(h*w, np.uint8)
                f1[:l] = f
                f1 = f1.reshape(h,w)
                cv2.imwrite(save_dir+'/'+ layer[0] + '/out.png', f1)


        else: # assume  map.ndim==3
            #save txt for debug
            np.savetxt(save_dir+'/'+ layer[0] + 'out', map[0], fmt='%0.6f', delimiter='\t', newline='\n')

            num_channels =len(map)
            for c in range(num_channels):
                f = map[c]
                fmax=np.max(f)
                fmin=np.min(f)
                f = ((f-fmin)/(fmax-fmin+1e-12)*255).astype(np.uint8)
                cv2.imwrite(save_dir+'/'+ layer[0] + '/out%03d.png'%c, f)

def add_hook(net,layer_name,func):
    '''
    Add a hook function in the layers you specified.
    Hook will be called during forward propagate at the layer you specified.

    :param net: The model you defined
    :param layer_name: Specify which layer you want to hook, currently you can hook 'all', 'maxpool', 'relu'
    :param func: Specify which hook function you want to hook while forward propagate
    :return: this function will return the model that hooked the function you specified in specific layer
    '''

    if layer_name=='maxpool':
        for m in net.features:
            if isinstance(m, nn.MaxPool2d):
                m.register_forward_hook(func)
        return net

    if layer_name == 'relu':
        for index, m in enumerate(net.features):
            if isinstance(m, nn.ReLU):
                type_name = str(type(m)).replace("<'", '').replace("'>", '').split('.')[-1]
                name = 'features' + '-' + str(index) + '-' + type_name
                hook = m.register_forward_hook(func)
                layers.append((name, m))
                hooks.append(hook)
        return net

    if layer_name == 'all':
        for index, m in enumerate(net.features):
            type_name = str(type(m)).replace("<'", '').replace("'>", '').split('.')[-1]
            name = 'features' + '-' + str(index) + '-' + type_name
            hook = m.register_forward_hook(func)
            layers.append((name, m))
            hooks.append(hook)
        return net


def visualbackprop(layers,maps):

    '''
    :param layers: the saved layers
    :param maps: the saved maps
    :return: return the final mask
    '''

    num_layers = len(maps)
    avgs = []
    mask = None
    ups  = []

    for n in range(num_layers-1,0,-1):
        cur_layer=layers[n][1]
        if type(cur_layer) in [torch.nn.MaxPool2d]:
            ##########################
            # Get and set attributes #
            ##########################
            relu = maps[n-1]
            conv = maps[n-2]

            ###########################################
            # Average filters and multiply pixel-wise #
            ###########################################

            # Average filters
            avg = relu.mean(dim=1)
            avgs.append(avg)

            if mask is not None:
                mask = mask * avg
            else:
                mask = avg

            # upsampling : see http://pytorch.org/docs/nn.html#convtranspose2d
            weight = Variable(torch.ones(1, 1, 3, 3))

            up = F.conv_transpose2d(Variable(mask), weight, stride=2, padding=1, output_padding=1)
            mask = up.data
            ups.append(mask)

    return ups


if __name__ == "__main__":

    '''
    Load image, resize the image to 224 x 224, then transfer the loaded image numpy array.
    The transfer included 1.ArrayToTensor and 2.Normalization.
    After image transformation, we need to define the model.
    The model used here is VGG19, you can choose whatever model you like.
    After the model is defined, the pre-trained model is loaded.
    
    Before we forward the image through VGG, we need to define our hook first.
    The function "add_hook" provide you an easy way to add hook to layer.
    You have to specify: 
        1. the model you want to hook 
        2. the layer you want to hook
        3. the function you want to hook
    
    Since I want to save the feature maps to a list.
    I create "save_image_maps" as my hook function
    This function will help me to extract the layer output.
    After the outputs are extracted, I want to output the extracted feature maps to image.
    So by calling "save_image_maps", we call save all the maps as image locally.
    '''


    BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__dir__')))
    IMG_DIR = BASE_DIR + '/image'
    MODEL_DIR = BASE_DIR + '/pretrained_model'
    IMG_NAME = 'photo.jpg'


    image = cv2.imread(IMG_DIR+'/'+IMG_NAME)
    image = cv2.resize(image, (224, 224))

    im = Image.fromarray(image)
    im.save(IMG_DIR+'/'+ 'resized_' +IMG_NAME)

    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = loader(image)
    img = img.contiguous().view(1, 3, 224, 224)
    model = vgg19()

    load_valid(model, MODEL_DIR + "/vgg19-dcbb9e9d.pth", None)
    x = Variable(img)


    add_hook(model,'all',save_feature_maps)
    logits, probs = model.forward(x)


    masks = visualbackprop(layers, maps)

    mask_num = len(masks)
    print(masks[mask_num-2].size())

    vis_single_square(masks[mask_num-2],IMG_DIR+'/'+ 'mask_' +IMG_NAME)