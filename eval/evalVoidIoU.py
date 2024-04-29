# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time
from erfnet import ERFNet
from enet import ENet
from bisenet import BiSeNet
from PIL import Image
from argparse import ArgumentParser
from erfnet_pruned import prune_and_return_model

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

def load_my_state_dict(model, state_dict, model_name):  #custom function to load model when not all dict elements
        if(model_name == 'ERFNet'):
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    if name.startswith("module."):
                        own_state[name.split("module.")[-1]].copy_(param)
                    else:
                        print(name, " not loaded")
                        continue
                else:
                    own_state[name].copy_(param)
        else:
            model = model.load_state_dict(state_dict)
        return model

def main(args):

    

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)
    
    if(args.model == 'ENet'):
        model = ENet(NUM_CLASSES)
    elif(args.model == 'BiSeNet'):
        model = BiSeNet(NUM_CLASSES)
    else:
        model = ERFNet(NUM_CLASSES)
    if args.pruned == True:
      print("in loop")
      model = prune_and_return_model(model, 0.7)
      print("prunato baby")
    print(args.pruned)

    #model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    

    state_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
    if(args.model == 'ENet' or args.model == 'BiSeNet'):
        #print(state_dict)
        state_dict = {k if k.startswith("module.") else "module." + k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        model = load_my_state_dict(model, state_dict, args.model)
    #print(model)
    print ("Model and weights LOADED successfully")

    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        # print(step)
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)

        if(model == 'Enet'):
            outputs = torch.roll(outputs, -1, 1)
        

        if(model != 'BiSeNet'):
          iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels)
        else:
          iouEvalVal.addBatch(outputs[0].max(1)[1].unsqueeze(1).data, labels)
        
        filenameSave = filename[0].split("leftImg8bit/")[1]

        print(step, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    
    for i in range(iou_classes.size(0)):
        #iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iouStr = '{:0.2f}'.format(iou_classes[i]*100)
        iou_classes_str.append(iouStr)

    if not os.path.exists('voidIoUResults.txt'):
        open('voidIoUResults.txt', 'w').close()
    with open('voidIoUResults.txt', 'a') as f:
        print("---------------------------------------", file=f)
        if args.pruned:
          print("Model", args.model, "with pruning", "Took", time.time() - start, "seconds", file=f)
        else:
          print("Model", args.model, "no pruning ", "Took", time.time() - start, "seconds", file=f)
        print("=======================================", file=f)
        # print("TOTAL IOU: ", iou * 100, "%", file=f)
        print("Per-Class IoU:", file=f)
        print(iou_classes_str[0], "Road", file=f)
        print(iou_classes_str[1], "sidewalk", file=f)
        print(iou_classes_str[2], "building", file=f)
        print(iou_classes_str[3], "wall", file=f)
        print(iou_classes_str[4], "fence", file=f)
        print(iou_classes_str[5], "pole", file=f)
        print(iou_classes_str[6], "traffic light", file=f)
        print(iou_classes_str[7], "traffic sign", file=f)
        print(iou_classes_str[8], "vegetation", file=f)
        print(iou_classes_str[9], "terrain", file=f)
        print(iou_classes_str[10], "sky", file=f)
        print(iou_classes_str[11], "person", file=f)
        print(iou_classes_str[12], "rider", file=f)
        print(iou_classes_str[13], "car", file=f)
        print(iou_classes_str[14], "truck", file=f)
        print(iou_classes_str[15], "bus", file=f)
        print(iou_classes_str[16], "train", file=f)
        print(iou_classes_str[17], "motorcycle", file=f)
        print(iou_classes_str[18], "bicycle", file=f)
        print("=======================================", file=f)
        # iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
        iouStr = '{:0.2f}'.format(iouVal * 100)
        print("MEAN IoU: ", iouStr, "%", file=f)

    #file.close()
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method', default='msp')
    parser.add_argument('--model', default = 'ENet')
    parser.add_argument('--pruned', action = 'store_true')
    main(parser.parse_args())