# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method', default = 'msp')
    parser.add_argument('--temperature', default = 1.0)
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
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
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()
    
    #ADDED CODE FOR TEMPERATURE GRID SEARCH
    """
    auprc_scores = []
    fpr_scores = []
    temperature_range = np.linspace(0.1, 2.0, 20)
    for temperature in temperature_range:
      print(temperature)
      for path in glob.glob(os.path.expanduser(str(args.input[0]))):
          print(path)
          images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
          images = images.permute(0,3,1,2)
          with torch.no_grad():
              result = model(images)
          print(result)
          if args.method == "msp":
              softmax_probs = torch.nn.functional.softmax(result.squeeze(0)/temperature, dim=0)
              anomaly_result = 1.0 - np.max(softmax_probs.data.cpu().numpy(), axis=0)
          elif args.method == "max_logit":
              anomaly_result = 1.0 - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)
          elif args.method == "max_entropy":
              # softmax_probs = torch.nn.functional.softmax(result.squeeze(0), dim=0)
              # anomaly_result = torch.div(-torch.sum(softmax_probs * torch.nn.functional.log_softmax(result.squeeze(0), dim=0), dim=0), torch.log(torch.tensor(result.shape[1]))).data.cpu().numpy()
              softmax_probs = torch.nn.functional.softmax(result.squeeze(0), dim=0)
              log_softmax_probs = torch.nn.functional.log_softmax(result.squeeze(0), dim=0)
              anomaly_result = torch.div(-torch.sum(softmax_probs * log_softmax_probs, dim=0),torch.log(torch.tensor(result.shape[1]))).data.cpu().numpy()
          pathGT = path.replace("images", "labels_masks")                
          if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
          if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")                
          if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")  

          mask = Image.open(pathGT)
          ood_gts = np.array(mask)

          if "RoadAnomaly" in pathGT:
              ood_gts = np.where((ood_gts==2), 1, ood_gts)
          if "LostAndFound" in pathGT:
              ood_gts = np.where((ood_gts==0), 255, ood_gts)
              ood_gts = np.where((ood_gts==1), 0, ood_gts)
              ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

          if "Streethazard" in pathGT:
              ood_gts = np.where((ood_gts==14), 255, ood_gts)
              ood_gts = np.where((ood_gts<20), 0, ood_gts)
              ood_gts = np.where((ood_gts==255), 1, ood_gts)

          if 1 not in np.unique(ood_gts):
              continue              
          else:
              ood_gts_list.append(ood_gts)
              anomaly_score_list.append(anomaly_result)
          del result, anomaly_result, ood_gts, mask
          torch.cuda.empty_cache()

      file.write( "\n")

      ood_gts = np.array(ood_gts_list)
      anomaly_scores = np.array(anomaly_score_list)

      ood_mask = (ood_gts == 1)
      ind_mask = (ood_gts == 0)

      ood_out = anomaly_scores[ood_mask]
      ind_out = anomaly_scores[ind_mask]

      ood_label = np.ones(len(ood_out))
      ind_label = np.zeros(len(ind_out))
      
      val_out = np.concatenate((ind_out, ood_out))
      val_label = np.concatenate((ind_label, ood_label))

      prc_auc = average_precision_score(val_label, val_out)
      fpr = fpr_at_95_tpr(val_out, val_label)

      print(f'AUPRC score: {prc_auc*100.0}')
      print(f'FPR@TPR95: {fpr*100.0}')
      auprc_scores.append(prc_auc)
      fpr_scores.append(fpr)
      #file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
      #file.close()
    print("Punteggi AUPRC per ogni temperatura:")
    for temperature, auprc_score in zip(temperature_range, auprc_scores):
        print(f'Temperatura: {temperature}, AUPRC: {auprc_score}')

    print("\nPunteggi FPR per ogni temperatura:")
    for temperature, fpr_score in zip(temperature_range, fpr_scores):
        print(f'Temperatura: {temperature}, FPR: {fpr_score}')
    res = sorted(auprc_scores, reverse=True)
    res_min = sorted(fpr_scores)
    max_indices = []
    min_indices = []

    for index, value in enumerate(auprc_scores):  # Using enumerate to get both index and value
        max_indices.append(value - fpr_scores[index])


    print("Indices list of max N elements is : " + str(max_indices.index(np.max(max_indices))))
    """

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
          print(path)
          images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
          images = images.permute(0,3,1,2)
          with torch.no_grad():
              result = model(images)
          if args.method == "msp":
              temperature = float(args.temperature)
              softmax_probs = torch.nn.functional.softmax(result.squeeze(0)/temperature, dim=0)
              anomaly_result = 1.0 - np.max(softmax_probs.data.cpu().numpy(), axis=0)
          elif args.method == "max_logit":
              anomaly_result = 1.0 - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)
          elif args.method == "max_entropy":
              # softmax_probs = torch.nn.functional.softmax(result.squeeze(0), dim=0)
              # anomaly_result = torch.div(-torch.sum(softmax_probs * torch.nn.functional.log_softmax(result.squeeze(0), dim=0), dim=0), torch.log(torch.tensor(result.shape[1]))).data.cpu().numpy()
              softmax_probs = torch.nn.functional.softmax(result.squeeze(0), dim=0)
              log_softmax_probs = torch.nn.functional.log_softmax(result.squeeze(0), dim=0)
              anomaly_result = torch.div(-torch.sum(softmax_probs * log_softmax_probs, dim=0),torch.log(torch.tensor(result.shape[1]))).data.cpu().numpy()
          pathGT = path.replace("images", "labels_masks")                
          if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
          if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")                
          if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")  

          mask = Image.open(pathGT)
          ood_gts = np.array(mask)

          if "RoadAnomaly" in pathGT:
              ood_gts = np.where((ood_gts==2), 1, ood_gts)
          if "LostAndFound" in pathGT:
              ood_gts = np.where((ood_gts==0), 255, ood_gts)
              ood_gts = np.where((ood_gts==1), 0, ood_gts)
              ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

          if "Streethazard" in pathGT:
              ood_gts = np.where((ood_gts==14), 255, ood_gts)
              ood_gts = np.where((ood_gts<20), 0, ood_gts)
              ood_gts = np.where((ood_gts==255), 1, ood_gts)

          if 1 not in np.unique(ood_gts):
              continue              
          else:
              ood_gts_list.append(ood_gts)
              anomaly_score_list.append(anomaly_result)
          del result, anomaly_result, ood_gts, mask
          torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
      
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')
    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()
if __name__ == '__main__':
    main()