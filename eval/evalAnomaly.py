# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
from erfnet_quantized import Net
import os.path as osp
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from matplotlib import pyplot as plt

seed = 42

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)

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

    if args.loadModel == 'erfnet.py':
        model = ERFNet(NUM_CLASSES)
    # elif args.loadMmodel == 'quantized_erfnet.py':
    #     model = Net(NUM_CLASSES)
    #     model = quantize_model(model, args, False)

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
    if args.loadModel == 'erfnet.py':
        model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    # elif args.loadModel == 'quantized_erfnet.py':
    #     model = load_quant_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
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
              anomaly_result = -np.max(result.squeeze(0).data.cpu().numpy(), axis=0)
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
          images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float()
        #   images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        #   images = images.permute(0,3,1,2)
          with torch.no_grad():
              result = model(images).squeeze(0)
          print(result.shape)
          result = result[:-1]
          print(result.shape)
          if args.method == "msp":
              temperature = float(args.temperature)
              softmax_probs = torch.nn.functional.softmax(result.squeeze(0)/temperature, dim=0)
              anomaly_result = 1.0 - np.max(softmax_probs.data.cpu().numpy(), axis=0)
          elif args.method == "max_logit":
              anomaly_result = -torch.max(result, dim=0)[0]
              anomaly_result = anomaly_result.data.cpu().numpy()
            #   anomaly_result = -np.max(result.squeeze(0).data.cpu().numpy(), axis=0)
          elif args.method == "max_entropy":
              # softmax_probs = torch.nn.functional.softmax(result.squeeze(0), dim=0)
              # anomaly_result = torch.div(-torch.sum(softmax_probs * torch.nn.functional.log_softmax(result.squeeze(0), dim=0), dim=0), torch.log(torch.tensor(result.shape[1]))).data.cpu().numpy()
              softmax_probs = torch.nn.functional.softmax(result.squeeze(0), dim=0)
              log_softmax_probs = torch.nn.functional.log_softmax(result.squeeze(0), dim=0)
              anomaly_result = torch.div(-torch.sum(softmax_probs * log_softmax_probs, dim=0),torch.log(torch.tensor(result.shape[1]))).data.cpu().numpy()
        #   anomaly_result = anomaly_result.data.cpu().numpy()
          pathGT = path.replace("images", "labels_masks")                
          if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
          if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")                
          if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")  


          mask = Image.open(pathGT)
          mask = target_transform(mask)
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

          def select_treshold(proba, target):
                precision, recall, thresholds = precision_recall_curve(target, proba)
                best_treshold = sorted(
                    list(zip(np.abs(precision - recall), thresholds)), key=lambda i: i[0], reverse=False
                )[0][1]

                return best_treshold

          labels = np.where(ood_gts.flatten() != 1, -1, ood_gts.flatten())
        #   print(ood_gts.flatten())
        #   print(labels)
          results = anomaly_result.flatten()
          best_treshold = select_treshold(results, labels)
          save_path = f'.\\save_color\\{args.method}\\' + path.split('Validation_Dataset\\')[2]
          save_path2 = f'.\\save_color\\{args.method}\\gt\\' + path.split('Validation_Dataset\\')[2]
          if not os.path.exists("\\".join(save_path.split("\\")[:-1:])):
              os.makedirs("\\".join(save_path.split("\\")[:-1:]))
          if not os.path.exists("\\".join(save_path2.split("\\")[:-1:])):
              os.makedirs("\\".join(save_path2.split("\\")[:-1:]))

        #   print(f"Threshold: {best_treshold}")
        #   print(f"Scores: {anomaly_result}")

          norm_an_res = (anomaly_result - np.min(anomaly_result)) / (np.max(anomaly_result) - np.min(anomaly_result))
          norm_th = (best_treshold - np.min(anomaly_result)) / (np.max(anomaly_result) - np.min(anomaly_result)) 

        # USED TO SAVE THE IMAGES
        #   plt.imsave(
        #         save_path,
        #         norm_an_res - norm_th,
        #         cmap='bwr',
        #   )

        # #   print(ood_gts)

        #   plt.imsave(
        #         save_path2,
        #         (ood_gts == 1) * 255,
        #         cmap='bwr',
        #   )

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