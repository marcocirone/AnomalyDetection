import os
import importlib
from erfnet_quantized import Net
import torch
from torch import nn
from torch.ao.quantization import quantize_fx, observer, get_default_qconfig_mapping
# import torch.ao.quantization as quantization
from torch.utils.data import DataLoader
from torch.autograd import Variable
from flopth import flopth
from dataset import cityscapes
from copy import deepcopy
from argparse import ArgumentParser
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from transform import Relabel, ToLabel, Colorize

backend = "x86"

class MyCoTransform(object):
    def __init__(self, enc, model='erfnet', augment=True, height=512):
        self.enc=enc
        self.augment = augment
        self.height = height
        self.model = model
        pass
    def __call__(self, input):
        # do something to both images
        if self.model == 'erfnet' or self.model == 'bisenet':
            input = Resize(self.height, Image.BILINEAR)(input)
            target = Resize(self.height, Image.NEAREST)(target)
        elif self.model == 'enet':
            input = Resize((self.height, self.height), Image.BILINEAR)(input)
            target = Resize((self.height, self.height), Image.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY)) 

        input = ToTensor()(input)
        if (self.enc):
            target = Resize(int(self.height/8), Image.NEAREST)(target)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target

def calibrate(model):
    dataset_val = cityscapes(args.datadir, subset='val')
    loader_val = DataLoader(dataset_val, num_workers=4, batch_size=args.batch_size, shuffle=False)
    print(len(loader_val))
    with torch.inference_mode():
        for step, (images, _) in enumerate(loader_val ):
            print(f"step {step + 1}")
            print(images.shape)
            if not args.cpu:
                images = images.cuda()
            images = Variable(images)
            model(images)

def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith('module.'):
                own_state[name.split('module.')[-1]].copy_(param)
            else:
                print(name, ' not loaded')
                continue
        else:
            own_state[name].copy_(param)
    return model

def load_quant_dict(model, state_dict):
    own_state = model.state_dict()
    for name, params in state_dict.items():
        own_state[name]=deepcopy(params)
    return model

def quantize_model(model, args, calibrate):
    m = deepcopy(model)
    # flops, params = flopth(model, in_size=((3, 512, 1024),))
    # print("NUMBER OF FLOPS")
    # print(flops, params)
    # print("PRE FUSION")
    # total_params = sum(p.numel() for p in m.parameters() if torch.any(p != 0))
    # print(total_params)
    modules_to_fuse = [[f"encoder.layers.{i}.conv1x3_1", f"encoder.layers.{i}.bn1"] for i in range(1, 6)]
    modules_to_fuse += [[f"encoder.layers.{i}.conv1x3_1", f"encoder.layers.{i}.bn1"] for i in range(7, 15)]
    modules_to_fuse += [[f"decoder.layers.{i}.conv1x3_1", f"decoder.layers.{i}.bn1"] for i in range(1, 3)]
    modules_to_fuse += [[f"decoder.layers.{i}.conv1x3_1", f"decoder.layers.{i}.bn1"] for i in range(4, 6)]
    # for f in modules_to_fuse:
    #     print(f)
    m.eval()
    if not args.cpu:
        m.cuda()
    config = torch.ao.quantization.qconfig.QConfig(
        activation = torch.ao.quantization.MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric),
        weight = torch.ao.quantization.MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.qint8)
    )
    m.qconfig = config
    # m.qconfig = config
    print()
    # print(m.qconfig)
    print()
    fused_model = torch.ao.quantization.fuse_modules(m, modules_to_fuse)
    print("POST FUSION")
    total_params = sum(p.numel() for p in fused_model.parameters() if torch.any(p != 0))
    print(total_params)
    model_prepared = torch.ao.quantization.prepare(fused_model)
    if calibrate:
        input_transform_cityscapes = Compose([
            Resize(512, Image.BILINEAR),
            ToTensor(),
        ])
        target_transform_cityscapes = Compose([
            Resize(512, Image.NEAREST),
            ToLabel(),
            Relabel(255, 19),   #ignore label to 19
        ])
        dataset_val = cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset='val')
        loader_val = DataLoader(dataset_val, num_workers=4, batch_size=args.batch_size, shuffle=False)
        print(len(loader_val))
        with torch.inference_mode():
            for step, (images, _, _, _) in enumerate(loader_val):
                if step == 2:
                    break
                print(f"step {step + 1}")
                print(images.shape)
                if not args.cpu:
                    images = images.cuda()
                images = Variable(images)
                model_prepared(images)
    model_quantized = torch.ao.quantization.convert(model_prepared.cpu(), remove_qconfig=False)
    # total_params = sum(p.numel() for p in model_quantized.parameters() if torch.any(p != 0))
    # print(total_params)
    # flops, params = flopth(model_quantized, in_size=((3, 512, 1024),))
    # print("NUMBER OF FLOPS")
    # print(flops, params)
    torch.save(model_quantized.state_dict(), "./quantized_model.pth")
    return model_quantized

def main(args, calibrate=True):
    savedir = f"{args.savedir}/quantized_model.pth"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    model = Net(20)
    model = load_my_state_dict(model, torch.load(f"../trained_models/{args.model}_pretrained.pth", map_location=torch.device('cpu')))
    total_params = sum(p.numel() for p in model.parameters() if torch.any(p != 0))
    print(total_params)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()
    qmodel = quantize_model(model, args, calibrate)
    if calibrate:
        torch.save(qmodel.state_dict(), "./quantized_model.pth")
    return qmodel

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--model', default='erfnet')
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--savedir', default="/content/Drive/MyDrive/save")
    parser.add_argument('--datadir', default="/content/Drive/MyDrive/cityscapes")
    parser.add_argument('--resume', action='store_true')
    
    args = parser.parse_args()
    
    model = main(args, calibrate=args.resume)  # first execution initializes the model
    total_params = sum(p.numel() for p in model.parameters() if torch.any(p != 0))
    print(f"Total Params: {total_params}")  

    # dict = torch.load("quantized_model.pth", map_location=lambda storage, loc: storage)
    # # model = load_quant_dict(model, dict)
    # model.load_state_dict(dict)
    # print("dict loaded")
    file1 = open("file.txt", "w")
    for name, _ in model.state_dict().items():
        file1.write(f"{name}\n")
    # summary(model, input_size=(3, 512, 1024))
    # for p in model.parameters():
    #     print(p)
    # with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    #     with profiler.record_function("model_inference"):
    #         model(torch.randn(1, 3, 512, 1024))
    # result = prof.key_averages().table(sort_by="self_cpu_time_total")
    # if "Flops" in result:
    #     print(f"Numero di FLOPS: {result["Flops"]}")
    # else:
    #     print("Statistiche dei FLOPS non disponibili nel profilo.")
    #     print(result)

    # flops, macs = get_model_complexity_info(model, (3, 512, 1024), as_strings=True, print_per_layer_stat=True)
    # print(f"FLOPS: {flops}")
    # print(f"Moltiplicazioni e addizioni (MACs): {macs}")

    # flops, params = flopth(model, in_size=((3, 512, 1024),))
    # print(flops, params)