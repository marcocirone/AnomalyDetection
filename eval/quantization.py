import os
import importlib
from erfnet_quantized import Net
from erfnet import ERFNet
import torch
from torch import nn
from torch.ao.quantization import quantize_fx, QConfigMapping, observer, quantize_dynamic
# import torch.ao.quantization as quantization
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
from torchprofile import profile_macs
from dataset import cityscapes
from copy import deepcopy
from argparse import ArgumentParser

backend = "x86"

def calibrate(model, data_load):
    co_transform_val = MyCoTransform(False, augment=False, height=512)
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')
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

def quantize_model2(model, args):
    m = deepcopy(model)


def quantize_model(model, args, calibrate): #datadir should be the path to the cityscapes validation dataset
    m = deepcopy(model)
    example_input = torch.randn(1, 20, 512, 1024)
    my_qconfig = torch.ao.quantization.QConfig(
        weight = observer.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine),
        activation = observer.MinMaxObserver.with_args(dtype = torch.qint8, qscheme=torch.per_tensor_affine)
    )
    # qconfig = get_default_qconfig(backend)
    # qconfig_mapping = QConfigMapping().set_global(my_qconfig)
    # if not args.cpu:
    #     example_input = example_input.cuda()
    #     for key, value in qconfig_mapping.items():
    #         qconfig_mapping[key] = value.cuda()
    # model_prepared = quantize_fx.prepare_fx(m.eval(), qconfig_mapping, example_input)
    model_quantized = quantize_dynamic(model, dtype=torch.float16)
    # model_quantized = quantization.quantize_static(model, weight_precision=16)
    # if calibrate:
    #     co_transform_val = MyCoTransform(False, augment=False, height=512)
    #     dataset_val = cityscapes(args.datadir, co_transform_val, 'val')
    #     loader_val = DataLoader(dataset_val, num_workers=4, batch_size=args.batch_size, shuffle=False)
    #     print(len(loader_val))
    #     with torch.inference_mode():
    #         for step, (images, _) in enumerate(loader_val ):
    #             print(f"step {step + 1}")
    #             print(images.shape)
    #             if not args.cpu:
    #                 images = images.cuda()
    #             images = Variable(images)
    #             model_prepared(images)
    # model_quantized = quantize_fx.convert_fx(model_prepared)
    return model_quantized

def main(args, calibrate=True):
    savedir = f"{args.savedir}/quantized_model.pth"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    model = ERFNet(20)
    model = load_my_state_dict(model, torch.load(f"../trained_models/{args.model}_pretrained.pth", map_location=torch.device('cpu')))
    for name, _  in model.state_dict().items():
        print(name)
    flops = profile_macs(model, torch.randn(1, 3, 512, 1024))
    print(f"FLOPS initial model: {flops / 10**9:.2f} GFLOPS")
    total_params = sum(p.numel() for p in model.parameters() if torch.any(p != 0))
    print(total_params)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()
    qmodel = quantize_model(model, args, calibrate)
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
    
    model = main(args, calibrate=False)  # first execution initializes the model
    total_params = sum(p.numel() for p in model.parameters() if torch.any(p != 0))
    print(f"Total Params: {total_params}")  
    # if not args.resume:  #second execution (if needed) calibrates the model
    #     main(parser.parse_args(), calibrate=True)

    dict = torch.load("quantized_model.pth", map_location=lambda storage, loc: storage)
    # model = load_quant_dict(model, dict)
    model.load_state_dict(dict)
    file1 = open("file.txt", "a")
    for name,  params in model.state_dict().items():
        file1.write(f"{name}: {params}\n")
    input = torch.randn(1, 3, 512, 1024)
    print(input.shape)
    summary(model, input_size=(3, 512, 1024))
    # for p in model.parameters():
    #     print(p)
    flops = profile_macs(model, input)
    print(f"FLOPS: {flops / 10**9:.2f} GFLOPS")
    